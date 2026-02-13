"""Interactive dataset preparation + manual annotation tool for robot sumo.

This script is designed for non-technical users:
1) Sample frames from videos into YOLO dataset folders
2) Annotate robot bbox + blade/front orientation with simple keyboard/mouse steps

Output:
- YOLO labels (class 0=robot, class 1=blade_front)
- metadata JSON per image with robot IDs + orientation vectors
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class RobotAnnotation:
    robot_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]
    front_xy: Tuple[int, int]
    heading_deg: float


def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV is required for this script. Install with: pip install opencv-python"
        ) from e


def _ensure_dataset(out_dir: Path) -> None:
    for p in [
        out_dir / "images" / "train",
        out_dir / "images" / "val",
        out_dir / "labels" / "train",
        out_dir / "labels" / "val",
        out_dir / "meta" / "train",
        out_dir / "meta" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _sample_frame_indices(total: int, wanted: int, seed: int) -> List[int]:
    wanted = min(max(1, wanted), max(1, total))
    rnd = random.Random(seed)

    # Mix uniform coverage + random sampling for diversity.
    uniform = [int(i * (total - 1) / max(1, wanted - 1)) for i in range(wanted)]
    jittered = []
    for idx in uniform:
        j = idx + rnd.randint(-3, 3)
        jittered.append(min(total - 1, max(0, j)))

    return sorted(set(jittered))


def prepare_frames(videos: List[Path], out_dir: Path, frames_per_video: int, val_ratio: float, seed: int) -> int:
    cv2 = _require_cv2()
    _ensure_dataset(out_dir)

    total_saved = 0
    rnd = random.Random(seed)

    for v in videos:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            continue

        indices = _sample_frame_indices(total, frames_per_video, seed + hash(v.name) % 10000)

        for i, fidx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok:
                continue

            split = "val" if rnd.random() < val_ratio else "train"
            stem = f"{v.stem}_f{fidx:06d}"
            img_path = out_dir / "images" / split / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            total_saved += 1

        cap.release()

    return total_saved


def _heading_deg(center: Tuple[int, int], front: Tuple[int, int]) -> float:
    import math

    return float(math.degrees(math.atan2(front[1] - center[1], front[0] - center[0])))


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def _front_point_box(px: int, py: int, w: int, h: int, size_ratio: float = 0.02) -> Tuple[int, int, int, int]:
    side = max(4, int(min(w, h) * size_ratio))
    x1 = max(0, px - side // 2)
    y1 = max(0, py - side // 2)
    x2 = min(w - 1, x1 + side)
    y2 = min(h - 1, y1 + side)
    return x1, y1, x2, y2


def annotate_split(out_dir: Path, split: str, start_index: int = 0) -> int:
    cv2 = _require_cv2()

    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    meta_dir = out_dir / "meta" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.glob("*.jpg")])
    if not images:
        print(f"No images found in {img_dir}")
        return 0

    print("\nAnnotation controls:")
    print("  - For each robot: ROI selector appears (drag + Enter). ESC cancels robot")
    print("  - Then click FRONT point in preview window, press any key to confirm")
    print("  - Keys in main window: n=save+next, r=redo frame, q=quit\n")

    saved = 0
    for idx, img_path in enumerate(images[start_index:], start=start_index):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        ann: List[RobotAnnotation] = []

        # annotate up to 2 robots per frame
        for rid in [1, 2]:
            roi = cv2.selectROI("Select robot bbox (Enter confirm, ESC skip)", frame, fromCenter=False, showCrosshair=True)
            x, y, bw, bh = [int(v) for v in roi]
            if bw <= 0 or bh <= 0:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            front_pt = [center[0], center[1]]
            clicked = {"ok": False}

            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, f"R{rid}: click FRONT point", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            def on_mouse(event, mx, my, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    front_pt[0], front_pt[1] = int(mx), int(my)
                    clicked["ok"] = True

            cv2.namedWindow("Front point picker", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Front point picker", on_mouse)

            while True:
                vis = preview.copy()
                cv2.circle(vis, (front_pt[0], front_pt[1]), 4, (255, 255, 0), -1)
                cv2.line(vis, center, (front_pt[0], front_pt[1]), (255, 255, 0), 2)
                cv2.imshow("Front point picker", vis)
                k = cv2.waitKey(20) & 0xFF
                if clicked["ok"] and k != 255:
                    break

            heading = _heading_deg(center, (front_pt[0], front_pt[1]))
            ann.append(
                RobotAnnotation(
                    robot_id=rid,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=center,
                    front_xy=(front_pt[0], front_pt[1]),
                    heading_deg=heading,
                )
            )

            cv2.destroyWindow("Front point picker")

        # review overlay
        review = frame.copy()
        for a in ann:
            x1, y1, x2, y2 = a.bbox_xyxy
            cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(review, a.front_xy, 4, (255, 255, 0), -1)
            cv2.line(review, a.center_xy, a.front_xy, (255, 255, 0), 2)
            cv2.putText(review, f"R{a.robot_id} {a.heading_deg:.1f}deg", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame review (n=save, r=redo, q=quit)", review)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            continue

        # Save labels + metadata
        label_lines = []
        for a in ann:
            x1, y1, x2, y2 = a.bbox_xyxy
            cx, cy, bw, bh = _xyxy_to_yolo(x1, y1, x2, y2, w, h)
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            fx1, fy1, fx2, fy2 = _front_point_box(a.front_xy[0], a.front_xy[1], w, h)
            fcx, fcy, fbw, fbh = _xyxy_to_yolo(fx1, fy1, fx2, fy2, w, h)
            label_lines.append(f"1 {fcx:.6f} {fcy:.6f} {fbw:.6f} {fbh:.6f}")

        label_path = lbl_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        meta_path = meta_dir / f"{img_path.stem}.json"
        payload = {
            "image": str(img_path.name),
            "width": w,
            "height": h,
            "annotations": [asdict(a) for a in ann],
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        saved += 1
        print(f"[{idx+1}/{len(images)}] saved {img_path.name}")

    cv2.destroyAllWindows()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare + annotate robot-sumo dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="sample frames from videos into dataset structure")
    p_prepare.add_argument("--videos", nargs="+", required=True, help="video file paths")
    p_prepare.add_argument("--out-dir", default="data/robot_dataset")
    p_prepare.add_argument("--frames-per-video", type=int, default=120)
    p_prepare.add_argument("--val-ratio", type=float, default=0.2)
    p_prepare.add_argument("--seed", type=int, default=42)

    p_annot = sub.add_parser("annotate", help="manually annotate prepared frames")
    p_annot.add_argument("--out-dir", default="data/robot_dataset")
    p_annot.add_argument("--split", choices=["train", "val"], default="train")
    p_annot.add_argument("--start-index", type=int, default=0)

    args = parser.parse_args()

    if args.cmd == "prepare":
        videos = [Path(v) for v in args.videos]
        out_dir = Path(args.out_dir)
        saved = prepare_frames(videos, out_dir, args.frames_per_video, args.val_ratio, args.seed)
        print(f"Prepared {saved} frames in {out_dir}")
        return 0

    if args.cmd == "annotate":
        out_dir = Path(args.out_dir)
        saved = annotate_split(out_dir, args.split, start_index=args.start_index)
        print(f"Annotated {saved} frames for split={args.split}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
