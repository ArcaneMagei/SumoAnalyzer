"""Robot Sumo Training Studio (interactive)

A practical, local-first workflow:
1) Trim short action clips from long videos
2) Extract frames for labeling
3) Annotate 2 robots + front/blade direction quickly
4) Save normalized annotations for training export

This is intentionally OpenCV-based so it runs without web setup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple


def require_cv2():
    try:
        import cv2

        return cv2
    except Exception as e:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from e


@dataclass
class RobotLabel:
    robot_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]
    blade_front_xy: Tuple[int, int]
    visible: bool


@dataclass
class FrameLabel:
    image_name: str
    width: int
    height: int
    robots: List[RobotLabel]


def ensure_dirs(base: Path) -> None:
    for p in [
        base / "clips",
        base / "frames",
        base / "annotations",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def trim_video(video: Path, out_clip: Path) -> int:
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Could not open: {video}")
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        print("Video has no frames")
        cap.release()
        return 3

    start_f = 0
    end_f = total - 1
    cur = 0

    cv2.namedWindow("Trim Studio", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", "Trim Studio", 0, max(1, total - 1), lambda x: None)

    print("Trim controls:")
    print("  j/l = -/+ 1 frame, a/d = -/+ 15 frames")
    print("  i = set IN, o = set OUT")
    print("  s = save clip, q = quit")

    while True:
        cv2.setTrackbarPos("frame", "Trim Studio", int(cur))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cur))
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        cv2.putText(vis, f"frame={cur}/{total-1}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"IN={start_f} OUT={end_f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Trim Studio", vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return 0
        if k == ord("j"):
            cur = max(0, cur - 1)
        elif k == ord("l"):
            cur = min(total - 1, cur + 1)
        elif k == ord("a"):
            cur = max(0, cur - 15)
        elif k == ord("d"):
            cur = min(total - 1, cur + 15)
        elif k == ord("i"):
            start_f = min(cur, end_f)
        elif k == ord("o"):
            end_f = max(cur, start_f)
        elif k == ord("s"):
            break
        else:
            tb = cv2.getTrackbarPos("frame", "Trim Studio")
            cur = int(tb)

    out_clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
    for _ in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved clip: {out_clip} ({start_f}..{end_f})")
    return 0


def extract_frames(video: Path, out_dir: Path, fps_out: float = 8.0) -> int:
    cv2 = require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Could not open: {video}")
        return 2

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps_in / max(0.1, fps_out))))

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = out_dir / f"{video.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(p), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")
    return 0


def xyxy_to_norm(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def annotate_frames(frames_dir: Path, out_json_dir: Path, start_index: int = 0) -> int:
    cv2 = require_cv2()

    imgs = sorted([*frames_dir.glob("*.jpg"), *frames_dir.glob("*.png"), *frames_dir.glob("*.jpeg")])
    if not imgs:
        print(f"No frames in {frames_dir}")
        return 2

    out_json_dir.mkdir(parents=True, exist_ok=True)

    print("Annotation controls:")
    print("  For each robot (R1 then R2):")
    print("    - draw bbox with ROI selector, Enter confirms, ESC skips")
    print("    - click blade/front point, press any key")
    print("  Review keys: n=save+next, r=redo frame, q=quit")

    for i, img_path in enumerate(imgs[start_index:], start=start_index):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        labels: List[RobotLabel] = []

        for rid in [1, 2]:
            roi = cv2.selectROI(f"R{rid} bbox (Enter confirm, ESC skip)", frame, fromCenter=False, showCrosshair=True)
            x, y, bw, bh = [int(v) for v in roi]
            if bw <= 0 or bh <= 0:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            blade = [center[0], center[1]]
            done = {"ok": False}

            panel = frame.copy()
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(panel, f"R{rid}: click blade/front point", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            def on_mouse(event, mx, my, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    blade[0], blade[1] = int(mx), int(my)
                    done["ok"] = True

            cv2.namedWindow("Blade point picker", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Blade point picker", on_mouse)

            while True:
                vis = panel.copy()
                cv2.circle(vis, (blade[0], blade[1]), 4, (255, 255, 0), -1)
                cv2.line(vis, center, (blade[0], blade[1]), (255, 255, 0), 2)
                cv2.imshow("Blade point picker", vis)
                key = cv2.waitKey(20) & 0xFF
                if done["ok"] and key != 255:
                    break

            labels.append(
                RobotLabel(
                    robot_id=rid,
                    class_name="robot",
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=center,
                    blade_front_xy=(blade[0], blade[1]),
                    visible=True,
                )
            )
            cv2.destroyWindow("Blade point picker")

        review = frame.copy()
        for lb in labels:
            x1, y1, x2, y2 = lb.bbox_xyxy
            cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(review, lb.blade_front_xy, 4, (255, 255, 0), -1)
            cv2.line(review, lb.center_xy, lb.blade_front_xy, (255, 255, 0), 2)
            cv2.putText(review, f"R{lb.robot_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Review (n/r/q)", review)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            continue

        rec = FrameLabel(image_name=img_path.name, width=w, height=h, robots=labels)
        outp = out_json_dir / f"{img_path.stem}.json"
        outp.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8")

        print(f"[{i+1}/{len(imgs)}] saved {outp.name}")

    cv2.destroyAllWindows()
    return 0


def export_yolo_from_json(frames_dir: Path, json_dir: Path, labels_out: Path) -> int:
    """Export labels for training:
    class 0 = robot bbox
    class 1 = blade_front point as tiny bbox
    """
    labels_out.mkdir(parents=True, exist_ok=True)
    items = sorted(json_dir.glob("*.json"))
    if not items:
        print(f"No annotation json files in {json_dir}")
        return 2

    exported = 0
    for jp in items:
        rec = json.loads(jp.read_text(encoding="utf-8"))
        w, h = int(rec["width"]), int(rec["height"])
        lines = []

        for rb in rec.get("robots", []):
            x1, y1, x2, y2 = rb["bbox_xyxy"]
            cx, cy, bw, bh = xyxy_to_norm(int(x1), int(y1), int(x2), int(y2), w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            fx, fy = rb["blade_front_xy"]
            side = max(4, int(min(w, h) * 0.02))
            bx1, by1 = max(0, int(fx - side // 2)), max(0, int(fy - side // 2))
            bx2, by2 = min(w - 1, bx1 + side), min(h - 1, by1 + side)
            fcx, fcy, fbw, fbh = xyxy_to_norm(bx1, by1, bx2, by2, w, h)
            lines.append(f"1 {fcx:.6f} {fcy:.6f} {fbw:.6f} {fbh:.6f}")

        out_txt = labels_out / f"{jp.stem}.txt"
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported += 1

    print(f"Exported YOLO labels: {exported} files -> {labels_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Robot Sumo Training Studio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_trim = sub.add_parser("trim", help="trim action clip from long video")
    p_trim.add_argument("--video", required=True)
    p_trim.add_argument("--out", required=True)

    p_extract = sub.add_parser("extract", help="extract frames from action clip")
    p_extract.add_argument("--video", required=True)
    p_extract.add_argument("--out-dir", required=True)
    p_extract.add_argument("--fps", type=float, default=8.0)

    p_annot = sub.add_parser("annotate", help="annotate extracted frames")
    p_annot.add_argument("--frames-dir", required=True)
    p_annot.add_argument("--out-json-dir", required=True)
    p_annot.add_argument("--start-index", type=int, default=0)

    p_export = sub.add_parser("export-yolo", help="export YOLO txt labels from json annotations")
    p_export.add_argument("--frames-dir", required=True)
    p_export.add_argument("--json-dir", required=True)
    p_export.add_argument("--labels-out", required=True)

    args = parser.parse_args()

    if args.cmd == "trim":
        return trim_video(Path(args.video), Path(args.out))
    if args.cmd == "extract":
        return extract_frames(Path(args.video), Path(args.out_dir), fps_out=args.fps)
    if args.cmd == "annotate":
        return annotate_frames(Path(args.frames_dir), Path(args.out_json_dir), start_index=args.start_index)
    if args.cmd == "export-yolo":
        return export_yolo_from_json(Path(args.frames_dir), Path(args.json_dir), Path(args.labels_out))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
