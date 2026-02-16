"""Robot Sumo Training Studio (all-in-one local app)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional


def require_cv2():
    try:
        import cv2

        return cv2
    except Exception as e:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from e


def resolve_video_path(video_arg: str, search_roots: Optional[List[Path]] = None) -> Path:
    """Resolve video path robustly for Windows CLI usage.

    Handles relative paths, quoted paths, and searches common project folders.
    """
    raw = video_arg.strip().strip('"').strip("'")
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                Path.cwd() / p,
                Path.cwd() / "data" / "studio" / "raw_videos" / p.name,
                Path.cwd() / "data" / "studio" / "clips" / p.name,
                Path.cwd() / p.name,
            ]
        )
        if search_roots:
            for root in search_roots:
                candidates.append(root / p.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find video. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


def ensure_dirs(base: Path) -> None:
    for p in [base / "clips", base / "frames", base / "annotations", base / "exports", base / "raw_videos"]:
        p.mkdir(parents=True, exist_ok=True)


def open_video_capture(video: Path):
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        return cap, None

    msg = [
        f"OpenCV failed to open video: {video}",
        "Possible causes:",
        "  - wrong path/working directory",
        "  - MOV codec unsupported in this OpenCV build",
        "  - corrupted video",
    ]
    ffmpeg = shutil_which("ffmpeg")
    if ffmpeg:
        msg.append("Try transcoding first:")
        msg.append(f"  ffmpeg -y -i \"{video}\" -c:v libx264 -pix_fmt yuv420p -c:a aac \"{video.with_suffix('.mp4')}\"")
    return cap, "\n".join(msg)


def shutil_which(cmd: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""]
    if os.name == "nt":
        exts += [".exe", ".bat", ".cmd"]
    for d in paths:
        for e in exts:
            p = Path(d) / f"{cmd}{e}"
            if p.exists() and p.is_file():
                return str(p)
    return None


@dataclass
class RobotLabel:
    robot_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]
    blade_left_xy: Tuple[int, int]
    blade_right_xy: Tuple[int, int]
    blade_mid_xy: Tuple[int, int]
    heading_deg: float
    blade_width_px: float
    visible: bool


@dataclass
class FrameLabel:
    image_name: str
    width: int
    height: int
    robots: List[RobotLabel]


def _heading_deg(center: Tuple[int, int], front: Tuple[int, int]) -> float:
    import math

    return float(math.degrees(math.atan2(front[1] - center[1], front[0] - center[0])))


def trim_video(video: Path, out_clip: Path) -> int:
    cv2 = require_cv2()
    cap, err = open_video_capture(video)
    if not cap.isOpened():
        print(err)
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        print("Video has no frames")
        cap.release()
        return 3

    start_f, end_f, cur = 0, total - 1, 0
    cv2.namedWindow("Trim Studio", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", "Trim Studio", 0, max(1, total - 1), lambda x: None)

    print("Trim controls: j/l +/-1, a/d +/-15, i=set IN, o=set OUT, s=save, q=quit")

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
            cur = int(cv2.getTrackbarPos("frame", "Trim Studio"))

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

    cap, err = open_video_capture(video)
    if not cap.isOpened():
        print(err)
        return 2

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps_in / max(0.1, fps_out))))

    idx, saved = 0, 0
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


def _point_picker(base_img, title: str, color=(255, 255, 0)):
    cv2 = require_cv2()
    pt = [0, 0]
    clicked = {"ok": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[0], pt[1] = int(x), int(y)
            clicked["ok"] = True

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        vis = base_img.copy()
        if clicked["ok"]:
            cv2.circle(vis, (pt[0], pt[1]), 4, color, -1)
        cv2.imshow(title, vis)
        k = cv2.waitKey(20) & 0xFF
        if clicked["ok"] and k != 255:
            break

    cv2.destroyWindow(title)
    return int(pt[0]), int(pt[1])


def annotate_frames(frames_dir: Path, out_json_dir: Path, start_index: int = 0) -> int:
    cv2 = require_cv2()
    imgs = sorted([*frames_dir.glob("*.jpg"), *frames_dir.glob("*.png"), *frames_dir.glob("*.jpeg")])
    if not imgs:
        print(f"No frames in {frames_dir}")
        return 2

    out_json_dir.mkdir(parents=True, exist_ok=True)
    print("Annotate controls: ROI Enter confirm / ESC skip; review n=save r=redo q=quit")
    print("For each robot you will pick: bbox -> blade LEFT point -> blade RIGHT point")

    for i, img_path in enumerate(imgs[start_index:], start=start_index):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        labels: List[RobotLabel] = []
        for rid in [1, 2]:
            roi = cv2.selectROI(f"R{rid} bbox", frame, fromCenter=False, showCrosshair=True)
            x, y, bw, bh = [int(v) for v in roi]
            if bw <= 0 or bh <= 0:
                continue

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            panel = frame.copy()
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(panel, f"R{rid}: click BLADE LEFT then key", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            bl = _point_picker(panel, "Blade-left picker", color=(255, 200, 0))

            panel2 = frame.copy()
            cv2.rectangle(panel2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(panel2, bl, 4, (255, 200, 0), -1)
            cv2.putText(panel2, f"R{rid}: click BLADE RIGHT then key", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            br = _point_picker(panel2, "Blade-right picker", color=(0, 255, 255))

            mid = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)
            heading = _heading_deg(center, mid)
            blade_width = float(((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5)

            labels.append(
                RobotLabel(
                    robot_id=rid,
                    class_name="robot",
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=center,
                    blade_left_xy=bl,
                    blade_right_xy=br,
                    blade_mid_xy=mid,
                    heading_deg=heading,
                    blade_width_px=blade_width,
                    visible=True,
                )
            )

        review = frame.copy()
        for lb in labels:
            x1, y1, x2, y2 = lb.bbox_xyxy
            cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(review, lb.blade_left_xy, 4, (255, 200, 0), -1)
            cv2.circle(review, lb.blade_right_xy, 4, (0, 255, 255), -1)
            cv2.line(review, lb.blade_left_xy, lb.blade_right_xy, (255, 255, 0), 2)
            cv2.line(review, lb.center_xy, lb.blade_mid_xy, (255, 255, 0), 2)
            cv2.putText(review, f"R{lb.robot_id} {lb.heading_deg:.1f}deg", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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


def xyxy_to_norm(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def export_yolo_from_json(frames_dir: Path, json_dir: Path, labels_out: Path, blade_box_thickness_px: int = 8) -> int:
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

            blx, bly = rb["blade_left_xy"]
            brx, bry = rb["blade_right_xy"]
            bx1 = max(0, min(int(blx), int(brx)))
            bx2 = min(w - 1, max(int(blx), int(brx)))
            by_mid = int((int(bly) + int(bry)) / 2)
            by1 = max(0, by_mid - blade_box_thickness_px // 2)
            by2 = min(h - 1, by_mid + blade_box_thickness_px // 2)
            fcx, fcy, fbw, fbh = xyxy_to_norm(bx1, by1, bx2, by2, w, h)
            lines.append(f"1 {fcx:.6f} {fcy:.6f} {fbw:.6f} {fbh:.6f}")

        out_txt = labels_out / f"{jp.stem}.txt"
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported += 1

    print(f"Exported YOLO labels: {exported} files -> {labels_out}")
    return 0


def run_training(dataset_dir: Path, model: str, epochs: int, imgsz: int, batch: int, device: str, workers: int) -> int:
    cmd = [
        sys.executable,
        "scripts/train_robot_detector.py",
        "--dataset-dir",
        str(dataset_dir),
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--device",
        device,
        "--workers",
        str(workers),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def infer_video(weights: Path, video: Path, out_video: Path, conf: float = 0.2, imgsz: int = 960) -> int:
    cv2 = require_cv2()
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"Ultralytics missing: {e}")
        return 2

    model = YOLO(str(weights))
    cap, err = open_video_capture(video)
    if not cap.isOpened():
        print(err)
        return 3

    ok, frame = cap.read()
    if not ok:
        print("Could not open video")
        return 3

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False, max_det=6)
        robots = []
        blades = []
        if res and len(res[0].boxes) > 0:
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cls = int(b.cls[0]) if b.cls is not None else -1
                cf = float(b.conf[0])
                if cls == 0:
                    robots.append((x1, y1, x2, y2, cf))
                elif cls == 1:
                    blades.append((x1, y1, x2, y2, cf))

        for x1, y1, x2, y2, cf in robots:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            nearest = None
            best_d = 1e12
            for bx1, by1, bx2, by2, bcf in blades:
                bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                d = (bcx - cx) ** 2 + (bcy - cy) ** 2
                if d < best_d:
                    best_d = d
                    nearest = (bcx, bcy)

            if nearest is not None:
                cv2.circle(frame, nearest, 4, (255, 255, 0), -1)
                cv2.line(frame, (cx, cy), nearest, (255, 255, 0), 2)

            cv2.putText(frame, f"robot {cf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for bx1, by1, bx2, by2, bcf in blades:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
            cv2.putText(frame, f"blade {bcf:.2f}", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote inference preview: {out_video}")
    return 0


def doctor() -> int:
    """Quick environment and path diagnostics for local setup."""
    print("== Training Studio Doctor ==")
    print(f"Python: {sys.version.split()[0]}")

    try:
        import cv2  # noqa: F401
        print("OpenCV: OK")
    except Exception as e:
        print(f"OpenCV: MISSING ({e})")

    try:
        import ultralytics  # noqa: F401
        print("Ultralytics: OK")
    except Exception as e:
        print(f"Ultralytics: MISSING ({e})")

    ff = shutil_which("ffmpeg")
    print(f"FFmpeg: {'OK' if ff else 'MISSING'}{f' ({ff})' if ff else ''}")

    for d in [Path('data/studio/raw_videos'), Path('data/studio/clips'), Path('data/studio/frames'), Path('data/studio/annotations'), Path('artifacts')]:
        d.mkdir(parents=True, exist_ok=True)
        ok = os.access(d, os.W_OK)
        print(f"Writable dir: {d} -> {'OK' if ok else 'NO'}")

    print("Doctor finished.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Robot Sumo Training Studio (all-in-one)")
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
    p_export.add_argument("--blade-box-thickness-px", type=int, default=8)

    p_train = sub.add_parser("train", help="launch robust training wrapper")
    p_train.add_argument("--dataset-dir", default="data/robot_dataset")
    p_train.add_argument("--model", default="yolov8n.pt")
    p_train.add_argument("--epochs", type=int, default=120)
    p_train.add_argument("--imgsz", type=int, default=960)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--workers", type=int, default=0)

    p_infer = sub.add_parser("infer-video", help="run trained model on test video")
    p_infer.add_argument("--weights", default="models/weights/robot_sumo.pt")
    p_infer.add_argument("--video", required=True)
    p_infer.add_argument("--out", default="artifacts/training_studio_infer.mp4")
    p_infer.add_argument("--conf", type=float, default=0.2)
    p_infer.add_argument("--imgsz", type=int, default=960)

    sub.add_parser("doctor", help="check dependencies/paths/local environment")

    args = parser.parse_args()

    try:
        if args.cmd == "trim":
            v = resolve_video_path(args.video)
            return trim_video(v, Path(args.out))
        if args.cmd == "extract":
            v = resolve_video_path(args.video)
            return extract_frames(v, Path(args.out_dir), fps_out=args.fps)
        if args.cmd == "annotate":
            return annotate_frames(Path(args.frames_dir), Path(args.out_json_dir), start_index=args.start_index)
        if args.cmd == "export-yolo":
            return export_yolo_from_json(
                Path(args.frames_dir), Path(args.json_dir), Path(args.labels_out), blade_box_thickness_px=args.blade_box_thickness_px
            )
        if args.cmd == "train":
            return run_training(Path(args.dataset_dir), args.model, args.epochs, args.imgsz, args.batch, args.device, args.workers)
        if args.cmd == "infer-video":
            v = resolve_video_path(args.video)
            return infer_video(Path(args.weights), v, Path(args.out), conf=args.conf, imgsz=args.imgsz)
        if args.cmd == "doctor":
            return doctor()
    except FileNotFoundError as e:
        print(e)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
