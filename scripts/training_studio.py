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


def _named_window(name: str):
    cv2 = require_cv2()
    flags = cv2.WINDOW_AUTOSIZE
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(name, flags)


@dataclass
class RobotLabel:
    robot_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]
    blade_left_xy: Optional[Tuple[int, int]] = None
    blade_right_xy: Optional[Tuple[int, int]] = None
    blade_mid_xy: Optional[Tuple[int, int]] = None
    heading_deg: float = 0.0
    blade_width_px: float = 0.0
    extension_type: str = "none"
    extension_side: str = "none"
    visible: bool = True


@dataclass
class FrameLabel:
    image_name: str
    width: int
    height: int
    robots: List[RobotLabel]
    dohyo_ellipse: Optional[Tuple[float, float, float, float, float]] = None  # cx,cy,axis_w,axis_h,angle


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

    trim_meta = out_clip.with_suffix(out_clip.suffix + ".trim.json")
    start_f, end_f = 0, total - 1
    if trim_meta.exists():
        try:
            meta = json.loads(trim_meta.read_text(encoding="utf-8"))
            start_f = int(meta.get("start_f", start_f))
            end_f = int(meta.get("end_f", end_f))
            start_f = max(0, min(total - 1, start_f))
            end_f = max(start_f, min(total - 1, end_f))
            print(f"Loaded previous trim: IN={start_f} OUT={end_f}")
        except Exception:
            pass

    cur = start_f
    _named_window("Trim Studio")
    cv2.createTrackbar("frame", "Trim Studio", int(cur), max(1, total - 1), lambda x: None)

    print("Trim controls: j/l +/-1, a/d +/-15, i=set IN, o=set OUT, s=save, q=quit")
    saved = False

    while True:
        if cv2.getWindowProperty("Trim Studio", cv2.WND_PROP_VISIBLE) < 1:
            break

        cv2.setTrackbarPos("frame", "Trim Studio", int(cur))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cur))
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        cv2.putText(vis, f"frame={cur}/{total-1}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"IN={start_f} OUT={end_f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, "j/l:+-1  a/d:+-15  i:set IN  o:set OUT  s:save  q:quit", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
        cv2.imshow("Trim Studio", vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
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
            saved = True
            break

    if saved:
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
        for _ in range(start_f, end_f + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()

        trim_meta.write_text(json.dumps({"start_f": int(start_f), "end_f": int(end_f), "total": int(total)}, indent=2), encoding="utf-8")
        print(f"Saved clip: {out_clip} ({start_f}..{end_f})")
    else:
        print("Trim canceled (no new save).")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def extract_frames(video: Path, out_dir: Path, fps_out: float = 8.0, clear_existing: bool = True) -> int:
    cv2 = require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        removed = 0
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            for f in out_dir.glob(pat):
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            print(f"Cleared {removed} existing frames from {out_dir}")

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


def _overlay_lines(img, lines: List[str], origin=(20, 30), line_h: int = 28):
    cv2 = require_cv2()
    x, y = origin
    for ln in lines:
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        y += line_h


def _point_picker(
    base_img,
    title: str,
    color=(255, 255, 0),
    prompt: Optional[str] = None,
    frame_progress: Optional[str] = None,
):
    cv2 = require_cv2()
    pt = [0, 0]
    clicked = {"ok": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[0], pt[1] = int(x), int(y)
            clicked["ok"] = True

    _named_window(title)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        vis = base_img.copy()
        lines = []
        if frame_progress:
            lines.append(frame_progress)
        if prompt:
            lines.append(prompt)
        lines.append("Mouse: left click to place point | Enter/Space: confirm | R: reset")
        _overlay_lines(vis, lines)
        if clicked["ok"]:
            cv2.circle(vis, (pt[0], pt[1]), 4, color, -1)
        cv2.imshow(title, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("r"), ord("R")):
            clicked["ok"] = False
        if clicked["ok"] and k in (13, 32):
            break

    cv2.destroyWindow(title)
    return int(pt[0]), int(pt[1])


def _line_editor(
    base_img,
    title: str,
    left_pt: Tuple[int, int],
    right_pt: Tuple[int, int],
    frame_progress: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Adjust a two-point line by dragging endpoints for faster/blind blade labeling."""
    cv2 = require_cv2()
    left = [int(left_pt[0]), int(left_pt[1])]
    right = [int(right_pt[0]), int(right_pt[1])]
    active = {"name": None}

    def _dist2(x: int, y: int, p: List[int]) -> int:
        return (x - p[0]) ** 2 + (y - p[1]) ** 2

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if _dist2(x, y, left) <= _dist2(x, y, right):
                active["name"] = "left"
            else:
                active["name"] = "right"
        elif event == cv2.EVENT_MOUSEMOVE and active["name"] is not None:
            if active["name"] == "left":
                left[0], left[1] = int(x), int(y)
            else:
                right[0], right[1] = int(x), int(y)
        elif event == cv2.EVENT_LBUTTONUP:
            active["name"] = None

    _named_window(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        vis = base_img.copy()
        lines = []
        if frame_progress:
            lines.append(frame_progress)
        if prompt:
            lines.append(prompt)
        lines.append("Drag closest endpoint to adjust blade line | Enter/Space: confirm")
        _overlay_lines(vis, lines)
        cv2.circle(vis, (left[0], left[1]), 5, (255, 200, 0), -1)
        cv2.circle(vis, (right[0], right[1]), 5, (0, 255, 255), -1)
        cv2.line(vis, (left[0], left[1]), (right[0], right[1]), (255, 255, 0), 2)
        cv2.imshow(title, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):
            break

    cv2.destroyWindow(title)
    return (left[0], left[1]), (right[0], right[1])


def _extension_boxes(rb: dict, w: int, h: int) -> List[Tuple[int, int, int, int, int]]:
    """Generate YOLO extension boxes.

    Returns (cls_id, x1, y1, x2, y2)
    class ids:
      2: flag_left, 3: flag_right, 4: flag_both,
      5: blade_ext_left, 6: blade_ext_right, 7: blade_ext_both
    """
    x1, y1, x2, y2 = [int(v) for v in rb["bbox_xyxy"]]
    ext_type = str(rb.get("extension_type", "none")).lower()
    ext_side = str(rb.get("extension_side", "none")).lower()
    if ext_type not in {"flag", "blade"} or ext_side not in {"left", "right", "both"}:
        return []

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side_w = max(2, int(round(bw * 0.2)))

    if ext_type == "flag":
        y_top = max(0, y1 - int(round(bh * 0.5)))
        y_bot = min(h - 1, y1 + int(round(bh * 0.2)))
        cls_map = {"left": 2, "right": 3, "both": 4}
    else:
        y_top = max(0, y1 + int(round(bh * 0.35)))
        y_bot = min(h - 1, y2)
        cls_map = {"left": 5, "right": 6, "both": 7}

    boxes: List[Tuple[int, int, int, int, int]] = []
    if ext_side in {"left", "both"}:
        lx1 = max(0, x1 - side_w)
        lx2 = min(w - 1, x1 + side_w)
        boxes.append((cls_map[ext_side], lx1, y_top, lx2, y_bot))
    if ext_side in {"right", "both"}:
        rx1 = max(0, x2 - side_w)
        rx2 = min(w - 1, x2 + side_w)
        boxes.append((cls_map[ext_side], rx1, y_top, rx2, y_bot))
    return boxes




def _pick_dohyo_ellipse(
    frame,
    window_name: str,
    frame_progress: str,
    prev_ellipse: Optional[Tuple[float, float, float, float, float]] = None,
    auto_ellipse: Optional[Tuple[float, float, float, float, float]] = None,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Pick/adjust dohyo ellipse with auto+prev options."""
    cv2 = require_cv2()

    vis = frame.copy()
    if auto_ellipse is not None:
        cx, cy, aw, ah, ang = auto_ellipse
        cv2.ellipse(vis, (int(cx), int(cy)), (max(1, int(aw / 2)), max(1, int(ah / 2))), float(ang), 0, 360, (80, 255, 80), 2)
    if prev_ellipse is not None:
        cx, cy, aw, ah, ang = prev_ellipse
        cv2.ellipse(vis, (int(cx), int(cy)), (max(1, int(aw / 2)), max(1, int(ah / 2))), float(ang), 0, 360, (200, 255, 0), 2)

    _overlay_lines(
        vis,
        [
            frame_progress,
            "Dohyo: A=accept auto | G=use previous | Enter=draw ROI | E=edit current",
        ],
    )
    cv2.imshow(window_name, vis)
    k = cv2.waitKey(0) & 0xFF

    if k in (ord('a'), ord('A')) and auto_ellipse is not None:
        return auto_ellipse
    if k in (ord('g'), ord('G')) and prev_ellipse is not None:
        return prev_ellipse

    base = auto_ellipse if auto_ellipse is not None else prev_ellipse
    if base is not None and k in (ord('e'), ord('E')):
        cx, cy, aw, ah, ang = base
        x = max(0, int(cx - aw / 2))
        y = max(0, int(cy - ah / 2))
        w = int(aw)
        h = int(ah)
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        rx, ry, rw, rh = [int(v) for v in roi]
        if rw > 0 and rh > 0:
            x, y, w, h = rx, ry, rw, rh
        return (float(x + w / 2.0), float(y + h / 2.0), float(w), float(h), float(ang))

    roi_vis = frame.copy()
    _overlay_lines(roi_vis, [frame_progress, "Dohyo: draw ROI around ring (Enter confirm, ESC keep previous/auto)"])
    roi = cv2.selectROI(window_name, roi_vis, fromCenter=False, showCrosshair=True)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return auto_ellipse if auto_ellipse is not None else prev_ellipse
    return (float(x + w / 2.0), float(y + h / 2.0), float(w), float(h), 0.0)


def _find_robot_by_id(labels: List[RobotLabel], rid: int) -> Optional[RobotLabel]:
    for lb in labels:
        if lb.robot_id == rid:
            return lb
    return None


def _annotate_one_robot(
    frame,
    frame_progress: str,
    window_name: str,
    rid: int,
    annotate_blade: bool,
    prev_robot: Optional[RobotLabel] = None,
    ai_robot: Optional[RobotLabel] = None,
) -> Optional[RobotLabel]:
    cv2 = require_cv2()
    h, w = frame.shape[:2]

    prompt = frame.copy()
    lines = [frame_progress, f"Robot {rid}/2: Enter=manual bbox"]
    if prev_robot is not None:
        lines.append("G=use previous robot annotation")
    if ai_robot is not None:
        lines.append("A=use AI prefill")
    _overlay_lines(prompt, lines)
    cv2.imshow(window_name, prompt)
    k = cv2.waitKey(0) & 0xFF

    if k in (ord('g'), ord('G')) and prev_robot is not None:
        return RobotLabel(**asdict(prev_robot))
    if k in (ord('a'), ord('A')) and ai_robot is not None:
        return RobotLabel(**asdict(ai_robot))

    roi_src = frame.copy()
    _overlay_lines(roi_src, [frame_progress, f"Robot {rid}/2: draw BODY bbox, Enter=confirm, ESC=skip robot"])
    roi = cv2.selectROI(window_name, roi_src, fromCenter=False, showCrosshair=True)
    x, y, bw, bh = [int(v) for v in roi]
    if bw <= 0 or bh <= 0:
        return None

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
    default_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    panel_center = frame.copy()
    cv2.rectangle(panel_center, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(panel_center, default_center, 4, (0, 255, 0), -1)
    center = _point_picker(panel_center, window_name, color=(0, 255, 0), prompt=f"Robot {rid}/2: click BODY center.", frame_progress=frame_progress)

    bl = br = mid = None
    heading = 0.0
    blade_width = 0.0
    if annotate_blade:
        panel = frame.copy()
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bl = _point_picker(panel, window_name, color=(255, 200, 0), prompt=f"Robot {rid}/2: click blade LEFT endpoint.", frame_progress=frame_progress)

        panel2 = frame.copy()
        cv2.rectangle(panel2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(panel2, bl, 4, (255, 200, 0), -1)
        br = _point_picker(panel2, window_name, color=(0, 255, 255), prompt=f"Robot {rid}/2: click blade RIGHT endpoint.", frame_progress=frame_progress)
        bl, br = _line_editor(panel2, window_name, bl, br, frame_progress=frame_progress, prompt=f"Robot {rid}/2: drag points to align blade.")
        mid = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)
        heading = _heading_deg(center, mid)
        blade_width = float(((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5)

    return RobotLabel(
        robot_id=rid,
        class_name="robot",
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=center,
        blade_left_xy=bl,
        blade_right_xy=br,
        blade_mid_xy=mid,
        heading_deg=heading,
        blade_width_px=blade_width,
        extension_type="none",
        extension_side="none",
        visible=True,
    )


def _ai_prefill_robots(frame, model) -> List[RobotLabel]:
    if model is None:
        return []
    out = []
    try:
        pred = model.predict(frame, conf=0.25, imgsz=960, verbose=False, max_det=6)
        boxes = []
        if pred and len(pred[0].boxes) > 0:
            for b in pred[0].boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if cls != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cf = float(b.conf[0])
                boxes.append((cf, x1, y1, x2, y2))
        boxes.sort(reverse=True)
        for rid, (_, x1, y1, x2, y2) in zip([1, 2], boxes[:2]):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            out.append(RobotLabel(robot_id=rid, class_name="robot", bbox_xyxy=(x1, y1, x2, y2), center_xy=(cx, cy)))
    except Exception:
        return []
    return out


def annotate_frames(
    frames_dir: Path,
    out_json_dir: Path,
    start_index: int = 0,
    frame_step: int = 1,
    max_frames: int = 0,
    annotate_blade: bool = True,
    skip_existing: bool = True,
) -> int:
    cv2 = require_cv2()
    imgs = sorted([*frames_dir.glob("*.jpg"), *frames_dir.glob("*.png"), *frames_dir.glob("*.jpeg")])
    if not imgs:
        print(f"No frames in {frames_dir}")
        return 2

    out_json_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(frame_step))
    selected = imgs[start_index::step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        print("No frames selected for annotation after applying start/step/max.")
        return 2

    existing = {p.stem for p in out_json_dir.glob("*.json")}
    if skip_existing:
        selected = [im for im in selected if im.stem not in existing]
        if not selected:
            print("All selected frames already annotated. Nothing to do.")
            return 0

    print("Annotation controls: A annotate / P previous frame / S skip / F finish / X exit")
    print("Review controls: Enter approve, R redo, 1/2 edit robot, E edit dohyo")

    window_name = "Annotation Studio"
    _named_window(window_name)

    prev_labels: List[RobotLabel] = []
    prev_dohyo: Optional[Tuple[float, float, float, float, float]] = None

    # optional auto helpers
    auto_dohyo = None
    try:
        from models.dohyo import DohyoDetector
        auto_dohyo = DohyoDetector()
    except Exception:
        auto_dohyo = None

    yolo_model = None
    try:
        from ultralytics import YOLO
        w = Path("models/weights/robot_sumo.pt")
        if w.exists():
            yolo_model = YOLO(str(w))
    except Exception:
        yolo_model = None

    for i, img_path in enumerate(selected, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        frame_progress = f"Frame {i}/{len(selected)} - {img_path.name}"

        while True:
            choice_img = frame.copy()
            _overlay_lines(choice_img, [frame_progress, "A=Annotate | P=Use previous | S=Skip | F=Finish | X=Exit"])
            cv2.imshow(window_name, choice_img)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('x'), ord('X')):
                cv2.destroyWindow(window_name)
                return 0
            if key in (ord('f'), ord('F')):
                cv2.destroyWindow(window_name)
                return 0
            if key in (ord('s'), ord('S')):
                print(f"[{i}/{len(selected)}] skipped {img_path.name}")
                break

            auto_ellipse = None
            if auto_dohyo is not None:
                try:
                    auto_dohyo.track(frame)
                    info = getattr(auto_dohyo, "last_detection_info", None)
                    if info is not None:
                        cx, cy = info["center"]
                        aw, ah = info["axes"]
                        auto_ellipse = (float(cx), float(cy), float(aw), float(ah), float(info["angle"]))
                except Exception:
                    auto_ellipse = None

            if key in (ord('p'), ord('P')) and prev_labels:
                labels = [RobotLabel(**asdict(lb)) for lb in prev_labels]
                dohyo_ellipse = prev_dohyo
            else:
                labels = []
                ai_pref = _ai_prefill_robots(frame, yolo_model)
                dohyo_ellipse = _pick_dohyo_ellipse(frame, window_name, frame_progress, prev_ellipse=prev_dohyo, auto_ellipse=auto_ellipse)

                for rid in [1, 2]:
                    prev_robot = _find_robot_by_id(prev_labels, rid)
                    ai_robot = _find_robot_by_id(ai_pref, rid)
                    rb = _annotate_one_robot(frame, frame_progress, window_name, rid, annotate_blade, prev_robot=prev_robot, ai_robot=ai_robot)
                    if rb is not None:
                        labels = [l for l in labels if l.robot_id != rid]
                        labels.append(rb)

            labels = sorted(labels, key=lambda z: z.robot_id)

            # review + optional editing
            while True:
                review = frame.copy()
                if dohyo_ellipse is not None:
                    cx, cy, aw, ah, ang = dohyo_ellipse
                    cv2.ellipse(review, (int(cx), int(cy)), (max(1, int(aw / 2)), max(1, int(ah / 2))), float(ang), 0, 360, (255, 255, 0), 2)

                for lb in labels:
                    x1, y1, x2, y2 = lb.bbox_xyxy
                    cv2.rectangle(review, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if lb.blade_left_xy and lb.blade_right_xy:
                        cv2.circle(review, lb.blade_left_xy, 4, (255, 200, 0), -1)
                        cv2.circle(review, lb.blade_right_xy, 4, (0, 255, 255), -1)
                        cv2.line(review, lb.blade_left_xy, lb.blade_right_xy, (255, 255, 0), 2)
                    if lb.center_xy and lb.blade_mid_xy:
                        cv2.line(review, lb.center_xy, lb.blade_mid_xy, (255, 255, 0), 2)
                    cv2.putText(review, f"R{lb.robot_id} heading={lb.heading_deg:.1f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                _overlay_lines(review, [frame_progress, "Review: Enter=Approve | R=Redo | 1/2 Edit robot | E Edit dohyo | S Skip"])
                cv2.imshow(window_name, review)
                rk = cv2.waitKey(0) & 0xFF
                if rk in (13, 32):
                    rec = FrameLabel(image_name=img_path.name, width=w, height=h, robots=labels, dohyo_ellipse=dohyo_ellipse)
                    outp = out_json_dir / f"{img_path.stem}.json"
                    if outp.exists() and skip_existing:
                        print(f"[{i}/{len(selected)}] exists, skipped write {outp.name}")
                    else:
                        outp.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8")
                        print(f"[{i}/{len(selected)}] approved {outp.name}")
                    prev_labels = [RobotLabel(**asdict(lb)) for lb in labels]
                    prev_dohyo = dohyo_ellipse
                    break
                if rk in (ord('s'), ord('S')):
                    print(f"[{i}/{len(selected)}] skipped {img_path.name}")
                    break
                if rk in (ord('r'), ord('R')):
                    break
                if rk in (ord('e'), ord('E')):
                    dohyo_ellipse = _pick_dohyo_ellipse(frame, window_name, frame_progress, prev_ellipse=dohyo_ellipse, auto_ellipse=auto_ellipse)
                    continue
                if rk in (ord('1'), ord('2')):
                    rid = 1 if rk == ord('1') else 2
                    prev_robot = _find_robot_by_id(prev_labels, rid)
                    ai_pref = _ai_prefill_robots(frame, yolo_model)
                    ai_robot = _find_robot_by_id(ai_pref, rid)
                    edited = _annotate_one_robot(frame, frame_progress, window_name, rid, annotate_blade, prev_robot=prev_robot, ai_robot=ai_robot)
                    if edited is not None:
                        labels = [l for l in labels if l.robot_id != rid]
                        labels.append(edited)
                        labels = sorted(labels, key=lambda z: z.robot_id)
                    continue
                if rk in (ord('f'), ord('F'), ord('x'), ord('X')):
                    cv2.destroyWindow(window_name)
                    return 0

            # if redo frame, restart annotation choice
            if rk in (ord('r'), ord('R')):
                continue
            break

    cv2.destroyWindow(window_name)
    return 0


def xyxy_to_norm(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def export_yolo_from_json(
    frames_dir: Path,
    json_dir: Path,
    labels_out: Path,
    blade_box_thickness_px: int = 8,
    include_extension_classes: bool = False,
    filename_prefix: str = "",
) -> int:
    labels_out.mkdir(parents=True, exist_ok=True)
    items = sorted(json_dir.glob("*.json"))
    if not items:
        print(f"No annotation json files in {json_dir}")
        return 2

    exported = 0
    dataset_root = None
    images_train_dir = None
    parts = [x.lower() for x in labels_out.parts]
    if len(parts) >= 2 and parts[-2:] == ["labels", "train"]:
        dataset_root = labels_out.parent.parent
        images_train_dir = dataset_root / "images" / "train"
        images_train_dir.mkdir(parents=True, exist_ok=True)

    for jp in items:
        rec = json.loads(jp.read_text(encoding="utf-8"))
        w, h = int(rec["width"]), int(rec["height"])
        lines = []

        for rb in rec.get("robots", []):
            x1, y1, x2, y2 = rb["bbox_xyxy"]
            cx, cy, bw, bh = xyxy_to_norm(int(x1), int(y1), int(x2), int(y2), w, h)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            bl = rb.get("blade_left_xy")
            br = rb.get("blade_right_xy")
            if bl and br:
                blx, bly = bl
                brx, bry = br
                bx1 = max(0, min(int(blx), int(brx)))
                bx2 = min(w - 1, max(int(blx), int(brx)))
                by_mid = int((int(bly) + int(bry)) / 2)
                by1 = max(0, by_mid - blade_box_thickness_px // 2)
                by2 = min(h - 1, by_mid + blade_box_thickness_px // 2)
                fcx, fcy, fbw, fbh = xyxy_to_norm(bx1, by1, bx2, by2, w, h)
                lines.append(f"1 {fcx:.6f} {fcy:.6f} {fbw:.6f} {fbh:.6f}")

            if include_extension_classes:
                for cls_id, ex1, ey1, ex2, ey2 in _extension_boxes(rb, w, h):
                    ecx, ecy, ebw, ebh = xyxy_to_norm(ex1, ey1, ex2, ey2, w, h)
                    lines.append(f"{cls_id} {ecx:.6f} {ecy:.6f} {ebw:.6f} {ebh:.6f}")

        stem = f"{filename_prefix}{jp.stem}" if filename_prefix else jp.stem
        out_txt = labels_out / f"{stem}.txt"
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if images_train_dir is not None:
            src_img = frames_dir / rec["image_name"]
            if src_img.exists():
                dst_name = f"{filename_prefix}{src_img.name}" if filename_prefix else src_img.name
                dst_img = images_train_dir / dst_name
                if not dst_img.exists():
                    dst_img.write_bytes(src_img.read_bytes())

        exported += 1

    print(f"Exported YOLO labels: {exported} files -> {labels_out}")
    if images_train_dir is not None:
        print(f"Prepared training images: {images_train_dir}")
    return 0


def run_training(dataset_dir: Path, model: str, epochs: int, imgsz: int, batch: int, device: str, workers: int) -> int:
    classes = "robot,blade_front"
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
        "--classes",
        classes,
    ]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    assert proc.stdout is not None
    while True:
        raw = proc.stdout.readline()
        if not raw:
            break
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:
            line = str(raw).rstrip()
        print(line)
    return int(proc.wait())



def infer_video(weights: Path, video: Path, out_video: Path, conf: float = 0.2, imgsz: int = 960) -> int:
    cv2 = require_cv2()
    try:
        import numpy as np
    except Exception as e:
        print(f"NumPy missing: {e}")
        return 2
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"Ultralytics missing: {e}")
        return 2

    if not Path(weights).exists():
        print(f"Weights file not found: {weights}")
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

    dohyo = None
    try:
        from models.dohyo import DohyoDetector
        dohyo = DohyoDetector()
    except Exception:
        dohyo = None

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks = {
        1: {"center": None, "vel": (0.0, 0.0), "miss": 0, "bbox": None, "blade": None, "blade_vec": None, "conf": 0.0},
        2: {"center": None, "vel": (0.0, 0.0), "miss": 0, "bbox": None, "blade": None, "blade_vec": None, "conf": 0.0},
    }

    def _inside(mask, cx, cy):
        return 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0] and mask[cy, cx] > 0

    def _pred_center(tk):
        if tk["center"] is None:
            return None
        return (tk["center"][0] + tk["vel"][0], tk["center"][1] + tk["vel"][1])

    def _norm(vx, vy):
        n = float(np.hypot(vx, vy))
        if n < 1e-5:
            return None
        return (vx / n, vy / n)

    def _expected_front_dir(tk):
        v = _norm(tk["vel"][0], tk["vel"][1])
        if v is not None and np.hypot(tk["vel"][0], tk["vel"][1]) > 1.4:
            return v
        if tk["blade_vec"] is not None:
            bv = _norm(tk["blade_vec"][0], tk["blade_vec"][1])
            if bv is not None:
                return bv
        return None

    def _clamp_pt(x, y):
        return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        dohyo_mask = None
        dohyo_ellipse = None
        inner_ellipse = None
        if dohyo is not None:
            dohyo.track(frame)
            info = getattr(dohyo, "last_detection_info", None)
            if info is not None:
                cx, cy = [int(v) for v in info["center"]]
                ax, ay = [max(1, int(v / 2)) for v in info["axes"]]
                ang = float(info["angle"])
                dohyo_ellipse = (cx, cy, ax, ay, ang)
                inner_ellipse = (cx, cy, max(1, int(ax * (154.0 / 164.0))), max(1, int(ay * (154.0 / 164.0))), ang)
                dohyo_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(dohyo_mask, (cx, cy), (max(1, int(ax * 0.92)), max(1, int(ay * 0.92))), ang, 0, 360, 255, -1)

        res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False, max_det=16)
        raw_robots = []
        raw_blades = []
        if res and len(res[0].boxes) > 0:
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cls = int(b.cls[0]) if b.cls is not None else -1
                cf = float(b.conf[0])
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area = max(1, (x2 - x1) * (y2 - y1))
                if cls == 0:
                    if dohyo_mask is not None and not _inside(dohyo_mask, cx, cy):
                        continue
                    if area < 250 or area > int(0.20 * w * h):
                        continue
                    raw_robots.append((x1, y1, x2, y2, cf, cx, cy))
                elif cls == 1:
                    if dohyo_mask is not None and not _inside(dohyo_mask, cx, cy):
                        continue
                    raw_blades.append((x1, y1, x2, y2, cf, cx, cy))

        raw_robots = sorted(raw_robots, key=lambda z: z[4], reverse=True)[:6]

        used = set()
        for rid in [1, 2]:
            tk = tracks[rid]
            pred = _pred_center(tk)
            best_j = None
            best_cost = 1e9
            for j, r in enumerate(raw_robots):
                if j in used:
                    continue
                _, _, _, _, cf, cx, cy = r
                if pred is None:
                    cost = -cf * 50.0
                else:
                    d = float(np.hypot(cx - pred[0], cy - pred[1]))
                    if d > 140:
                        continue  # anti-teleport gate
                    cost = d - cf * 20.0
                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            if best_j is None:
                tk["miss"] += 1
                continue

            used.add(best_j)
            x1, y1, x2, y2, cf, cx, cy = raw_robots[best_j]
            if tk["center"] is not None:
                vx = cx - tk["center"][0]
                vy = cy - tk["center"][1]
                tk["vel"] = (0.70 * tk["vel"][0] + 0.30 * vx, 0.70 * tk["vel"][1] + 0.30 * vy)
            tk["center"] = (cx, cy)
            tk["bbox"] = (x1, y1, x2, y2)
            tk["conf"] = cf
            tk["miss"] = 0

        for rid in [1, 2]:
            tk = tracks[rid]
            if tk["center"] is None and raw_robots:
                for j, r in enumerate(raw_robots):
                    if j in used:
                        continue
                    x1, y1, x2, y2, cf, cx, cy = r
                    tk["center"] = (cx, cy)
                    tk["bbox"] = (x1, y1, x2, y2)
                    tk["conf"] = cf
                    tk["vel"] = (0.0, 0.0)
                    tk["miss"] = 0
                    used.add(j)
                    break

        # better blade assignment: choose blades consistent with expected front direction and distance
        for rid in [1, 2]:
            tk = tracks[rid]
            if tk["center"] is None or tk["bbox"] is None:
                continue
            cx, cy = tk["center"]
            x1, y1, x2, y2 = tk["bbox"]
            bw = max(8, x2 - x1)
            bh = max(8, y2 - y1)
            reach = 0.80 * max(bw, bh)
            exp_dir = _expected_front_dir(tk)

            best = None
            best_score = -1e9
            for bx1, by1, bx2, by2, bcf, bcx, bcy in raw_blades:
                dvx, dvy = (bcx - cx), (bcy - cy)
                dist = float(np.hypot(dvx, dvy))
                if dist < 3 or dist > reach * 1.4:
                    continue
                dirv = _norm(dvx, dvy)
                align = 0.0
                if exp_dir is not None and dirv is not None:
                    align = exp_dir[0] * dirv[0] + exp_dir[1] * dirv[1]
                # prefer confident, close-ish, front-aligned candidates
                score = (2.2 * bcf) + (1.2 * align) - (0.006 * dist)
                if score > best_score:
                    best_score = score
                    best = (bcx, bcy)

            if best is not None:
                bx, by = best
                tk["blade"] = (int(bx), int(by))
                vec = (bx - cx, by - cy)
                if tk["blade_vec"] is None:
                    tk["blade_vec"] = vec
                else:
                    tk["blade_vec"] = (
                        0.75 * tk["blade_vec"][0] + 0.25 * vec[0],
                        0.75 * tk["blade_vec"][1] + 0.25 * vec[1],
                    )
            else:
                # fallback: preserve smoothed blade orientation, then motion vector
                dirv = _expected_front_dir(tk)
                if dirv is None:
                    continue
                bx = cx + dirv[0] * (0.48 * max(bw, bh))
                by = cy + dirv[1] * (0.48 * max(bw, bh))
                tk["blade"] = _clamp_pt(bx, by)
                tk["blade_vec"] = (tk["blade"][0] - cx, tk["blade"][1] - cy)

        # draw overlays: dohyo contour validation + robots + blade/front
        if dohyo_ellipse is not None:
            dcx, dcy, dax, day, dang = dohyo_ellipse
            cv2.ellipse(frame, (dcx, dcy), (dax, day), dang, 0, 360, (255, 255, 0), 2)
            if inner_ellipse is not None:
                icx, icy, iax, iay, iang = inner_ellipse
                cv2.ellipse(frame, (icx, icy), (iax, iay), iang, 0, 360, (120, 255, 255), 2)
            cv2.drawMarker(frame, (dcx, dcy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        for rid, tk in tracks.items():
            if tk["center"] is None or tk["bbox"] is None:
                continue
            if tk["miss"] > 15:
                continue
            x1, y1, x2, y2 = tk["bbox"]
            color = (0, 220, 0) if rid == 1 else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = tk["center"]
            cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
            if tk["blade"] is not None:
                bx, by = tk["blade"]
                cv2.circle(frame, (int(bx), int(by)), 4, (255, 255, 0), -1)
                cv2.line(frame, (int(cx), int(cy)), (int(bx), int(by)), (255, 255, 0), 2)
            cv2.putText(frame, f"R{rid} {tk['conf']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Progress: {frame_idx}/{max(frame_idx, total)}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

        if frame_idx % 30 == 0:
            pct = (100.0 * frame_idx / total) if total > 0 else 0.0
            print(f"Infer progress: {frame_idx}/{total if total > 0 else '?'} ({pct:.1f}%)")

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
    p_annot.add_argument("--frame-step", type=int, default=1, help="annotate every Nth frame (speedup)")
    p_annot.add_argument("--max-frames", type=int, default=0, help="stop after annotating this many selected frames")
    p_annot.add_argument("--bbox-only", action="store_true", help="annotate body bbox/center only, skip blade points")
    p_annot.add_argument("--allow-overwrite", action="store_true", help="allow overwriting existing annotation JSON files")

    p_export = sub.add_parser("export-yolo", help="export YOLO txt labels from json annotations")
    p_export.add_argument("--frames-dir", required=True)
    p_export.add_argument("--json-dir", required=True)
    p_export.add_argument("--labels-out", required=True)
    p_export.add_argument("--blade-box-thickness-px", type=int, default=8)
    p_export.add_argument("--include-extension-classes", action="store_true", help="include extension classes (2..7) if present in JSON")

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
            return annotate_frames(
                Path(args.frames_dir),
                Path(args.out_json_dir),
                start_index=args.start_index,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                annotate_blade=not args.bbox_only,
                skip_existing=not args.allow_overwrite,
            )
        if args.cmd == "export-yolo":
            return export_yolo_from_json(
                Path(args.frames_dir),
                Path(args.json_dir),
                Path(args.labels_out),
                blade_box_thickness_px=args.blade_box_thickness_px,
                include_extension_classes=args.include_extension_classes,
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
