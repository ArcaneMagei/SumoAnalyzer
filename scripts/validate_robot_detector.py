"""Quick inference validator for trained robot detector weights."""

from __future__ import annotations

import argparse
from pathlib import Path



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/weights/robot_sumo.pt")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="artifacts/ai_detector_preview.mp4")
    parser.add_argument("--conf", type=float, default=0.2)
    args = parser.parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:
        print(f"Ultralytics not installed: {e}")
        return 2

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        print("Could not read video")
        return 3

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = model.predict(frame, conf=args.conf, verbose=False, imgsz=960)
        if r and len(r[0].boxes) > 0:
            for b in r[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cf = float(b.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
