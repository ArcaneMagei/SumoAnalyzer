# Training App Blueprint (for your exact use case)

This is the recommended stack to get to **reliable robot + blade direction detection** fast.

## Why previous attempts felt weak

Your problem is not just “object detection”. It combines:
- moving camera + angled view,
- short action windows,
- high-speed collisions,
- partial blade occlusion,
- black robots on scratched dark dohyo.

So you need a **data-first training app** and a model stack designed for orientation.

---

## Recommended architecture (practical)

### A) Training Studio app (implemented here)
Use `scripts/training_studio.py` for a clean data pipeline:
1. Trim clips to only the few seconds of real action.
2. Extract frames from clips.
3. Annotate 2 robots + front/blade point.
4. Export YOLO labels.

This avoids wasting labeling time on pre/post-match dead frames.

### B) Model strategy
Train two models incrementally:
1. **Detector model** (`robot`, `blade_front`) for robust localization.
2. Optional **pose/heading model** (keypoints for blade/front and rear center).

Then track with your existing temporal tracker.

### C) Training cadence
- Start with 500–800 high-quality frames.
- Train v1 model.
- Run on new videos.
- Collect failure frames.
- Label only failures.
- Retrain v2.

This iterative loop beats brute-force labeling thousands blindly.

---

## How to use the Training Studio now

## 1) Trim action clips

```bash
python scripts/training_studio.py trim --video /path/full_match.mov --out data/studio/clips/match1_action.mp4
```

## 2) Extract frames

```bash
python scripts/training_studio.py extract \
  --video data/studio/clips/match1_action.mp4 \
  --out-dir data/studio/frames \
  --fps 8
```

## 3) Annotate robot + blade front

```bash
python scripts/training_studio.py annotate \
  --frames-dir data/studio/frames \
  --out-json-dir data/studio/annotations
```

## 4) Export YOLO labels

```bash
python scripts/training_studio.py export-yolo \
  --frames-dir data/studio/frames \
  --json-dir data/studio/annotations \
  --labels-out data/robot_dataset/labels/train
```

(Place/copy images into `data/robot_dataset/images/train` accordingly; keep a validation split too.)

---

## UX recommendations (next upgrade)

If you want an even easier UI than OpenCV windows:
- Use **CVAT** with SAM-assisted polygons + keypoints.
- Keep this repo scripts for trim/extract/export automation.

This hybrid gives the best balance: fast UI + reproducible pipeline.

---

## What “great” labels look like

Per visible robot in each frame:
- 1 tight bbox around robot body/extensions.
- 1 front point at blade attack direction.
- Keep consistent ID (R1/R2) if possible in metadata.

For occlusion frames:
- still label what is visible,
- do not invent hidden geometry,
- include many such cases (critical for robustness).
