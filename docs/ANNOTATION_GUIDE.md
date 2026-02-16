# Robot Sumo Annotation Guide (Beginner Friendly)

Goal: produce high-quality labels for:
- robot localization,
- blade orientation (most important).

This guide uses the unified tool:

```bash
python scripts/training_studio.py --help
```

---

## 1) Prepare action clip and frames

```bash
python scripts/training_studio.py trim --video /path/full_match.mov --out data/studio/clips/m1.mp4
python scripts/training_studio.py extract --video data/studio/clips/m1.mp4 --out-dir data/studio/frames --fps 10
```

---

## 2) Annotate frames

```bash
python scripts/training_studio.py annotate --frames-dir data/studio/frames --out-json-dir data/studio/annotations
```

For each robot (R1 then R2):
1. Draw robot bbox (Enter confirms, ESC skips).
2. Click **blade left endpoint**.
3. Click **blade right endpoint**.
4. Review and press:
   - `n` save + next
   - `r` redo frame
   - `q` quit

Why endpoints and not only a point?
- better orientation learning,
- blade width supervision,
- stronger signal under rotation.

---

## 3) Export YOLO labels

```bash
python scripts/training_studio.py export-yolo \
  --frames-dir data/studio/frames \
  --json-dir data/studio/annotations \
  --labels-out data/robot_dataset/labels/train
```

Label classes:
- class `0` = robot bbox
- class `1` = blade line region (derived from endpoint segment)

---

## 4) Quality checklist (critical)

Per visible robot:
- bbox tight around body/extensions,
- blade endpoints match real blade width and front,
- avoid guessing hidden geometry during occlusion.

Include difficult frames:
- collisions,
- partial occlusion,
- stationary start,
- bright reflections/scratches,
- motion blur.

---

## 5) Train + test

Train:

```bash
python scripts/training_studio.py train \
  --dataset-dir data/robot_dataset \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --workers 0
```

Test on unseen video:

```bash
python scripts/training_studio.py infer-video \
  --weights models/weights/robot_sumo.pt \
  --video /path/unseen.mov \
  --out artifacts/infer_unseen.mp4
```

---

## 6) If training fails very early

Use safe debug config first:

```bash
python scripts/training_studio.py train \
  --dataset-dir data/robot_dataset \
  --model yolov8n.pt \
  --epochs 10 \
  --imgsz 640 \
  --batch 4 \
  --device cpu \
  --workers 0
```

If this works, switch to GPU/full settings.
