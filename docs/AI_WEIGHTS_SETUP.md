# AI Robot Detector Setup (Full Functional Workflow)

This project supports AI-first detection using:

```text
models/weights/robot_sumo.pt
```

## Install

```bash
pip install ultralytics opencv-python
```

---

## Preferred workflow: Training Studio (single app)

Use one tool for all steps:

```bash
python scripts/training_studio.py --help
```

### Typical sequence

1) Trim action clip:

```bash
python scripts/training_studio.py trim --video /path/full_match.mov --out data/studio/clips/m1.mp4
```

2) Extract frames:

```bash
python scripts/training_studio.py extract --video data/studio/clips/m1.mp4 --out-dir data/studio/frames --fps 10
```

3) Annotate robots + blade endpoints:

```bash
python scripts/training_studio.py annotate --frames-dir data/studio/frames --out-json-dir data/studio/annotations
```

4) Export YOLO labels:

```bash
python scripts/training_studio.py export-yolo \
  --frames-dir data/studio/frames \
  --json-dir data/studio/annotations \
  --labels-out data/robot_dataset/labels/train
```

5) Train:

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

6) Test on unseen video:

```bash
python scripts/training_studio.py infer-video \
  --weights models/weights/robot_sumo.pt \
  --video /path/unseen.mov \
  --out artifacts/infer_unseen.mp4
```

---

## If training stalls at epoch 1 (common)

Use debug-safe settings first:

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

Then switch back to GPU and full settings.

---

## More detailed blueprint

See:

```text
docs/TRAINING_APP_BLUEPRINT.md
```

And detailed annotation UX guide:

```text
docs/ANNOTATION_GUIDE.md
```
