# AI Robot Detector Setup (Full Functional Workflow)

This project now supports AI-first detection through `models/weights/robot_sumo.pt`.

## 1) Install dependencies

On your machine (RTX 4090 recommended):

```bash
pip install ultralytics opencv-python
```

## 2) Prepare dataset (YOLO format)

Create:

```text
data/robot_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Label class `0` as `robot` (mandatory). Optional class `1`: `blade_front`.

YOLO label line format:

```text
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

## 3) Train weights

```bash
python scripts/train_robot_detector.py \
  --dataset-dir data/robot_dataset \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

After training, best weights are copied automatically to:

```text
models/weights/robot_sumo.pt
```

## 4) Validate quickly

```bash
python scripts/validate_robot_detector.py \
  --weights models/weights/robot_sumo.pt \
  --video /path/to/match.mov \
  --out artifacts/ai_detector_preview.mp4 \
  --conf 0.2
```

## 5) Use in app

- Enable **Use AI robot detector**.
- Set **AI weights path** to `models/weights/robot_sumo.pt`.
- Adjust **AI confidence** to ~0.15–0.30.

## Practical recommendation

Start with 300–600 labeled frames from your own footage, then add hard examples (collisions, occlusions, scratches, reflections). This dramatically improves stationary robot detection.


## Manual annotation helper (recommended)

If you are not sure how to annotate by hand, use:

```bash
python scripts/annotate_robot_dataset.py --help
```

Detailed walkthrough is in:

```text
docs/ANNOTATION_GUIDE.md
```


## Troubleshooting: training stops at Epoch 1 with no weights

If you see output like:

```text
Epoch 1/120 ... 0G ... 0/14
```

and then it stops, usually this means dataloader/runtime issue (not successful training completion).

Use this safer command first:

```bash
python scripts/train_robot_detector.py   --dataset-dir data/robot_dataset   --model yolov8n.pt   --epochs 120   --imgsz 640   --batch 4   --device cpu   --workers 0
```

Then move to GPU:

```bash
python scripts/train_robot_detector.py   --dataset-dir data/robot_dataset   --model yolov8n.pt   --epochs 120   --imgsz 960   --batch 16   --device 0   --workers 0
```

Notes:
- `0G` means you are not using CUDA GPU memory (likely CPU path).
- this training script now validates labels before training and will fail early on bad annotation format.
- if `best.pt` is not produced, script will fallback to `last.pt` when available and still copy to `models/weights/robot_sumo.pt`.


## Dedicated training app workflow

For a full training-first workflow (trim clips -> extract frames -> annotate robot + blade -> export), see:

```text
docs/TRAINING_APP_BLUEPRINT.md
```

Main tool:

```bash
python scripts/training_studio.py --help
```
