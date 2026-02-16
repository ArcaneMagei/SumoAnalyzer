# Robot Sumo AI Annotation Guide (Step-by-step, beginner friendly)

This guide explains exactly how to create high-quality training labels for robot position and blade/front orientation.

---

## 1) Install tools

On your training machine:

```bash
pip install opencv-python ultralytics
```

---

## 2) Prepare frames from your match videos

Use the built-in sampler to extract diverse frames from one or more videos:

```bash
python scripts/annotate_robot_dataset.py prepare \
  --videos /path/match1.mov /path/match2.mov \
  --out-dir data/robot_dataset \
  --frames-per-video 150 \
  --val-ratio 0.2
```

This creates YOLO-ready folders:

```text
data/robot_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  meta/train/
  meta/val/
```

---

## 3) Annotate train split

```bash
python scripts/annotate_robot_dataset.py annotate \
  --out-dir data/robot_dataset \
  --split train
```

### Annotation flow per frame

1. **Select robot bbox** (drag rectangle, press Enter).
2. **Click front point** (blade/front direction) in the popup.
3. Repeat for second robot (or ESC to skip if only one visible).
4. Review overlay and press:
   - `n` = save and next frame
   - `r` = redo frame
   - `q` = quit

Do the same for validation split:

```bash
python scripts/annotate_robot_dataset.py annotate --out-dir data/robot_dataset --split val
```

---

## 4) Label quality checklist (critical)

For **every robot**:
- Bounding box should tightly include chassis + relevant extensions.
- Front point should be at true attacking front (blade tip direction).
- If robots overlap, still annotate visible one as accurately as possible.

Include hard cases:
- stationary robots,
- very fast bursts,
- collisions/occlusions,
- scratches/reflections on black dohyo,
- camera shake/zoom.

---

## 5) Train the detector

```bash
python scripts/train_robot_detector.py \
  --dataset-dir data/robot_dataset \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --workers 0
```

After training, best weights are copied to:

```text
models/weights/robot_sumo.pt
```

---

## 6) Validate detector output quickly

```bash
python scripts/validate_robot_detector.py \
  --weights models/weights/robot_sumo.pt \
  --video /path/match_test.mov \
  --out artifacts/ai_detector_preview.mp4 \
  --conf 0.2
```

Open `artifacts/ai_detector_preview.mp4` and check if boxes are stable on both moving and still robots.

---

## 7) Use in app

In Streamlit sidebar:
- Enable **Use AI robot detector**
- Set **AI weights path** = `models/weights/robot_sumo.pt`
- Start with **AI confidence** around `0.15–0.30`

---

## 8) Recommended dataset size

For strong performance on your case:
- **MVP:** 400–800 annotated frames
- **Good:** 1,500+ frames across many tournaments/lighting conditions

Label variety matters more than raw count.


### If training stops very early

Start with a debug run:

```bash
python scripts/train_robot_detector.py \
  --dataset-dir data/robot_dataset \
  --model yolov8n.pt \
  --epochs 10 \
  --imgsz 640 \
  --batch 4 \
  --device cpu \
  --workers 0
```

If that works, switch to GPU and larger settings.
