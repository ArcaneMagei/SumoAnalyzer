# Bulletproof Training Blueprint for Robot + Blade + Dohyo

This is the recommended production workflow for your exact constraints:
- short bouts (few seconds),
- side-angle camera variation,
- robots rotating rapidly,
- blade orientation most important,
- blade is ~robot width (20cm), at front, touching dohyo unless robot is airborne/out.

---

## 1) Use one app for the entire pipeline

Use **one command app**:

```bash
python scripts/training_studio.py --help
```

It includes all required stages:
- `trim` -> cut only action seconds
- `extract` -> sample frames from clips
- `annotate` -> label robot bbox + blade left/right endpoints
- `export-yolo` -> produce labels for training
- `train` -> run robust training wrapper
- `infer-video` -> test trained model visually on unseen video

This is the “single app” workflow you requested.

---

## 2) Why blade-left/right annotation is critical

Instead of a single front point only, Studio now asks for:
- **blade left endpoint**
- **blade right endpoint**

From these it derives:
- blade midpoint (front direction)
- heading vector (center -> blade midpoint)
- blade width in pixels (quality signal)

This is far more stable for learning true front orientation.

---

## 3) Practical “bulletproof” data recipe

For each tournament/camera style:
1. Trim all videos to action-only clips.
2. Extract at 8–12 FPS (no wasted dead frames).
3. Label every visible robot in those frames.
4. Ensure difficult cases are well represented:
   - collisions/occlusions
   - rotations
   - stationary starts
   - scratched/reflective dohyo

Target dataset sizes:
- v1: 700–1200 labeled frames
- v2+: add only model failure frames (active learning)

---

## 4) End-to-end commands

## A) Trim

```bash
python scripts/training_studio.py trim \
  --video /path/full_match.mov \
  --out data/studio/clips/match1_action.mp4
```

## B) Extract

```bash
python scripts/training_studio.py extract \
  --video data/studio/clips/match1_action.mp4 \
  --out-dir data/studio/frames \
  --fps 10
```

## C) Annotate

```bash
python scripts/training_studio.py annotate \
  --frames-dir data/studio/frames \
  --out-json-dir data/studio/annotations
```

## D) Export YOLO labels

```bash
python scripts/training_studio.py export-yolo \
  --frames-dir data/studio/frames \
  --json-dir data/studio/annotations \
  --labels-out data/robot_dataset/labels/train \
  --blade-box-thickness-px 8
```

## E) Train

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

## F) Test model on unseen video

```bash
python scripts/training_studio.py infer-video \
  --weights models/weights/robot_sumo.pt \
  --video /path/unseen_match.mov \
  --out artifacts/infer_unseen.mp4 \
  --conf 0.2
```

---

## 5) Quality checks that matter most

Your success metric should be:
- robot center stable and accurate,
- blade direction stable and correct,
- recovery through collisions,
- low false positives on scratches/highlights.

Reject training versions that only look good on easy frames.

---

## 6) Recommended next model upgrade

After detector baseline is stable:
- add keypoint model for blade endpoints directly,
- keep detector for robust presence + bbox,
- fuse both in tracker for strongest orientation accuracy.

This gives best final reliability for strategy analysis.
