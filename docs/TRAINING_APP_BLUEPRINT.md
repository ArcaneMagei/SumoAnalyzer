# Bulletproof Training Blueprint for Robot + Blade + Dohyo

This workflow is designed for your exact match conditions:
- short action windows,
- side-angle camera variation,
- rapid robot rotation and collisions,
- blade direction as the key signal.

## Use one app for everything (no CLI memorization)

### Desktop UI (recommended)

```bash
python scripts/training_studio_ui.py
```

This UI persists project state in:

```text
data/studio/project.json
```

and lets you run the full pipeline by buttons:
1. Trim clip
2. Extract frames
3. Annotate robots + blade endpoints
4. Export YOLO labels
5. Train
6. Infer on test video

### CLI (also available)

```bash
python scripts/training_studio.py --help
```

---

## Blade-specific supervision strategy

You now annotate **blade-left** and **blade-right** points per robot (not just one point).
This derives:
- blade midpoint/front direction,
- heading vector,
- blade width constraint.

This makes orientation learning much stronger and more consistent with your robot geometry.

---

## Best-practice data recipe

1. Trim to action-only clips (remove dead intro/outro).
2. Extract 8-12 FPS frames.
3. Label all visible robots in those frames.
4. Ensure difficult scenes are included:
   - collision/occlusion,
   - stationary starts,
   - scratches/reflections,
   - motion blur.

Target size:
- v1: 700-1200 frames
- v2+: add only failure frames (active-learning loop)

---

## Troubleshooting video open issues

If you get `Could not open video`:
- use absolute path, or add video via UI file picker,
- if file exists but still fails, transcode MOV to MP4:

```bash
ffmpeg -y -i "IMG_1399.mov" -c:v libx264 -pix_fmt yuv420p -c:a aac "IMG_1399.mp4"
```

The CLI now prints detailed path attempts and codec hints.

---

## End-to-end CLI examples

```bash
python scripts/training_studio.py trim --video /path/full_match.mov --out data/studio/clips/m1.mp4
python scripts/training_studio.py extract --video data/studio/clips/m1.mp4 --out-dir data/studio/frames --fps 10
python scripts/training_studio.py annotate --frames-dir data/studio/frames --out-json-dir data/studio/annotations
python scripts/training_studio.py export-yolo --frames-dir data/studio/frames --json-dir data/studio/annotations --labels-out data/robot_dataset/labels/train
python scripts/training_studio.py train --dataset-dir data/robot_dataset --model yolov8n.pt --epochs 120 --imgsz 960 --batch 16 --device 0 --workers 0
python scripts/training_studio.py infer-video --weights models/weights/robot_sumo.pt --video /path/unseen.mov --out artifacts/infer_unseen.mp4
```


## UI launch note (Windows)

Both launch modes are supported:

```bash
python scripts/training_studio_ui.py
# or
python -m scripts.training_studio_ui
```
