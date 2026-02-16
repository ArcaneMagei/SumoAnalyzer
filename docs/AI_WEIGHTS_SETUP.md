# AI Robot Detector Setup (Single-App Workflow)

## 1) Install dependencies

```bash
pip install ultralytics opencv-python
```

## 2) Launch training app UI (recommended)

```bash
python scripts/training_studio_ui.py
```

This UI is persistent and stores project state in `data/studio/project.json`.

Use buttons in order:
1. Trim clip
2. Extract frames
3. Annotate
4. Export YOLO labels
5. Train model
6. Infer on test video

## 3) If a video cannot open

- Use absolute path in UI/CLI.
- MOV codec may fail in OpenCV builds; transcode once:

```bash
ffmpeg -y -i "IMG_1399.mov" -c:v libx264 -pix_fmt yuv420p -c:a aac "IMG_1399.mp4"
```

## 4) If training stalls at epoch 1

Run safe debug config first:

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

Then switch to GPU/full settings.

## 5) Deeper docs

- `docs/TRAINING_APP_BLUEPRINT.md`
- `docs/ANNOTATION_GUIDE.md`


## 6) Quick health-check

Run:

```bash
python scripts/training_studio.py doctor
```

This checks OpenCV/Ultralytics/FFmpeg and writable project folders.


## UX/persistence notes
- The UI now creates isolated per-match work folders under `data/studio/matches/<match_id>/` so clips/frames/annotations from one video do not mix with another.
- Annotation resumes by default and skips already-annotated frames unless you explicitly use CLI `--allow-overwrite`.
- Training defaults are tuned safer for local iteration in UI (`imgsz=640`, `batch=8`) and live output is streamed into the log.
