# Robot Sumo Annotation Guide (UI-first)

## Start app

```bash
python scripts/training_studio_ui.py
```

## Workflow in UI

1. Add videos (file picker), set current video.
2. Click **Trim clip** and mark IN/OUT around action.
3. Click **Extract frames**.
4. Click **Annotate**:
   - For each robot: draw bbox,
   - click blade LEFT endpoint,
   - click blade RIGHT endpoint.
5. Click **Export YOLO labels**.
6. Click **Train model**.
7. Click **Test on video** (Infer) and inspect output MP4.

## Why blade endpoints?

Endpoints provide stronger supervision than a single point:
- true front line direction,
- blade width signal,
- more stable under rotation/occlusion.

## Quality checklist

- Tight robot bbox around chassis/extensions.
- Blade endpoints match actual blade contact edge.
- Do not invent hidden shape during occlusions.
- Include hard frames: blur, collisions, reflections, scratches.

## If video won’t open

Use absolute path or transcode MOV to MP4:

```bash
ffmpeg -y -i "input.mov" -c:v libx264 -pix_fmt yuv420p -c:a aac "input.mp4"
```

## If training fails early

Set in UI/CLI: `workers=0`, `device=cpu`, smaller `imgsz/batch`, then move back to GPU once stable.
