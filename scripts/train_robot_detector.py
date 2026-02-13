"""Train custom YOLO model for robot sumo detection.

Usage:
  python scripts/train_robot_detector.py \
      --dataset-dir data/robot_dataset \
      --model yolov8n.pt \
      --epochs 120 \
      --imgsz 960

Expected dataset layout (YOLO format):
  data/robot_dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt

Class mapping (required):
  0 robot
Optional:
  1 blade_front
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def ensure_data_yaml(dataset_dir: Path, classes: list[str]) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    content = (
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + names_lines + "\n"
    )
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def validate_structure(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset dirs:\n" + "\n".join(str(m) for m in missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/robot_dataset")
    parser.add_argument("--model", default="yolov8n.pt", help="base model or existing checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="e.g. 0, 0,1, cpu, mps")
    parser.add_argument("--project", default="runs/sumo")
    parser.add_argument("--name", default="robot_detector")
    parser.add_argument("--classes", default="robot,blade_front")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    validate_structure(dataset_dir)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("At least one class is required", file=sys.stderr)
        return 2

    data_yaml = ensure_data_yaml(dataset_dir, classes)

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        print("Ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        return 3

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cache=True,
        workers=8,
        close_mosaic=10,
        patience=30,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        target = Path("models/weights/robot_sumo.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(best.read_bytes())
        print(f"Saved ready-to-use weights to {target}")
    else:
        print("Training finished, but best.pt not found. Check run logs.", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
