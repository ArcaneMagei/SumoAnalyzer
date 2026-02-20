"""Train custom YOLO model for robot sumo detection.

This version includes dataset sanity checks and safer defaults for Windows/CPU
machines where dataloader workers can crash at epoch start.
"""

from __future__ import annotations

import argparse
import os
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


def _image_files(folder: Path) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    out: list[Path] = []
    for e in exts:
        out.extend(folder.glob(e))
    return sorted(out)




def ensure_structure_and_val_split(dataset_dir: Path, val_ratio: float = 0.15) -> None:
    """Create missing dirs and auto-build val split from train when absent."""
    train_img = dataset_dir / "images" / "train"
    val_img = dataset_dir / "images" / "val"
    train_lbl = dataset_dir / "labels" / "train"
    val_lbl = dataset_dir / "labels" / "val"

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    train_images = _image_files(train_img)
    if not train_images:
        raise FileNotFoundError(f"No training images found in {train_img}")

    val_images = _image_files(val_img)
    if val_images:
        return

    target = max(1, int(round(len(train_images) * max(0.05, min(0.4, val_ratio)))))
    step = max(1, len(train_images) // target)
    selected = [img for idx, img in enumerate(train_images) if idx % step == 0][:target]

    for img in selected:
        dst_img = val_img / img.name
        if not dst_img.exists():
            dst_img.write_bytes(img.read_bytes())

        src_lbl = train_lbl / f"{img.stem}.txt"
        dst_lbl = val_lbl / src_lbl.name
        if src_lbl.exists() and not dst_lbl.exists():
            dst_lbl.write_bytes(src_lbl.read_bytes())

    print(f"Auto-created val split: {len(selected)} images in {val_img}")


def validate_labels(dataset_dir: Path, num_classes: int) -> None:
    """Hard-fail on malformed YOLO labels that often crash training."""
    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        imgs = _image_files(img_dir)
        if not imgs:
            raise ValueError(f"No images found in {img_dir}")

        bad = []
        for img in imgs:
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            lines = [ln.strip() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for li, ln in enumerate(lines, start=1):
                parts = ln.split()
                if len(parts) != 5:
                    bad.append(f"{label}:{li} expected 5 columns, got {len(parts)}")
                    continue
                try:
                    cls = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except Exception:
                    bad.append(f"{label}:{li} non-numeric values")
                    continue

                if cls < 0 or cls >= num_classes:
                    bad.append(f"{label}:{li} class {cls} out of range [0,{num_classes-1}]")
                x, y, w, h = vals
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bad.append(f"{label}:{li} invalid normalized bbox {vals}")

        if bad:
            preview = "\n".join(bad[:25])
            more = "" if len(bad) <= 25 else f"\n... and {len(bad)-25} more"
            raise ValueError(f"Label validation failed for split={split}:\n{preview}{more}")




def _find_latest_weights(project_dir: Path, run_name: str, save_dir_hint: Path | None = None) -> Path | None:
    """Find newest best/last checkpoint across common Ultralytics layouts."""
    candidates = []

    # canonical expectation
    preferred = project_dir / run_name / "weights"
    for p in [preferred / "best.pt", preferred / "last.pt"]:
        if p.exists():
            candidates.append(p)

    # if train() returned save_dir, trust it first
    if save_dir_hint is not None:
        hint = Path(save_dir_hint)
        for p in [hint / "weights" / "best.pt", hint / "weights" / "last.pt", hint / "best.pt", hint / "last.pt"]:
            if p.exists():
                candidates.append(p)

    # search under requested project dir
    if project_dir.exists():
        for pat in ["**/weights/best.pt", "**/weights/last.pt"]:
            for p in project_dir.glob(pat):
                if p.is_file():
                    candidates.append(p)

    # search common alt roots used by ultralytics detect task
    for alt in [Path("runs"), Path("runs/detect"), Path.cwd()]:
        if not alt.exists():
            continue
        for pat in ["**/weights/best.pt", "**/weights/last.pt"]:
            for p in alt.glob(pat):
                if not p.is_file():
                    continue
                # prioritize paths containing run name/project hint
                s = str(p).lower()
                if run_name.lower() in s or project_dir.name.lower() in s:
                    candidates.append(p)

    if not candidates:
        return None

    uniq = list({str(c.resolve()): c.resolve() for c in candidates}.values())
    uniq.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return uniq[0]


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/robot_dataset")
    parser.add_argument("--model", default="yolov8n.pt", help="base model or existing checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, 0, 0,1, cpu, mps")
    parser.add_argument("--project", default="runs/sumo")
    parser.add_argument("--name", default="robot_detector")
    parser.add_argument("--classes", default="robot,blade_front")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 first on Windows to avoid dataloader hangs")
    parser.add_argument("--cache", action="store_true", help="Enable caching images in RAM")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--save-period", type=int, default=5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ensure_structure_and_val_split(dataset_dir)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("At least one class is required", file=sys.stderr)
        return 2

    validate_labels(dataset_dir, len(classes))
    data_yaml = ensure_data_yaml(dataset_dir, classes)

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        print("Ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        return 3

    chosen_device = detect_device(args.device)
    print(f"Training device: {chosen_device}")
    print(f"Dataset: {dataset_dir.resolve()}")
    print(f"Workers: {args.workers} | Cache: {args.cache}")

    model = YOLO(args.model)
    train_result = None
    save_dir_hint = None
    try:
        train_result = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=chosen_device,
            project=args.project,
            name=args.name,
            cache=args.cache,
            workers=args.workers,
            close_mosaic=10,
            patience=30,
            save_period=args.save_period,
            resume=args.resume,
        )
        try:
            save_dir_hint = Path(getattr(model, "trainer", None).save_dir) if getattr(model, "trainer", None) is not None else None
        except Exception:
            save_dir_hint = None
        if save_dir_hint is not None:
            print(f"Ultralytics save_dir: {save_dir_hint}")
    except Exception as e:
        print("\nTraining crashed.", file=sys.stderr)
        print("Most common fixes:", file=sys.stderr)
        print("  1) keep --workers 0 (especially on Windows)", file=sys.stderr)
        print("  2) use --device cpu for debugging", file=sys.stderr)
        print("  3) reduce --imgsz to 640 and --batch to 4", file=sys.stderr)
        print("  4) check labels with this script (already enforced)", file=sys.stderr)
        print(f"Original error: {e}", file=sys.stderr)
        return 5

    target = Path("models/weights/robot_sumo.pt")
    target.parent.mkdir(parents=True, exist_ok=True)

    latest = _find_latest_weights(Path(args.project), args.name, save_dir_hint=save_dir_hint)
    if latest is None:
        print("Training finished, but no weights found (best.pt/last.pt missing).", file=sys.stderr)
        print(f"Searched project root: {Path(args.project).resolve()}", file=sys.stderr)
        print(f"Searched cwd recursively: {Path.cwd().resolve()}", file=sys.stderr)
        if save_dir_hint is not None:
            print(f"Ultralytics save_dir hint: {save_dir_hint}", file=sys.stderr)
        return 4

    target.write_bytes(latest.read_bytes())
    print(f"Saved ready-to-use weights to {target} (from {latest})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
