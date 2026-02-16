"""Desktop UI for Robot Sumo Training Studio.

Run:
  python scripts/training_studio_ui.py
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import sys

# Support both:
#   python scripts/training_studio_ui.py
#   python -m scripts.training_studio_ui
try:
    from scripts import training_studio as studio
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts import training_studio as studio


PROJECT_FILE = Path("data/studio/project.json")


def load_project() -> dict:
    if PROJECT_FILE.exists():
        try:
            return json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "videos": [],
        "current_video": "",
        "clip_path": "data/studio/clips/current_action.mp4",
        "frames_dir": "data/studio/frames",
        "annotations_dir": "data/studio/annotations",
        "dataset_dir": "data/robot_dataset",
        "weights": "models/weights/robot_sumo.pt",
        "infer_out": "artifacts/training_studio_infer.mp4",
    }


def save_project(data: dict) -> None:
    PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Robot Sumo Training Studio")
        self.geometry("960x720")

        self.state = load_project()

        self._build_ui()
        self._refresh_videos()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Video Library", padding=8)
        top.pack(fill="x", pady=6)

        self.video_list = tk.Listbox(top, height=6)
        self.video_list.pack(side="left", fill="x", expand=True)

        btns = ttk.Frame(top)
        btns.pack(side="left", padx=8)
        ttk.Button(btns, text="Add videos", command=self.add_videos).pack(fill="x", pady=2)
        ttk.Button(btns, text="Set selected", command=self.set_selected_video).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove selected", command=self.remove_selected_video).pack(fill="x", pady=2)

        paths = ttk.LabelFrame(frm, text="Paths", padding=8)
        paths.pack(fill="x", pady=6)

        self.current_var = tk.StringVar(value=self.state.get("current_video", ""))
        self.clip_var = tk.StringVar(value=self.state.get("clip_path", "data/studio/clips/current_action.mp4"))
        self.frames_var = tk.StringVar(value=self.state.get("frames_dir", "data/studio/frames"))
        self.ann_var = tk.StringVar(value=self.state.get("annotations_dir", "data/studio/annotations"))
        self.data_var = tk.StringVar(value=self.state.get("dataset_dir", "data/robot_dataset"))
        self.weights_var = tk.StringVar(value=self.state.get("weights", "models/weights/robot_sumo.pt"))
        self.infer_var = tk.StringVar(value=self.state.get("infer_out", "artifacts/training_studio_infer.mp4"))

        self._add_labeled_entry(paths, "Current video", self.current_var)
        self._add_labeled_entry(paths, "Clip path", self.clip_var)
        self._add_labeled_entry(paths, "Frames dir", self.frames_var)
        self._add_labeled_entry(paths, "Annotations dir", self.ann_var)
        self._add_labeled_entry(paths, "Dataset dir", self.data_var)
        self._add_labeled_entry(paths, "Weights", self.weights_var)
        self._add_labeled_entry(paths, "Infer output", self.infer_var)

        actions = ttk.LabelFrame(frm, text="Workflow Actions", padding=8)
        actions.pack(fill="x", pady=6)

        row1 = ttk.Frame(actions)
        row1.pack(fill="x", pady=3)
        ttk.Button(row1, text="1) Trim clip", command=self.run_trim).pack(side="left", padx=4)
        ttk.Button(row1, text="2) Extract frames", command=self.run_extract).pack(side="left", padx=4)
        ttk.Button(row1, text="3) Annotate", command=self.run_annotate).pack(side="left", padx=4)

        row2 = ttk.Frame(actions)
        row2.pack(fill="x", pady=3)
        ttk.Button(row2, text="4) Export YOLO labels", command=self.run_export).pack(side="left", padx=4)
        ttk.Button(row2, text="5) Train model", command=self.run_train).pack(side="left", padx=4)
        ttk.Button(row2, text="6) Test on video", command=self.run_infer).pack(side="left", padx=4)

        train_opts = ttk.Frame(actions)
        train_opts.pack(fill="x", pady=6)
        self.fps_var = tk.StringVar(value="10")
        self.epochs_var = tk.StringVar(value="120")
        self.imgsz_var = tk.StringVar(value="960")
        self.batch_var = tk.StringVar(value="16")
        self.device_var = tk.StringVar(value="auto")
        self.workers_var = tk.StringVar(value="0")

        for label, var, width in [
            ("extract fps", self.fps_var, 6),
            ("epochs", self.epochs_var, 6),
            ("imgsz", self.imgsz_var, 6),
            ("batch", self.batch_var, 6),
            ("device", self.device_var, 8),
            ("workers", self.workers_var, 6),
        ]:
            ttk.Label(train_opts, text=label).pack(side="left", padx=3)
            ttk.Entry(train_opts, textvariable=var, width=width).pack(side="left", padx=3)

        logf = ttk.LabelFrame(frm, text="Log", padding=8)
        logf.pack(fill="both", expand=True, pady=6)
        self.log = tk.Text(logf, height=18)
        self.log.pack(fill="both", expand=True)

        ttk.Button(frm, text="Save project", command=self.save_state).pack(anchor="e", pady=6)

    def _add_labeled_entry(self, parent, label, var):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text=label, width=14).pack(side="left")
        ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True)

    def _refresh_videos(self):
        self.video_list.delete(0, tk.END)
        for v in self.state.get("videos", []):
            self.video_list.insert(tk.END, v)

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def save_state(self):
        self.state["current_video"] = self.current_var.get().strip()
        self.state["clip_path"] = self.clip_var.get().strip()
        self.state["frames_dir"] = self.frames_var.get().strip()
        self.state["annotations_dir"] = self.ann_var.get().strip()
        self.state["dataset_dir"] = self.data_var.get().strip()
        self.state["weights"] = self.weights_var.get().strip()
        self.state["infer_out"] = self.infer_var.get().strip()
        save_project(self.state)
        self._log("Project saved.")

    def add_videos(self):
        files = filedialog.askopenfilenames(title="Select match videos")
        if not files:
            return
        vids = set(self.state.get("videos", []))
        vids.update(files)
        self.state["videos"] = sorted(vids)
        self._refresh_videos()
        self.save_state()

    def set_selected_video(self):
        sel = self.video_list.curselection()
        if not sel:
            return
        v = self.video_list.get(sel[0])
        self.current_var.set(v)
        self.save_state()

    def remove_selected_video(self):
        sel = self.video_list.curselection()
        if not sel:
            return
        v = self.video_list.get(sel[0])
        self.state["videos"] = [x for x in self.state.get("videos", []) if x != v]
        if self.current_var.get() == v:
            self.current_var.set("")
        self._refresh_videos()
        self.save_state()

    def _run_bg(self, title: str, fn):
        def worker():
            try:
                self._log(f"== {title} ==")
                rc = fn()
                self._log(f"Done ({title}), rc={rc}")
            except Exception as e:
                self._log(f"ERROR ({title}): {e}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def run_trim(self):
        def fn():
            self.save_state()
            v = studio.resolve_video_path(self.current_var.get())
            return studio.trim_video(v, Path(self.clip_var.get()))

        self._run_bg("Trim", fn)

    def run_extract(self):
        def fn():
            self.save_state()
            v = studio.resolve_video_path(self.clip_var.get())
            fps = float(self.fps_var.get())
            return studio.extract_frames(v, Path(self.frames_var.get()), fps_out=fps)

        self._run_bg("Extract", fn)

    def run_annotate(self):
        def fn():
            self.save_state()
            return studio.annotate_frames(Path(self.frames_var.get()), Path(self.ann_var.get()), start_index=0)

        self._run_bg("Annotate", fn)

    def run_export(self):
        def fn():
            self.save_state()
            labels_out = Path(self.data_var.get()) / "labels" / "train"
            return studio.export_yolo_from_json(Path(self.frames_var.get()), Path(self.ann_var.get()), labels_out)

        self._run_bg("Export YOLO", fn)

    def run_train(self):
        def fn():
            self.save_state()
            return studio.run_training(
                Path(self.data_var.get()),
                "yolov8n.pt",
                int(self.epochs_var.get()),
                int(self.imgsz_var.get()),
                int(self.batch_var.get()),
                self.device_var.get(),
                int(self.workers_var.get()),
            )

        self._run_bg("Train", fn)

    def run_infer(self):
        def fn():
            self.save_state()
            v = studio.resolve_video_path(self.current_var.get())
            return studio.infer_video(Path(self.weights_var.get()), v, Path(self.infer_var.get()))

        self._run_bg("Infer", fn)


if __name__ == "__main__":
    App().mainloop()
