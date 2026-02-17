"""Desktop UI for Robot Sumo Training Studio.

Run:
  python scripts/training_studio_ui.py
"""

from __future__ import annotations

import hashlib
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


def _slug(s: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")
    return clean or "match"


def _match_id(video_path: str) -> str:
    p = Path(video_path)
    digest = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{_slug(p.stem)}_{digest}"


def _match_paths(video_path: str) -> dict:
    mid = _match_id(video_path)
    base = Path("data/studio/matches") / mid
    return {
        "match_id": mid,
        "clip_path": str(base / "clip" / f"{Path(video_path).stem}_action.mp4"),
        "frames_dir": str(base / "frames"),
        "annotations_dir": str(base / "annotations"),
        "manifest": str(base / "manifest.json"),
    }


def _count_files(folder: Path, patterns: tuple[str, ...]) -> int:
    if not folder.exists():
        return 0
    n = 0
    for pat in patterns:
        n += len(list(folder.glob(pat)))
    return n


def _get_progress(frames_dir: Path, ann_dir: Path) -> tuple[int, int]:
    total_frames = _count_files(frames_dir, ("*.jpg", "*.jpeg", "*.png"))
    done = _count_files(ann_dir, ("*.json",))
    return done, total_frames


def _find_latest_weight_candidate() -> Path | None:
    direct = Path("models/weights/robot_sumo.pt")
    if direct.exists():
        return direct

    candidates = []
    for pat in ["runs/sumo/**/weights/best.pt", "runs/sumo/**/weights/last.pt"]:
        for p in Path(".").glob(pat):
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_project() -> dict:
    if PROJECT_FILE.exists():
        try:
            data = json.loads(PROJECT_FILE.read_text(encoding="utf-8"))
            data.setdefault("videos", [])
            data.setdefault("matches", {})
            data.setdefault("current_video", "")
            data.setdefault("dataset_dir", "data/robot_dataset")
            data.setdefault("weights", "models/weights/robot_sumo.pt")
            data.setdefault("infer_out", "artifacts/training_studio_infer.mp4")
            return data
        except Exception:
            pass
    return {
        "videos": [],
        "matches": {},
        "current_video": "",
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
        self.geometry("1080x760")

        self.state = load_project()
        self._running = False
        self.action_buttons = []

        self._build_ui()
        self._refresh_videos()
        self._sync_ui_from_current_video()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Video Library", padding=8)
        top.pack(fill="x", pady=6)

        self.video_list = tk.Listbox(top, height=7)
        self.video_list.pack(side="left", fill="x", expand=True)
        self.video_list.bind("<<ListboxSelect>>", self._on_video_select)
        self._video_index_to_path = {}

        btns = ttk.Frame(top)
        btns.pack(side="left", padx=8)
        ttk.Button(btns, text="Add videos", command=self.add_videos).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove selected", command=self.remove_selected_video).pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value="No video selected")
        ttk.Label(frm, textvariable=self.status_var).pack(anchor="w", pady=(0, 6))

        paths = ttk.LabelFrame(frm, text="Per-match Paths (auto-managed)", padding=8)
        paths.pack(fill="x", pady=6)

        self.current_var = tk.StringVar(value=self.state.get("current_video", ""))
        self.clip_var = tk.StringVar(value="")
        self.frames_var = tk.StringVar(value="")
        self.ann_var = tk.StringVar(value="")
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

        self.step_var = tk.StringVar(value="Current step: 1) Trim clip")
        ttk.Label(actions, textvariable=self.step_var).pack(anchor="w", pady=(0, 6))

        row1 = ttk.Frame(actions)
        row1.pack(fill="x", pady=3)
        b1 = ttk.Button(row1, text="1) Trim clip", command=self.run_trim)
        b1.pack(side="left", padx=4)
        b2 = ttk.Button(row1, text="2) Extract frames", command=self.run_extract)
        b2.pack(side="left", padx=4)
        b3 = ttk.Button(row1, text="3) Annotate", command=self.run_annotate)
        b3.pack(side="left", padx=4)

        row2 = ttk.Frame(actions)
        row2.pack(fill="x", pady=3)
        b4 = ttk.Button(row2, text="4) Export YOLO labels", command=self.run_export)
        b4.pack(side="left", padx=4)
        b5 = ttk.Button(row2, text="5) Train model", command=self.run_train)
        b5.pack(side="left", padx=4)
        b6 = ttk.Button(row2, text="6) Test on video", command=self.run_infer)
        b6.pack(side="left", padx=4)
        self.action_buttons = [b1, b2, b3, b4, b5, b6]

        train_opts = ttk.Frame(actions)
        train_opts.pack(fill="x", pady=6)
        self.fps_var = tk.StringVar(value="10")
        self.epochs_var = tk.StringVar(value="80")
        self.imgsz_var = tk.StringVar(value="640")
        self.batch_var = tk.StringVar(value="8")
        self.device_var = tk.StringVar(value="auto")
        self.workers_var = tk.StringVar(value="0")
        self.frame_step_var = tk.StringVar(value="1")
        self.max_frames_var = tk.StringVar(value="0")
        self.bbox_only_var = tk.BooleanVar(value=False)
        self.include_ext_var = tk.BooleanVar(value=False)
        self.export_all_var = tk.BooleanVar(value=True)

        for label, var, width in [
            ("extract fps", self.fps_var, 6),
            ("frame step", self.frame_step_var, 6),
            ("max frames", self.max_frames_var, 6),
            ("epochs", self.epochs_var, 6),
            ("imgsz", self.imgsz_var, 6),
            ("batch", self.batch_var, 6),
            ("device", self.device_var, 8),
            ("workers", self.workers_var, 6),
        ]:
            ttk.Label(train_opts, text=label).pack(side="left", padx=3)
            ttk.Entry(train_opts, textvariable=var, width=width).pack(side="left", padx=3)

        guide = ttk.LabelFrame(frm, text="Workflow Guidance", padding=8)
        guide.pack(fill="x", pady=6)
        guide_text = (
            "1) Select one match video (app auto-creates dedicated clip/frames/annotations folders).\n"
            "2) Trim and extract.\n"
            "3) Annotate: each frame can be Annotate / Propagate previous / Skip / Finish match.\n"
            "4) Export labels (current match or all prepared matches) and train.\n"
            "Tip: frame-step > 1 speeds up annotation; app resumes existing annotations by default."
        )
        ttk.Label(guide, text=guide_text, justify="left").pack(anchor="w")

        opts_row = ttk.Frame(guide)
        opts_row.pack(fill="x", pady=4)
        ttk.Checkbutton(opts_row, text="BBox-only annotation (skip blade points)", variable=self.bbox_only_var).pack(side="left", padx=4)
        ttk.Checkbutton(opts_row, text="Export extension classes", variable=self.include_ext_var).pack(side="left", padx=4)
        ttk.Checkbutton(opts_row, text="Export ALL prepared matches", variable=self.export_all_var).pack(side="left", padx=4)

        help_row = ttk.Frame(guide)
        help_row.pack(fill="x", pady=4)
        ttk.Button(help_row, text="Trim key help", command=self.show_trim_help).pack(side="left", padx=4)
        ttk.Button(help_row, text="Annotation help", command=self.show_annot_help).pack(side="left", padx=4)

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
        self._video_index_to_path = {}
        for idx, v in enumerate(self.state.get("videos", [])):
            m = self.state.get("matches", {}).get(v, {})
            done, total = _get_progress(Path(m.get("frames_dir", "")), Path(m.get("annotations_dir", "")))
            mark = "✅" if (total > 0 and done >= total) else "🟨"
            label = f"{mark} {done}/{total}  {Path(v).name}"
            self._video_index_to_path[idx] = v
            self.video_list.insert(tk.END, label)

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _ensure_match_entry(self, video_path: str) -> dict:
        matches = self.state.setdefault("matches", {})
        if video_path not in matches:
            m = _match_paths(video_path)
            matches[video_path] = {
                **m,
                "annotate_completed": False,
                "last_action": "created",
            }
        return matches[video_path]

    def _sync_ui_from_current_video(self):
        video = self.current_var.get().strip()
        if not video:
            return
        m = self._ensure_match_entry(video)
        self.clip_var.set(m["clip_path"])
        self.frames_var.set(m["frames_dir"])
        self.ann_var.set(m["annotations_dir"])

        done, total = _get_progress(Path(m["frames_dir"]), Path(m["annotations_dir"]))
        completed = bool(m.get("annotate_completed", False))
        self.status_var.set(
            f"Match: {Path(video).name} | Progress: {done}/{total} annotated"
            + (" | COMPLETE" if completed else " | IN PROGRESS")
        )

    def save_state(self):
        self.state["current_video"] = self.current_var.get().strip()
        self.state["dataset_dir"] = self.data_var.get().strip()
        self.state["weights"] = self.weights_var.get().strip()
        self.state["infer_out"] = self.infer_var.get().strip()

        cur = self.current_var.get().strip()
        if cur:
            m = self._ensure_match_entry(cur)
            m["clip_path"] = self.clip_var.get().strip()
            m["frames_dir"] = self.frames_var.get().strip()
            m["annotations_dir"] = self.ann_var.get().strip()

        save_project(self.state)
        self._sync_ui_from_current_video()
        self._log("Project saved.")

    def add_videos(self):
        files = filedialog.askopenfilenames(title="Select match videos")
        if not files:
            return
        vids = set(self.state.get("videos", []))
        vids.update(files)
        self.state["videos"] = sorted(vids)
        for v in files:
            self._ensure_match_entry(v)
        self._refresh_videos()
        self.save_state()


    def _on_video_select(self, event=None):
        sel = self.video_list.curselection()
        if not sel:
            return
        v = self._video_index_to_path.get(sel[0])
        if not v:
            return
        self.current_var.set(v)
        self._sync_ui_from_current_video()
        self.save_state()

    def set_selected_video(self):
        self._on_video_select()

    def remove_selected_video(self):
        sel = self.video_list.curselection()
        if not sel:
            return
        v = self._video_index_to_path.get(sel[0])
        if not v:
            return
        self.state["videos"] = [x for x in self.state.get("videos", []) if x != v]
        self.state.setdefault("matches", {}).pop(v, None)
        if self.current_var.get() == v:
            self.current_var.set("")
        self._refresh_videos()
        self.save_state()

    def show_trim_help(self):
        messagebox.showinfo(
            "Trim controls",
            "Trim window keys:\n"
            "  j/l = -/+ 1 frame\n"
            "  a/d = -/+ 15 frames\n"
            "  i = set IN\n"
            "  o = set OUT\n"
            "  s = save clip\n"
            "  q = quit trim"
        )

    def show_annot_help(self):
        messagebox.showinfo(
            "Annotation controls",
            "Frame decision:\n"
            "  a = annotate now\n"
            "  p = propagate previous labels\n"
            "  s = skip frame\n"
            "  f = finish this match\n"
            "  x = exit annotator\n\n"
            "Review step:\n"
            "  Enter = approve frame\n"
            "  r = redo frame\n"
            "  s = skip frame\n"
            "  f = finish this match\n"
            "  x = exit annotator"
        )

    def _set_running(self, running: bool):
        self._running = running
        state = "disabled" if running else "normal"
        for btn in self.action_buttons:
            btn.config(state=state)

    def _run_bg(self, title: str, fn, next_step: str):
        if self._running:
            self._log("A workflow step is already running. Please wait for completion.")
            return

        def worker():
            try:
                self._log(f"== {title} ==")
                rc = fn()
                self._log(f"Done ({title}), rc={rc}")
                if rc == 0:
                    self.after(0, lambda: self.step_var.set(f"Current step: {next_step}"))
            except Exception as e:
                self._log(f"ERROR ({title}): {e}")
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
            finally:
                self.after(0, self._sync_ui_from_current_video)
                self.after(0, lambda: self._set_running(False))

        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()

    def run_trim(self):
        def fn():
            self.save_state()
            video = self.current_var.get().strip()
            if not video:
                raise ValueError("Select a current video first")
            m = self._ensure_match_entry(video)
            m["last_action"] = "trim"
            return studio.trim_video(studio.resolve_video_path(video), Path(m["clip_path"]))

        self._run_bg("Trim", fn, next_step="2) Extract frames")

    def run_extract(self):
        def fn():
            self.save_state()
            video = self.current_var.get().strip()
            if not video:
                raise ValueError("Select a current video first")
            m = self._ensure_match_entry(video)
            fps = float(self.fps_var.get())
            rc = studio.extract_frames(Path(m["clip_path"]), Path(m["frames_dir"]), fps_out=fps)
            if rc == 0:
                m["annotate_completed"] = False
                m["last_action"] = "extract"
                save_project(self.state)
            return rc

        self._run_bg("Extract", fn, next_step="3) Annotate")

    def run_annotate(self):
        def fn():
            self.save_state()
            video = self.current_var.get().strip()
            if not video:
                raise ValueError("Select a current video first")
            m = self._ensure_match_entry(video)
            done_before, total_before = _get_progress(Path(m["frames_dir"]), Path(m["annotations_dir"]))
            self._log(f"Annotation resume state: {done_before}/{total_before} already annotated")
            rc = studio.annotate_frames(
                Path(m["frames_dir"]),
                Path(m["annotations_dir"]),
                start_index=0,
                frame_step=int(self.frame_step_var.get()),
                max_frames=int(self.max_frames_var.get()),
                annotate_blade=not self.bbox_only_var.get(),
                skip_existing=True,
            )
            done, total = _get_progress(Path(m["frames_dir"]), Path(m["annotations_dir"]))
            m["annotate_completed"] = total > 0 and done >= total
            m["last_action"] = "annotate"
            save_project(self.state)
            return rc

        self._run_bg("Annotate", fn, next_step="4) Export YOLO labels")

    def run_export(self):
        def fn():
            self.save_state()
            labels_out = Path(self.data_var.get()) / "labels" / "train"

            if self.export_all_var.get():
                total = 0
                for video, m in self.state.get("matches", {}).items():
                    ann_dir = Path(m.get("annotations_dir", ""))
                    frm_dir = Path(m.get("frames_dir", ""))
                    if not ann_dir.exists() or not any(ann_dir.glob("*.json")):
                        continue
                    prefix = f"{m.get('match_id', 'match')}_"
                    rc = studio.export_yolo_from_json(
                        frm_dir,
                        ann_dir,
                        labels_out,
                        include_extension_classes=self.include_ext_var.get(),
                        filename_prefix=prefix,
                    )
                    if rc != 0:
                        return rc
                    total += 1
                    m["last_action"] = "export"
                save_project(self.state)
                self._log(f"Exported labels from {total} prepared matches")
                return 0

            video = self.current_var.get().strip()
            if not video:
                raise ValueError("Select a current video first")
            m = self._ensure_match_entry(video)
            rc = studio.export_yolo_from_json(
                Path(m["frames_dir"]),
                Path(m["annotations_dir"]),
                labels_out,
                include_extension_classes=self.include_ext_var.get(),
                filename_prefix=f"{m.get('match_id', 'match')}_",
            )
            if rc == 0:
                m["last_action"] = "export"
                save_project(self.state)
            return rc

        self._run_bg("Export YOLO", fn, next_step="5) Train model")

    def run_train(self):
        def fn():
            self.save_state()
            rc = studio.run_training(
                Path(self.data_var.get()),
                "yolov8n.pt",
                int(self.epochs_var.get()),
                int(self.imgsz_var.get()),
                int(self.batch_var.get()),
                self.device_var.get(),
                int(self.workers_var.get()),
            )
            latest = _find_latest_weight_candidate()
            if latest is not None:
                self.weights_var.set(str(latest))
                self._log(f"Using weights: {latest}")
            return rc

        self._run_bg("Train", fn, next_step="6) Test on video")

    def run_infer(self):
        def fn():
            self.save_state()
            w = Path(self.weights_var.get())
            if not w.exists():
                latest = _find_latest_weight_candidate()
                if latest is None:
                    raise FileNotFoundError(
                        "No trained weights found. Train first or set Weights path manually. "
                        "Expected models/weights/robot_sumo.pt or runs/sumo/**/weights/best.pt"
                    )
                self.weights_var.set(str(latest))
                self._log(f"Auto-selected latest weights: {latest}")
            v = studio.resolve_video_path(self.current_var.get())
            return studio.infer_video(Path(self.weights_var.get()), v, Path(self.infer_var.get()))

        self._run_bg("Infer", fn, next_step="1) Trim clip (new video)")


if __name__ == "__main__":
    App().mainloop()
