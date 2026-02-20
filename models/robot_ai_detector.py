"""Optional AI detector wrapper for robot-sumo.

Uses Ultralytics YOLO weights if available.
Expected training classes include robot body class at index 0 by default.
"""

from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np


class AIRobotDetector:
    def __init__(self, weights_path: str, conf: float = 0.2, class_id: Optional[int] = None):
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.class_id = class_id
        self.ready = False
        self.error = None

        try:
            from ultralytics import YOLO
            if not self.weights_path.exists():
                self.error = f"weights not found: {self.weights_path}"
                return
            self.model = YOLO(str(self.weights_path))
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self.ready = False

    def detect(self, frame_bgr: np.ndarray, max_det: int = 4) -> List[Dict]:
        if not self.ready:
            return []
        try:
            results = self.model.predict(frame_bgr, conf=self.conf, verbose=False, imgsz=960, max_det=max_det)
            if not results or len(results[0].boxes) == 0:
                return []
            detections = []
            for b in results[0].boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if self.class_id is not None and cls != self.class_id:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                if w <= 2 or h <= 2:
                    continue
                detections.append(
                    {
                        "bbox_norm": (x, y, w, h),
                        "center_norm": (x + w // 2, y + h // 2),
                        "ai_confidence": float(b.conf[0]),
                        "cls": cls,
                    }
                )
            return detections
        except Exception:
            return []
