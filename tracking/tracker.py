"""Robust two-robot tracker using dohyo-normalized view."""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RobotState:
    id: int
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # in original frame
    confidence: float
    frames_lost: int
    history: Deque[Tuple[int, int]]


class RobotTracker:
    """Track exactly two robots and return stable bounding boxes."""

    def __init__(self, max_lost_frames: int = 25):
        self.max_lost_frames = max_lost_frames
        self.tracks: Dict[int, RobotState] = {}
        self.frame_idx = 0
        self.background: Optional[np.ndarray] = None  # on normalized dohyo view

        self.out_size = 640
        self.circle_center = (self.out_size // 2, self.out_size // 2)
        self.circle_radius = 250
        self.inner_scale = 154.0 / 164.0

    @staticmethod
    def _ellipse_points(center: Tuple[float, float], axes: Tuple[float, float], angle_deg: float) -> np.ndarray:
        cx, cy = center
        a = axes[0] / 2.0
        b = axes[1] / 2.0
        t = np.deg2rad(angle_deg)
        pts = []
        for phi in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
            x = a * np.cos(phi)
            y = b * np.sin(phi)
            xr = x * np.cos(t) - y * np.sin(t)
            yr = x * np.sin(t) + y * np.cos(t)
            pts.append([cx + xr, cy + yr])
        return np.array(pts, dtype=np.float32)

    def _compute_homography(self, dohyo_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        src = self._ellipse_points(dohyo_info["center"], dohyo_info["axes"], dohyo_info["angle"])
        c = self.circle_center
        r = self.circle_radius
        dst = np.array([[c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]], [c[0], c[1] - r]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        Hinv = cv2.getPerspectiveTransform(dst, src)
        return H, Hinv

    def _norm_candidates(self, norm_frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        inner_r = int(self.circle_radius * self.inner_scale)
        cv2.circle(mask, self.circle_center, inner_r, 255, -1)

        # running background on normalized dohyo
        if self.background is None:
            self.background = gray.astype(np.float32)
        cv2.accumulateWeighted(gray, self.background, 0.03)
        bg = cv2.convertScaleAbs(self.background)
        motion = cv2.absdiff(gray, bg)
        _, motion = cv2.threshold(motion, 18, 255, cv2.THRESH_BINARY)

        # dark/contrast cue for robots on gray dohyo
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        fused = cv2.bitwise_or(motion, dark)
        fused = cv2.bitwise_and(fused, mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # expected 20x20cm robot in normalized frame
        cm_per_px = (154.0 / 2.0) / float(inner_r)
        robot_side_px = 20.0 / cm_per_px
        expected_area = robot_side_px * robot_side_px
        min_area = expected_area * 0.25
        max_area = expected_area * 3.0

        cands = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 5 or h <= 5:
                continue
            ar = w / float(h)
            if ar < 0.35 or ar > 2.8:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cands.append({"center": (cx, cy), "bbox": (x, y, w, h), "area": a})

        cands.sort(key=lambda z: z["area"], reverse=True)
        return cands[:4]

    @staticmethod
    def _bbox_from_points(pts: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
        x_max = min(w - 1, int(np.ceil(np.max(pts[:, 0]))))
        y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
        y_max = min(h - 1, int(np.ceil(np.max(pts[:, 1]))))
        return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

    def _norm_bbox_to_original(self, bbox: Tuple[int, int, int, int], Hinv: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ], dtype=np.float32).reshape(-1, 1, 2)
        back = cv2.perspectiveTransform(corners, Hinv).reshape(-1, 2)
        return self._bbox_from_points(back, frame_shape)

    def _assign_tracks(self, candidates: List[Dict]) -> Dict[int, Optional[int]]:
        ids = [1, 2]
        if not self.tracks:
            return {}

        if not candidates:
            return {1: None, 2: None}

        cost = np.full((2, len(candidates)), 1e6, dtype=np.float32)
        for i, rid in enumerate(ids):
            if rid not in self.tracks:
                continue
            cx, cy = self.tracks[rid].center
            for j, c in enumerate(candidates):
                dx = cx - c["center"][0]
                dy = cy - c["center"][1]
                cost[i, j] = np.hypot(dx, dy)

        assign = {1: None, 2: None}
        if len(candidates) == 1:
            r = int(np.argmin([cost[0, 0], cost[1, 0]]))
            if cost[r, 0] < 120:
                assign[ids[r]] = 0
            return assign

        # brute force for two tracks / up to 4 candidates
        best = (None, None, 1e9)
        for j1 in range(len(candidates)):
            for j2 in range(len(candidates)):
                if j1 == j2:
                    continue
                csum = cost[0, j1] + cost[1, j2]
                if csum < best[2]:
                    best = (j1, j2, csum)
        if best[0] is not None and cost[0, best[0]] < 120:
            assign[1] = int(best[0])
        if best[1] is not None and cost[1, best[1]] < 120:
            assign[2] = int(best[1])
        return assign

    def _init_if_needed(self, candidates: List[Dict], Hinv: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        if self.tracks or len(candidates) < 2:
            return

        for rid, c in zip([1, 2], candidates[:2]):
            orig_bbox = self._norm_bbox_to_original(c["bbox"], Hinv, frame_shape)
            self.tracks[rid] = RobotState(
                id=rid,
                center=c["center"],
                bbox=orig_bbox,
                confidence=0.8,
                frames_lost=0,
                history=deque([c["center"]], maxlen=40),
            )

    def update(self, frame: np.ndarray, dohyo_info: Dict) -> Dict[int, RobotState]:
        self.frame_idx += 1
        H, Hinv = self._compute_homography(dohyo_info)
        norm = cv2.warpPerspective(frame, H, (self.out_size, self.out_size))

        cands = self._norm_candidates(norm)
        self._init_if_needed(cands, Hinv, frame.shape[:2])

        if not self.tracks:
            return self.tracks

        assign = self._assign_tracks(cands)

        for rid in [1, 2]:
            if rid not in self.tracks:
                continue
            t = self.tracks[rid]
            j = assign.get(rid)
            if j is None:
                t.frames_lost += 1
                t.confidence = max(0.05, t.confidence - 0.08)
                if t.frames_lost > self.max_lost_frames:
                    del self.tracks[rid]
                continue

            c = cands[j]
            t.center = c["center"]
            t.history.append(c["center"])
            t.frames_lost = 0
            t.confidence = min(1.0, t.confidence + 0.03)
            t.bbox = self._norm_bbox_to_original(c["bbox"], Hinv, frame.shape[:2])

        return self.tracks

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for rid in [1, 2]:
            if rid not in self.tracks:
                continue
            t = self.tracks[rid]
            color = (0, 255, 0) if t.frames_lost == 0 else (0, 255, 255)
            x, y, w, h = t.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f"R{rid} {t.confidence:.2f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
