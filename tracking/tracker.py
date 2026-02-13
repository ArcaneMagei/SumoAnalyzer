"""
Two-robot tracker for robot sumo matches.
- Keeps stable IDs (1 and 2 only)
- Handles temporary occlusions
- Estimates front/blade point per robot
"""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RobotState:
    id: int
    position: Tuple[int, int]
    velocity: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    confidence: float
    frames_lost: int
    position_history: Deque[Tuple[int, int]]
    color_histogram: np.ndarray
    front_point: Tuple[int, int]
    heading_deg: float
    occluded: bool


class RobotTracker:
    """Tracks exactly two robots inside dohyo and exposes blade/front estimate."""

    def __init__(
        self,
        dohyo_center: Tuple[int, int],
        dohyo_radius: int,
        max_lost_frames: int = 30,
        position_history_size: int = 20,
    ):
        self.dohyo_center = dohyo_center
        self.dohyo_radius = dohyo_radius
        self.max_lost_frames = max_lost_frames
        self.position_history_size = position_history_size

        self.robots: Dict[int, RobotState] = {}
        self.frame_count = 0
        self.prev_gray: Optional[np.ndarray] = None

        # Tuned for 3kg mega sumo visual footprint in common 1080p-ish recordings.
        self.min_area = 300
        self.max_area = int(np.pi * (dohyo_radius * 0.35) ** 2)

    def _inside_dohyo(self, p: Tuple[int, int], margin: float = 0.98) -> bool:
        d = np.hypot(p[0] - self.dohyo_center[0], p[1] - self.dohyo_center[1])
        return d <= self.dohyo_radius * margin

    def _estimate_front_point(
        self, frame: np.ndarray, contour: np.ndarray, center: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], float]:
        """Estimate front using minAreaRect axis + shiny metal cue."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)

        v1 = box[1] - box[0]
        v2 = box[2] - box[1]
        axis = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
        axis = axis.astype(np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-6)

        # Prefer outward-facing direction from dohyo center.
        radial = np.array(
            [center[0] - self.dohyo_center[0], center[1] - self.dohyo_center[1]],
            dtype=np.float32,
        )
        if np.dot(axis, radial) < 0:
            axis = -axis

        pts = contour.reshape(-1, 2)
        geom_front = pts[int(np.argmax(pts @ axis))]

        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y : y + h, x : x + w]
        blade_front = geom_front
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Low sat + high value catches shiny metallic blades.
            bright = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 90, 255]))
            if np.count_nonzero(bright) > (w * h * 0.008):
                ys, xs = np.where(bright > 0)
                bright_pts = np.stack([xs + x, ys + y], axis=1)
                blade_front = bright_pts[int(np.argmax(bright_pts @ axis))]

        front = np.round(0.65 * geom_front + 0.35 * blade_front).astype(int)
        heading_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return (int(front[0]), int(front[1])), heading_deg

    def detect_robots(self, frame: np.ndarray) -> List[Dict]:
        """Return up to two robot candidates from fused segmentation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Dohyo circular mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, self.dohyo_center, self.dohyo_radius, 255, -1)

        # Black robots on black dohyo is hard: fuse motion and edge-rich areas.
        edges = cv2.Canny(gray, 60, 160)
        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        motion = np.zeros_like(gray)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray

        fused = cv2.bitwise_or(dark, motion)
        fused = cv2.bitwise_or(fused, edges)
        fused = cv2.bitwise_and(fused, mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            ar = w / float(h)
            if ar < 0.25 or ar > 4.0:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if not self._inside_dohyo((cx, cy)):
                continue

            roi_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
            hist = cv2.calcHist([hsv], [0, 1], roi_mask, [32, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            front_point, heading_deg = self._estimate_front_point(frame, cnt, (cx, cy))
            candidates.append(
                {
                    "position": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": area,
                    "hist": hist,
                    "front_point": front_point,
                    "heading_deg": heading_deg,
                }
            )

        candidates.sort(key=lambda d: d["area"], reverse=True)
        return candidates[:2]

    def _init_tracks(self, detections: List[Dict]) -> None:
        """Initialize exactly two tracks when possible."""
        if len(detections) < 2 or self.robots:
            return
        for rid, det in enumerate(detections[:2], start=1):
            self.robots[rid] = RobotState(
                id=rid,
                position=det["position"],
                velocity=(0.0, 0.0),
                bbox=det["bbox"],
                confidence=0.8,
                frames_lost=0,
                position_history=deque([det["position"]], maxlen=self.position_history_size),
                color_histogram=det["hist"],
                front_point=det["front_point"],
                heading_deg=det["heading_deg"],
                occluded=False,
            )

    def _match_two_tracks(self, detections: List[Dict]) -> Dict[int, Optional[int]]:
        """Return mapping track_id -> detection index (or None)."""
        if not self.robots:
            return {}

        track_ids = sorted(self.robots.keys())[:2]
        if not detections:
            return {tid: None for tid in track_ids}

        costs = []
        for tid in track_ids:
            r = self.robots[tid]
            pred = (r.position[0] + r.velocity[0], r.position[1] + r.velocity[1])
            row = []
            for d in detections:
                pos_dist = np.hypot(pred[0] - d["position"][0], pred[1] - d["position"][1])
                pos_cost = pos_dist / 140.0
                color_sim = cv2.compareHist(r.color_histogram, d["hist"], cv2.HISTCMP_CORREL)
                color_cost = 1.0 - max(-1.0, min(1.0, color_sim))
                row.append(0.75 * pos_cost + 0.25 * color_cost)
            costs.append(row)

        # Since we have max 2x2, brute-force assignments.
        mapping = {tid: None for tid in track_ids}
        if len(track_ids) == 1:
            best_j = int(np.argmin(costs[0]))
            if costs[0][best_j] < 0.85:
                mapping[track_ids[0]] = best_j
            return mapping

        if len(detections) == 1:
            best_tid = track_ids[int(np.argmin([costs[0][0], costs[1][0]]))]
            if min(costs[0][0], costs[1][0]) < 0.85:
                mapping[best_tid] = 0
            return mapping

        c00, c01 = costs[0][0], costs[0][1]
        c10, c11 = costs[1][0], costs[1][1]
        if (c00 + c11) <= (c01 + c10):
            if c00 < 0.85:
                mapping[track_ids[0]] = 0
            if c11 < 0.85:
                mapping[track_ids[1]] = 1
        else:
            if c01 < 0.85:
                mapping[track_ids[0]] = 1
            if c10 < 0.85:
                mapping[track_ids[1]] = 0
        return mapping

    def update(self, frame: np.ndarray) -> Dict[int, RobotState]:
        self.frame_count += 1
        detections = self.detect_robots(frame)
        self._init_tracks(detections)

        # Track IDs are fixed (1 and 2 once initialized); never spawn 3rd+ robot IDs.
        if not self.robots:
            return self.robots

        assign = self._match_two_tracks(detections)

        for rid, robot in self.robots.items():
            det_idx = assign.get(rid, None)
            if det_idx is None:
                robot.frames_lost += 1
                robot.occluded = True
                robot.confidence = max(0.05, robot.confidence - 0.06)
                px = int(robot.position[0] + robot.velocity[0])
                py = int(robot.position[1] + robot.velocity[1])
                robot.position = (px, py)
                robot.position_history.append(robot.position)
                continue

            det = detections[det_idx]
            old = robot.position
            new = det["position"]
            vx = 0.65 * robot.velocity[0] + 0.35 * (new[0] - old[0])
            vy = 0.65 * robot.velocity[1] + 0.35 * (new[1] - old[1])

            robot.position = new
            robot.velocity = (vx, vy)
            robot.bbox = det["bbox"]
            robot.frames_lost = 0
            robot.occluded = False
            robot.confidence = min(1.0, robot.confidence + 0.04)
            robot.position_history.append(new)
            robot.front_point = det["front_point"]
            robot.heading_deg = det["heading_deg"]
            robot.color_histogram = 0.9 * robot.color_histogram + 0.1 * det["hist"]

        # keep tracks present even if briefly lost, but hard reset if totally gone too long
        if all(r.frames_lost > self.max_lost_frames for r in self.robots.values()):
            self.robots = {}

        return self.robots

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for rid in sorted(self.robots.keys()):
            r = self.robots[rid]
            color = (0, 255, 0) if not r.occluded else (0, 255, 255)
            x, y, w, h = r.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.circle(out, r.position, 5, (0, 0, 255), -1)
            cv2.circle(out, r.front_point, 4, (255, 255, 0), -1)
            cv2.line(out, r.position, r.front_point, (255, 255, 0), 2)
            label = f"R{rid} {r.confidence:.2f}"
            if r.occluded:
                label += " OCC"
            cv2.putText(out, label, (r.position[0] - 30, r.position[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if len(r.position_history) > 1:
                pts = np.array(list(r.position_history), dtype=np.int32)
                cv2.polylines(out, [pts], False, color, 1)

        cv2.circle(out, self.dohyo_center, self.dohyo_radius, (0, 255, 0), 3)
        cv2.putText(out, f"Frame: {self.frame_count} | Tracks: {len(self.robots)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out
