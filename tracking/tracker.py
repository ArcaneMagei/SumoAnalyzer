"""High-accuracy two-robot tracking on a dohyo-normalized plane.

Design goals for robot sumo:
- Exactly two persistent tracks (R1/R2)
- Robust to fast motion and temporary standstill
- Front-facing estimation (velocity + shape/blade cue)
- Out-of-dohyo (loss) detection
"""

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RobotState:
    id: int
    center_norm: Tuple[int, int]
    center_img: Tuple[int, int]
    bbox_img: Tuple[int, int, int, int]
    confidence: float
    frames_lost: int
    heading_deg: float
    front_img: Tuple[int, int]
    in_dohyo: bool
    eliminated: bool
    history_norm: Deque[Tuple[int, int]]


class RobotTracker:
    """Hybrid detector-tracker on canonical top-down dohyo space."""

    def __init__(self, max_lost_frames: int = 20):
        self.max_lost_frames = max_lost_frames
        self.frame_idx = 0

        # canonical space
        self.norm_size = 800
        self.norm_center = (self.norm_size // 2, self.norm_size // 2)
        self.norm_outer_r = 330
        self.inner_scale = 154.0 / 164.0
        self.norm_inner_r = int(self.norm_outer_r * self.inner_scale)

        # robot geometry in canonical space (20x20 cm)
        cm_per_px = 77.0 / self.norm_inner_r
        self.robot_side_px = 20.0 / cm_per_px
        self.robot_area_expected = self.robot_side_px * self.robot_side_px

        self.tracks: Dict[int, RobotState] = {}
        self.bg_norm: Optional[np.ndarray] = None
        self.prev_norm_gray: Optional[np.ndarray] = None

        # light Kalman filters for center smoothing/prediction
        self.kalman: Dict[int, cv2.KalmanFilter] = {}

    @staticmethod
    def _ellipse_four_points(center: Tuple[float, float], axes: Tuple[float, float], angle_deg: float) -> np.ndarray:
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

    def _homography(self, dohyo_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        src = self._ellipse_four_points(dohyo_info["center"], dohyo_info["axes"], dohyo_info["angle"])
        c = self.norm_center
        r = self.norm_outer_r
        dst = np.array([[c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]], [c[0], c[1] - r]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        Hinv = cv2.getPerspectiveTransform(dst, src)
        return H, Hinv

    def _make_ring_mask(self) -> np.ndarray:
        m = np.zeros((self.norm_size, self.norm_size), dtype=np.uint8)
        cv2.circle(m, self.norm_center, self.norm_inner_r, 255, -1)
        return m

    def _detect_candidates(self, norm_frame: np.ndarray) -> List[Dict]:
        """Detect moving/stationary robots in normalized top-down view."""
        gray = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2HSV)
        mask = self._make_ring_mask()

        # 1) background subtraction catches motion (fast robots)
        if self.bg_norm is None:
            self.bg_norm = gray.astype(np.float32)
        cv2.accumulateWeighted(gray, self.bg_norm, 0.02)
        bg = cv2.convertScaleAbs(self.bg_norm)
        bg_diff = cv2.absdiff(gray, bg)
        _, m_bg = cv2.threshold(bg_diff, 14, 255, cv2.THRESH_BINARY)

        # 2) frame-to-frame motion catches fast bursts
        m_ff = np.zeros_like(gray)
        if self.prev_norm_gray is not None:
            d = cv2.absdiff(gray, self.prev_norm_gray)
            _, m_ff = cv2.threshold(d, 10, 255, cv2.THRESH_BINARY)
        self.prev_norm_gray = gray

        # 3) static robot cue on gray dohyo: dark + textured compact objects
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, m_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 70, 170)

        fused = cv2.bitwise_or(m_bg, m_ff)
        fused = cv2.bitwise_or(fused, m_dark)
        fused = cv2.bitwise_or(fused, edges)
        fused = cv2.bitwise_and(fused, mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.robot_area_expected * 0.25
        max_area = self.robot_area_expected * 3.5

        cands: List[Dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 8 or h < 8:
                continue

            ar = w / float(h)
            if ar < 0.3 or ar > 3.5:
                continue

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            # simple appearance descriptor (helps ID stability)
            roi_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
            hist = cv2.calcHist([hsv], [0, 1], roi_mask, [24, 24], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # front hint from geometry + shiny blade cue
            front_norm, heading = self._estimate_front(norm_frame, cnt, (cx, cy))

            cands.append(
                {
                    "center_norm": (cx, cy),
                    "bbox_norm": (x, y, w, h),
                    "area": area,
                    "hist": hist,
                    "front_norm": front_norm,
                    "heading_deg": heading,
                }
            )

        cands.sort(key=lambda z: z["area"], reverse=True)
        return cands[:6]

    def _estimate_front(self, norm_frame: np.ndarray, contour: np.ndarray, center: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)
        v1 = box[1] - box[0]
        v2 = box[2] - box[1]
        axis = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
        axis = axis.astype(np.float32)
        axis /= (np.linalg.norm(axis) + 1e-6)

        radial = np.array([center[0] - self.norm_center[0], center[1] - self.norm_center[1]], dtype=np.float32)
        if np.dot(axis, radial) < 0:
            axis = -axis

        pts = contour.reshape(-1, 2)
        front_geom = pts[int(np.argmax(pts @ axis))]

        x, y, w, h = cv2.boundingRect(contour)
        roi = norm_frame[y:y + h, x:x + w]
        front_blade = front_geom
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 85, 255]))
            if np.count_nonzero(bright) > (w * h * 0.01):
                ys, xs = np.where(bright > 0)
                bpts = np.stack([xs + x, ys + y], axis=1)
                front_blade = bpts[int(np.argmax(bpts @ axis))]

        front = np.round(0.65 * front_geom + 0.35 * front_blade).astype(int)
        heading = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return (int(front[0]), int(front[1])), heading

    @staticmethod
    def _map_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        p = pts.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, H).reshape(-1, 2)

    @staticmethod
    def _bbox_from_points(pts: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = frame_shape
        x1 = max(0, int(np.floor(np.min(pts[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pts[:, 1]))))
        x2 = min(w - 1, int(np.ceil(np.max(pts[:, 0]))))
        y2 = min(h - 1, int(np.ceil(np.max(pts[:, 1]))))
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _bbox_norm_to_img(self, bbox_norm: Tuple[int, int, int, int], Hinv: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox_norm
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        mapped = self._map_points(pts, Hinv)
        return self._bbox_from_points(mapped, frame_shape)

    def _point_norm_to_img(self, p: Tuple[int, int], Hinv: np.ndarray) -> Tuple[int, int]:
        mapped = self._map_points(np.array([[p[0], p[1]]], dtype=np.float32), Hinv)[0]
        return int(round(mapped[0])), int(round(mapped[1]))

    def _init_kalman(self, rid: int, p: Tuple[int, int]) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.4
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[p[0]], [p[1]], [0], [0]], dtype=np.float32)
        self.kalman[rid] = kf

    def _predict(self, rid: int) -> Tuple[float, float]:
        pred = self.kalman[rid].predict()
        return float(pred[0][0]), float(pred[1][0])

    def _correct(self, rid: int, p: Tuple[int, int]) -> None:
        m = np.array([[np.float32(p[0])], [np.float32(p[1])]])
        self.kalman[rid].correct(m)

    def _init_tracks_if_needed(self, cands: List[Dict], Hinv: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        if self.tracks or len(cands) < 2:
            return
        for rid, c in zip([1, 2], cands[:2]):
            center_img = self._point_norm_to_img(c["center_norm"], Hinv)
            bbox_img = self._bbox_norm_to_img(c["bbox_norm"], Hinv, frame_shape)
            front_img = self._point_norm_to_img(c["front_norm"], Hinv)
            self.tracks[rid] = RobotState(
                id=rid,
                center_norm=c["center_norm"],
                center_img=center_img,
                bbox_img=bbox_img,
                confidence=0.8,
                frames_lost=0,
                heading_deg=c["heading_deg"],
                front_img=front_img,
                in_dohyo=True,
                eliminated=False,
                history_norm=deque([c["center_norm"]], maxlen=120),
            )
            self._init_kalman(rid, c["center_norm"])

    def _assign(self, cands: List[Dict]) -> Dict[int, Optional[int]]:
        assign = {1: None, 2: None}
        if not self.tracks:
            return assign
        if not cands:
            return assign

        preds = {}
        for rid in [1, 2]:
            if rid in self.kalman:
                preds[rid] = self._predict(rid)

        # cost = position distance + heading disagreement
        costs = np.full((2, len(cands)), 1e6, dtype=np.float32)
        for i, rid in enumerate([1, 2]):
            if rid not in self.tracks:
                continue
            px, py = preds.get(rid, self.tracks[rid].center_norm)
            track_heading = self.tracks[rid].heading_deg
            for j, c in enumerate(cands):
                dx = px - c["center_norm"][0]
                dy = py - c["center_norm"][1]
                d = np.hypot(dx, dy)
                hd = abs(((track_heading - c["heading_deg"] + 180) % 360) - 180)
                costs[i, j] = d + 0.25 * hd

        if len(cands) == 1:
            r = int(np.argmin([costs[0, 0], costs[1, 0]]))
            if min(costs[0, 0], costs[1, 0]) < 120:
                assign[[1, 2][r]] = 0
            return assign

        best = (None, None, 1e9)
        for j1 in range(len(cands)):
            for j2 in range(len(cands)):
                if j1 == j2:
                    continue
                s = costs[0, j1] + costs[1, j2]
                if s < best[2]:
                    best = (j1, j2, s)

        if best[0] is not None and costs[0, best[0]] < 120:
            assign[1] = int(best[0])
        if best[1] is not None and costs[1, best[1]] < 120:
            assign[2] = int(best[1])
        return assign

    def _check_dohyo_status(self, p_norm: Tuple[int, int]) -> Tuple[bool, bool]:
        d = np.hypot(p_norm[0] - self.norm_center[0], p_norm[1] - self.norm_center[1])
        in_dohyo = d <= self.norm_inner_r
        # eliminated if center clearly outside by safety margin
        eliminated = d > self.norm_inner_r * 1.05
        return in_dohyo, eliminated

    def update(self, frame: np.ndarray, dohyo_info: Dict) -> Dict[int, RobotState]:
        self.frame_idx += 1

        H, Hinv = self._homography(dohyo_info)
        norm = cv2.warpPerspective(frame, H, (self.norm_size, self.norm_size))

        cands = self._detect_candidates(norm)
        self._init_tracks_if_needed(cands, Hinv, frame.shape[:2])
        if not self.tracks:
            return self.tracks

        assign = self._assign(cands)

        for rid in [1, 2]:
            if rid not in self.tracks:
                continue
            t = self.tracks[rid]
            j = assign.get(rid)

            if j is None:
                t.frames_lost += 1
                t.confidence = max(0.05, t.confidence - 0.08)
                if rid in self.kalman:
                    px, py = self._predict(rid)
                    t.center_norm = (int(px), int(py))
                t.center_img = self._point_norm_to_img(t.center_norm, Hinv)
                in_d, elim = self._check_dohyo_status(t.center_norm)
                t.in_dohyo = in_d
                t.eliminated = elim
                if t.frames_lost > self.max_lost_frames:
                    del self.tracks[rid]
                    if rid in self.kalman:
                        del self.kalman[rid]
                continue

            c = cands[j]
            t.center_norm = c["center_norm"]
            t.history_norm.append(c["center_norm"])
            t.center_img = self._point_norm_to_img(c["center_norm"], Hinv)
            t.bbox_img = self._bbox_norm_to_img(c["bbox_norm"], Hinv, frame.shape[:2])
            t.front_img = self._point_norm_to_img(c["front_norm"], Hinv)

            # heading: prefer velocity heading when moving, else geometry heading
            if len(t.history_norm) >= 4:
                p0 = t.history_norm[-4]
                p1 = t.history_norm[-1]
                vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                speed = np.hypot(vx, vy)
                if speed > 6:
                    t.heading_deg = float(np.degrees(np.arctan2(vy, vx)))
                else:
                    t.heading_deg = c["heading_deg"]
            else:
                t.heading_deg = c["heading_deg"]

            t.frames_lost = 0
            t.confidence = min(1.0, t.confidence + 0.03)
            self._correct(rid, c["center_norm"])

            in_d, elim = self._check_dohyo_status(c["center_norm"])
            t.in_dohyo = in_d
            t.eliminated = elim

        return self.tracks

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for rid in [1, 2]:
            if rid not in self.tracks:
                continue
            t = self.tracks[rid]
            if t.eliminated:
                color = (0, 0, 255)
            elif t.in_dohyo:
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)

            x, y, w, h = t.bbox_img
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.circle(out, t.center_img, 4, color, -1)
            cv2.circle(out, t.front_img, 4, (255, 255, 0), -1)
            cv2.line(out, t.center_img, t.front_img, (255, 255, 0), 2)

            status = "OUT" if not t.in_dohyo else "IN"
            if t.eliminated:
                status = "LOST"
            label = f"R{rid} {status} {t.confidence:.2f}"
            cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        return out
