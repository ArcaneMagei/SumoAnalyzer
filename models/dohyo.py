"""Dohyo detection/tracking with ellipse support and overlay helpers."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class DohyoDetector:
    """Detect and track dohyo ellipse frame-by-frame."""

    def __init__(self, expected_diameter_cm: float = 154.0, border_width_cm: float = 5.0):
        self.expected_diameter = expected_diameter_cm
        self.border_width = border_width_cm
        self.total_diameter = expected_diameter_cm + 2.0 * border_width_cm
        self.last_detection_info: Optional[Dict] = None
        self.smooth_alpha = 0.35

    def _white_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # robust white under varied lighting
        m_hsv = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 95, 255]))

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        l_blur = cv2.GaussianBlur(l, (0, 0), 2.2)
        _, m_adapt = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # emphasize strong bright edges around ring
        gradx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grady = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gradx), 0.5, cv2.convertScaleAbs(grady), 0.5, 0))
        _, m_grad = cv2.threshold(mag, 42, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(m_hsv, m_adapt)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 55]))))
        mask = cv2.bitwise_or(mask, m_grad)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
        return mask

    @staticmethod
    def _ellipse_points(center: Tuple[float, float], axes: Tuple[float, float], angle_deg: float, scale: float = 1.0) -> np.ndarray:
        cx, cy = center
        a = (axes[0] / 2.0) * scale
        b = (axes[1] / 2.0) * scale
        t = np.deg2rad(angle_deg)
        pts = []
        for phi in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
            x = a * np.cos(phi)
            y = b * np.sin(phi)
            xr = x * np.cos(t) - y * np.sin(t)
            yr = x * np.sin(t) + y * np.cos(t)
            pts.append([cx + xr, cy + yr])
        return np.array(pts, dtype=np.float32)

    def _candidate_score(self, cand: Dict, white_mask: np.ndarray, prev: Optional[Dict]) -> float:
        h, w = white_mask.shape[:2]
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        in_mask = np.zeros((h, w), dtype=np.uint8)
        center = cand["center"]
        axes = cand["axes"]
        angle = cand["angle"]

        outer_pts = self._ellipse_points(center, axes, angle, scale=1.0)
        inner_scale = self.expected_diameter / self.total_diameter  # 154/164
        inner_pts = self._ellipse_points(center, axes, angle, scale=inner_scale)
        core_pts = self._ellipse_points(center, axes, angle, scale=inner_scale * 0.82)

        cv2.fillConvexPoly(ring_mask, outer_pts.astype(np.int32), 255)
        cv2.fillConvexPoly(ring_mask, inner_pts.astype(np.int32), 0)
        cv2.fillConvexPoly(in_mask, core_pts.astype(np.int32), 255)

        ring_area = np.count_nonzero(ring_mask)
        inner_area = np.count_nonzero(in_mask)
        if ring_area < 300 or inner_area < 500:
            return -1e9

        white_on_ring = cv2.bitwise_and(white_mask, white_mask, mask=ring_mask)
        white_on_inner = cv2.bitwise_and(white_mask, white_mask, mask=in_mask)
        whiteness_ring = np.count_nonzero(white_on_ring) / float(ring_area)
        whiteness_inner = np.count_nonzero(white_on_inner) / float(inner_area)

        # ring should be brighter than interior (black dohyo center)
        ring_contrast = whiteness_ring - 0.65 * whiteness_inner

        d1, d2 = axes
        ar = max(d1, d2) / (min(d1, d2) + 1e-6)
        shape_penalty = 0.0 if ar < 4.0 else -0.22 * (ar - 4.0)

        score = (1.25 * whiteness_ring) + (1.60 * ring_contrast) + shape_penalty

        if prev is not None:
            pcx, pcy = prev["center"]
            dc = np.hypot(center[0] - pcx, center[1] - pcy)
            pd1, pd2 = prev["axes"]
            dd = abs(d1 - pd1) + abs(d2 - pd2)
            score -= 0.0028 * dc
            score -= 0.0012 * dd

        return score

    def _detect_candidates(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        min_axis = min(h, w) * 0.35
        max_axis = min(h, w) * 1.05

        white_mask = self._white_mask(frame)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands: List[Dict] = []
        for c in contours:
            if len(c) < 5:
                continue
            area = cv2.contourArea(c)
            if area < (h * w * 0.02):
                continue
            ellipse = cv2.fitEllipse(c)
            (cx, cy), (d1, d2), angle = ellipse
            if d1 < min_axis and d2 < min_axis:
                continue
            if d1 > max_axis or d2 > max_axis:
                continue
            cands.append(
                {
                    "center": (float(cx), float(cy)),
                    "axes": (float(d1), float(d2)),
                    "angle": float(angle),
                    "contour": c,
                    "white_mask": white_mask,
                }
            )

        # fallback: try Hough circles for near top-down shots
        if not cands:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 3,
                                       param1=120, param2=40,
                                       minRadius=int(min(h, w) * 0.2), maxRadius=int(min(h, w) * 0.52))
            if circles is not None:
                for cr in circles[0]:
                    cx, cy, r = cr
                    cands.append(
                        {
                            "center": (float(cx), float(cy)),
                            "axes": (float(2 * r), float(2 * r)),
                            "angle": 0.0,
                            "contour": None,
                            "white_mask": white_mask,
                        }
                    )

        return cands

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        if frame is None or frame.size == 0:
            self.last_detection_info = None
            return None, None

        cands = self._detect_candidates(frame)
        if not cands:
            self.last_detection_info = None
            return None, None

        prev = self.last_detection_info
        best = None
        best_score = -1e9
        for c in cands:
            s = self._candidate_score(c, c["white_mask"], prev)
            if s > best_score:
                best_score = s
                best = c

        if best is None:
            self.last_detection_info = None
            return None, None

        center = (int(round(best["center"][0])), int(round(best["center"][1])))
        axes = (float(best["axes"][0]), float(best["axes"][1]))
        angle = float(best["angle"])
        radius = int(round((axes[0] + axes[1]) / 4.0))

        if prev is not None:
            # temporal smoothing to stabilize noisy edge fluctuations
            pcx, pcy = prev["center"]
            pd1, pd2 = prev["axes"]
            pangle = float(prev.get("angle", angle))
            a = self.smooth_alpha
            center = (int(round((1 - a) * pcx + a * center[0])), int(round((1 - a) * pcy + a * center[1])))
            axes = ((1 - a) * pd1 + a * axes[0], (1 - a) * pd2 + a * axes[1])
            angle = float((1 - a) * pangle + a * angle)
            radius = int(round((axes[0] + axes[1]) / 4.0))

        self.last_detection_info = {
            "shape": "ellipse",
            "center": center,
            "axes": axes,
            "angle": angle,
            "radius": radius,
            "score": best_score,
            "inner_scale": self.expected_diameter / self.total_diameter,
        }
        return center, radius

    def track(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Per-frame detection with temporal preference through last_detection_info."""
        return self.detect(frame)

    def draw_overlay(self, frame: np.ndarray, color_outer=(0, 255, 0), color_inner=(255, 255, 255), color_center=(0, 0, 255)) -> np.ndarray:
        out = frame.copy()
        if not self.last_detection_info:
            return out

        info = self.last_detection_info
        center = info["center"]
        axes = (int(round(info["axes"][0])), int(round(info["axes"][1])))
        angle = info["angle"]
        inner_scale = info.get("inner_scale", self.expected_diameter / self.total_diameter)

        cv2.ellipse(out, center, (axes[0] // 2, axes[1] // 2), angle, 0, 360, color_outer, 2)
        inner_axes = (int(round(axes[0] * inner_scale)), int(round(axes[1] * inner_scale)))
        cv2.ellipse(out, center, (inner_axes[0] // 2, inner_axes[1] // 2), angle, 0, 360, color_inner, 2)

        cv2.drawMarker(out, center, color_center, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.putText(out, f"Dohyo center: {center}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return out

    def get_physical_scale(self, radius_pixels: int) -> float:
        return radius_pixels / (self.expected_diameter / 2.0)
