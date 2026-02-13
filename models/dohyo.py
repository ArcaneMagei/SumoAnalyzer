"""
Robust Zero-Calibration Dohyo Detector with ELLIPSE support
Works with any camera angle, any lighting, no setup needed
Uses adaptive thresholding and multiple validation passes
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict


class DohyoDetector:
    """
    Zero-calibration dohyo detector
    Finds white border regardless of camera angle or lighting
    Supports both circles (top-down) and ellipses (angled)
    """
    
    def __init__(self, 
                 expected_diameter_cm: float = 154.0,
                 border_width_cm: float = 5.0):
        self.expected_diameter = expected_diameter_cm
        self.border_width = border_width_cm
        self.total_diameter = expected_diameter_cm + (2 * border_width_cm)
        
        # Store last detection info for ellipse drawing
        self.last_detection_info = None
        
    def detect(self, frame: np.ndarray, 
               confidence: float = 0.5,
               min_radius: int = 100,
               max_radius: int = None) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        Detect dohyo using adaptive multi-method approach
        Works with any camera angle, no calibration needed
        
        Returns:
            center: (x, y) tuple
            radius: int (average radius for display)
        
        Note: For ellipse parameters, check self.last_detection_info after calling
        """
        if frame is None or frame.size == 0:
            return None, None
        
        h, w = frame.shape[:2]
        if max_radius is None:
            max_radius = min(w, h) // 2
        
        # Try multiple thresholds and detection methods
        candidates = []
        
        # Method 1: Adaptive threshold (works in variable lighting)
        candidates.extend(self._detect_adaptive_threshold(frame, min_radius, max_radius))
        
        # Method 2: Otsu's threshold (automatic binarization)
        candidates.extend(self._detect_otsu_threshold(frame, min_radius, max_radius))
        
        # Method 3: High brightness threshold (pure white detection)
        candidates.extend(self._detect_high_brightness(frame, min_radius, max_radius))
        
        # Method 4: Edge-based with multiple parameters
        candidates.extend(self._detect_multi_edge(frame, min_radius, max_radius))
        
        if not candidates:
            self.last_detection_info = None
            return None, None
        
        # Score all candidates and pick best
        best_candidate = self._select_best_candidate(candidates, frame)
        
        if best_candidate:
            # Store full detection info
            self.last_detection_info = best_candidate
            return best_candidate['center'], best_candidate['radius']
        
        self.last_detection_info = None
        return None, None
    
    def _detect_adaptive_threshold(self, frame: np.ndarray, 
                                   min_radius: int, 
                                   max_radius: int) -> List[dict]:
        """Adaptive threshold - works with uneven lighting"""
        candidates = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -5
        )
        
        # Invert so white becomes foreground
        thresh = cv2.bitwise_not(thresh)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Try both circle and ellipse fitting
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if len(contour) < 5:
                continue
            
            # Try ellipse fit
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (d1, d2), angle = ellipse
                
                avg_radius = (d1 + d2) / 4
                if min_radius <= avg_radius <= max_radius:
                    candidates.append({
                        'center': (int(cx), int(cy)),
                        'radius': int(avg_radius),
                        'method': 'adaptive_ellipse',
                        'shape': 'ellipse',
                        'params': (int(d1), int(d2), angle),
                        'contour': contour
                    })
            except:
                pass
            
            # Try circle fit
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if min_radius <= radius <= max_radius:
                candidates.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'method': 'adaptive_circle',
                    'shape': 'circle',
                    'contour': contour
                })
        
        return candidates
    
    def _detect_otsu_threshold(self, frame: np.ndarray,
                               min_radius: int,
                               max_radius: int) -> List[dict]:
        """Otsu's method - automatic threshold selection"""
        candidates = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Otsu's thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if len(contour) < 5:
                continue
            
            # Ellipse fit
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (d1, d2), angle = ellipse
                
                avg_radius = (d1 + d2) / 4
                if min_radius <= avg_radius <= max_radius:
                    candidates.append({
                        'center': (int(cx), int(cy)),
                        'radius': int(avg_radius),
                        'method': 'otsu_ellipse',
                        'shape': 'ellipse',
                        'params': (int(d1), int(d2), angle),
                        'contour': contour
                    })
            except:
                pass
        
        return candidates
    
    def _detect_high_brightness(self, frame: np.ndarray,
                                min_radius: int,
                                max_radius: int) -> List[dict]:
        """High brightness threshold - finds very bright white"""
        candidates = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple brightness thresholds
        for threshold in [200, 220, 240]:
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                if len(contour) < 5:
                    continue
                
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (cx, cy), (d1, d2), angle = ellipse
                    
                    avg_radius = (d1 + d2) / 4
                    if min_radius <= avg_radius <= max_radius:
                        candidates.append({
                            'center': (int(cx), int(cy)),
                            'radius': int(avg_radius),
                            'method': f'brightness_{threshold}',
                            'shape': 'ellipse',
                            'params': (int(d1), int(d2), angle),
                            'contour': contour
                        })
                except:
                    pass
        
        return candidates
    
    def _detect_multi_edge(self, frame: np.ndarray,
                          min_radius: int,
                          max_radius: int) -> List[dict]:
        """Multiple edge detection parameters"""
        candidates = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Try different Canny thresholds
        for low, high in [(30, 100), (50, 150), (70, 200)]:
            edges = cv2.Canny(enhanced, low, high)
            
            # Dilate to connect edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                if len(contour) < 5:
                    continue
                
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (cx, cy), (d1, d2), angle = ellipse
                    
                    avg_radius = (d1 + d2) / 4
                    if min_radius <= avg_radius <= max_radius:
                        candidates.append({
                            'center': (int(cx), int(cy)),
                            'radius': int(avg_radius),
                            'method': f'edge_{low}_{high}',
                            'shape': 'ellipse',
                            'params': (int(d1), int(d2), angle),
                            'contour': contour
                        })
                except:
                    pass
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[dict], frame: np.ndarray) -> Optional[dict]:
        """Select best candidate from all detections using comprehensive scoring"""
        if not candidates:
            return None
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        scored_candidates = []
        
        for candidate in candidates:
            score = 0.0
            center = candidate['center']
            radius = candidate['radius']
            cx, cy = center
            
            # Score 1: Size appropriateness (0-30 points)
            min_expected = min(w, h) * 0.15
            max_expected = min(w, h) * 0.50
            if min_expected <= radius <= max_expected:
                sweet_min = min(w, h) * 0.25
                sweet_max = min(w, h) * 0.45
                if sweet_min <= radius <= sweet_max:
                    score += 30
                else:
                    score += 15
            
            # Score 2: Centrality (0-20 points)
            frame_center = np.array([w/2, h/2])
            center_dist = np.linalg.norm([cx - frame_center[0], cy - frame_center[1]])
            max_dist = np.sqrt(w**2 + h**2) / 2
            centrality = 1.0 - (center_dist / max_dist)
            score += centrality * 20
            
            # Score 3: Brightness contrast (0-30 points)
            try:
                # Create masks
                border_mask = np.zeros((h, w), dtype=np.uint8)
                center_mask = np.zeros((h, w), dtype=np.uint8)
                
                if candidate['shape'] == 'ellipse' and 'params' in candidate:
                    d1, d2, angle = candidate['params']
                    cv2.ellipse(border_mask, center, (d1//2, d2//2), angle, 0, 360, 255, 
                               thickness=max(1, int(max(d1, d2)/15)))
                    cv2.ellipse(center_mask, center, (d1//4, d2//4), angle, 0, 360, 255, thickness=-1)
                else:
                    cv2.circle(border_mask, center, radius, 255, thickness=max(1, radius//10))
                    cv2.circle(center_mask, center, radius//2, 255, thickness=-1)
                
                border_pixels = gray[border_mask > 0]
                center_pixels = gray[center_mask > 0]
                
                if len(border_pixels) > 10 and len(center_pixels) > 10:
                    border_brightness = np.mean(border_pixels)
                    center_brightness = np.mean(center_pixels)
                    brightness_diff = border_brightness - center_brightness
                    
                    # High contrast = high score
                    if brightness_diff > 80:
                        score += 30
                    elif brightness_diff > 50:
                        score += 20
                    elif brightness_diff > 20:
                        score += 10
            except:
                pass
            
            # Score 4: Contour quality (0-20 points)
            if 'contour' in candidate:
                contour = candidate['contour']
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Circularity / ellipticity
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)
                    if compactness > 0.7:
                        score += 20
                    elif compactness > 0.5:
                        score += 10
            
            candidate['score'] = score
            scored_candidates.append(candidate)
        
        # Return highest scoring candidate
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        best = scored_candidates[0]
        
        # Require minimum score
        if best['score'] < 30:
            return None
        
        return best
    
    def draw_detection(self, frame: np.ndarray,
                      center: Tuple[int, int],
                      radius: int,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 3) -> np.ndarray:
        """
        Draw detected dohyo
        Automatically uses ellipse if last detection was ellipse
        """
        result = frame.copy()
        
        # Check if we have ellipse info
        if self.last_detection_info and self.last_detection_info['shape'] == 'ellipse':
            # Draw as ellipse
            center = self.last_detection_info['center']
            d1, d2, angle = self.last_detection_info['params']
            
            # Outer ellipse (white border)
            cv2.ellipse(result, center, (d1//2, d2//2), angle, 0, 360, color, thickness)
            
            # Inner ellipse (black area)
            ratio = self.expected_diameter / self.total_diameter
            inner_d1 = int(d1 * ratio / 2)
            inner_d2 = int(d2 * ratio / 2)
            cv2.ellipse(result, center, (inner_d1, inner_d2), angle, 0, 360, color, max(1, thickness//2))
            
            # Center point
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            # Text
            cv2.putText(result, f"Dohyo Ellipse: {d1//2}x{d2//2}px @ {angle:.0f}deg", 
                       (center[0] - 150, center[1] - max(d1, d2)//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Draw as circle (original code)
            cv2.circle(result, center, radius, color, thickness)
            inner_radius = int(radius * (self.expected_diameter / self.total_diameter))
            cv2.circle(result, center, inner_radius, color, max(1, thickness//2))
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            cv2.putText(result, f"Dohyo: r={radius}px", (center[0] - 100, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return result
    
    def get_physical_scale(self, radius_pixels: int) -> float:
        """Calculate pixels per cm scale factor"""
        diameter_pixels = 2 * radius_pixels
        return diameter_pixels / self.total_diameter


# Test standalone
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dohyo.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video")
        sys.exit(1)
    
    detector = DohyoDetector()
    center, radius = detector.detect(frame)
    
    if center and radius:
        print(f"✅ Dohyo detected!")
        print(f"   Center: {center}")
        print(f"   Radius: {radius} pixels")
        
        if detector.last_detection_info and detector.last_detection_info['shape'] == 'ellipse':
            d1, d2, angle = detector.last_detection_info['params']
            print(f"   Shape: ELLIPSE {d1//2}x{d2//2}px at {angle:.1f}°")
        else:
            print(f"   Shape: CIRCLE")
        
        print(f"   Scale: {detector.get_physical_scale(radius):.2f} px/cm")
        
        result = detector.draw_detection(frame, center, radius)
        cv2.imwrite("dohyo_detected.jpg", result)
        print("   Saved: dohyo_detected.jpg")
    else:
        print("❌ Dohyo not detected")
