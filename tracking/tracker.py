"""
Enhanced Robot Tracker with Advanced Features
Improves tracking robustness for occlusions, fast motion, and collisions
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class RobotState:
    """Enhanced robot state with tracking history"""
    id: int
    position: Tuple[int, int]
    velocity: Tuple[float, float]
    color_histogram: np.ndarray
    confidence: float
    frames_tracked: int
    frames_lost: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    position_history: deque  # Last N positions for smoothing
    front_point: Tuple[int, int]
    heading_deg: float
    occluded: bool
    
    def __post_init__(self):
        if not isinstance(self.position_history, deque):
            self.position_history = deque(maxlen=10)


class RobotTracker:
    """
    Advanced robot tracker with:
    - Kalman filtering for motion prediction
    - Color histogram matching for re-identification
    - Occlusion handling
    - Collision detection and recovery
    """
    
    def __init__(self, 
                 dohyo_center: Tuple[int, int],
                 dohyo_radius: int,
                 max_lost_frames: int = 15,
                 position_history_size: int = 10):
        self.dohyo_center = dohyo_center
        self.dohyo_radius = dohyo_radius
        self.max_lost_frames = max_lost_frames
        self.position_history_size = position_history_size
        
        self.robots: Dict[int, RobotState] = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Detection parameters
        self.min_robot_size = 20  # pixels
        self.max_robot_size = dohyo_radius // 2
        self.color_match_threshold = 0.7
        self.prev_gray: Optional[np.ndarray] = None
        self.max_track_count = 2
        
    def _estimate_front_point(self, frame: np.ndarray, contour: np.ndarray, center: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
        """Estimate robot front edge by combining geometric heading and shiny blade cue."""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)

        # Prefer the longer side as heading axis, then orient it to point away from dohyo center.
        v1 = box[1] - box[0]
        v2 = box[2] - box[1]
        axis = v1 if np.linalg.norm(v1) >= np.linalg.norm(v2) else v2
        axis = axis.astype(np.float32)
        axis_norm = np.linalg.norm(axis) + 1e-6
        axis /= axis_norm

        radial = np.array([center[0] - self.dohyo_center[0], center[1] - self.dohyo_center[1]], dtype=np.float32)
        if np.dot(axis, radial) < 0:
            axis = -axis

        # Geometry candidate: contour point furthest along heading axis.
        contour_pts = contour.reshape(-1, 2)
        proj = contour_pts @ axis
        front_geom = contour_pts[int(np.argmax(proj))]

        # Blade candidate: brightest pixel cluster (shiny metal wedges).
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y + h, x:x + w]
        front_blade = front_geom
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 70, 255]))
            if np.count_nonzero(bright) > (w * h * 0.01):
                ys, xs = np.where(bright > 0)
                blade_points = np.stack([xs + x, ys + y], axis=1)
                blade_proj = blade_points @ axis
                front_blade = blade_points[int(np.argmax(blade_proj))]

        # Blend robustly between geometric and shiny-blade cues.
        front = np.round(0.6 * front_geom + 0.4 * front_blade).astype(int)
        heading_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
        return (int(front[0]), int(front[1])), heading_deg

    def detect_robots(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect robot candidates in frame
        Returns list of detections with position, size, color histogram
        """
        # Create dohyo mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, self.dohyo_center, self.dohyo_radius, 255, -1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect darker structure and moving objects (handles bright robots too)
        _, dark_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        motion_mask = np.zeros_like(gray)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray

        combined = cv2.bitwise_or(dark_thresh, motion_mask)
        
        # Apply morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Mask to dohyo area only
        combined = cv2.bitwise_and(combined, mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering
            if area < self.min_robot_size**2 or area > self.max_robot_size**2:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio check (robots are roughly circular/square)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                continue
            
            # Center position
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if inside dohyo
            dist_from_center = np.sqrt((cx - self.dohyo_center[0])**2 + 
                                      (cy - self.dohyo_center[1])**2)
            if dist_from_center > self.dohyo_radius * 0.95:
                continue
            
            # Extract color histogram for tracking
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(roi_mask, [contour], 0, 255, -1)
            
            # Color histogram in HSV space (more robust than BGR)
            hist = cv2.calcHist([hsv], [0, 1], roi_mask, [30, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            front_point, heading_deg = self._estimate_front_point(frame, contour, (cx, cy))

            detections.append({
                'position': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'contour': contour,
                'color_histogram': hist,
                'front_point': front_point,
                'heading_deg': heading_deg,
            })

        detections.sort(key=lambda d: d['area'], reverse=True)
        detections = detections[:self.max_track_count]
        
        return detections
    
    def match_detections_to_tracks(self, 
                                   detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Match detections to existing tracks using:
        1. Predicted position (motion model)
        2. Color histogram similarity
        3. Size consistency
        
        Returns: (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not self.robots:
            return [], list(range(len(detections))), []
        
        # Predict robot positions based on velocity
        predictions = {}
        for robot_id, robot in self.robots.items():
            pred_x = robot.position[0] + robot.velocity[0]
            pred_y = robot.position[1] + robot.velocity[1]
            predictions[robot_id] = (pred_x, pred_y)
        
        # Compute cost matrix: distance + color difference
        cost_matrix = np.zeros((len(self.robots), len(detections)))
        robot_ids = list(self.robots.keys())
        
        for i, robot_id in enumerate(robot_ids):
            robot = self.robots[robot_id]
            pred_pos = predictions[robot_id]
            
            for j, detection in enumerate(detections):
                det_pos = detection['position']
                
                # Position distance (normalized by expected motion range)
                pos_dist = np.sqrt((pred_pos[0] - det_pos[0])**2 + 
                                  (pred_pos[1] - det_pos[1])**2)
                pos_cost = pos_dist / 100.0  # Normalize
                
                # Color histogram similarity (using correlation)
                color_sim = cv2.compareHist(robot.color_histogram, 
                                           detection['color_histogram'],
                                           cv2.HISTCMP_CORREL)
                color_cost = 1.0 - color_sim  # Convert similarity to cost
                
                # Combined cost (weighted)
                cost_matrix[i, j] = 0.6 * pos_cost + 0.4 * color_cost
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches with high cost (likely wrong matches)
        matched_pairs = []
        max_acceptable_cost = 0.5
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_acceptable_cost:
                matched_pairs.append((robot_ids[r], c))
        
        # Unmatched detections and tracks
        matched_detection_indices = [c for _, c in matched_pairs]
        unmatched_detections = [i for i in range(len(detections)) 
                               if i not in matched_detection_indices]
        
        matched_robot_ids = [rid for rid, _ in matched_pairs]
        unmatched_tracks = [rid for rid in robot_ids 
                           if rid not in matched_robot_ids]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def update(self, frame: np.ndarray) -> Dict[int, RobotState]:
        """
        Update tracking for current frame
        
        Returns: Dictionary of active robot tracks {id: RobotState}
        """
        self.frame_count += 1
        
        # Detect robots in current frame
        detections = self.detect_robots(frame)
        
        # Match detections to existing tracks
        matched_pairs, unmatched_det, unmatched_tracks = \
            self.match_detections_to_tracks(detections)

        # Occlusion handling: one merged contour while we expect two robots.
        if len(detections) == 1 and len(self.robots) >= 2:
            for robot in self.robots.values():
                robot.frames_lost += 1
                robot.occluded = True
                robot.confidence = max(0.2, robot.confidence - 0.08)
                pred_x = robot.position[0] + robot.velocity[0]
                pred_y = robot.position[1] + robot.velocity[1]
                robot.position = (int(pred_x), int(pred_y))
            return self.robots
        
        # Update matched tracks
        for robot_id, det_idx in matched_pairs:
            detection = detections[det_idx]
            robot = self.robots[robot_id]
            
            # Update position
            new_pos = detection['position']
            old_pos = robot.position
            
            # Calculate velocity (smoothed)
            vx = 0.7 * robot.velocity[0] + 0.3 * (new_pos[0] - old_pos[0])
            vy = 0.7 * robot.velocity[1] + 0.3 * (new_pos[1] - old_pos[1])
            
            # Update state
            robot.position = new_pos
            robot.velocity = (vx, vy)
            robot.bbox = detection['bbox']
            robot.position_history.append(new_pos)
            robot.front_point = detection['front_point']
            robot.heading_deg = detection['heading_deg']
            robot.occluded = False
            robot.frames_tracked += 1
            robot.frames_lost = 0
            robot.confidence = min(1.0, robot.confidence + 0.1)
            
            # Update color histogram (slow adaptation for lighting changes)
            robot.color_histogram = 0.9 * robot.color_histogram + \
                                   0.1 * detection['color_histogram']
        
        # Handle unmatched tracks (potentially occluded or left dohyo)
        for robot_id in unmatched_tracks:
            robot = self.robots[robot_id]
            robot.frames_lost += 1
            robot.occluded = True
            robot.confidence = max(0.0, robot.confidence - 0.2)
            
            # Predict position using velocity
            pred_x = robot.position[0] + robot.velocity[0]
            pred_y = robot.position[1] + robot.velocity[1]
            robot.position = (int(pred_x), int(pred_y))
            
            # Remove track if lost too long
            if robot.frames_lost > self.max_lost_frames:
                del self.robots[robot_id]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_det:
            detection = detections[det_idx]
            
            new_robot = RobotState(
                id=self.next_id,
                position=detection['position'],
                velocity=(0.0, 0.0),
                color_histogram=detection['color_histogram'],
                confidence=0.5,
                frames_tracked=1,
                frames_lost=0,
                bbox=detection['bbox'],
                position_history=deque([detection['position']], 
                                      maxlen=self.position_history_size),
                front_point=detection['front_point'],
                heading_deg=detection['heading_deg'],
                occluded=False,
            )
            
            self.robots[self.next_id] = new_robot
            self.next_id += 1
        
        return self.robots
    
    def get_smoothed_position(self, robot_id: int) -> Optional[Tuple[int, int]]:
        """Get smoothed position using position history"""
        if robot_id not in self.robots:
            return None
        
        robot = self.robots[robot_id]
        if len(robot.position_history) == 0:
            return robot.position
        
        # Average last N positions
        positions = list(robot.position_history)
        avg_x = int(np.mean([p[0] for p in positions]))
        avg_y = int(np.mean([p[1] for p in positions]))
        
        return (avg_x, avg_y)
    
    def draw_tracks(self, frame: np.ndarray, 
                   show_trails: bool = True,
                   show_ids: bool = True) -> np.ndarray:
        """Draw tracking visualization"""
        result = frame.copy()
        
        for robot_id, robot in self.robots.items():
            # Color based on confidence
            if robot.confidence > 0.7:
                color = (0, 255, 0)  # Green = high confidence
            elif robot.confidence > 0.4:
                color = (0, 255, 255)  # Yellow = medium
            else:
                color = (0, 165, 255)  # Orange = low confidence
            
            # Draw bounding box
            x, y, w, h = robot.bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(result, robot.position, 5, color, -1)
            
            # Draw ID
            if show_ids:
                occ = " OCC" if robot.occluded else ""
                cv2.putText(result, f"R{robot_id}{occ}", 
                           (robot.position[0] - 10, robot.position[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Front indicator and heading vector
            cv2.circle(result, robot.front_point, 4, (255, 255, 0), -1)
            cv2.line(result, robot.position, robot.front_point, (255, 255, 0), 2)
            
            # Draw velocity vector
            if np.linalg.norm(robot.velocity) > 1:
                end_x = int(robot.position[0] + robot.velocity[0] * 3)
                end_y = int(robot.position[1] + robot.velocity[1] * 3)
                cv2.arrowedLine(result, robot.position, (end_x, end_y), 
                               color, 2, tipLength=0.3)
            
            # Draw trail
            if show_trails and len(robot.position_history) > 1:
                points = np.array(list(robot.position_history), dtype=np.int32)
                cv2.polylines(result, [points], False, color, 1)
        
        # Draw info
        info_text = f"Tracking: {len(self.robots)} robots | Frame: {self.frame_count}"
        cv2.putText(result, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
