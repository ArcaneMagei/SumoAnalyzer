"""
Homography Transform Module
Converts side-view perspective to top-down (bird's eye) view
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict


class HomographyTransform:
    """
    Transforms perspective from camera view to top-down view
    Uses dohyo as reference for calibration
    """

    def __init__(self, dohyo_detection):
        # Allow tuple or dict (as we did before)
        if isinstance(dohyo_detection, tuple):
            center, radius = dohyo_detection
            scale = radius / 77.0
            dohyo_detection = {
                "center": center,
                "radius": radius,
                "scale": scale,
                "shape": "circle",
            }

        self.dohyo_center = dohyo_detection["center"]
        self.dohyo_radius_px = dohyo_detection["radius"]
        self.scale = dohyo_detection["scale"]  # px/cm
        self.dohyo_radius_cm = 154 / 2  # 77 cm

        # Define output_size BEFORE using it
        self.output_size = int(self.dohyo_radius_cm * 2.5 * self.scale)

        # Now safe to compute homography
        self.homography_matrix = self._calculate_homography()
    
    def _calculate_homography(self) -> np.ndarray:
        """
        Calculate homography matrix from dohyo circle
        
        Returns:
            3x3 homography matrix
        """
        cx, cy = self.dohyo_center
        r = self.dohyo_radius_px
        
        # Source points (4 points on dohyo circle in camera view)
        # Top, Right, Bottom, Left of circle
        src_points = np.float32([
            [cx, cy - r],      # Top
            [cx + r, cy],      # Right
            [cx, cy + r],      # Bottom
            [cx - r, cy]       # Left
        ])
        
        # Destination points (perfect circle in top-down view)
        # Centered in output image
        output_center = self.output_size // 2
        output_radius = int(self.dohyo_radius_cm * self.scale)
        
        dst_points = np.float32([
            [output_center, output_center - output_radius],  # Top
            [output_center + output_radius, output_center],  # Right
            [output_center, output_center + output_radius],  # Bottom
            [output_center - output_radius, output_center]   # Left
        ])
        
        # Calculate homography matrix
        H, _ = cv2.findHomography(src_points, dst_points)
        
        return H
    
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Transform entire frame to top-down view
        
        Args:
            frame: Input frame from camera
        
        Returns:
            Top-down view frame
        """
        top_down = cv2.warpPerspective(
            frame,
            self.homography_matrix,
            (self.output_size, self.output_size)
        )
        
        return top_down
    
    def transform_point(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """
        Transform single point from camera view to top-down coordinates
        
        Args:
            point: (x, y) in camera view
        
        Returns:
            (x, y) in top-down view (in centimeters from center)
        """
        # Convert to homogeneous coordinates
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        pt = np.array([pt])
        
        # Apply homography
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        tx, ty = transformed[0][0]
        
        # Convert to centimeters relative to dohyo center
        center = self.output_size / 2
        x_cm = (tx - center) / self.scale
        y_cm = (ty - center) / self.scale
        
        return (x_cm, y_cm)
    
    def transform_points(self, points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Transform multiple points at once
        
        Args:
            points: List of (x, y) tuples in camera view
        
        Returns:
            List of (x, y) tuples in top-down view (cm)
        """
        if len(points) == 0:
            return []
        
        # Convert to numpy array
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply homography
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        
        # Convert to cm coordinates
        center = self.output_size / 2
        result = []
        for pt in transformed:
            x_cm = (pt[0][0] - center) / self.scale
            y_cm = (pt[0][1] - center) / self.scale
            result.append((x_cm, y_cm))
        
        return result
    
    def draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw measurement grid on top-down view
        
        Args:
            frame: Top-down view frame
        
        Returns:
            Frame with grid overlay
        """
        output = frame.copy()
        center = self.output_size // 2
        
        # Draw dohyo circles
        dohyo_radius_px = int(self.dohyo_radius_cm * self.scale)
        white_line_radius_px = int((self.dohyo_radius_cm - 5) * self.scale)
        
        # Outer circle (white line outer edge)
        cv2.circle(output, (center, center), dohyo_radius_px, (0, 255, 0), 2)
        
        # Inner circle (black dohyo edge)
        cv2.circle(output, (center, center), white_line_radius_px, (255, 0, 0), 2)
        
        # Center point
        cv2.circle(output, (center, center), 5, (0, 0, 255), -1)
        
        # Draw grid lines every 10cm
        for distance in range(10, int(self.dohyo_radius_cm), 10):
            radius_px = int(distance * self.scale)
            cv2.circle(output, (center, center), radius_px, (200, 200, 200), 1)
        
        # Draw axes
        cv2.line(output, (center, 0), (center, self.output_size), (150, 150, 150), 1)
        cv2.line(output, (0, center), (self.output_size, center), (150, 150, 150), 1)
        
        # Add scale info
        cv2.putText(output, f"Scale: {self.scale:.2f} px/cm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output


def test_homography():
    """Test homography transform"""
    from dohyo import DohyoDetector
    
    detector = DohyoDetector()
    cap = cv2.VideoCapture(0)
    
    # Detect dohyo in first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    
    center, radius = detector.detect(frame)
    if center is None or radius is None:
        print("Dohyo not detected")
        return

    try:
        scale = detector.get_physical_scale(radius)
    except AttributeError:
        scale = radius / 77.0  # fallback

    dohyo = {
        "center": center,
        "radius": radius,
        "scale": scale,
        "shape": "circle",
    }
    
    # Initialize homography
    transform = HomographyTransform(dohyo)
    
    print(f"Homography initialized. Output size: {transform.output_size}x{transform.output_size}")
    
    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Transform to top-down
        top_down = transform.transform_frame(frame)
        
        # Draw grid
        top_down_with_grid = transform.draw_grid(top_down)
        
        # Display both views
        cv2.imshow('Camera View', frame)
        cv2.imshow('Top-Down View', top_down_with_grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_homography()
