"""
Robot Detection Module
YOLO-based detection for sumo robots with body and extension tracking
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class RobotDetector:
    """
    Robot detector using YOLOv8
    Detects robot bodies, blades, and flags
    """
    
    def __init__(self, confidence=0.5, model_size='n'):
        """
        Initialize robot detector
        
        Args:
            confidence: Detection confidence threshold (0.0-1.0)
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
                       'n' = nano (fastest), 'x' = extra large (most accurate)
        """
        self.confidence = confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pre-trained YOLO model
        # Will start with COCO pre-trained, fine-tune later with user annotations
        model_path = Path(__file__).parent / 'weights' / f'yolov8{model_size}.pt'
        
        if not model_path.exists():
            print(f"Downloading YOLOv8{model_size} model...")
            model_path.parent.mkdir(exist_ok=True, parents=True)
        
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model.to(self.device)
        
        # Custom class mapping
        # Initially use COCO classes, will add custom classes after annotation
        self.robot_classes = {
            'body': 0,      # Will map to appropriate COCO class initially
            'blade': 1,     # Custom class (after fine-tuning)
            'flag': 2       # Custom class (after fine-tuning)
        }
        
        print(f"Robot detector initialized on {self.device}")
    
    def detect(self, frame):
        """
        Detect robots in a frame
        
        Args:
            frame: Input image (BGR format from OpenCV)
        
        Returns:
            List of detections, each containing:
                - bbox: [x, y, w, h]
                - center: (cx, cy)
                - confidence: detection confidence
                - class_name: 'body', 'blade', or 'flag'
                - type: robot component type
        """
        # Run YOLO inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            imgsz=640  # Input size (can adjust for speed/accuracy tradeoff)
        )
        
        detections = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Convert to [x, y, w, h] format
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                cx, cy = x + w // 2, y + h // 2
                
                # Classify detection type
                # Initially, we'll use size and shape heuristics
                # After fine-tuning, use actual class predictions
                detection_type = self._classify_detection(w, h, frame[y:y+h, x:x+w])
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'center': (cx, cy),
                    'confidence': conf,
                    'class_name': detection_type,
                    'type': detection_type,
                    'area': w * h
                })
        
        # Post-process: group body + extensions
        detections = self._group_detections(detections)
        
        return detections
    
    def _classify_detection(self, width, height, patch):
        """
        Classify detection as body, blade, or flag based on characteristics
        
        Args:
            width, height: Bounding box dimensions
            patch: Image patch of detection
        
        Returns:
            Classification: 'body', 'blade', or 'flag'
        """
        aspect_ratio = width / height if height > 0 else 1.0
        area = width * height
        
        # Analyze color
        if patch.size > 0:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            
            # Check if predominantly white (flags)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            white_ratio = np.sum(white_mask > 0) / (width * height)
            
            if white_ratio > 0.6:  # Mostly white
                return 'flag'
        
        # Size-based classification
        # Body: roughly square, 20x20cm (will calibrate based on dohyo scale)
        # Blade: elongated, extends from body
        # Flag: very elongated, white
        
        if 0.7 < aspect_ratio < 1.3 and 1000 < area < 10000:
            # Square-ish, medium size -> likely body
            return 'body'
        elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
            # Very elongated -> likely blade or flag
            return 'blade'  # Will refine with color info
        else:
            # Default to body
            return 'body'
    
    def _group_detections(self, detections):
        """
        Group body detections with their extensions (blades/flags)
        
        Args:
            detections: List of individual detections
        
        Returns:
            List of grouped robot detections
        """
        # Separate bodies and extensions
        bodies = [d for d in detections if d['type'] == 'body']
        extensions = [d for d in detections if d['type'] in ['blade', 'flag']]
        
        # For each body, find nearby extensions
        grouped = []
        for body in bodies:
            body_cx, body_cy = body['center']
            body_extensions = []
            
            for ext in extensions:
                ext_cx, ext_cy = ext['center']
                distance = np.sqrt((ext_cx - body_cx)**2 + (ext_cy - body_cy)**2)
                
                # If extension is close to body (within 100 pixels), associate it
                if distance < 100:
                    body_extensions.append(ext)
            
            grouped.append({
                'body': body,
                'extensions': body_extensions,
                'center': body['center'],
                'bbox': body['bbox'],
                'confidence': body['confidence']
            })
        
        return grouped
    
    def fine_tune(self, annotated_images_dir, epochs=50):
        """
        Fine-tune model on user-annotated sumo robot data
        
        Args:
            annotated_images_dir: Path to YOLO format annotations
            epochs: Number of training epochs
        """
        print(f"Fine-tuning model on {annotated_images_dir}...")
        
        # Train model
        results = self.model.train(
            data=f"{annotated_images_dir}/data.yaml",
            epochs=epochs,
            imgsz=640,
            device=self.device,
            project='sumo_training',
            name='robot_detector'
        )
        
        print("Fine-tuning complete!")
        return results
    
    def draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections from detect()
        
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        colors = {
            'body': (0, 255, 0),    # Green
            'blade': (255, 0, 0),   # Blue
            'flag': (0, 0, 255)     # Red
        }
        
        for detection in detections:
            # Draw body
            body = detection['body']
            x, y, w, h = body['bbox']
            cv2.rectangle(output, (x, y), (x + w, y + h), colors['body'], 2)
            
            # Draw label
            label = f"Robot {body['confidence']:.2f}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['body'], 2)
            
            # Draw extensions
            for ext in detection['extensions']:
                x, y, w, h = ext['bbox']
                color = colors[ext['type']]
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                ext_label = f"{ext['type']} {ext['confidence']:.2f}"
                cv2.putText(output, ext_label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return output


def test_detector():
    """Test the detector with a sample image or video"""
    detector = RobotDetector(confidence=0.5)
    
    # Test with webcam or video file
    cap = cv2.VideoCapture(0)  # 0 for webcam, or path to video file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect robots
        detections = detector.detect(frame)
        
        # Draw detections
        annotated = detector.draw_detections(frame, detections)
        
        # Display
        cv2.imshow('Robot Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_detector()
