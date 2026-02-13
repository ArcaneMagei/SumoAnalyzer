"""
Enhanced Test for Robust Dohyo Detection with Ellipse Support
Tests multi-method detector and saves ALL candidates for review
"""

import cv2
import numpy as np
import sys
from pathlib import Path

try:
    from models.dohyo import DohyoDetector
except:
    print("ERROR: Could not import DohyoDetector")
    print("Solution: Copy 'models-dohyo-ellipse-fixed.py' to 'models/dohyo.py'")
    sys.exit(1)


def test_robust_detection(video_path):
    """Test robust multi-method detection with full candidate visualization"""
    
    print(f"Testing robust dohyo detection on: {video_path}")
    print("=" * 60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Could not open video file")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print()
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ ERROR: Could not read first frame")
        return
    
    # Initialize detector
    detector = DohyoDetector()
    
    print("Running robust detection (trying 40+ combinations)...")
    print("-" * 60)
    
    # Get detection candidates (access internal methods for debugging)
    min_radius = 100
    max_radius = min(width, height) // 2
    
    all_candidates = []
    
    # Method 1: Adaptive threshold
    print("Method 1: Adaptive threshold... ", end="", flush=True)
    candidates_adaptive = detector._detect_adaptive_threshold(frame, min_radius, max_radius)
    all_candidates.extend(candidates_adaptive)
    print(f"Found {len(candidates_adaptive)} candidates")
    
    # Method 2: Otsu's threshold
    print("Method 2: Otsu's threshold... ", end="", flush=True)
    candidates_otsu = detector._detect_otsu_threshold(frame, min_radius, max_radius)
    all_candidates.extend(candidates_otsu)
    print(f"Found {len(candidates_otsu)} candidates")
    
    # Method 3: High brightness
    print("Method 3: High brightness (3 thresholds)... ", end="", flush=True)
    candidates_bright = detector._detect_high_brightness(frame, min_radius, max_radius)
    all_candidates.extend(candidates_bright)
    print(f"Found {len(candidates_bright)} candidates")
    
    # Method 4: Multi-edge
    print("Method 4: Multi-edge detection (3 params)... ", end="", flush=True)
    candidates_edge = detector._detect_multi_edge(frame, min_radius, max_radius)
    all_candidates.extend(candidates_edge)
    print(f"Found {len(candidates_edge)} candidates")
    
    print()
    print(f"Total candidates found: {len(all_candidates)}")
    print()
    
    if not all_candidates:
        print("❌ No candidates found by any method")
        print("This is unusual - the white border should be detectable")
        cv2.imwrite("first_frame_debug.jpg", frame)
        print("Saved first frame to: first_frame_debug.jpg")
        return
    
    # Score all candidates
    print("Scoring all candidates...")
    print("-" * 60)
    
    best_candidate = detector._select_best_candidate(all_candidates, frame)
    
    if best_candidate:
        print(f"✅ Best candidate selected:")
        print(f"   Method: {best_candidate['method']}")
        print(f"   Center: {best_candidate['center']}")
        print(f"   Radius: {best_candidate['radius']}px")
        
        if best_candidate['shape'] == 'ellipse' and 'params' in best_candidate:
            d1, d2, angle = best_candidate['params']
            print(f"   Shape: ELLIPSE {d1//2}x{d2//2}px at {angle:.1f}°")
        else:
            print(f"   Shape: CIRCLE")
        
        print(f"   Score: {best_candidate['score']:.1f}/100")
        print()
        
        # Show score breakdown
        if best_candidate['score'] > 70:
            print("   Quality: ✅ EXCELLENT")
        elif best_candidate['score'] > 50:
            print("   Quality: ✅ GOOD")
        elif best_candidate['score'] > 30:
            print("   Quality: ⚠️  ACCEPTABLE")
        else:
            print("   Quality: ❌ POOR")
    else:
        print("❌ No candidate passed minimum score threshold")
        print("All detections were rejected as unreliable")
        print()
    
    print()
    print("-" * 60)
    
    # Run main detection (uses all methods + scoring)
    print("Running main detect() method... ", end="", flush=True)
    center_main, radius_main = detector.detect(frame)
    
    if center_main and radius_main:
        print("✅ SUCCESS!")
        print()
        print(f"Final Detection:")
        print(f"  Center: {center_main}")
        print(f"  Radius: {radius_main} pixels (average)")
        
        if detector.last_detection_info and detector.last_detection_info['shape'] == 'ellipse':
            d1, d2, angle = detector.last_detection_info['params']
            print(f"  Shape: ELLIPSE {d1//2}x{d2//2}px at {angle:.1f}°")
        else:
            print(f"  Shape: CIRCLE")
        
        print(f"  Scale: {detector.get_physical_scale(radius_main):.2f} pixels/cm")
        
        # Draw detection
        result = detector.draw_detection(frame, center_main, radius_main)
        cv2.imwrite("dohyo_detected_main.jpg", result)
        print()
        print("✅ Main detection saved to: dohyo_detected_main.jpg")
    else:
        print("❌ FAILED")
        print()
        print("Main detection rejected all candidates")
    
    print()
    print("-" * 60)
    
    # Save ALL candidates for review (TOP 10)
    print()
    print("Saving visualizations of ALL top candidates...")
    scored = sorted(all_candidates, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, candidate in enumerate(scored[:10], 1):
        result = frame.copy()
        center = candidate['center']
        radius = candidate['radius']
        score = candidate.get('score', 0)
        method = candidate['method']
        shape = candidate['shape']
        
        # Draw detection based on shape
        if shape == 'ellipse' and 'params' in candidate:
            d1, d2, angle = candidate['params']
            
            # Outer ellipse (white border)
            cv2.ellipse(result, center, (d1//2, d2//2), angle, 0, 360, (0, 255, 0), 3)
            
            # Inner ellipse (black area) - 154/160 ratio
            ratio = 0.963
            inner_d1 = int(d1 * ratio / 2)
            inner_d2 = int(d2 * ratio / 2)
            cv2.ellipse(result, center, (inner_d1, inner_d2), angle, 0, 360, (0, 255, 0), 2)
            
            # Center point
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            # Text
            cv2.putText(result, f"#{i} {method} - Score: {score:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result, f"Ellipse: {d1//2}x{d2//2}px @ {angle:.0f}deg", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Circle
            cv2.circle(result, center, radius, (0, 255, 0), 3)
            inner_radius = int(radius * 0.963)
            cv2.circle(result, center, inner_radius, (0, 255, 0), 2)
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            cv2.putText(result, f"#{i} {method} - Score: {score:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result, f"Circle: r={radius}px", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        filename = f"candidate_{i:02d}_{method}_score{score:.0f}.jpg"
        cv2.imwrite(filename, result)
        print(f"  {i:2d}. {filename:<50} [{shape.upper():8}] Score: {score:5.1f}")
    
    print()
    print("=" * 60)
    print("REVIEW INSTRUCTIONS:")
    print("=" * 60)
    print()
    print("1. Open candidate_01_*.jpg (BEST detection)")
    print("2. Check if ellipse/circle matches the dohyo border exactly")
    print("3. Compare with candidate_02_*.jpg and candidate_03_*.jpg")
    print()
    print("Expected results:")
    print("  ✅ Top-down view: Circle should match white border")
    print("  ✅ Angled view: Ellipse should match flattened dohyo")
    print()
    print("If candidate #1 is almost perfect:")
    print("  → Detection is working correctly!")
    print("  → Copy models-dohyo-ellipse-fixed.py to models/dohyo.py")
    print("  → Restart Docker and test")
    print()
    print("If candidate #1 is wrong but #2 or #3 looks better:")
    print("  → We need to adjust scoring weights")
    print("  → Share the images for fine-tuning")
    print()
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test-dohyo-detection.py <video_path>")
        print()
        print("Example:")
        print("  python test-dohyo-detection.py IMG_1426.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    test_robust_detection(video_path)


if __name__ == "__main__":
    main()
