"""
Strategy Classification Module
Classifies robot strategies: positioning, search, attack, special
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class StrategyClassifier:
    """
    Classifies robot strategies based on movement patterns
    """
    
    def __init__(self, positioning_speed=5, search_speed=(10, 40), 
                 attack_speed=50, special_rotation=180):
        """
        Initialize strategy classifier
        
        Args:
            positioning_speed: Max speed for positioning phase (cm/s)
            search_speed: (min, max) speed range for search (cm/s)
            attack_speed: Min speed for attack phase (cm/s)
            special_rotation: Min rotation rate for special moves (deg/s)
        """
        self.positioning_speed = positioning_speed
        self.search_speed_min = search_speed[0]
        self.search_speed_max = search_speed[1]
        self.attack_speed = attack_speed
        self.special_rotation = special_rotation
        
        # For temporal smoothing
        self.window_size = 5  # frames
    
    def classify_match(self, all_detections: List[Dict]) -> Dict:
        """
        Classify strategies for entire match
        
        Args:
            all_detections: List of frame detections with tracked robots
        
        Returns:
            Match analysis data including strategies, statistics, etc.
        """
        if len(all_detections) == 0:
            return self._empty_result()
        
        # Extract robot trajectories
        robot_trajectories = self._extract_trajectories(all_detections)
        
        # Calculate velocities and accelerations
        robot_kinematics = {}
        for robot_id, trajectory in robot_trajectories.items():
            robot_kinematics[robot_id] = self._calculate_kinematics(trajectory)
        
        # Classify strategies for each robot
        robot_strategies = {}
        for robot_id, kinematics in robot_kinematics.items():
            strategies = self._classify_robot_strategies(
                kinematics,
                robot_trajectories[robot_id]
            )
            robot_strategies[robot_id] = strategies
        
        # Calculate statistics
        robot_stats = {}
        for robot_id in robot_trajectories.keys():
            robot_stats[robot_id] = self._calculate_statistics(
                robot_trajectories[robot_id],
                robot_kinematics[robot_id],
                robot_strategies[robot_id]
            )
        
        # Build result
        duration = all_detections[-1]['timestamp'] if all_detections else 0
        
        result = {
            'duration': duration,
            'total_frames': len(all_detections),
            'robots': robot_stats,
            'strategy_changes': sum(len(s) - 1 for s in robot_strategies.values()),
            'export_csv': lambda: self._export_csv(all_detections, robot_strategies)
        }
        
        return result
    
    def _extract_trajectories(self, all_detections: List[Dict]) -> Dict[int, List[Tuple]]:
        """Extract trajectory for each robot"""
        trajectories = {}
        
        for frame_data in all_detections:
            for robot in frame_data['robots']:
                robot_id = robot['id']
                
                if robot_id not in trajectories:
                    trajectories[robot_id] = []
                
                trajectories[robot_id].append({
                    'frame': frame_data['frame'],
                    'timestamp': frame_data['timestamp'],
                    'position': robot['top_down_pos'],
                    'center': robot['center']
                })
        
        return trajectories
    
    def _calculate_kinematics(self, trajectory: List[Dict]) -> List[Dict]:
        """Calculate velocity, acceleration, rotation for trajectory"""
        kinematics = []
        
        for i in range(len(trajectory)):
            frame_data = trajectory[i]
            
            # Calculate velocity (requires at least 2 points)
            if i > 0:
                prev = trajectory[i-1]
                dt = frame_data['timestamp'] - prev['timestamp']
                
                if dt > 0:
                    dx = frame_data['position'][0] - prev['position'][0]
                    dy = frame_data['position'][1] - prev['position'][1]
                    
                    speed = np.sqrt(dx**2 + dy**2) / dt  # cm/s
                    direction = np.arctan2(dy, dx)  # radians
                else:
                    speed = 0
                    direction = 0
            else:
                speed = 0
                direction = 0
            
            # Calculate acceleration (requires at least 3 points)
            if i > 1:
                prev_speed = kinematics[i-1]['speed']
                dt = frame_data['timestamp'] - trajectory[i-1]['timestamp']
                acceleration = (speed - prev_speed) / dt if dt > 0 else 0
            else:
                acceleration = 0
            
            # Calculate rotation rate (requires at least 2 points)
            if i > 0:
                prev_direction = kinematics[i-1]['direction']
                dt = frame_data['timestamp'] - trajectory[i-1]['timestamp']
                
                # Handle angle wrapping
                angle_diff = direction - prev_direction
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                rotation_rate = np.degrees(abs(angle_diff) / dt) if dt > 0 else 0
            else:
                rotation_rate = 0
            
            # Distance from center
            distance_from_center = np.sqrt(
                frame_data['position'][0]**2 + frame_data['position'][1]**2
            )
            
            kinematics.append({
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                'speed': speed,
                'direction': direction,
                'acceleration': acceleration,
                'rotation_rate': rotation_rate,
                'distance_from_center': distance_from_center
            })
        
        return kinematics
    
    def _classify_robot_strategies(self, kinematics: List[Dict], 
                                   trajectory: List[Dict]) -> List[Dict]:
        """Classify strategy for each frame"""
        strategies = []
        current_strategy = None
        strategy_start = 0
        
        for i, kin in enumerate(kinematics):
            # Determine strategy based on rules
            if kin['timestamp'] < 2.0 and kin['speed'] < self.positioning_speed:
                strategy = 'positioning'
            
            elif kin['rotation_rate'] > self.special_rotation:
                strategy = 'special'
            
            elif kin['speed'] > self.attack_speed and kin['acceleration'] > 0:
                strategy = 'attack'
            
            elif self.search_speed_min <= kin['speed'] <= self.search_speed_max:
                strategy = 'search'
            
            elif kin['speed'] < self.search_speed_min:
                strategy = 'positioning'
            
            else:
                strategy = 'search'  # default
            
            # Track strategy changes
            if strategy != current_strategy:
                if current_strategy is not None:
                    strategies.append({
                        'strategy': current_strategy,
                        'start': kinematics[strategy_start]['timestamp'],
                        'end': kin['timestamp'],
                        'start_frame': strategy_start,
                        'end_frame': i
                    })
                
                current_strategy = strategy
                strategy_start = i
        
        # Add final strategy
        if current_strategy is not None:
            strategies.append({
                'strategy': current_strategy,
                'start': kinematics[strategy_start]['timestamp'],
                'end': kinematics[-1]['timestamp'],
                'start_frame': strategy_start,
                'end_frame': len(kinematics) - 1
            })
        
        return strategies
    
    def _calculate_statistics(self, trajectory: List[Dict], 
                             kinematics: List[Dict],
                             strategies: List[Dict]) -> Dict:
        """Calculate statistics for a robot"""
        if len(kinematics) == 0:
            return self._empty_robot_stats()
        
        speeds = [k['speed'] for k in kinematics]
        distances_from_center = [k['distance_from_center'] for k in kinematics]
        
        # Count attacks
        attack_count = sum(1 for s in strategies if s['strategy'] == 'attack')
        
        # Time in center (within 30cm of center)
        center_frames = sum(1 for d in distances_from_center if d < 30)
        total_frames = len(kinematics)
        center_time = (center_frames / total_frames) * kinematics[-1]['timestamp'] if total_frames > 0 else 0
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]['position']
            curr = trajectory[i]['position']
            distance = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            total_distance += distance
        
        return {
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'attack_count': attack_count,
            'center_time': center_time,
            'total_distance': total_distance,
            'trajectory': [t['position'] for t in trajectory],
            'strategies': strategies,
            'strategy_timeline': strategies
        }
    
    def _empty_robot_stats(self) -> Dict:
        """Empty statistics template"""
        return {
            'avg_speed': 0,
            'max_speed': 0,
            'attack_count': 0,
            'center_time': 0,
            'total_distance': 0,
            'trajectory': [],
            'strategies': [],
            'strategy_timeline': []
        }
    
    def _empty_result(self) -> Dict:
        """Empty result template"""
        return {
            'duration': 0,
            'total_frames': 0,
            'robots': {},
            'strategy_changes': 0,
            'export_csv': lambda: ""
        }
    
    def _export_csv(self, all_detections: List[Dict], 
                   robot_strategies: Dict[int, List[Dict]]) -> str:
        """Export match data to CSV format"""
        import io
        
        output = io.StringIO()
        
        # Header
        output.write("frame,timestamp,robot_id,x_cm,y_cm,strategy\n")
        
        # Data rows
        for frame_data in all_detections:
            for robot in frame_data['robots']:
                robot_id = robot['id']
                x, y = robot['top_down_pos']
                
                # Find current strategy
                strategy = 'unknown'
                if robot_id in robot_strategies:
                    for strat in robot_strategies[robot_id]:
                        if strat['start_frame'] <= frame_data['frame'] <= strat['end_frame']:
                            strategy = strat['strategy']
                            break
                
                output.write(f"{frame_data['frame']},{frame_data['timestamp']:.3f},"
                           f"{robot_id},{x:.2f},{y:.2f},{strategy}\n")
        
        return output.getvalue()


def test_classifier():
    """Test strategy classifier with synthetic data"""
    classifier = StrategyClassifier()
    
    # Create synthetic match data
    all_detections = []
    
    for frame in range(300):  # 10 seconds at 30 fps
        timestamp = frame / 30.0
        
        # Simulate 2 robots
        robots = [
            {
                'id': 0,
                'center': (100 + frame, 200),
                'top_down_pos': (frame * 0.5, 10)
            },
            {
                'id': 1,
                'center': (300, 150 + frame),
                'top_down_pos': (-10, frame * 0.5)
            }
        ]
        
        all_detections.append({
            'frame': frame,
            'timestamp': timestamp,
            'robots': robots
        })
    
    # Classify
    result = classifier.classify_match(all_detections)
    
    print(f"Match Duration: {result['duration']:.2f}s")
    print(f"Total Frames: {result['total_frames']}")
    print(f"Robots Detected: {len(result['robots'])}")
    
    for robot_id, stats in result['robots'].items():
        print(f"\nRobot {robot_id}:")
        print(f"  Avg Speed: {stats['avg_speed']:.2f} cm/s")
        print(f"  Max Speed: {stats['max_speed']:.2f} cm/s")
        print(f"  Attacks: {stats['attack_count']}")
        print(f"  Center Time: {stats['center_time']:.2f}s")
        print(f"  Total Distance: {stats['total_distance']:.2f} cm")
        print(f"  Strategies: {len(stats['strategies'])}")


if __name__ == '__main__':
    test_classifier()
