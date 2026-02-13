"""
Result Collector for Detection Results
Collects and samples detection results, organizing frames by action category
"""

import os
import json
import cv2
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np


class ResultCollector:
    """
    Collects detection results and samples representative frames per action

    Features:
    - Collects frame-level detection results
    - Samples frames (max N per action) to avoid excessive disk usage
    - Saves frames as images for later review
    - Provides statistics and export functionality
    """

    def __init__(self,
                 save_dir: str = "outputs/results",
                 max_frames_per_action: int = 10,
                 save_frame_images: bool = True,
                 session_id: Optional[str] = None):
        """
        Initialize result collector

        Args:
            save_dir: Directory to save results
            max_frames_per_action: Maximum number of frames to save per action
            save_frame_images: Whether to save frame images to disk
            session_id: Unique session identifier (auto-generated if None)
        """
        self.save_dir = Path(save_dir)
        self.max_frames_per_action = max_frames_per_action
        self.save_frame_images = save_frame_images

        # Generate session ID
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"sess_{timestamp}"
        self.session_id = session_id

        # Create directories
        self.frames_dir = self.save_dir / "frames" / self.session_id
        if self.save_frame_images:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Data structures
        self.results: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'confidence_sum': 0.0,
            'frames': []  # List of (frame_idx, timestamp, confidence, frame_path)
        })

        # Session metadata
        self.start_time = time.time()
        self.total_frames = 0
        self.total_detected_frames = 0
        self.video_source = "unknown"

        # Current frame index
        self.frame_idx = 0

    def set_video_source(self, source: str):
        """Set video source description"""
        self.video_source = source

    def add_result(self,
                   frame: np.ndarray,
                   action: str,
                   confidence: float,
                   timestamp: float) -> Optional[str]:
        """
        Add a detection result

        Args:
            frame: Frame image (numpy array)
            action: Detected action label
            confidence: Detection confidence
            timestamp: Frame timestamp

        Returns:
            Path to saved frame (if saved), None otherwise
        """
        self.total_frames += 1
        self.frame_idx += 1

        # Skip if no action detected
        if not action or action in ["Unknown", "Detecting...", "Collecting..."]:
            return None

        self.total_detected_frames += 1

        # Update statistics
        result_dict = self.results[action]
        result_dict['count'] += 1
        result_dict['confidence_sum'] += confidence

        # Decide whether to save this frame (sampling strategy)
        frame_path = None
        if self.save_frame_images and len(result_dict['frames']) < self.max_frames_per_action:
            # Save frame image
            frame_filename = f"{action}_{self.frame_idx}_{confidence:.2f}.jpg"
            # Sanitize action name for filename
            safe_action = "".join(c if c.isalnum() else "_" for c in action)
            frame_filename = f"{safe_action}_{self.frame_idx}.jpg"
            frame_path = str(self.frames_dir / frame_filename)

            # Save frame
            cv2.imwrite(frame_path, frame)

            # Add to frames list
            result_dict['frames'].append({
                'frame_idx': self.frame_idx,
                'timestamp': timestamp,
                'confidence': confidence,
                'frame_path': frame_path
            })
        elif self.save_frame_images and len(result_dict['frames']) >= self.max_frames_per_action:
            # Sampling strategy: replace lowest confidence frame if current is better
            min_confidence_idx = min(
                range(len(result_dict['frames'])),
                key=lambda i: result_dict['frames'][i]['confidence']
            )
            if confidence > result_dict['frames'][min_confidence_idx]['confidence']:
                # Remove old frame
                old_frame_path = result_dict['frames'][min_confidence_idx]['frame_path']
                if os.path.exists(old_frame_path):
                    os.remove(old_frame_path)

                # Save new frame
                safe_action = "".join(c if c.isalnum() else "_" for c in action)
                frame_filename = f"{safe_action}_{self.frame_idx}.jpg"
                frame_path = str(self.frames_dir / frame_filename)
                cv2.imwrite(frame_path, frame)

                # Replace in list
                result_dict['frames'][min_confidence_idx] = {
                    'frame_idx': self.frame_idx,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'frame_path': frame_path
                }

        return frame_path

    def get_statistics(self) -> Dict:
        """
        Get detection statistics

        Returns:
            Dictionary with statistics
        """
        elapsed = time.time() - self.start_time

        # Build statistics for each action
        actions_stats = {}
        for action, data in self.results.items():
            if data['count'] > 0:
                avg_confidence = data['confidence_sum'] / data['count']
                percentage = (data['count'] / self.total_detected_frames * 100) if self.total_detected_frames > 0 else 0

                actions_stats[action] = {
                    'count': data['count'],
                    'percentage': percentage,
                    'confidence_avg': avg_confidence,
                    'saved_frames': len(data['frames']),
                    'frames': data['frames']
                }

        # Sort by count (descending)
        sorted_actions = dict(sorted(
            actions_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))

        return {
            'session_id': self.session_id,
            'start_time': datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'video_source': self.video_source,
            'total_frames': self.total_frames,
            'total_detected_frames': self.total_detected_frames,
            'duration': elapsed,
            'sampling_config': {
                'max_frames_per_action': self.max_frames_per_action,
                'sampling_method': 'uniform_with_confidence_priority'
            },
            'actions': sorted_actions
        }

    def get_action_frames(self, action_name: str) -> List[Dict]:
        """
        Get all saved frames for a specific action

        Args:
            action_name: Name of the action

        Returns:
            List of frame information dictionaries
        """
        if action_name in self.results:
            return self.results[action_name]['frames']
        return []

    def export_results(self, output_path: str, format: str = 'json') -> bool:
        """
        Export results to file

        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')

        Returns:
            True if successful, False otherwise
        """
        try:
            stats = self.get_statistics()

            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                import csv

                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Write header
                    writer.writerow(['Action', 'Count', 'Percentage', 'Avg Confidence',
                                   'Saved Frames', 'Frame Indices'])

                    # Write action statistics
                    for action, data in stats['actions'].items():
                        frame_indices = [f['frame_idx'] for f in data['frames']]
                        writer.writerow([
                            action,
                            data['count'],
                            f"{data['percentage']:.2f}%",
                            f"{data['confidence_avg']:.4f}",
                            data['saved_frames'],
                            ', '.join(map(str, frame_indices))
                        ])

                # Also save session metadata
                metadata_path = output_path.replace('.csv', '_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

    def clear(self):
        """Clear all results"""
        self.results.clear()
        self.total_frames = 0
        self.total_detected_frames = 0
        self.frame_idx = 0
        self.start_time = time.time()

    def get_summary(self) -> str:
        """
        Get a text summary of results

        Returns:
            Summary string
        """
        stats = self.get_statistics()
        lines = [
            f"Session: {stats['session_id']}",
            f"Video Source: {stats['video_source']}",
            f"Total Frames: {stats['total_frames']}",
            f"Detected Frames: {stats['total_detected_frames']}",
            f"Duration: {stats['duration']:.1f}s",
            "",
            "Action Statistics:"
        ]

        for action, data in stats['actions'].items():
            lines.append(
                f"  {action}: {data['count']} frames "
                f"({data['percentage']:.1f}%, avg conf: {data['confidence_avg']:.2f})"
            )

        return '\n'.join(lines)
