"""
Main Processing Pipeline
Coordinates all components for real-time behavior detection
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import torch
from .video_writer import FrameOverlay


class PredictionSmoother:
    """Smooth predictions across frames using exponential moving average"""

    def __init__(self, alpha: float = 0.3, history_length: int = 5):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more influence from history.
            history_length: Number of previous predictions to keep.
        """
        self.alpha = alpha
        self.history = deque(maxlen=history_length)

    def update(self, prediction: np.ndarray) -> np.ndarray:
        """
        Update with new prediction and return smoothed result.

        Args:
            prediction: New prediction (probability distribution over classes)

        Returns:
            Smoothed prediction
        """
        self.history.append(prediction)
        if len(self.history) < 2:
            return prediction

        # Exponential moving average
        smoothed = self.history[-1].copy()
        for hist in list(self.history)[-2::-1]:
            smoothed = self.alpha * hist + (1 - self.alpha) * smoothed

        return smoothed


class DetectionPipeline:
    """
    Main pipeline for real-time behavior detection
    Supports multi-person detection and action classification
    """

    # Colors for different people (BGR format)
    PERSON_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]

    def __init__(self, checkpoint_path: str, yolo_model: str = 'yolov5s.pt',
                 output_path: str = None, fps: float = 30.0,
                 show_display: bool = True, save_video: bool = False,
                 max_persons: int = 5,
                 enable_result_collection: bool = False,
                 results_dir: str = "outputs/results"):
        """
        Initialize detection pipeline

        Args:
            checkpoint_path: Path to trained TSN model checkpoint
            yolo_model: YOLO model path
            output_path: Output video path (None for no output)
            fps: Target FPS
            show_display: Show real-time display
            save_video: Save output video
            max_persons: Maximum number of persons to track simultaneously
            enable_result_collection: Enable detection result collection
            results_dir: Directory to save results
        """
        self.show_display = show_display
        self.save_video = save_video or (output_path is not None)
        self.max_persons = max_persons

        # Store frame resolution
        self.frame_resolution = None

        # Confidence threshold for displaying predictions (set to 0 to show all predictions)
        self.confidence_threshold = 0.0

        # Result collection
        self.enable_result_collection = enable_result_collection
        self.result_collector = None
        if enable_result_collection:
            from .result_collector import ResultCollector
            self.result_collector = ResultCollector(
                save_dir=results_dir,
                max_frames_per_action=10,
                save_frame_images=True
            )

        # Initialize components
        print("Initializing components...")

        # Human detector
        from .human_detector import HumanDetector
        self.detector = HumanDetector(
            model_path=yolo_model,
            confidence=0.5,
            device='auto'
        )

        # Temporal processor (updated to 5 segments x 5 frames = 25 total)
        from .temporal_processor import TemporalProcessor
        self.temporal_processor = TemporalProcessor(
            num_segments=5,
            frames_per_segment=5,
            buffer_size=30,
            max_memory_mb=500.0  # Memory limit for multi-person tracking
        )

        # MULTIPLE PERSON SUPPORT: Prediction smoother for each track
        self.prediction_smoothers: Dict[int, PredictionSmoother] = {}

        # Track current action for each person
        self.current_actions: Dict[int, str] = {}
        self.current_confidences: Dict[int, float] = {}
        self.current_detections: Dict[int, dict] = {}

        # Track ID counter for assigning IDs to new detections
        self.next_track_id = 0

        # Action classifier
        from .action_classifier import load_classifier
        self.classifier = load_classifier(
            checkpoint_path=checkpoint_path,
            device='auto'
        )

        # Video writer
        from .video_writer import VideoWriter, FrameOverlay
        self.video_writer = None
        if self.save_video and output_path:
            self.video_writer = VideoWriter(
                output_path=output_path,
                fps=fps
            )

        # Initialize display window
        if self.show_display:
            cv2.namedWindow('Real-time Behavior Detection', cv2.WINDOW_NORMAL)

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }

        # Current action state
        self.current_action = "Unknown"
        self.current_confidence = 0.0
        self.last_detection_time = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Process a single frame with multi-person detection support

        Args:
            frame: Input frame
            timestamp: Frame timestamp

        Returns:
            Processed frame with overlays
        """
        # Store frame resolution
        if self.frame_resolution is None:
            self.frame_resolution = (frame.shape[1], frame.shape[0])

        # Detect humans
        detections = self.detector.detect(frame)
        self.stats['detections'] += len(detections)

        # Limit number of detections to max_persons
        detections = detections[:self.max_persons]

        # Process each detection
        active_track_ids = []

        for idx, detection in enumerate(detections):
            # Assign track ID based on detection index (simplified tracking)
            # In production, use proper tracking like DeepSORT
            track_id = idx

            # Initialize prediction smoother for new track
            if track_id not in self.prediction_smoothers:
                self.prediction_smoothers[track_id] = PredictionSmoother(alpha=0.3, history_length=5)

            active_track_ids.append(track_id)

            # Crop person frame
            person_crop = self.detector.crop_person(frame, detection)

            # Add cropped frame to temporal buffer
            frame_ready = self.temporal_processor.add_frame(
                person_crop, track_id, timestamp
            )

            if frame_ready:
                # Get temporal segments for classification
                segment_frames = self.temporal_processor.get_temporal_segments(track_id)

                if segment_frames:
                    # Preprocess for TSN model
                    input_tensor = self.temporal_processor.preprocess_frames(segment_frames)

                    # Get probability distribution and apply temporal smoothing
                    probs = self.classifier.predict_proba(input_tensor)
                    smoother = self.prediction_smoothers[track_id]
                    smoothed_probs = smoother.update(probs)

                    # Get top prediction from smoothed probabilities
                    top_idx = np.argmax(smoothed_probs)
                    action = self.classifier.ucf101_classes[top_idx]
                    confidence = float(smoothed_probs[top_idx])

                    # Apply confidence threshold
                    if confidence >= self.confidence_threshold:
                        display_action = action
                        display_confidence = confidence
                    else:
                        display_action = "Detecting..."
                        display_confidence = 0.0

                    # Update state for this track
                    self.current_actions[track_id] = action
                    self.current_confidences[track_id] = confidence
                    self.current_detections[track_id] = detection
                    self.stats['classifications'] += 1

                    # Add to result collector if enabled
                    if self.result_collector:
                        self.result_collector.add_result(
                            frame=person_crop,
                            action=action,
                            confidence=confidence,
                            timestamp=timestamp
                        )
            else:
                # Not enough frames yet
                self.current_actions[track_id] = "Collecting..."
                self.current_confidences[track_id] = 0.0
                self.current_detections[track_id] = detection

        # Remove inactive tracks
        all_track_ids = list(self.current_actions.keys())
        for track_id in all_track_ids:
            if track_id not in active_track_ids:
                # Keep track data for a few frames before removing
                # (for better handling of temporary occlusions)
                pass

        # Draw all detections with multi-person support
        frame = self._draw_multi_person_detections(frame, active_track_ids)

        # Clean up old tracks periodically
        if self.stats['frames_processed'] % 30 == 0:
            self.temporal_processor.remove_old_tracks(timestamp, max_age=2.0)
            # Also remove prediction smoothers for inactive tracks
            active_tracks = set(self.temporal_processor.frame_buffers.keys())
            inactive_tracks = set(self.prediction_smoothers.keys()) - active_tracks
            for track_id in inactive_tracks:
                del self.prediction_smoothers[track_id]
                if track_id in self.current_actions:
                    del self.current_actions[track_id]
                if track_id in self.current_confidences:
                    del self.current_confidences[track_id]
                if track_id in self.current_detections:
                    del self.current_detections[track_id]

        # Draw overlays
        info = self._get_info_dict(timestamp)
        frame = FrameOverlay.draw_info_panel(frame, info)

        # Draw timestamp
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
        frame = FrameOverlay.draw_timestamp(frame, timestamp_str)

        # Update statistics
        self.stats['frames_processed'] += 1

        # Save frame if video writer is enabled
        if self.video_writer:
            self.video_writer.write_frame(frame)

        # Show frame if display is enabled
        if self.show_display:
            cv2.imshow('Real-time Behavior Detection', frame)

        return frame

    def _draw_multi_person_detections(self, frame: np.ndarray, track_ids: List[int]) -> np.ndarray:
        """
        Draw detection boxes and labels for multiple people

        Args:
            frame: Input frame
            track_ids: List of active track IDs

        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()

        for track_id in track_ids:
            if track_id not in self.current_detections:
                continue

            detection = self.current_detections[track_id]
            action = self.current_actions.get(track_id, "Unknown")
            confidence = self.current_confidences.get(track_id, 0.0)

            # Get color for this person (cycle through colors)
            color = self.PERSON_COLORS[track_id % len(self.PERSON_COLORS)]

            # Get bounding box
            x1, y1, x2, y2 = detection['bbox']

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_thickness = 1
            line_height = 18

            # Prepare labels
            labels = []
            labels.append(f"Person {track_id + 1}: {detection['confidence']:.2f}")

            if action and confidence is not None:
                labels.append(f"Action: {action}")
                labels.append(f"Conf: {confidence:.2f}")

            # Draw labels above the bounding box
            for i, label in enumerate(labels):
                y_pos = y1 - 10 - (len(labels) - i - 1) * line_height

                # Ensure labels don't go off screen
                y_pos = max(y_pos, line_height)

                # Draw text with background for better visibility
                (text_width, text_height) = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                cv2.rectangle(output_frame,
                            (x1, y_pos - text_height - 2),
                            (x1 + text_width + 4, y_pos + 2),
                            color, -1)
                cv2.putText(output_frame, label, (x1 + 2, y_pos),
                          font, font_scale, (255, 255, 255), text_thickness)

        return output_frame

    def _get_info_dict(self, timestamp: float) -> Dict:
        """Get information dictionary for overlay with multi-person support"""
        elapsed = timestamp - self.stats['start_time']
        current_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

        # Format resolution as width×height using standard multiplication sign
        resolution_str = f"{self.frame_resolution[0]}x{self.frame_resolution[1]}" if self.frame_resolution else "N/A"

        # Count active persons
        person_count = len(self.current_actions)

        # For display, show the action of the most confident person
        if person_count > 0:
            # Find person with highest confidence
            best_track_id = max(self.current_confidences.items(),
                             key=lambda x: x[1])[0] if self.current_confidences else None
            if best_track_id is not None:
                display_action = self.current_actions.get(best_track_id, "Unknown")
                display_confidence = self.current_confidences.get(best_track_id, 0.0)
            else:
                display_action = "Detecting..."
                display_confidence = 0.0
        else:
            display_action = "No person detected"
            display_confidence = 0.0

        return {
            'time': f"{elapsed:.1f}s",
            'fps': f"{current_fps:.1f}",
            'resolution': resolution_str,
            'action': display_action,
            'confidence': display_confidence,
            'person_count': person_count
        }

    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """
        Process video file

        Args:
            video_path: Input video path
            output_path: Output video path (overrides instance setting)

        Returns:
            Processing statistics
        """
        from .video_writer import VideoWriter, FrameOverlay

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return {}

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Update video writer if output path provided
        video_writer = None
        if output_path or self.save_video:
            final_output = output_path or self.video_writer.output_path
            video_writer = VideoWriter(
                output_path=final_output,
                fps=fps,
                frame_size=(width, height)
            )

        # Reset statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }
        self.temporal_processor.clear_buffer()

        # Process frames
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            timestamp = frame_count / fps
            processed_frame = self.process_frame(frame, timestamp)

            # Save frame
            if video_writer:
                video_writer.write_frame(processed_frame)

            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")

            frame_count += 1

            # Check for quit
            if self.show_display and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.close()

        if self.show_display:
            cv2.destroyAllWindows()

        # Update final statistics
        elapsed = time.time() - self.stats['start_time']
        self.stats['total_time'] = elapsed
        self.stats['average_fps'] = frame_count / elapsed

        print(f"\nVideo processing complete!")
        print(f"Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {self.stats['average_fps']:.1f}")

        return self.stats

    def process_webcam(self, camera_index: int = 0, output_path: str = None) -> Dict:
        """
        Process webcam feed

        Args:
            camera_index: Camera device index
            output_path: Output video path

        Returns:
            Processing statistics
        """
        from .video_writer import VideoWriter, FrameOverlay

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return {}

        # Get camera properties
        fps = 30.0  # Assume 30 FPS for webcam
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing webcam feed (Camera {camera_index})")
        print(f"Resolution: {width}x{height}")

        # Update video writer if output path provided
        video_writer = None
        if output_path or self.save_video:
            final_output = output_path or self.video_writer.output_path
            video_writer = VideoWriter(
                output_path=final_output,
                fps=fps,
                frame_size=(width, height)
            )

        # Reset statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }
        self.temporal_processor.clear_buffer()

        # Process frames
        start_time = time.time()

        print("Press 'q' to quit...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            timestamp = time.time() - start_time
            processed_frame = self.process_frame(frame, timestamp)

            # Save frame
            if video_writer:
                video_writer.write_frame(processed_frame)

            # Check for quit
            if self.show_display and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.close()

        if self.show_display:
            cv2.destroyAllWindows()

        # Update final statistics
        elapsed = time.time() - self.stats['start_time']
        self.stats['total_time'] = elapsed
        self.stats['average_fps'] = self.stats['frames_processed'] / elapsed

        print(f"\nWebcam processing complete!")
        print(f"Processed {self.stats['frames_processed']} frames in {elapsed:.1f}s")
        print(f"Average FPS: {self.stats['average_fps']:.1f}")

        return self.stats

    def close(self):
        """Clean up resources with multi-person support"""
        if self.video_writer:
            self.video_writer.close()
        if self.show_display:
            cv2.destroyAllWindows()
        self.temporal_processor.clear_buffer()

        # Clean up multi-person tracking data
        self.prediction_smoothers.clear()
        self.current_actions.clear()
        self.current_confidences.clear()
        self.current_detections.clear()
