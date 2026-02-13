"""
Temporal Processing Module
Maintains frame buffers for consistent action classification
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import torch
import sys
import gc


class TemporalProcessor:
    """
    Processes temporal segments for consistent action recognition
    Maintains frame buffers and applies TSN sampling strategy
    """

    def __init__(self, num_segments: int = 3, frames_per_segment: int = 5,
                 buffer_size: int = 30, target_size: tuple = (224, 224),
                 max_memory_mb: float = 500.0):
        """
        Initialize temporal processor

        Args:
            num_segments: Number of temporal segments for TSN
            frames_per_segment: Frames per segment
            buffer_size: Maximum frames to store in buffer
            target_size: Target resize size (H, W)
            max_memory_mb: Maximum memory usage in MB before triggering cleanup
        """
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment
        self.buffer_size = buffer_size
        self.target_size = target_size
        self.max_memory_mb = max_memory_mb

        # Frame buffer for each detected person
        # Structure: {track_id: deque of frames}
        self.frame_buffers: Dict[int, deque] = {}

        # Timestamp buffer for temporal consistency
        self.timestamp_buffers: Dict[int, deque] = {}

        # Frame count for each track
        self.frame_counts: Dict[int, int] = {}

        # Memory monitoring
        self.memory_warnings = 0
        self.max_memory_threshold_reached = False

    def add_frame(self, frame: np.ndarray, track_id: int, timestamp: float) -> bool:
        """
        Add frame to buffer for specific track

        Args:
            frame: Input frame (BGR format)
            track_id: Track ID for person
            timestamp: Frame timestamp

        Returns:
            True if buffer is ready for processing, False otherwise
        """
        # Initialize track if not exists
        if track_id not in self.frame_buffers:
            self.frame_buffers[track_id] = deque(maxlen=self.buffer_size)
            self.timestamp_buffers[track_id] = deque(maxlen=self.buffer_size)
            self.frame_counts[track_id] = 0

        # CRITICAL FIX: Intelligent frame scaling to reduce memory usage
        # If frame is larger than target size, scale it down before storing
        # This prevents high-resolution person crops from consuming excessive memory
        processed_frame = self._scale_frame_intelligently(frame)

        # Add processed frame to buffer
        self.frame_buffers[track_id].append(processed_frame)
        self.timestamp_buffers[track_id].append(timestamp)
        self.frame_counts[track_id] += 1

        # Periodically check memory usage and cleanup if needed
        if self.frame_counts[track_id] % 10 == 0:
            self._check_memory_and_cleanup()

        # Check if we have enough frames
        return len(self.frame_buffers[track_id]) >= self.total_frames

    def _scale_frame_intelligently(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Intelligently scale frame to reduce memory usage while preserving aspect ratio

        Args:
            frame: Input frame (BGR format)
            max_size: Maximum dimension size (width or height)

        Returns:
            Scaled frame if needed, otherwise original frame
        """
        h, w = frame.shape[:2]

        # Check if scaling is needed
        if h <= max_size and w <= max_size:
            # Frame is small enough, return as-is
            return frame

        # Calculate scale factor to fit within max_size while preserving aspect ratio
        scale = min(max_size / h, max_size / w)

        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Scale frame
        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return scaled_frame

    def get_temporal_segments(self, track_id: int) -> Optional[List[np.ndarray]]:
        """
        Extract temporal segments for TSN processing

        Args:
            track_id: Track ID

        Returns:
            List of frames in TSN format or None if not enough frames
        """
        if track_id not in self.frame_buffers or len(self.frame_buffers[track_id]) < self.total_frames:
            return None

        frames = list(self.frame_buffers[track_id])
        timestamps = list(self.timestamp_buffers[track_id])

        # Apply TSN sampling strategy
        # For test/validation mode: uniform sampling within segments
        segment_frames = []

        for seg_idx in range(self.num_segments):
            start_idx = int(seg_idx * len(frames) / self.num_segments)
            end_idx = int((seg_idx + 1) * len(frames) / self.num_segments)
            segment_frame_count = self.frames_per_segment

            # Sample frames from this segment
            if self.frames_per_segment == 1:
                # Single frame: use center frame of segment
                frame_idx = (start_idx + end_idx) // 2
                selected_frame = frames[frame_idx]
            else:
                # Multiple frames: distribute evenly
                segment_frames_list = []
                for i in range(segment_frame_count):
                    # Calculate position within segment
                    pos = start_idx + (end_idx - start_idx) * i / (segment_frame_count - 1)
                    frame_idx = int(pos)
                    frame_idx = min(frame_idx, end_idx - 1)
                    segment_frames_list.append(frames[frame_idx])

                # Average segment frames if needed
                selected_frame = self._average_frames(segment_frames_list)

            segment_frames.append(selected_frame)

        return segment_frames

    def _average_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Average multiple frames together

        Args:
            frames: List of frames to average

        Returns:
            Averaged frame
        """
        # CRITICAL: Resize all frames to target size before stacking
        # This handles person crops with different sizes
        frames_resized = [cv2.resize(frame, self.target_size) for frame in frames]

        # Convert to float for averaging
        frames_float = [frame.astype(np.float32) for frame in frames_resized]

        # Stack and average
        stacked = np.stack(frames_float, axis=0)
        averaged = np.mean(stacked, axis=0)

        # Convert back to uint8
        return averaged.astype(np.uint8)

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess frames for TSN model input
        Must match training preprocessing exactly:
        Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)

        Args:
            frames: List of frames (BGR format)

        Returns:
            Preprocessed tensor (1, T, C, H, W)
        """
        processed_frames = []

        # ImageNet normalization constants (use float32 to match model)
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 1: Resize to 256 (matches training)
            resized_256 = cv2.resize(rgb_frame, (256, 256))

            # Step 2: Center crop to 224 (matches training)
            h, w = resized_256.shape[:2]
            top = (h - 224) // 2
            left = (w - 224) // 2
            cropped_frame = resized_256[top:top+224, left:left+224]

            # Step 3: Normalize to [0, 1]
            normalized_frame = cropped_frame.astype(np.float32) / 255.0

            # Step 4: Apply ImageNet normalization
            imagenet_normalized = (normalized_frame - IMAGENET_MEAN) / IMAGENET_STD

            # Convert to tensor and add channel dimension (HWC -> CHW)
            tensor_frame = torch.from_numpy(imagenet_normalized).permute(2, 0, 1)

            processed_frames.append(tensor_frame)

        # Stack frames: (T, C, H, W) -> (1, T, C, H, W)
        stacked_frames = torch.stack(processed_frames, dim=0)
        batch_frames = stacked_frames.unsqueeze(0)

        return batch_frames

    def remove_old_tracks(self, current_time: float, max_age: float = 2.0):
        """
        Remove tracks that haven't been updated recently

        Args:
            current_time: Current timestamp
            max_age: Maximum age in seconds before removing track
        """
        tracks_to_remove = []

        for track_id in self.frame_buffers:
            if self.timestamp_buffers[track_id]:
                last_update = self.timestamp_buffers[track_id][-1]
                if current_time - last_update > max_age:
                    tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.frame_buffers[track_id]
            del self.timestamp_buffers[track_id]
            del self.frame_counts[track_id]
            print(f"Removed track {track_id} (inactive for {max_age:.1f}s)")

    def get_track_stats(self) -> Dict[int, Dict]:
        """
        Get statistics for all active tracks

        Returns:
            Dictionary with track statistics
        """
        stats = {}

        for track_id in self.frame_buffers:
            stats[track_id] = {
                'frame_count': self.frame_counts[track_id],
                'buffer_size': len(self.frame_buffers[track_id]),
                'ready_for_processing': len(self.frame_buffers[track_id]) >= self.total_frames
            }

        return stats

    def clear_buffer(self, track_id: int = None):
        """
        Clear frame buffer for specific track or all tracks

        Args:
            track_id: Specific track ID, None for all tracks
        """
        if track_id is not None:
            if track_id in self.frame_buffers:
                del self.frame_buffers[track_id]
                del self.timestamp_buffers[track_id]
                del self.frame_counts[track_id]
        else:
            self.frame_buffers.clear()
            self.timestamp_buffers.clear()
            self.frame_counts.clear()

    def _estimate_memory_usage(self) -> float:
        """
        Estimate current memory usage of frame buffers in MB

        Returns:
            Estimated memory usage in MB
        """
        total_bytes = 0
        for track_id, buffer in self.frame_buffers.items():
            for frame in buffer:
                # Estimate size: height * width * channels * 3 (uint8 -> potential expansion)
                if hasattr(frame, 'nbytes'):
                    total_bytes += frame.nbytes
                else:
                    # Fallback estimation
                    h, w = frame.shape[:2] if len(frame.shape) >= 2 else (224, 224)
                    c = frame.shape[2] if len(frame.shape) == 3 else 3
                    total_bytes += h * w * c * 4  # Assume float32 worst case

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _check_memory_and_cleanup(self):
        """
        Check memory usage and perform cleanup if needed
        """
        estimated_memory = self._estimate_memory_usage()

        if estimated_memory > self.max_memory_mb:
            if not self.max_memory_threshold_reached:
                self.max_memory_threshold_reached = True
                self.memory_warnings += 1
                print(f"WARNING: Memory usage ({estimated_memory:.1f} MB) exceeds threshold ({self.max_memory_mb} MB)")
                print("Performing aggressive cleanup...")

                # Force garbage collection
                gc.collect()

                # Reduce buffer sizes for all tracks
                for track_id in self.frame_buffers:
                    current_size = len(self.frame_buffers[track_id])
                    if current_size > self.total_frames:
                        # Trim buffer to minimum required size
                        excess = current_size - self.total_frames
                        for _ in range(excess):
                            self.frame_buffers[track_id].popleft()
                            self.timestamp_buffers[track_id].popleft()

                # Remove inactive tracks more aggressively
                import time
                current_time = time.time()
                self.remove_old_tracks(current_time, max_age=1.0)  # More aggressive

                print(f"Cleanup complete. Memory after cleanup: {self._estimate_memory_usage():.1f} MB")

        elif estimated_memory > self.max_memory_mb * 0.8 and not self.max_memory_threshold_reached:
            # Warning threshold at 80%
            if self.memory_warnings == 0:
                print(f"INFO: Memory usage at {estimated_memory:.1f} MB ({estimated_memory/self.max_memory_mb*100:.0f}% of threshold)")
                self.memory_warnings += 1
