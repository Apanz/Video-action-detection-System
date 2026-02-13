"""
Video Output Module
Handles video writing with overlays and metadata
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class VideoWriter:
    """
    Handles video output with overlay support
    """

    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Tuple[int, int] = None,
                 codec: str = 'mp4v', quality: int = 95):
        """
        Initialize video writer

        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: Video frame size (width, height)
            codec: FourCC codec code
            quality: Video quality (0-100)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.quality = quality

        self.writer = None
        self.is_initialized = False
        self.frame_count = 0
        self.start_time = None

    def initialize(self, frame: np.ndarray):
        """
        Initialize video writer with sample frame

        Args:
            frame: Sample frame to get dimensions
        """
        if self.is_initialized:
            return

        if self.frame_size is None:
            # Get frame size from input
            height, width = frame.shape[:2]
            self.frame_size = (width, height)

        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Set quality for mp4 codec
        if self.codec == 'mp4v':
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps,
                self.frame_size, isColor=True
            )

        self.is_initialized = True
        self.start_time = time.time()

        print(f"Video writer initialized: {self.output_path}")
        print(f"Resolution: {self.frame_size[0]}x{self.frame_size[1]}")
        print(f"FPS: {self.fps}")

    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video

        Args:
            frame: Frame to write
        """
        if not self.is_initialized:
            self.initialize(frame)

        if self.writer is not None and self.is_initialized:
            # Ensure frame size matches
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                frame = cv2.resize(frame, self.frame_size)

            self.writer.write(frame)
            self.frame_count += 1

            # Print progress every 30 frames
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"Written {self.frame_count} frames ({current_fps:.1f} FPS)")

    def close(self):
        """
        Close video writer
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None

            elapsed = time.time() - self.start_time
            final_fps = self.frame_count / elapsed if elapsed > 0 else 0

            print(f"\nVideo writing complete!")
            print(f"Total frames: {self.frame_count}")
            print(f"Final FPS: {final_fps:.1f}")
            print(f"Output file: {self.output_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class FrameOverlay:
    """
    Handles overlay drawing on frames
    """

    @staticmethod
    def draw_info_panel(frame: np.ndarray, info: dict) -> np.ndarray:
        """
        Draw information panel on frame (top-right corner)
        Shows Resolution, Action, Persons with smaller font

        Args:
            frame: Input frame
            info: Dictionary with information to display

        Returns:
            Frame with info panel
        """
        output_frame = frame.copy()

        # Frame dimensions
        height, width = output_frame.shape[:2]

        # Text settings (smaller font size)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # Smaller font size
        line_height = 18
        text_thickness = 1

        # Draw info panel at top-right corner (without FPS)
        panel_width = 180
        panel_height = 75
        panel_color = (0, 0, 0, 150)

        # Calculate right-top position
        panel_x = width - panel_width - 10
        panel_y = 10

        # Draw panel background
        cv2.rectangle(output_frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     panel_color, -1)

        # Draw info text at right-top (without FPS)
        texts = [
            f"Resolution: {info.get('resolution', 'N/A')}",
            f"Action: {info.get('action', 'N/A')}",
            f"Persons: {info.get('person_count', 0)}"
        ]

        y_offset = panel_y + 25
        x_offset = panel_x + 10

        for i, text in enumerate(texts):
            y_pos = y_offset + i * line_height
            cv2.putText(output_frame, text, (x_offset, y_pos),
                       font, font_scale, (255, 255, 255), text_thickness)

        return output_frame

    @staticmethod
    def draw_action_label(frame: np.ndarray, label: str, confidence: float,
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draw action label above bounding box

        Args:
            frame: Input frame
            label: Action label
            confidence: Confidence score
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Frame with action label
        """
        output_frame = frame.copy()

        x1, y1, x2, y2 = bbox

        # Create label text
        label_text = f"{label}: {confidence:.2f}"

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )

        # Calculate label position
        label_x = x1
        label_y = y1 - 10

        # Draw background rectangle
        cv2.rectangle(output_frame,
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y),
                     (0, 255, 0), -1)

        # Draw text
        cv2.putText(output_frame, label_text, (label_x, label_y),
                   font, font_scale, (0, 0, 0), text_thickness)

        return output_frame

    @staticmethod
    def draw_timestamp(frame: np.ndarray, timestamp: str) -> np.ndarray:
        """
        Draw timestamp on frame (left-top corner with smaller font)

        Args:
            frame: Input frame
            timestamp: Timestamp string

        Returns:
            Frame with timestamp
        """
        output_frame = frame.copy()

        # Text settings (smaller font size)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # Smaller font size
        thickness = 1

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            timestamp, font, font_scale, thickness
        )

        # Position at left-top corner
        x = 10
        y = text_height + 10

        # Draw background
        cv2.rectangle(output_frame,
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + 5),
                     (0, 0, 0, 150), -1)

        # Draw text
        cv2.putText(output_frame, timestamp, (x, y),
                   font, font_scale, (255, 255, 255), thickness)

        return output_frame

    @staticmethod
    def draw_confidence_bar(frame: np.ndarray, confidence: float,
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draw confidence bar on bounding box

        Args:
            frame: Input frame
            confidence: Confidence score (0-1)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Frame with confidence bar
        """
        output_frame = frame.copy()

        x1, y1, x2, y2 = bbox

        # Bar dimensions
        bar_width = x2 - x1
        bar_height = 5
        bar_x = x1
        bar_y = y2 + 5

        # Draw background
        cv2.rectangle(output_frame,
                     (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # Draw confidence bar
        conf_width = int(bar_width * confidence)
        if conf_width > 0:
            cv2.rectangle(output_frame,
                         (bar_x, bar_y),
                         (bar_x + conf_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        return output_frame
