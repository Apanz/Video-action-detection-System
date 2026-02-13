"""
Video processing worker thread for GUI
Handles real-time video detection without blocking the UI
"""

import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from typing import Optional


class VideoProcessingThread(QThread):
    """
    Worker thread for video processing
    Runs detection pipeline in background and sends frames to UI
    """

    # Signals for communication with UI
    frame_ready = pyqtSignal(np.ndarray, dict)  # Frame and detection info
    processing_finished = pyqtSignal(dict)  # Final statistics
    error_occurred = pyqtSignal(str)  # Error message
    result_ready = pyqtSignal(dict)  # Detection results statistics

    def __init__(self, pipeline, mode='webcam', video_path=None, camera_index=0):
        """
        Initialize video processing thread

        Args:
            pipeline: DetectionPipeline instance
            mode: 'webcam' or 'video'
            video_path: Path to video file (for video mode)
            camera_index: Camera index (for webcam mode)
        """
        super().__init__()
        self.pipeline = pipeline
        self.mode = mode
        self.video_path = video_path
        self.camera_index = camera_index
        self.is_running = False
        self.mutex = QMutex()

    def run(self):
        """Main processing loop"""
        self.is_running = True

        try:
            if self.mode == 'webcam':
                self._process_webcam()
            elif self.mode == 'video':
                self._process_video_file()
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def _process_webcam(self):
        """Process webcam feed"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_index}")
            return

        try:
            # Track start time for FPS calculation
            start_time = time.time()

            # Get camera information
            camera_backend = cap.getBackendName()
            camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            device_info = f"Camera {self.camera_index} ({camera_backend})"

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                timestamp = time.time() - start_time
                processed_frame = self.pipeline.process_frame(frame, timestamp)

                # Get detection info
                elapsed = time.time() - start_time
                info = {
                    'action': self.pipeline.current_action,
                    'confidence': self.pipeline.current_confidence,
                    'fps': self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                    'device': device_info,
                    'resolution': f"{camera_width}x{camera_height}"
                }

                # Emit frame to UI
                self.frame_ready.emit(processed_frame, info)

                # Emit result statistics if collection is enabled
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    # Emit stats every 30 frames to avoid overwhelming UI
                    if self.pipeline.stats['frames_processed'] % 30 == 0:
                        stats = self.pipeline.result_collector.get_statistics()
                        self.result_ready.emit(stats)

                # Small sleep to control frame rate and allow UI to process
                time.sleep(0.01)

                # Check if stopped
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break

        finally:
            cap.release()
            # CRITICAL FIX: Close video_writer to release file lock
            if self.pipeline.video_writer:
                self.pipeline.video_writer.close()
                self.pipeline.video_writer = None

    def _process_video_file(self):
        """Process video file"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open video file: {self.video_path}")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Track start time for FPS calculation
            start_time = time.time()

            # Get device info
            device_info = f"Video File ({video_width}x{video_height} @ {fps:.1f}fps)"

            frame_count = 0
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                timestamp = frame_count / fps
                processed_frame = self.pipeline.process_frame(frame, timestamp)

                # Get detection info
                elapsed = time.time() - start_time
                info = {
                    'action': self.pipeline.current_action,
                    'confidence': self.pipeline.current_confidence,
                    'fps': self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                    'device': device_info,
                    'resolution': f"{video_width}x{video_height}",
                    'progress': frame_count / total_frames if total_frames > 0 else 0
                }

                # Emit frame to UI
                self.frame_ready.emit(processed_frame, info)

                # Emit result statistics if collection is enabled
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    # Emit stats every 30 frames to avoid overwhelming UI
                    if frame_count % 30 == 0:
                        stats = self.pipeline.result_collector.get_statistics()
                        self.result_ready.emit(stats)

                frame_count += 1

                # Check if stopped
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break

        finally:
            cap.release()
            # CRITICAL FIX: Close video_writer to release file lock
            if self.pipeline.video_writer:
                self.pipeline.video_writer.close()
                self.pipeline.video_writer = None

        # Emit completion signal
        stats = self.pipeline.stats
        elapsed = time.time() - start_time
        stats['average_fps'] = self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        stats['total_frames'] = frame_count if self.mode == 'video' else self.pipeline.stats['frames_processed']
        self.processing_finished.emit(stats)

    def stop(self):
        """Stop processing"""
        with QMutexLocker(self.mutex):
            self.is_running = False

        # Wait for thread to finish with timeout (non-blocking)
        if self.isRunning():
            self.wait(1000)  # Wait up to 1 second
