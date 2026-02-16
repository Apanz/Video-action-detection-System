"""
GUI的视频处理工作线程
处理实时视频检测而不阻塞UI
"""

import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from typing import Optional


class VideoProcessingThread(QThread):
    """
    用于视频处理的工作线程
    在后台运行检测流水线并将帧发送到UI
    """

    # 用于与UI通信的信号
    frame_ready = pyqtSignal(np.ndarray, dict)  # 帧和检测信息
    processing_finished = pyqtSignal(dict)  # 最终统计
    error_occurred = pyqtSignal(str)  # 错误消息
    result_ready = pyqtSignal(dict)  # 检测结果统计

    def __init__(self, pipeline, mode='webcam', video_path=None, camera_index=0):
        """
        初始化视频处理线程

        Args:
            pipeline: DetectionPipeline实例
            mode: 'webcam' 或 'video'
            video_path: 视频文件路径（用于视频模式）
            camera_index: 摄像头索引（用于摄像头模式）
        """
        super().__init__()
        self.pipeline = pipeline
        self.mode = mode
        self.video_path = video_path
        self.camera_index = camera_index
        self.is_running = False
        self.mutex = QMutex()

    def run(self):
        """主处理循环"""
        self.is_running = True

        try:
            if self.mode == 'webcam':
                self._process_webcam()
            elif self.mode == 'video':
                self._process_video_file()
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def _process_webcam(self):
        """处理摄像头馈送"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_index}")
            return

        try:
            # 跟踪开始时间以计算FPS
            start_time = time.time()

            # 获取摄像头信息
            camera_backend = cap.getBackendName()
            camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            device_info = f"Camera {self.camera_index} ({camera_backend})"

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                timestamp = time.time() - start_time
                processed_frame = self.pipeline.process_frame(frame, timestamp)

                # 获取检测信息
                elapsed = time.time() - start_time
                info = {
                    'action': self.pipeline.current_action,
                    'confidence': self.pipeline.current_confidence,
                    'fps': self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                    'device': device_info,
                    'resolution': f"{camera_width}x{camera_height}"
                }

                # 向UI发送帧
                self.frame_ready.emit(processed_frame, info)

                # 如果启用了结果收集则发送结果统计
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    # 每30帧发送一次统计信息以避免压倒UI
                    if self.pipeline.stats['frames_processed'] % 30 == 0:
                        stats = self.pipeline.result_collector.get_statistics()
                        self.result_ready.emit(stats)

                # 小睡以控制帧率并允许UI处理
                time.sleep(0.01)

                # 检查是否已停止
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break

        finally:
            cap.release()
            # 关键修复：关闭video_writer以释放文件锁
            if self.pipeline.video_writer:
                self.pipeline.video_writer.close()
                self.pipeline.video_writer = None

    def _process_video_file(self):
        """处理视频文件"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open video file: {self.video_path}")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 跟踪开始时间以计算FPS
            start_time = time.time()

            # 获取设备信息
            device_info = f"Video File ({video_width}x{video_height} @ {fps:.1f}fps)"

            frame_count = 0
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                timestamp = frame_count / fps
                processed_frame = self.pipeline.process_frame(frame, timestamp)

                # 获取检测信息
                elapsed = time.time() - start_time
                info = {
                    'action': self.pipeline.current_action,
                    'confidence': self.pipeline.current_confidence,
                    'fps': self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                    'device': device_info,
                    'resolution': f"{video_width}x{video_height}",
                    'progress': frame_count / total_frames if total_frames > 0 else 0
                }

                # 向UI发送帧
                self.frame_ready.emit(processed_frame, info)

                # 如果启用了结果收集，则发送结果统计信息
                if hasattr(self.pipeline, 'result_collector') and self.pipeline.result_collector:
                    # 每30帧发送一次统计信息以避免压倒UI
                    if frame_count % 30 == 0:
                        stats = self.pipeline.result_collector.get_statistics()
                        self.result_ready.emit(stats)

                frame_count += 1

                # 检查是否已停止
                with QMutexLocker(self.mutex):
                    if not self.is_running:
                        break

        finally:
            cap.release()
            # 关键修复：关闭video_writer以释放文件锁
            if self.pipeline.video_writer:
                self.pipeline.video_writer.close()
                self.pipeline.video_writer = None

        # 发送完成信号
        stats = self.pipeline.stats
        elapsed = time.time() - start_time
        stats['average_fps'] = self.pipeline.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        stats['total_frames'] = frame_count if self.mode == 'video' else self.pipeline.stats['frames_processed']
        self.processing_finished.emit(stats)

    def stop(self):
        """停止处理"""
        with QMutexLocker(self.mutex):
            self.is_running = False

        # 等待线程完成，带有超时（非阻塞）
        if self.isRunning():
            self.wait(1000)  # 最多等待1秒
