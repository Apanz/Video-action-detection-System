"""
主处理流水线
协调所有组件进行实时行为检测
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import torch
from .video_writer import FrameOverlay

from core.config import DetectionConfig


class PredictionSmoother:
    """使用指数移动平均跨帧平滑预测"""

    def __init__(self, alpha: float = 0.3, history_length: int = 5):
        """
        Args:
            alpha: 平滑因子(0-1)。越高 = 来自历史的影响越大。
            history_length: 保留的先前预测数量。
        """
        self.alpha = alpha
        self.history = deque(maxlen=history_length)

    def update(self, prediction: np.ndarray) -> np.ndarray:
        """
        使用新预测更新并返回平滑结果。

        Args:
            prediction: 新预测（类别上的概率分布）

        Returns:
            平滑后的预测
        """
        self.history.append(prediction)
        if len(self.history) < 2:
            return prediction

        # 指数移动平均
        smoothed = self.history[-1].copy()
        for hist in list(self.history)[-2::-1]:
            smoothed = self.alpha * hist + (1 - self.alpha) * smoothed

        return smoothed


class DetectionPipeline:
    """
    实时行为检测的主流水线
    支持多人检测和动作分类
    """

    # 不同人员的颜色（BGR格式）
    PERSON_COLORS = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红色
    ]

    def __init__(self, checkpoint_path: str, yolo_model=None,
                 output_path: str = None, fps: float = 30.0,
                 show_display: bool = True, save_video: bool = False,
                 max_persons: int = 5,
                 enable_result_collection: bool = False,
                 results_dir: str = "outputs/results",
                 label_file=None):
        """
        初始化检测流水线

        Args:
            checkpoint_path: 训练好的TSN模型检查点路径
            yolo_model: YOLO模型路径 (default: from DetectionConfig)
            output_path: 输出视频路径（None表示不输出）
            fps: 目标FPS
            show_display: 显示实时显示
            save_video: 保存输出视频
            max_persons: 同时跟踪的最大人数
            enable_result_collection: 启用检测结果收集
            results_dir: 保存结果的目录
            label_file: 可选的标签文件路径
        """
        # Use config defaults if not specified
        if yolo_model is None:
            yolo_model = DetectionConfig.YOLO_MODEL

        # Map model name to full path if needed
        if yolo_model in DetectionConfig.DEFAULT_YOLO_MODELS:
            yolo_model = DetectionConfig.DEFAULT_YOLO_MODELS[yolo_model]
        elif not yolo_model.endswith('.pt'):
            # If it's just a name without .pt, try to add it and look up
            yolo_model_with_ext = yolo_model + '.pt'
            if yolo_model_with_ext in DetectionConfig.DEFAULT_YOLO_MODELS:
                yolo_model = DetectionConfig.DEFAULT_YOLO_MODELS[yolo_model_with_ext]

        self.show_display = show_display
        self.save_video = save_video or (output_path is not None)
        self.max_persons = max_persons

        # 存储帧分辨率
        self.frame_resolution = None

        # 显示预测的置信度阈值（使用DetectionConfig）
        self.confidence_threshold = DetectionConfig.DETECTION_CONFIDENCE

        # 结果收集
        self.enable_result_collection = enable_result_collection
        self.result_collector = None
        if enable_result_collection:
            from .result_collector import ResultCollector
            self.result_collector = ResultCollector(
                save_dir=results_dir,
                max_frames_per_action=10,
                save_frame_images=True
            )

        # 初始化组件
        print("Initializing components...")

        # 人体检测器
        from .human_detector import HumanDetector
        self.detector = HumanDetector(
            model_path=yolo_model,
            confidence=0.5,
            device='auto'
        )

        # 多人支持：每个轨迹的预测平滑器
        self.prediction_smoothers: Dict[int, PredictionSmoother] = {}

        # 跟踪每个人的当前动作
        self.current_actions: Dict[int, str] = {}
        self.current_confidences: Dict[int, float] = {}
        self.current_detections: Dict[int, dict] = {}

        # 轨迹ID计数器，用于为新检测分配ID
        self.next_track_id = 0

        # 动作分类器
        from .action_classifier import load_classifier
        self.classifier = load_classifier(
            checkpoint_path=checkpoint_path,
            device='auto',
            label_file=label_file
        )

        # 时序处理器（使用分类器的时序参数以确保一致性）
        from .temporal_processor import TemporalProcessor
        self.temporal_processor = TemporalProcessor(
            num_segments=self.classifier.num_segments,
            frames_per_segment=self.classifier.frames_per_segment,
            max_memory_mb=500.0  # 多人跟踪的内存限制
        )

        # 视频写入器
        from .video_writer import VideoWriter, FrameOverlay
        self.video_writer = None
        if self.save_video and output_path:
            self.video_writer = VideoWriter(
                output_path=output_path,
                fps=fps
            )

        # 初始化显示窗口
        if self.show_display:
            cv2.namedWindow('Real-time Behavior Detection', cv2.WINDOW_NORMAL)

        # 统计
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }

        # 当前动作状态
        self.current_action = "Unknown"
        self.current_confidence = 0.0
        self.last_detection_time = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        处理单个帧，支持多人检测

        Args:
            frame: 输入帧
            timestamp: 帧时间戳

        Returns:
            带有叠加层的处理后帧
        """
        # 存储帧分辨率
        if self.frame_resolution is None:
            self.frame_resolution = (frame.shape[1], frame.shape[0])

        # 检测人体
        detections = self.detector.detect(frame)
        self.stats['detections'] += len(detections)

        # 限制检测数量为max_persons
        detections = detections[:self.max_persons]

        # 处理每个检测
        active_track_ids = []

        for idx, detection in enumerate(detections):
            # 基于检测索引分配轨迹ID（简化跟踪）
            # 生产环境中应使用DeepSORT等适当的跟踪
            track_id = idx

            # 为新轨迹初始化预测平滑器
            if track_id not in self.prediction_smoothers:
                self.prediction_smoothers[track_id] = PredictionSmoother(alpha=0.3, history_length=5)

            active_track_ids.append(track_id)

            # 裁剪人员帧
            person_crop = self.detector.crop_person(frame, detection)

            # 将裁剪的帧添加到时序缓冲区
            frame_ready = self.temporal_processor.add_frame(
                person_crop, track_id, timestamp
            )

            if frame_ready:
                # 获取用于分类的时序片段
                segment_frames = self.temporal_processor.get_temporal_segments(track_id)

                if segment_frames:
                    # 为TSN模型进行预处理
                    input_tensor = self.temporal_processor.preprocess_frames(segment_frames)

                    # 获取概率分布并应用时序平滑
                    probs = self.classifier.predict_proba(input_tensor)
                    smoother = self.prediction_smoothers[track_id]
                    smoothed_probs = smoother.update(probs)

                    # 从平滑概率中获取最佳预测
                    top_idx = np.argmax(smoothed_probs)
                    action = self.classifier.class_labels[top_idx]
                    confidence = float(smoothed_probs[top_idx])

                    # 应用置信度阈值
                    if confidence >= self.confidence_threshold:
                        display_action = action
                        display_confidence = confidence
                    else:
                        display_action = "Detecting..."
                        display_confidence = 0.0

                    # 更新此轨迹的状态
                    self.current_actions[track_id] = action
                    self.current_confidences[track_id] = confidence
                    self.current_detections[track_id] = detection
                    self.stats['classifications'] += 1

                    # 如果启用则添加到结果收集器
                    if self.result_collector:
                        self.result_collector.add_result(
                            frame=person_crop,
                            action=action,
                            confidence=confidence,
                            timestamp=timestamp
                        )
            else:
                # 帧数不足
                self.current_actions[track_id] = "Collecting..."
                self.current_confidences[track_id] = 0.0
                self.current_detections[track_id] = detection

        # 移除非活动轨迹
        all_track_ids = list(self.current_actions.keys())
        for track_id in all_track_ids:
            if track_id not in active_track_ids:
                # 在移除前保留几帧的轨迹数据
                #（以便更好地处理暂时遮挡）
                pass

        # 使用多人支持绘制所有检测
        frame = self._draw_multi_person_detections(frame, active_track_ids)

        # 定期清理旧轨迹
        if self.stats['frames_processed'] % 30 == 0:
            self.temporal_processor.remove_old_tracks(timestamp, max_age=2.0)
            # 同时移除非活动轨迹的预测平滑器
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

        # 绘制叠加层
        info = self._get_info_dict(timestamp)
        frame = FrameOverlay.draw_info_panel(frame, info)

        # 更新GUI显示用的当前动作和置信度
        # 使用与_get_info_dict相同的逻辑：显示置信度最高的人员的动作
        person_count = len(self.current_actions)
        if person_count > 0:
            # 查找置信度最高的人员
            best_track_id = max(self.current_confidences.items(),
                             key=lambda x: x[1])[0] if self.current_confidences else None
            if best_track_id is not None:
                self.current_action = self.current_actions.get(best_track_id, "Unknown")
                self.current_confidence = self.current_confidences.get(best_track_id, 0.0)
            else:
                self.current_action = "Detecting..."
                self.current_confidence = 0.0
        else:
            self.current_action = "No person detected"
            self.current_confidence = 0.0

        # 绘制时间戳
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
        frame = FrameOverlay.draw_timestamp(frame, timestamp_str)

        # 更新统计
        self.stats['frames_processed'] += 1

        # 如果启用了视频写入器则保存帧
        if self.video_writer:
            self.video_writer.write_frame(frame)

        # 如果启用了显示则显示帧
        if self.show_display:
            cv2.imshow('Real-time Behavior Detection', frame)

        return frame

    def _draw_multi_person_detections(self, frame: np.ndarray, track_ids: List[int]) -> np.ndarray:
        """
        为多个人绘制检测框和标签

        Args:
            frame: 输入帧
            track_ids: 活动轨迹ID列表

        Returns:
            绘制了检测结果的帧
        """
        output_frame = frame.copy()

        for track_id in track_ids:
            if track_id not in self.current_detections:
                continue

            detection = self.current_detections[track_id]
            action = self.current_actions.get(track_id, "Unknown")
            confidence = self.current_confidences.get(track_id, 0.0)

            # 获取此人员的颜色（循环使用颜色）
            color = self.PERSON_COLORS[track_id % len(self.PERSON_COLORS)]

            # 获取边界框
            x1, y1, x2, y2 = detection['bbox']

            # 绘制边界框
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # 文本设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_thickness = 1
            line_height = 18

            # 准备标签
            labels = []
            labels.append(f"Person {track_id + 1}: {detection['confidence']:.2f}")

            if action and confidence is not None:
                labels.append(f"Action: {action}")
                labels.append(f"Conf: {confidence:.2f}")

            # 在边界框上方绘制标签
            for i, label in enumerate(labels):
                y_pos = y1 - 10 - (len(labels) - i - 1) * line_height

                # 确保标签不会超出屏幕
                y_pos = max(y_pos, line_height)

                # 绘制带背景的文本以提高可见性
                (text_width, text_height) = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
                cv2.rectangle(output_frame,
                            (x1, y_pos - text_height - 2),
                            (x1 + text_width + 4, y_pos + 2),
                            color, -1)
                cv2.putText(output_frame, label, (x1 + 2, y_pos),
                          font, font_scale, (255, 255, 255), text_thickness)

        return output_frame

    def _get_info_dict(self, timestamp: float) -> Dict:
        """获取用于叠加层的信息字典，支持多人检测"""
        elapsed = timestamp - self.stats['start_time']
        current_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0

        # 使用标准乘号将分辨率格式化为宽度×高度
        resolution_str = f"{self.frame_resolution[0]}x{self.frame_resolution[1]}" if self.frame_resolution else "N/A"

        # 计算活动人员数量
        person_count = len(self.current_actions)

        # 用于显示，显示置信度最高的人员的动作
        if person_count > 0:
            # 查找置信度最高的人员
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
        处理视频文件

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（覆盖实例设置）

        Returns:
            处理统计信息
        """
        from .video_writer import VideoWriter, FrameOverlay

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return {}

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # 如果提供了输出路径，则更新视频写入器
        video_writer = None
        if output_path or self.save_video:
            final_output = output_path or self.video_writer.output_path
            video_writer = VideoWriter(
                output_path=final_output,
                fps=fps,
                frame_size=(width, height)
            )

        # 重置统计信息
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }
        self.temporal_processor.clear_buffer()

        # 处理帧
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            timestamp = frame_count / fps
            processed_frame = self.process_frame(frame, timestamp)

            # 保存帧
            if video_writer:
                video_writer.write_frame(processed_frame)

            # 打印进度
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")

            frame_count += 1

            # 检查是否退出
            if self.show_display and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 清理
        cap.release()
        if video_writer:
            video_writer.close()

        if self.show_display:
            cv2.destroyAllWindows()

        # 更新最终统计信息
        elapsed = time.time() - self.stats['start_time']
        self.stats['total_time'] = elapsed
        self.stats['average_fps'] = frame_count / elapsed

        print(f"\nVideo processing complete!")
        print(f"Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {self.stats['average_fps']:.1f}")

        return self.stats

    def process_webcam(self, camera_index: int = 0, output_path: str = None) -> Dict:
        """
        处理摄像头馈送

        Args:
            camera_index: 摄像头设备索引
            output_path: 输出视频路径

        Returns:
            处理统计信息
        """
        from .video_writer import VideoWriter, FrameOverlay

        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return {}

        # 获取摄像头属性
        fps = 30.0  # 假设摄像头为30 FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing webcam feed (Camera {camera_index})")
        print(f"Resolution: {width}x{height}")

        # 如果提供了输出路径，则更新视频写入器
        video_writer = None
        if output_path or self.save_video:
            final_output = output_path or self.video_writer.output_path
            video_writer = VideoWriter(
                output_path=final_output,
                fps=fps,
                frame_size=(width, height)
            )

        # 重置统计信息
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'start_time': time.time()
        }
        self.temporal_processor.clear_buffer()

        # 处理帧
        start_time = time.time()

        print("Press 'q' to quit...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            timestamp = time.time() - start_time
            processed_frame = self.process_frame(frame, timestamp)

            # 保存帧
            if video_writer:
                video_writer.write_frame(processed_frame)

            # 检查是否退出
            if self.show_display and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 清理
        cap.release()
        if video_writer:
            video_writer.close()

        if self.show_display:
            cv2.destroyAllWindows()

        # 更新最终统计信息
        elapsed = time.time() - self.stats['start_time']
        self.stats['total_time'] = elapsed
        self.stats['average_fps'] = self.stats['frames_processed'] / elapsed

        print(f"\nWebcam processing complete!")
        print(f"Processed {self.stats['frames_processed']} frames in {elapsed:.1f}s")
        print(f"Average FPS: {self.stats['average_fps']:.1f}")

        return self.stats

    def close(self):
        """清理资源，支持多人检测"""
        if self.video_writer:
            self.video_writer.close()
        if self.show_display:
            cv2.destroyAllWindows()
        self.temporal_processor.clear_buffer()

        # 清理多人跟踪数据
        self.prediction_smoothers.clear()
        self.current_actions.clear()
        self.current_confidences.clear()
        self.current_detections.clear()
