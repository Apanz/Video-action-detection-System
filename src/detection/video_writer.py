"""
视频输出模块
处理带有叠加层和元数据的视频写入
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class VideoWriter:
    """
    处理带有叠加层的视频输出
    """

    def __init__(self, output_path: str, fps: float = 30.0, frame_size: Tuple[int, int] = None,
                 codec: str = 'mp4v', quality: int = 95):
        """
        初始化视频写入器

        Args:
            output_path: 输出视频文件路径
            fps: 每秒帧数
            frame_size: 视频帧大小（宽度，高度）
            codec: FourCC编解码器代码
            quality: 视频质量（0-100）
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
        使用示例帧初始化视频写入器

        Args:
            frame: 用于获取尺寸的示例帧
        """
        if self.is_initialized:
            return

        if self.frame_size is None:
            # 从输入获取帧大小
            height, width = frame.shape[:2]
            self.frame_size = (width, height)

        # 初始化写入器
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # 为mp4编解码器设置质量
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
        将帧写入视频

        Args:
            frame: 要写入的帧
        """
        if not self.is_initialized:
            self.initialize(frame)

        if self.writer is not None and self.is_initialized:
            # 确保帧大小匹配
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                frame = cv2.resize(frame, self.frame_size)

            self.writer.write(frame)
            self.frame_count += 1

            # 每30帧打印一次进度
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"Written {self.frame_count} frames ({current_fps:.1f} FPS)")

    def close(self):
        """
        关闭视频写入器
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
    处理帧上的叠加层绘制
    """

    @staticmethod
    def draw_info_panel(frame: np.ndarray, info: dict) -> np.ndarray:
        """
        在帧上绘制信息面板（右上角）
        使用较小字体显示分辨率、动作、人员

        Args:
            frame: 输入帧
            info: 要显示的信息字典

        Returns:
            带有信息面板的帧
        """
        output_frame = frame.copy()

        # 帧尺寸
        height, width = output_frame.shape[:2]

        # 文本设置（较小字体大小）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # 较小字体大小
        line_height = 18
        text_thickness = 1

        # 在右上角绘制信息面板（不显示FPS）
        panel_width = 180
        panel_height = 75
        panel_color = (0, 0, 0, 150)

        # 计算右上位置
        panel_x = width - panel_width - 10
        panel_y = 10

        # 绘制面板背景
        cv2.rectangle(output_frame, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     panel_color, -1)

        # 在右上角绘制信息文本（不显示FPS）
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
        在边界框上方绘制动作标签

        Args:
            frame: 输入帧
            label: 动作标签
            confidence: 置信度分数
            bbox: 边界框 (x1, y1, x2, y2)

        Returns:
            带有动作标签的帧
        """
        output_frame = frame.copy()

        x1, y1, x2, y2 = bbox

        # 创建标签文本
        label_text = f"{label}: {confidence:.2f}"

        # 获取文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness
        )

        # 计算标签位置
        label_x = x1
        label_y = y1 - 10

        # 绘制背景矩形
        cv2.rectangle(output_frame,
                     (label_x, label_y - text_height - baseline),
                     (label_x + text_width, label_y),
                     (0, 255, 0), -1)

        # 绘制文本
        cv2.putText(output_frame, label_text, (label_x, label_y),
                   font, font_scale, (0, 0, 0), text_thickness)

        return output_frame

    @staticmethod
    def draw_timestamp(frame: np.ndarray, timestamp: str) -> np.ndarray:
        """
        在帧上绘制时间戳（左上角，较小字体）

        Args:
            frame: 输入帧
            timestamp: 时间戳字符串

        Returns:
            带有时间戳的帧
        """
        output_frame = frame.copy()

        # 文本设置（较小字体大小）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # 较小字体大小
        thickness = 1

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            timestamp, font, font_scale, thickness
        )

        # 定位在左上角
        x = 10
        y = text_height + 10

        # 绘制背景
        cv2.rectangle(output_frame,
                     (x - 5, y - text_height - 5),
                     (x + text_width + 5, y + 5),
                     (0, 0, 0, 150), -1)

        # 绘制文本
        cv2.putText(output_frame, timestamp, (x, y),
                   font, font_scale, (255, 255, 255), thickness)

        return output_frame

    @staticmethod
    def draw_confidence_bar(frame: np.ndarray, confidence: float,
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        在边界框上绘制置信度条

        Args:
            frame: 输入帧
            confidence: 置信度分数（0-1）
            bbox: 边界框 (x1, y1, x2, y2)

        Returns:
            带有置信度条的帧
        """
        output_frame = frame.copy()

        x1, y1, x2, y2 = bbox

        # 条形尺寸
        bar_width = x2 - x1
        bar_height = 5
        bar_x = x1
        bar_y = y2 + 5

        # 绘制背景
        cv2.rectangle(output_frame,
                     (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)

        # 绘制置信度条
        conf_width = int(bar_width * confidence)
        if conf_width > 0:
            cv2.rectangle(output_frame,
                         (bar_x, bar_y),
                         (bar_x + conf_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        return output_frame
