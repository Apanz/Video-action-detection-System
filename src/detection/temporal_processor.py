"""
时序处理模块
维护帧缓冲区以实现一致的动作分类
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import torch
import sys
import gc

from core.config import DetectionConfig, DataConfig


class TemporalProcessor:
    """
    为一致的动作识别处理时序片段
    维护帧缓冲区并应用TSN采样策略
    """

    def __init__(self, num_segments=None, frames_per_segment=None,
                 buffer_size=None, target_size=None,
                 max_memory_mb: float = 500.0):
        """
        初始化时序处理器

        Args:
            num_segments: TSN的时序片段数量 (default: from DetectionConfig)
            frames_per_segment: 每片段帧数 (default: from DetectionConfig)
            buffer_size: 缓冲区中存储的最大帧数 (default: from DetectionConfig)
            target_size: 目标调整大小 (H, W) (default: from DataConfig)
            max_memory_mb: 触发清理前的最大内存使用量（MB）
        """
        # Use config defaults if not specified
        if num_segments is None:
            num_segments = DetectionConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DetectionConfig.FRAMES_PER_SEGMENT
        if buffer_size is None:
            buffer_size = DetectionConfig.TEMPORAL_BUFFER_SIZE
        if target_size is None:
            target_size = (DataConfig.INPUT_SIZE, DataConfig.INPUT_SIZE)

        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment
        self.buffer_size = buffer_size
        self.target_size = target_size
        self.max_memory_mb = max_memory_mb

        # 每个检测人员的帧缓冲区
        # 结构：{track_id: 帧的双端队列}
        self.frame_buffers: Dict[int, deque] = {}

        # 时间一致性时间戳缓冲区
        self.timestamp_buffers: Dict[int, deque] = {}

        # 每个轨迹的帧计数
        self.frame_counts: Dict[int, int] = {}

        # 内存监控
        self.memory_warnings = 0
        self.max_memory_threshold_reached = False

    def add_frame(self, frame: np.ndarray, track_id: int, timestamp: float) -> bool:
        """
        将帧添加到特定轨迹的缓冲区

        Args:
            frame: 输入帧（BGR格式）
            track_id: 人员的轨迹ID
            timestamp: 帧时间戳

        Returns:
            如果缓冲区准备好处理则返回True，否则返回False
        """
        # 如果不存在则初始化轨迹
        if track_id not in self.frame_buffers:
            self.frame_buffers[track_id] = deque(maxlen=self.buffer_size)
            self.timestamp_buffers[track_id] = deque(maxlen=self.buffer_size)
            self.frame_counts[track_id] = 0

        # 关键修复：智能帧缩放以减少内存使用
        # 如果帧大于目标大小，在存储前将其缩小
        # 这可以防止高分辨率的人员裁剪消耗过多内存
        processed_frame = self._scale_frame_intelligently(frame)

        # 将处理后的帧添加到缓冲区
        self.frame_buffers[track_id].append(processed_frame)
        self.timestamp_buffers[track_id].append(timestamp)
        self.frame_counts[track_id] += 1

        # 定期检查内存使用情况并在需要时进行清理
        if self.frame_counts[track_id] % 10 == 0:
            self._check_memory_and_cleanup()

        # 检查我们是否有足够的帧
        return len(self.frame_buffers[track_id]) >= self.total_frames

    def _scale_frame_intelligently(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        智能缩放帧以减少内存使用，同时保持纵横比

        Args:
            frame: 输入帧（BGR格式）
            max_size: 最大维度大小（宽度或高度）

        Returns:
            如果需要则返回缩放后的帧，否则返回原始帧
        """
        h, w = frame.shape[:2]

        # 检查是否需要缩放
        if h <= max_size and w <= max_size:
            # 帧足够小，按原样返回
            return frame

        # 计算缩放因子以适应max_size，同时保持纵横比
        scale = min(max_size / h, max_size / w)

        # 计算新尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 缩放帧
        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return scaled_frame

    def get_temporal_segments(self, track_id: int) -> Optional[List[np.ndarray]]:
        """
        提取用于TSN处理的时序片段

        Args:
            track_id: 轨迹ID

        Returns:
            TSN格式的帧列表，如果帧数不足则返回None
        """
        if track_id not in self.frame_buffers or len(self.frame_buffers[track_id]) < self.total_frames:
            return None

        frames = list(self.frame_buffers[track_id])
        timestamps = list(self.timestamp_buffers[track_id])

        # 应用TSN采样策略
        # 对于测试/验证模式：在片段内均匀采样
        segment_frames = []

        for seg_idx in range(self.num_segments):
            start_idx = int(seg_idx * len(frames) / self.num_segments)
            end_idx = int((seg_idx + 1) * len(frames) / self.num_segments)
            segment_frame_count = self.frames_per_segment

            # 从此片段中采样帧
            if self.frames_per_segment == 1:
                # 单帧：使用片段的中心帧
                frame_idx = (start_idx + end_idx) // 2
                selected_frame = frames[frame_idx]
            else:
                # 多帧：均匀分布
                segment_frames_list = []
                for i in range(segment_frame_count):
                    # 计算片段内的位置
                    pos = start_idx + (end_idx - start_idx) * i / (segment_frame_count - 1)
                    frame_idx = int(pos)
                    frame_idx = min(frame_idx, end_idx - 1)
                    segment_frames_list.append(frames[frame_idx])

                # 如果需要，对片段帧进行平均
                selected_frame = self._average_frames(segment_frames_list)

            segment_frames.append(selected_frame)

        return segment_frames

    def _average_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        将多个帧平均在一起

        Args:
            frames: 要平均的帧列表

        Returns:
            平均后的帧
        """
        # 关键：在堆叠之前将所有帧调整为相同的目标尺寸
        # 这可以处理不同大小的人员裁剪
        frames_resized = [cv2.resize(frame, self.target_size) for frame in frames]

        # 转换为浮点数以进行平均
        frames_float = [frame.astype(np.float32) for frame in frames_resized]

        # 堆叠并平均
        stacked = np.stack(frames_float, axis=0)
        averaged = np.mean(stacked, axis=0)

        # 转换回uint8
        return averaged.astype(np.uint8)

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        为TSN模型输入预处理帧
        必须完全匹配训练预处理：
        Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)

        Args:
            frames: 帧列表（BGR格式）

        Returns:
            预处理后的张量 (1, T, C, H, W)
        """
        processed_frames = []

        # ImageNet归一化常数（使用float32以匹配模型）
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for frame in frames:
            # 将BGR转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 步骤1：调整大小为256（匹配训练）
            resized_256 = cv2.resize(rgb_frame, (256, 256))

            # 步骤2：中心裁剪为224（匹配训练）
            h, w = resized_256.shape[:2]
            top = (h - 224) // 2
            left = (w - 224) // 2
            cropped_frame = resized_256[top:top+224, left:left+224]

            # 步骤3：归一化到[0, 1]
            normalized_frame = cropped_frame.astype(np.float32) / 255.0

            # 步骤4：应用ImageNet归一化
            imagenet_normalized = (normalized_frame - IMAGENET_MEAN) / IMAGENET_STD

            # 转换为张量并添加通道维度（HWC -> CHW）
            tensor_frame = torch.from_numpy(imagenet_normalized).permute(2, 0, 1)

            processed_frames.append(tensor_frame)

        # 堆叠帧：(T, C, H, W) -> (1, T, C, H, W)
        stacked_frames = torch.stack(processed_frames, dim=0)
        batch_frames = stacked_frames.unsqueeze(0)

        return batch_frames

    def remove_old_tracks(self, current_time: float, max_age: float = 2.0):
        """
        移除最近未更新的轨迹

        Args:
            current_time: 当前时间戳
            max_age: 移除轨迹前的最大时间（秒）
        """
        tracks_to_remove = []

        for track_id in self.frame_buffers:
            if self.timestamp_buffers[track_id]:
                last_update = self.timestamp_buffers[track_id][-1]
                if current_time - last_update > max_age:
                    tracks_to_remove.append(track_id)

        # 移除旧轨迹
        for track_id in tracks_to_remove:
            del self.frame_buffers[track_id]
            del self.timestamp_buffers[track_id]
            del self.frame_counts[track_id]
            print(f"Removed track {track_id} (inactive for {max_age:.1f}s)")

    def get_track_stats(self) -> Dict[int, Dict]:
        """
        获取所有活动轨迹的统计信息

        Returns:
            包含轨迹统计信息的字典
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
        清除特定轨迹或所有轨迹的帧缓冲区

        Args:
            track_id: 特定轨迹ID，None表示所有轨迹
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
        估算帧缓冲区的当前内存使用量（MB）

        Returns:
            估算的内存使用量（MB）
        """
        total_bytes = 0
        for track_id, buffer in self.frame_buffers.items():
            for frame in buffer:
                # 估算大小：高度 * 宽度 * 通道数 * 3（uint8 -> 可能的扩展）
                if hasattr(frame, 'nbytes'):
                    total_bytes += frame.nbytes
                else:
                    # 回退估算（使用配置的输入尺寸）
                    fallback_size = DataConfig.INPUT_SIZE
                    h, w = frame.shape[:2] if len(frame.shape) >= 2 else (fallback_size, fallback_size)
                    c = frame.shape[2] if len(frame.shape) == 3 else 3
                    total_bytes += h * w * c * 4  # 假设最坏情况为float32

        return total_bytes / (1024 * 1024)  # 转换为MB

    def _check_memory_and_cleanup(self):
        """
        检查内存使用情况并在需要时执行清理
        """
        estimated_memory = self._estimate_memory_usage()

        if estimated_memory > self.max_memory_mb:
            if not self.max_memory_threshold_reached:
                self.max_memory_threshold_reached = True
                self.memory_warnings += 1
                print(f"WARNING: Memory usage ({estimated_memory:.1f} MB) exceeds threshold ({self.max_memory_mb} MB)")
                print("Performing aggressive cleanup...")

                # 强制垃圾回收
                gc.collect()

                # 减小所有轨迹的缓冲区大小
                for track_id in self.frame_buffers:
                    current_size = len(self.frame_buffers[track_id])
                    if current_size > self.total_frames:
                        # 将缓冲区修剪为最小所需大小
                        excess = current_size - self.total_frames
                        for _ in range(excess):
                            self.frame_buffers[track_id].popleft()
                            self.timestamp_buffers[track_id].popleft()

                # 更积极地移除非活动轨迹
                import time
                current_time = time.time()
                self.remove_old_tracks(current_time, max_age=1.0)  # 更积极

                print(f"Cleanup complete. Memory after cleanup: {self._estimate_memory_usage():.1f} MB")

        elif estimated_memory > self.max_memory_mb * 0.8 and not self.max_memory_threshold_reached:
            # 80%时的警告阈值
            if self.memory_warnings == 0:
                print(f"INFO: Memory usage at {estimated_memory:.1f} MB ({estimated_memory/self.max_memory_mb*100:.0f}% of threshold)")
                self.memory_warnings += 1
