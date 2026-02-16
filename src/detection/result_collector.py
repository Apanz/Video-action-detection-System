"""
检测结果收集器
收集和采样检测结果，按动作类别组织帧
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
    收集检测结果并为每个动作采样代表性帧

    功能：
    - 收集帧级别的检测结果
    - 采样帧（每个动作最多N帧）以避免过度磁盘使用
    - 将帧保存为图像供以后查看
    - 提供统计和导出功能
    """

    def __init__(self,
                 save_dir: str = "outputs/results",
                 max_frames_per_action: int = 10,
                 save_frame_images: bool = True,
                 session_id: Optional[str] = None):
        """
        初始化结果收集器

        Args:
            save_dir: 保存结果的目录
            max_frames_per_action: 每个动作保存的最大帧数
            save_frame_images: 是否将帧图像保存到磁盘
            session_id: 唯一会话标识符（如果为None则自动生成）
        """
        self.save_dir = Path(save_dir)
        self.max_frames_per_action = max_frames_per_action
        self.save_frame_images = save_frame_images

        # 生成会话ID
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"sess_{timestamp}"
        self.session_id = session_id

        # 创建目录
        self.frames_dir = self.save_dir / "frames" / self.session_id
        if self.save_frame_images:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        # 数据结构
        self.results: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'confidence_sum': 0.0,
            'frames': []  # (frame_idx, timestamp, confidence, frame_path)列表
        })

        # 会话元数据
        self.start_time = time.time()
        self.total_frames = 0
        self.total_detected_frames = 0
        self.video_source = "unknown"

        # 当前帧索引
        self.frame_idx = 0

    def set_video_source(self, source: str):
        """设置视频源描述"""
        self.video_source = source

    def add_result(self,
                   frame: np.ndarray,
                   action: str,
                   confidence: float,
                   timestamp: float) -> Optional[str]:
        """
        添加检测结果

        Args:
            frame: 帧图像（numpy数组）
            action: 检测到的动作标签
            confidence: 检测置信度
            timestamp: 帧时间戳

        Returns:
            保存的帧路径（如果已保存），否则返回None
        """
        self.total_frames += 1
        self.frame_idx += 1

        # 如果未检测到动作则跳过
        if not action or action in ["Unknown", "Detecting...", "Collecting..."]:
            return None

        self.total_detected_frames += 1

        # 更新统计信息
        result_dict = self.results[action]
        result_dict['count'] += 1
        result_dict['confidence_sum'] += confidence

        # 决定是否保存此帧（采样策略）
        frame_path = None
        if self.save_frame_images and len(result_dict['frames']) < self.max_frames_per_action:
            # 保存帧图像
            frame_filename = f"{action}_{self.frame_idx}_{confidence:.2f}.jpg"
            # 清理动作名称以用于文件名
            safe_action = "".join(c if c.isalnum() else "_" for c in action)
            frame_filename = f"{safe_action}_{self.frame_idx}.jpg"
            frame_path = str(self.frames_dir / frame_filename)

            # 保存帧
            cv2.imwrite(frame_path, frame)

            # 添加到帧列表
            result_dict['frames'].append({
                'frame_idx': self.frame_idx,
                'timestamp': timestamp,
                'confidence': confidence,
                'frame_path': frame_path
            })
        elif self.save_frame_images and len(result_dict['frames']) >= self.max_frames_per_action:
            # 采样策略：如果当前帧更好，则替换最低置信度帧
            min_confidence_idx = min(
                range(len(result_dict['frames'])),
                key=lambda i: result_dict['frames'][i]['confidence']
            )
            if confidence > result_dict['frames'][min_confidence_idx]['confidence']:
                # 移除旧帧
                old_frame_path = result_dict['frames'][min_confidence_idx]['frame_path']
                if os.path.exists(old_frame_path):
                    os.remove(old_frame_path)

                # 保存新帧
                safe_action = "".join(c if c.isalnum() else "_" for c in action)
                frame_filename = f"{safe_action}_{self.frame_idx}.jpg"
                frame_path = str(self.frames_dir / frame_filename)
                cv2.imwrite(frame_path, frame)

                # 在列表中替换
                result_dict['frames'][min_confidence_idx] = {
                    'frame_idx': self.frame_idx,
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'frame_path': frame_path
                }

        return frame_path

    def get_statistics(self) -> Dict:
        """
        获取检测统计信息

        Returns:
            包含统计信息的字典
        """
        elapsed = time.time() - self.start_time

        # 为每个动作构建统计信息
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

        # 按计数排序（降序）
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
        获取特定动作的所有已保存帧

        Args:
            action_name: 动作名称

        Returns:
            帧信息字典列表
        """
        if action_name in self.results:
            return self.results[action_name]['frames']
        return []

    def export_results(self, output_path: str, format: str = 'json') -> bool:
        """
        将结果导出到文件

        Args:
            output_path: 输出文件路径
            format: 导出格式（'json'或'csv'）

        Returns:
            成功返回True，否则返回False
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

                    # 写入标题
                    writer.writerow(['Action', 'Count', 'Percentage', 'Avg Confidence',
                                   'Saved Frames', 'Frame Indices'])

                    # 写入动作统计信息
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

                # 同时保存会话元数据
                metadata_path = output_path.replace('.csv', '_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

    def clear(self):
        """清除所有结果"""
        self.results.clear()
        self.total_frames = 0
        self.total_detected_frames = 0
        self.frame_idx = 0
        self.start_time = time.time()

    def get_summary(self) -> str:
        """
        获取结果的文本摘要

        Returns:
            摘要字符串
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
