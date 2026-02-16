"""
模型元数据提取器
从检查点文件提取和验证模型信息
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


class ModelMetadata:
    """
    从模型检查点文件提取元数据

    功能：
    - 提取模型架构信息
    - 验证模型文件
    - 生成人类可读的模型描述
    """

    # 已知模型签名
    UCF101_CLASSES = 101
    HMDB51_CLASSES = 51

    @staticmethod
    def extract_metadata(checkpoint_path: str) -> Dict:
        """
        从检查点文件提取模型元数据

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            包含模型元数据的字典
        """
        if not os.path.exists(checkpoint_path):
            return {
                'path': checkpoint_path,
                'is_valid': False,
                'error': 'File does not exist'
            }

        try:
            # 获取文件信息
            file_path = Path(checkpoint_path)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取元数据
            metadata = {
                'path': str(file_path.absolute()),
                'filename': file_path.name,
                'file_size_mb': round(file_size_mb, 2),
                'is_valid': True,
            }

            # 尝试提取模型状态字典
            if isinstance(checkpoint, dict):
                # 检查不同的检查点格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    ModelMetadata._extract_from_state_dict(state_dict, metadata, checkpoint)
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    ModelMetadata._extract_from_state_dict(state_dict, metadata, checkpoint)
                else:
                    # 假设检查点本身就是状态字典
                    ModelMetadata._extract_from_state_dict(checkpoint, metadata, {})
            else:
                # 检查点是模型对象
                metadata['error'] = 'Unknown checkpoint format'
                metadata['is_valid'] = False

            # 从num_classes推断数据集
            if 'num_classes' in metadata:
                # 确保num_classes是整数以便比较
                try:
                    num_classes = int(metadata['num_classes'])

                    if num_classes == ModelMetadata.UCF101_CLASSES:
                        metadata['dataset'] = 'ucf101'
                        print(f"[ModelMetadata] Inferred dataset: ucf101 (num_classes={num_classes})")
                    elif num_classes == ModelMetadata.HMDB51_CLASSES:
                        metadata['dataset'] = 'hmdb51'
                        print(f"[ModelMetadata] Inferred dataset: hmdb51 (num_classes={num_classes})")
                    else:
                        metadata['dataset'] = f'custom({num_classes}_classes)'
                        print(f"[ModelMetadata] Inferred dataset: custom (num_classes={num_classes})")
                except (ValueError, TypeError):
                    print(f"[ModelMetadata] Warning: Could not convert num_classes to int: {metadata['num_classes']}")
                    metadata['dataset'] = 'unknown'

            # 为缺失字段设置默认值
            metadata.setdefault('backbone', 'unknown')
            metadata.setdefault('num_classes', 'unknown')
            metadata.setdefault('num_segments', 'unknown')
            metadata.setdefault('frames_per_segment', 'unknown')
            metadata.setdefault('dataset', 'unknown')

            # 调试：打印提取的元数据以进行故障排除
            print(f"[ModelMetadata] Extracted from {metadata['filename']}:")
            print(f"  Num classes: {metadata.get('num_classes')}")
            print(f"  Dataset: {metadata.get('dataset')}")
            print(f"  Backbone: {metadata.get('backbone')}")
            print(f"  Num segments: {metadata.get('num_segments')}")
            print(f"  Frames per segment: {metadata.get('frames_per_segment')}")

            return metadata

        except Exception as e:
            return {
                'path': checkpoint_path,
                'filename': Path(checkpoint_path).name,
                'is_valid': False,
                'error': str(e)
            }

    @staticmethod
    def _extract_from_state_dict(state_dict: dict, metadata: dict, checkpoint: dict):
        """
        从模型状态字典提取信息

        Args:
            state_dict: 模型状态字典
            metadata: 要更新的元数据字典
            checkpoint: 完整检查点（可能包含训练信息）
        """

        # 首先检查配置部分（较新的检查点）
        if 'config' in checkpoint:
            config = checkpoint['config']
            metadata['backbone'] = config.get('backbone', 'unknown')
            metadata['num_classes'] = config.get('num_classes', 'unknown')
            metadata['num_segments'] = config.get('num_segments', 'unknown')
            metadata['frames_per_segment'] = config.get('frames_per_segment', 'unknown')
            metadata['dataset'] = config.get('dataset', 'unknown')
            # 仍在下面提取训练信息
        else:
            # 从层名称检测骨干网络（传统检查点）
            # 使用与action_classifier.py相同的检测逻辑
            state_keys = list(state_dict.keys())

            # 尝试检测骨干网络架构
            if any('backbone.layer' in k for k in state_keys):
                # 计算每个层中的块数以确定ResNet变体
                blocks = {}
                for key in state_dict.keys():
                    if 'backbone.layer' in key and '.conv1.weight' in key:
                        layer_name = key.split('.')[1]
                        block_idx = int(key.split('.')[2])
                        blocks[layer_name] = max(blocks.get(layer_name, 0), block_idx + 1)

                # 根据块数确定骨干网络
                total_blocks = sum(blocks.values())

                if total_blocks == 8:
                    metadata['backbone'] = 'resnet18'
                elif total_blocks == 16:
                    # ResNet-34和ResNet-50都有16个块
                    # 检查内核大小以确定块类型：
                    # BasicBlock（ResNet-34）：conv1使用3x3内核
                    # Bottleneck（ResNet-50）：conv1使用1x1内核
                    conv1_key = None
                    for key in state_dict.keys():
                        if 'backbone.layer1.0.conv1.weight' in key:
                            conv1_key = key
                            break

                    if conv1_key is not None:
                        kernel_size = state_dict[conv1_key].shape[-1]
                        if kernel_size == 1:
                            metadata['backbone'] = 'resnet50'
                        elif kernel_size == 3:
                            metadata['backbone'] = 'resnet34'
                        else:
                            metadata['backbone'] = 'unknown'
                    else:
                        metadata['backbone'] = 'unknown'
                elif total_blocks == 36:
                    metadata['backbone'] = 'resnet101'
                else:
                    metadata['backbone'] = f'unknown ({total_blocks} blocks)'
            elif any('inception' in k.lower() for k in state_keys):
                metadata['backbone'] = 'inception_v3'
            elif any('vgg' in k.lower() for k in state_keys):
                metadata['backbone'] = 'vgg16'
            elif any('mobilenet' in k.lower() for k in state_keys):
                metadata['backbone'] = 'mobilenet_v2'
            else:
                metadata['backbone'] = 'unknown'

            # 尝试从最终层提取num_classes
            # TSN模型可能使用不同的分类器键模式：
            # - 'fc.weight'（原始TSN）
            # - 'classifier.1.weight'（较新的TSN，带MLP头）
            # - 'classifier.weight'（通用）
            num_classes_key = None
            for key in state_keys:
                # 检查各种分类器模式
                if 'fc.weight' in key:
                    num_classes_key = key
                    break
                elif 'classifier' in key and 'weight' in key:
                    # 使用分类器中的最后一层（通常是最终线性层）
                    # 例如：'classifier.1.weight' 或 'classifier.3.weight'
                    if not num_classes_key or key > num_classes_key:
                        num_classes_key = key

            if num_classes_key:
                metadata['num_classes'] = int(state_dict[num_classes_key].shape[0])
                print(f"[ModelMetadata] Found num_classes={metadata['num_classes']} from key '{num_classes_key}'")

            # 尝试从new_weights形状提取num_segments
            # （在TSN中，new_weights的形状为[num_segments, C, 1, 1]）
            for key in state_keys:
                if 'new_weights' in key:
                    shape = state_dict[key].shape
                    if len(shape) >= 1:
                        metadata['num_segments'] = int(shape[0])
                    break

            # 如果num_segments仍然未知，尝试从模型结构推断
            if metadata.get('num_segments', 'unknown') == 'unknown':
                # 一些检查点以不同格式存储此信息
                # 检查我们是否可以从cons_weight或其他TSN特定层推断
                for key in state_keys:
                    if 'cons_weight' in key or 'new_weights' in key:
                        shape = state_dict[key].shape
                        if len(shape) >= 1 and shape[0] < 10:  # 合理的片段数
                            metadata['num_segments'] = int(shape[0])
                            break

            # 如果仍然未知，使用UCF101模型的通用默认值
            if metadata.get('num_segments', 'unknown') == 'unknown':
                # 大多数UCF101 TSN模型默认使用3个片段
                # 这与action_classifier.py中的默认值匹配
                metadata['num_segments'] = 3

        # 提取训练信息（如果可用）
        if 'epoch' in checkpoint:
            metadata['trained_epochs'] = checkpoint['epoch']

        if 'best_acc' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc'])
        elif 'best_acc1' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc1'])

        # 如果尚未设置frames_per_segment，则设置默认值
        if 'frames_per_segment' not in metadata or metadata['frames_per_segment'] == 'unknown':
            # 常见的TSN配置使用每片段1或5帧
            # 由于我们无法从检查点结构检测到这一点，
            # 基于num_segments使用合理的默认值
            if metadata.get('num_segments', 'unknown') != 'unknown':
                # 常见做法：较少片段 → 每段更多帧
                num_seg = metadata['num_segments']
                if num_seg <= 3:
                    metadata['frames_per_segment'] = 5  # 3x5 = 15帧
                elif num_seg <= 5:
                    metadata['frames_per_segment'] = 5  # 5x5 = 25帧
                else:
                    metadata['frames_per_segment'] = 1  # 更多片段，每段1帧
            else:
                # 如果无法确定num_segments，使用最常见的配置
                metadata['frames_per_segment'] = 5  # 大多数TSN模型使用5

    @staticmethod
    def validate_model(checkpoint_path: str) -> Tuple[bool, str]:
        """
        验证模型检查点文件

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            (is_valid, error_message)元组
        """
        if not os.path.exists(checkpoint_path):
            return False, "File does not exist"

        try:
            # 尝试加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 检查是否为有效格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                    return True, ""
                # 检查是否具有模型类似的键
                if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                    return True, ""
                return False, "Invalid checkpoint format (no model state dict found)"
            else:
                return False, "Unknown checkpoint format"

        except Exception as e:
            return False, f"Error loading checkpoint: {str(e)}"

    @staticmethod
    def get_model_description(metadata: dict) -> str:
        """
        生成人类可读的模型描述

        Args:
            metadata: 模型元数据字典

        Returns:
            描述字符串
        """
        if not metadata.get('is_valid', False):
            return f"Invalid model: {metadata.get('error', 'Unknown error')}"

        parts = [
            f"{metadata.get('backbone', 'Unknown').upper()}",
            f"{metadata.get('num_classes', '?')} classes",
        ]

        # 如果有时序信息则添加
        num_segments = metadata.get('num_segments')
        frames_per_segment = metadata.get('frames_per_segment')

        if num_segments != 'unknown' and frames_per_segment != 'unknown':
            total_frames = num_segments * frames_per_segment
            parts.append(f"{num_segments}×{frames_per_segment}={total_frames} frames")
        elif num_segments != 'unknown':
            parts.append(f"{num_segments} segments")

        # 如果有训练信息则添加
        if 'best_accuracy' in metadata:
            parts.append(f"acc: {metadata['best_accuracy']:.2f}%")

        return " | ".join(map(str, parts))

    @staticmethod
    def scan_models_directory(models_dir: str) -> Dict[str, list]:
        """
        扫描目录中的模型文件

        Args:
            models_dir: 要扫描的目录

        Returns:
            以类别为键、模型元数据列表为值的字典
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            return {}

        categories = {
            'ucf101': [],
            'custom': []
        }

        # 扫描子目录
        for category_dir in models_path.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name.lower()

            # 确定类别
            if category_name == 'ucf101':
                category = 'ucf101'
            else:
                category = 'custom'

            # 扫描.pth和.pt文件
            for model_file in category_dir.glob('*.pth'):
                metadata = ModelMetadata.extract_metadata(str(model_file))
                metadata['category'] = category
                categories[category].append(metadata)

            for model_file in category_dir.glob('*.pt'):
                metadata = ModelMetadata.extract_metadata(str(model_file))
                metadata['category'] = category
                categories[category].append(metadata)

        # 对每个类别按文件名排序
        for category in categories:
            categories[category].sort(key=lambda x: x['filename'])

        return categories
