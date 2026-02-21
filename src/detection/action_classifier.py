"""
动作分类器模块
集成训练好的TSN模型进行动作分类
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple

from core.config import DetectionConfig, ModelConfig


class ActionClassifier:
    """
    使用训练好的TSN模型的动作分类器
    """

    def __init__(self, checkpoint_path: str, device: str = 'auto',
                 num_segments=None, frames_per_segment=None,
                 backbone=None):
        """
        初始化动作分类器

        Args:
            checkpoint_path: 训练好的模型检查点路径
            device: 'cpu'、'cuda' 或 'auto'
            num_segments: 时序片段数量 (default: from DetectionConfig)
            frames_per_segment: 每片段帧数 (default: from DetectionConfig)
            backbone: 骨干网络架构 (default: from ModelConfig)
        """
        # Use config defaults if not specified
        if num_segments is None:
            num_segments = DetectionConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DetectionConfig.FRAMES_PER_SEGMENT
        if backbone is None:
            backbone = ModelConfig.BACKBONE
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading action classifier on {self.device}...")

        # 首先从检查点提取元数据以获取正确的时序参数
        from .model_metadata import ModelMetadata
        metadata = ModelMetadata.extract_metadata(checkpoint_path)

        # 使用元数据中的时序参数（如果可用）
        if metadata.get('num_segments', 'unknown') != 'unknown':
            num_segments = int(metadata['num_segments'])
            print(f"[ActionClassifier] Using num_segments={num_segments} from checkpoint metadata")
        if metadata.get('frames_per_segment', 'unknown') != 'unknown':
            frames_per_segment = int(metadata['frames_per_segment'])
            print(f"[ActionClassifier] Using frames_per_segment={frames_per_segment} from checkpoint metadata")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 获取模型配置
        if 'model_state_dict' in checkpoint:
            # 从检查点提取信息
            # 对于UCF101
            num_classes = 101
            dataset_name = 'ucf101'

            # 尝试从检查点元数据获取
            if 'config' in checkpoint:
                config = checkpoint['config']
                num_classes = config.get('num_classes', num_classes)
                backbone = config.get('backbone', backbone)
                num_segments = config.get('num_segments', num_segments)
                frames_per_segment = config.get('frames_per_segment', frames_per_segment)
            else:
                # 关键：从检查点结构推断骨干网络
                # 计算每层的块数
                state_dict = checkpoint['model_state_dict']
                blocks = {}
                for key in state_dict.keys():
                    if 'backbone.layer' in key and '.conv1.weight' in key:
                        layer_name = key.split('.')[1]
                        block_idx = int(key.split('.')[2])
                        blocks[layer_name] = max(blocks.get(layer_name, 0), block_idx + 1)

                # 根据块数确定骨干网络
                total_blocks = sum(blocks.values())
                print(f"[DEBUG classifier] Inferred blocks per layer: {dict(sorted(blocks.items()))}")
                print(f"[DEBUG classifier] Total blocks: {total_blocks}")

                if total_blocks == 8:
                    backbone = 'resnet18'
                    print("[DEBUG classifier] Detected backbone: ResNet-18")
                elif total_blocks == 16:
                    # ResNet-34和ResNet-50都有16个块
                    # 检查内核大小以确定块类型：
                    # BasicBlock（ResNet-34）：conv1使用3x3内核
                    # Bottleneck（ResNet-50）：conv1使用1x1内核
                    # 在layer1中查找conv1.weight并检查其形状
                    conv1_key = None
                    for key in state_dict.keys():
                        if 'backbone.layer1.0.conv1.weight' in key:
                            conv1_key = key
                            break

                    if conv1_key is not None:
                        kernel_size = state_dict[conv1_key].shape[-1]
                        print(f"[DEBUG classifier] Layer1 conv1 kernel size: {kernel_size}x{kernel_size}")

                        if kernel_size == 1:
                            backbone = 'resnet50'
                            print("[DEBUG classifier] Detected backbone: ResNet-50 (Bottleneck)")
                        elif kernel_size == 3:
                            backbone = 'resnet34'
                            print("[DEBUG classifier] Detected backbone: ResNet-34 (BasicBlock)")
                        else:
                            print(f"[WARNING classifier] Unknown kernel size {kernel_size}, using default: {backbone}")
                    else:
                        print(f"[WARNING classifier] Cannot determine block type, using default: {backbone}")
                elif total_blocks == 36:
                    backbone = 'resnet101'
                    print("[DEBUG classifier] Detected backbone: ResNet-101")
                else:
                    print(f"[WARNING classifier] Unknown architecture with {total_blocks} blocks, using default: {backbone}")

        else:
            # 如果检查点没有元数据，使用默认值
            num_classes = 101
            dataset_name = 'ucf101'

        # 创建模型
        from core import create_model
        self.model = create_model(
            dataset=dataset_name,
            backbone=backbone,
            pretrained=False,  # 我们正在加载训练好的权重
            num_segments=num_segments,
            frames_per_segment=frames_per_segment
        )

        # 调试：打印检查点键
        print(f"[DEBUG classifier] Checkpoint keys: {checkpoint.keys()}")
        if 'model_state_dict' in checkpoint:
            print(f"[DEBUG classifier] Checkpoint has model_state_dict with {len(checkpoint['model_state_dict'])} keys")
            # 打印前10个键
            for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")
        elif 'state_dict' in checkpoint:
            print(f"[DEBUG classifier] Checkpoint has state_dict with {len(checkpoint['state_dict'])} keys")
            # 打印前10个键
            for i, key in enumerate(list(checkpoint['state_dict'].keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")
        else:
            print(f"[DEBUG classifier] Checkpoint has {len(checkpoint)} keys (direct)")
            # 打印前10个键
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")

        # 调试：打印模型键
        model_state_dict = self.model.state_dict()
        print(f"[DEBUG classifier] Model has {len(model_state_dict)} keys")
        # 打印前10个键
        for i, key in enumerate(list(model_state_dict.keys())[:10]):
            print(f"  {i}: {key}")
        print("  ...")

        # 使用strict=False加载权重以查看缺失的内容
        if 'model_state_dict' in checkpoint:
            load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            load_result = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # 直接加载
            load_result = self.model.load_state_dict(checkpoint, strict=False)

        # 调试：打印加载结果
        print(f"[DEBUG classifier] Load result: {load_result}")
        print(f"[DEBUG classifier] Missing keys: {len(load_result.missing_keys)}")
        if load_result.missing_keys:
            print(f"[DEBUG classifier] Missing keys (first 10): {load_result.missing_keys[:10]}")
        print(f"[DEBUG classifier] Unexpected keys: {len(load_result.unexpected_keys)}")
        if load_result.unexpected_keys:
            print(f"[DEBUG classifier] Unexpected keys (first 10): {load_result.unexpected_keys[:10]}")

        # 关键：检查是否有任何权重包含NaN
        has_nan = False
        has_inf = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"[ERROR classifier] Parameter '{name}' contains NaN!")
                has_nan = True
            if torch.isinf(param.data).any():
                print(f"[ERROR classifier] Parameter '{name}' contains Inf!")
                has_inf = True

        if has_nan:
            print("[ERROR classifier] Model contains NaN weights! Checkpoint may be corrupted.")
        if has_inf:
            print("[ERROR classifier] Model contains Inf weights! Checkpoint may be corrupted.")

        # 关键：先将模型移动到正确的设备
        self.model.to(self.device)

        # 使用虚拟输入测试前向传播
        print("[DEBUG classifier] Testing forward pass with dummy input...")
        total_frames = num_segments * frames_per_segment
        dummy_input = torch.randn(1, total_frames, 3, 224, 224)
        dummy_input = dummy_input.to(self.device)
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
            print(f"[DEBUG classifier] Dummy output shape: {dummy_output.shape}")
            print(f"[DEBUG classifier] Dummy output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
            print(f"[DEBUG classifier] Dummy output contains NaN: {torch.isnan(dummy_output).any()}")
            print(f"[DEBUG classifier] Dummy output contains Inf: {torch.isinf(dummy_output).any()}")

        # 验证权重已加载 - 检查第一层
        first_conv_weight = None
        first_bn_weight = None
        classifier_weight = None

        for name, param in self.model.named_parameters():
            if 'backbone.conv1' in name and first_conv_weight is None:
                first_conv_weight = param.data
            elif 'backbone.bn1' in name and first_bn_weight is None:
                first_bn_weight = param.data
            elif 'classifier.1' in name:
                classifier_weight = param.data

        if first_conv_weight is not None:
            conv_msg = f"[VERIFY] First conv weight mean: {first_conv_weight.mean():.6f}"
            print(conv_msg)
        else:
            print("[VERIFY] First conv weight: None")

        if first_bn_weight is not None:
            bn_msg = f"[VERIFY] First bn weight mean: {first_bn_weight.mean():.6f}"
            print(bn_msg)
        else:
            print("[VERIFY] First bn weight: None")

        if classifier_weight is not None:
            cls_msg = f"[VERIFY] Classifier weight mean: {classifier_weight.mean():.6f}"
            print(cls_msg)
        else:
            print("[VERIFY] Classifier weight: None")

        # 检查权重是否看起来随机（未训练）
        # 注意：具有BatchNorm的训练模型通常具有均值≈0的权重，这是正常的
        # 此检查过于严格，导致了误报。删除以避免混淆。
        # 真正的验证是：(1) load_result显示所有键匹配，(2) 前向传播产生有效输出
        # 检查标准差以确保权重不是全零初始化
        if first_conv_weight is not None and first_conv_weight.std() < 0.01:
            # 标准差极小表示可能未训练（零初始化或接近零）
            print("[WARNING] First layer weights have very small std. Verify checkpoint is trained.")

        # 设置为评估模式
        self.model.eval()

        # UCF101类别名称
        self.ucf101_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BoxingPunchingBag',
            'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleaningAndFlossingTeeth',
            'CliffDiving', 'Diving', 'Drumming', 'Fencing', 'FrontCrawl',
            'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HeadMassage',
            'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
            'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
            'Knitting', 'Lunges', 'MagicTrick', 'Mixing', 'MoppingFloor',
            'Nunchucks', 'ParallelBars', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
            'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla',
            'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
            'PushUps', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
            'SkateBoarding', 'Skiing', 'SkiJet', 'Skydiving', 'SoccerJuggling',
            'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
            'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping',
            'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
        ]

        # Prediction smoothing
        self.prediction_history = []
        self.smoothing_window = DetectionConfig.SMOOTHING_WINDOW

        # Model parameters
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment

        print(f"Action classifier loaded successfully!")
        print(f"Model: {backbone}")
        print(f"Segments: {num_segments}, Frames per segment: {frames_per_segment}")
        print(f"Classes: {num_classes}")

    def classify(self, frames: torch.Tensor) -> Tuple[str, float]:
        """
        从时序帧中分类动作

        Args:
            frames: 输入张量 (1, T, C, H, W)

        Returns:
            (action_label, confidence) 元组
        """
        # DEBUG: 检查模型参数是否包含NaN
        if not hasattr(self, '_checked_params'):
            print("[DEBUG classifier] Checking model parameters for NaN...")
            for name, param in self.model.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"[ERROR classifier] Parameter '{name}' contains NaN!")
                if torch.isinf(param.data).any():
                    print(f"[ERROR classifier] Parameter '{name}' contains Inf!")
            self._checked_params = True

        with torch.no_grad():
            # Move to device
            frames = frames.to(self.device)

            # DEBUG: 记录输入信息
            print(f"[DEBUG classifier] Input tensor shape: {frames.shape}")
            print(f"[DEBUG classifier] Input tensor range: [{frames.min():.4f}, {frames.max():.4f}]")

            # Forward pass
            outputs = self.model(frames)

            # DEBUG: 记录模型输出
            print(f"[DEBUG classifier] Model output shape: {outputs.shape}")
            print(f"[DEBUG classifier] Model output range: min={outputs.min():.4f}, max={outputs.max():.4f}")

            # VALIDATION: Check for NaN or Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("[WARNING classifier] Model outputs contain NaN or Inf values!")
                print(f"[WARNING classifier] NaN count: {torch.isnan(outputs).sum()}")
                print(f"[WARNING classifier] Inf count: {torch.isinf(outputs).sum()}")
                # 返回安全的回退值
                return "Unknown", 0.0

            # Get probabilities
            try:
                probs = torch.softmax(outputs, dim=1)
                probs = probs.cpu().numpy()[0]  # Remove batch dim

                # DEBUG: 记录概率
                print(f"[DEBUG classifier] Probabilities shape: {probs.shape}")
                print(f"[DEBUG classifier] Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
                print(f"[DEBUG classifier] Probabilities sum: {probs.sum():.4f}")
            except Exception as e:
                print(f"[ERROR classifier] Softmax failed: {e}")
                return "Unknown", 0.0

            # 获取最佳预测
            top_idx = np.argmax(probs)
            confidence = float(probs[top_idx])
            action_label = self.ucf101_classes[top_idx]

            # DEBUG: 记录预测
            print(f"[DEBUG classifier] Predicted action: {action_label} (index: {top_idx})")
            print(f"[DEBUG classifier] Confidence: {confidence}")

            # 应用平滑处理
            self.prediction_history.append((action_label, confidence))
            if len(self.prediction_history) > self.smoothing_window:
                self.prediction_history.pop(0)

            # 平滑预测
            if len(self.prediction_history) > 1:
                # 使用最常见的动作和平均置信度
                action_counts = {}
                total_confidence = {}

                for action, conf in self.prediction_history:
                    action_counts[action] = action_counts.get(action, 0) + 1
                    total_confidence[action] = total_confidence.get(action, 0) + conf

                # 获取计数最高的动作
                best_action = max(action_counts, key=action_counts.get)
                avg_confidence = total_confidence[best_action] / action_counts[best_action]

                return best_action, avg_confidence

            return action_label, confidence

    def clear_prediction_history(self):
        """Clear prediction history for fresh start"""
        self.prediction_history.clear()

    def predict_proba(self, frames: torch.Tensor) -> np.ndarray:
        """
        Return probability distribution for all classes

        Args:
            frames: Input tensor (1, T, C, H, W)

        Returns:
            Probability array of shape (num_classes,)
        """
        with torch.no_grad():
            # Move to device
            frames = frames.to(self.device)

            # Forward pass
            outputs = self.model(frames)

            # VALIDATION: Check for NaN or Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("[WARNING classifier] Model outputs contain NaN or Inf values!")
                # 返回均匀分布
                num_classes = len(self.ucf101_classes)
                return np.ones(num_classes) / num_classes

            # Get probabilities
            try:
                probs = torch.softmax(outputs, dim=1)
                probs = probs.cpu().numpy()[0]  # Remove batch dim
                return probs
            except Exception as e:
                print(f"[ERROR classifier] Softmax failed: {e}")
                num_classes = len(self.ucf101_classes)
                return np.ones(num_classes) / num_classes


class SimpleClassifier:
    """
    Fallback classifier for when checkpoint is not available
    """

    def __init__(self):
        """Initialize simple classifier"""
        self.action_names = ["Unknown"]
        self.is_dummy = True
        # Add temporal attributes for compatibility
        self.num_segments = DetectionConfig.NUM_SEGMENTS
        self.frames_per_segment = DetectionConfig.FRAMES_PER_SEGMENT
        self.total_frames = self.num_segments * self.frames_per_segment
        # UCF101类别名称以保持兼容性
        self.ucf101_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BoxingPunchingBag',
            'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleaningAndFlossingTeeth',
            'CliffDiving', 'Diving', 'Drumming', 'Fencing', 'FrontCrawl',
            'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HeadMassage',
            'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
            'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
            'Knitting', 'Lunges', 'MagicTrick', 'Mixing', 'MoppingFloor',
            'Nunchucks', 'ParallelBars', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
            'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla',
            'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
            'PushUps', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
            'SkateBoarding', 'Skiing', 'SkiJet', 'Skydiving', 'SoccerJuggling',
            'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
            'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping',
            'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
        ]

    def classify(self, frames: torch.Tensor) -> Tuple[str, float]:
        """虚拟分类 - 返回未知"""
        return "Unknown", 0.0

    def predict_proba(self, frames: torch.Tensor) -> np.ndarray:
        """
        Return probability distribution for all classes

        Args:
            frames: Input tensor (1, T, C, H, W)

        Returns:
            Probability array of shape (num_classes,)
        """
        # 返回所有类别上的均匀分布
        num_classes = len(self.ucf101_classes)
        probs = np.ones(num_classes) / num_classes
        return probs


def load_classifier(checkpoint_path: str, device: str = 'auto',
                   num_segments=None, frames_per_segment=None,
                   backbone=None) -> ActionClassifier:
    """
    从检查点加载动作分类器

    Args:
        checkpoint_path: 检查点路径
        device: 要使用的设备
        num_segments: 时序片段数量 (default: from DetectionConfig)
        frames_per_segment: 每片段帧数 (default: from DetectionConfig)
        backbone: 骨干网络架构 (default: from ModelConfig)

    Returns:
        ActionClassifier 实例
    """
    # Use config defaults if not specified
    if num_segments is None:
        num_segments = DetectionConfig.NUM_SEGMENTS
    if frames_per_segment is None:
        frames_per_segment = DetectionConfig.FRAMES_PER_SEGMENT
    if backbone is None:
        backbone = ModelConfig.BACKBONE
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return SimpleClassifier()

    try:
        return ActionClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            backbone=backbone
        )
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return SimpleClassifier()
