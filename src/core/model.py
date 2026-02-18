"""
视频动作识别的TSN（时段网络）模型
使用预训练的CNN骨干和时间聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from core.config import DataConfig, ModelConfig


class TSN(nn.Module):
    """
    用于动作识别的时段网络（TSN），从时间段中抽样帧并汇总预测
    """

    def __init__(self, num_classes, backbone=None, pretrained=None,
                 dropout=None, num_segments=None, frames_per_segment=None):
        """
        Args:
            num_classes: Number of action classes
            backbone: CNN backbone architecture (default: from ModelConfig)
            pretrained: Use ImageNet pretrained weights (default: from ModelConfig)
            dropout: Dropout rate before classifier (default: from ModelConfig)
            num_segments: Number of temporal segments (default: from DataConfig)
            frames_per_segment: Frames per segment (default: from DataConfig)
        """
        # Use config defaults if not specified
        if backbone is None:
            backbone = ModelConfig.BACKBONE
        if pretrained is None:
            pretrained = ModelConfig.PRETRAINED
        if dropout is None:
            dropout = ModelConfig.DROPOUT
        if num_segments is None:
            num_segments = DataConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

        super(TSN, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment

        # 加载骨干网络
        self.backbone_name = backbone
        self.backbone = self._create_backbone(backbone, pretrained)

        # 从骨干网络获取特征维度
        if 'resnet' in backbone:
            feature_dim = self.backbone.fc.in_features
            # 移除原始分类层
            self.backbone.fc = nn.Identity()
        elif 'mobilenet' in backbone:
            feature_dim = self.backbone.classifier[1].in_features
            # 移除原始分类层
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_dim = feature_dim

        # 分类器头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def _create_backbone(self, backbone, pretrained):
        """Create CNN backbone from torchvision"""
        if backbone == 'resnet18':
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet34':
            return models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'mobilenet_v2':
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input video tensor of shape (B, T, C, H, W)
               where B = batch size, T = number of frames

        Returns:
            predictions: Logits of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # 重塑以单独处理每一帧
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(-1, C, H, W)

        # 从骨干网络提取特征
        features = self.backbone(x)  # 形状: (B*T, feature_dim)

        # 重塑回 (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)

        # 关键：使用实际帧数(T)而不是固定的num_segments * frames_per_segment
        # 这处理了 T < num_segments * frames_per_segment 的情况
        # 基于实际帧数动态计算片段
        segment_features = []
        frames_per_segment = T // self.num_segments

        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * frames_per_segment
            # 对于最后一个片段，包含所有剩余帧
            if seg_idx == self.num_segments - 1:
                end_idx = T
            else:
                end_idx = start_idx + frames_per_segment

            segment_feat = features[:, start_idx:end_idx, :]  # (B, 片段中的实际帧数, feature_dim)

            # 片段内平均池化（仅当非空时）
            if segment_feat.size(1) > 0:
                segment_feat = segment_feat.mean(dim=1)  # (B, feature_dim)
            else:
                # 如果片段为空，使用零向量
                segment_feat = torch.zeros(B, self.feature_dim, device=x.device)

            segment_features.append(segment_feat)

        # 堆叠片段特征: (B, num_segments, feature_dim)
        segment_features = torch.stack(segment_features, dim=1)

        # 跨片段平均池化（TSN共识）
        consensus_features = segment_features.mean(dim=1)  # (B, feature_dim)

        # 分类
        predictions = self.classifier(consensus_features)  # (B, num_classes)

        return predictions

    def get_features(self, x):
        """Extract features without classification"""
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, self.feature_dim)
        return features


class TSNWithConsensus(nn.Module):
    """
    TSN with different consensus strategies
    Options: 'avg', 'max', 'sum'
    """

    def __init__(self, num_classes, backbone=None, pretrained=None,
                 dropout=None, num_segments=None, frames_per_segment=None, consensus='avg'):
        # Use config defaults if not specified
        if backbone is None:
            backbone = ModelConfig.BACKBONE
        if pretrained is None:
            pretrained = ModelConfig.PRETRAINED
        if dropout is None:
            dropout = ModelConfig.DROPOUT
        if num_segments is None:
            num_segments = DataConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

        super(TSNWithConsensus, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment
        self.consensus = consensus

        # 加载骨干网络
        self.backbone_name = backbone
        self.backbone = self._create_backbone(backbone, pretrained)

        # 获取特征维度
        if 'resnet' in backbone:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'mobilenet' in backbone:
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.feature_dim = feature_dim

        # 分类器头（分别应用于每个片段）
        self.segment_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def _create_backbone(self, backbone, pretrained):
        """Create CNN backbone from torchvision"""
        if backbone == 'resnet18':
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet34':
            return models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'mobilenet_v2':
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        """
        Forward pass with segment-wise consensus
        """
        B, T, C, H, W = x.shape

        # 处理每一帧
        x = x.view(-1, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        features = features.view(B, T, self.feature_dim)

        # 获取每个片段的预测
        segment_predictions = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * self.frames_per_segment
            end_idx = start_idx + self.frames_per_segment
            segment_feat = features[:, start_idx:end_idx, :]
            segment_feat = segment_feat.mean(dim=1)  # 片段内平均
            seg_pred = self.segment_classifier(segment_feat)
            segment_predictions.append(seg_pred)

        # 堆叠并聚合
        segment_predictions = torch.stack(segment_predictions, dim=1)  # (B, num_segments, num_classes)

        if self.consensus == 'avg':
            predictions = segment_predictions.mean(dim=1)
        elif self.consensus == 'max':
            predictions = segment_predictions.max(dim=1)[0]
        elif self.consensus == 'sum':
            predictions = segment_predictions.sum(dim=1)
        else:
            raise ValueError(f"Unknown consensus: {self.consensus}")

        return predictions


def create_model(dataset='ucf101', backbone=None, pretrained=None, dropout=None,
                num_segments=None, frames_per_segment=None):
    """
    Create a TSN model for specific dataset

    Args:
        dataset: 'ucf101' or 'hmdb51'
        backbone: CNN backbone (default: from ModelConfig)
        pretrained: Use pretrained weights (default: from ModelConfig)
        dropout: Dropout rate (default: from ModelConfig)
        num_segments: Number of temporal segments (default: from DataConfig)
        frames_per_segment: Frames per segment (default: from DataConfig)

    Returns:
        TSN model
    """
    # Use config defaults if not specified
    if backbone is None:
        backbone = ModelConfig.BACKBONE
    if pretrained is None:
        pretrained = ModelConfig.PRETRAINED
    if dropout is None:
        dropout = ModelConfig.DROPOUT
    if num_segments is None:
        num_segments = DataConfig.NUM_SEGMENTS
    if frames_per_segment is None:
        frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

    if dataset.lower() == 'ucf101':
        num_classes = 101
    elif dataset.lower() == 'hmdb51':
        num_classes = 51
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model = TSN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        num_segments=num_segments,
        frames_per_segment=frames_per_segment
    )

    return model
