"""
TSN (Temporal Segment Networks) Model for Video Action Recognition
Uses pre-trained CNN backbone with temporal aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TSN(nn.Module):
    """
    Temporal Segment Networks (TSN) for action recognition
    Samples frames from temporal segments and aggregates predictions
    """

    def __init__(self, num_classes, backbone='resnet18', pretrained=True,
                 dropout=0.5, num_segments=3, frames_per_segment=5):
        """
        Args:
            num_classes: Number of action classes
            backbone: CNN backbone architecture ('resnet18', 'resnet34', 'resnet50', 'mobilenet_v2')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before classifier
            num_segments: Number of temporal segments (for TSN aggregation)
            frames_per_segment: Frames per segment
        """
        super(TSN, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment

        # Load backbone
        self.backbone_name = backbone
        self.backbone = self._create_backbone(backbone, pretrained)

        # Get feature dimension from backbone
        if 'resnet' in backbone:
            feature_dim = self.backbone.fc.in_features
            # Remove the original classification layer
            self.backbone.fc = nn.Identity()
        elif 'mobilenet' in backbone:
            feature_dim = self.backbone.classifier[1].in_features
            # Remove the original classification layer
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.feature_dim = feature_dim

        # Classifier head
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

        # Reshape to process each frame individually
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(-1, C, H, W)

        # Extract features from backbone
        features = self.backbone(x)  # Shape: (B*T, feature_dim)

        # Reshape back to (B, T, feature_dim)
        features = features.view(B, T, self.feature_dim)

        # CRITICAL: Use actual number of frames (T) instead of fixed num_segments * frames_per_segment
        # This handles cases where T < num_segments * frames_per_segment
        # Dynamic segment calculation based on actual frame count
        segment_features = []
        frames_per_segment = T // self.num_segments

        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * frames_per_segment
            # For the last segment, include all remaining frames
            if seg_idx == self.num_segments - 1:
                end_idx = T
            else:
                end_idx = start_idx + frames_per_segment

            segment_feat = features[:, start_idx:end_idx, :]  # (B, actual_frames_in_segment, feature_dim)

            # Average pooling within segment (only if non-empty)
            if segment_feat.size(1) > 0:
                segment_feat = segment_feat.mean(dim=1)  # (B, feature_dim)
            else:
                # If segment is empty, use zeros
                segment_feat = torch.zeros(B, self.feature_dim, device=x.device)

            segment_features.append(segment_feat)

        # Stack segment features: (B, num_segments, feature_dim)
        segment_features = torch.stack(segment_features, dim=1)

        # Average pooling across segments (TSN consensus)
        consensus_features = segment_features.mean(dim=1)  # (B, feature_dim)

        # Classification
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

    def __init__(self, num_classes, backbone='resnet18', pretrained=True,
                 dropout=0.5, num_segments=3, frames_per_segment=5, consensus='avg'):
        super(TSNWithConsensus, self).__init__()

        self.num_classes = num_classes
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment
        self.consensus = consensus

        # Load backbone
        self.backbone_name = backbone
        self.backbone = self._create_backbone(backbone, pretrained)

        # Get feature dimension
        if 'resnet' in backbone:
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'mobilenet' in backbone:
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.feature_dim = feature_dim

        # Classifier head (applied to each segment separately)
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

        # Process each frame
        x = x.view(-1, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        features = features.view(B, T, self.feature_dim)

        # Get per-segment predictions
        segment_predictions = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * self.frames_per_segment
            end_idx = start_idx + self.frames_per_segment
            segment_feat = features[:, start_idx:end_idx, :]
            segment_feat = segment_feat.mean(dim=1)  # Average within segment
            seg_pred = self.segment_classifier(segment_feat)
            segment_predictions.append(seg_pred)

        # Stack and aggregate
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


def create_model(dataset='ucf101', backbone='resnet18', pretrained=True, dropout=0.5,
                num_segments=3, frames_per_segment=5):
    """
    Create a TSN model for specific dataset

    Args:
        dataset: 'ucf101' or 'hmdb51'
        backbone: CNN backbone
        pretrained: Use pretrained weights
        dropout: Dropout rate
        num_segments: Number of temporal segments
        frames_per_segment: Frames per segment

    Returns:
        TSN model
    """
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
