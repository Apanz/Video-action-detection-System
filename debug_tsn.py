#!/usr/bin/env python3
"""
Debug script to trace TSN forward pass
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from src.core import create_model

# Load checkpoint
checkpoint_path = 'outputs/checkpoints/ucf101_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Infer backbone
state_dict = checkpoint['model_state_dict']
blocks = {}
for key in state_dict.keys():
    if 'backbone.layer' in key and '.conv1.weight' in key:
        layer_name = key.split('.')[1]
        block_idx = int(key.split('.')[2])
        blocks[layer_name] = max(blocks.get(layer_name, 0), block_idx + 1)

total_blocks = sum(blocks.values())
if total_blocks == 16:
    kernel_size = state_dict['backbone.layer1.0.conv1.weight'].shape[-1]
    backbone = 'resnet34' if kernel_size == 3 else 'resnet50'
else:
    backbone = 'resnet18'

# Create model and load weights
model = create_model(
    dataset='ucf101',
    backbone=backbone,
    pretrained=False,
    num_segments=3,
    frames_per_segment=5
)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Test with normalized input
dummy_input = torch.randn(1, 3, 3, 224, 224)

print('=== Tracing TSN Forward Pass ===\n')

B, T, C, H, W = dummy_input.shape
print(f'Input: B={B}, T={T}, C={C}, H={H}, W={W}')
print(f'Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]')
print(f'Input nan: {torch.isnan(dummy_input).any()}\n')

# Step 1: Reshape to (B*T, C, H, W)
x = dummy_input.view(-1, C, H, W)
print(f'1. After view(-1, C, H, W): shape={x.shape}, nan={torch.isnan(x).any()}')

# Step 2: Extract features from backbone
features = model.backbone(x)
print(f'2. After backbone: shape={features.shape}, nan={torch.isnan(features).any()}')

# Step 3: Reshape back to (B, T, feature_dim)
features = features.view(B, T, model.feature_dim)
print(f'3. After view(B, T, feature_dim): shape={features.shape}, nan={torch.isnan(features).any()}')

# Step 4: TSN aggregation
print(f'4. TSN aggregation:')
print(f'   num_segments={model.num_segments}, frames_per_segment={model.frames_per_segment}')

segment_features = []
for seg_idx in range(model.num_segments):
    start_idx = seg_idx * model.frames_per_segment
    end_idx = start_idx + model.frames_per_segment
    print(f'   Segment {seg_idx}: start_idx={start_idx}, end_idx={end_idx}')

    segment_feat = features[:, start_idx:end_idx, :]
    print(f'   segment_feat shape: {segment_feat.shape}, nan={torch.isnan(segment_feat).any()}')

    # Average pooling within segment
    segment_feat = segment_feat.mean(dim=1)
    print(f'   After mean(dim=1): shape={segment_feat.shape}, nan={torch.isnan(segment_feat).any()}')

    segment_features.append(segment_feat)

# Stack segment features
segment_features = torch.stack(segment_features, dim=1)
print(f'5. After stack: shape={segment_features.shape}, nan={torch.isnan(segment_features).any()}')

# Average pooling across segments (TSN consensus)
consensus_features = segment_features.mean(dim=1)
print(f'6. After consensus (mean): shape={consensus_features.shape}, nan={torch.isnan(consensus_features).any()}')
print(f'   Range: [{consensus_features.min():.4f}, {consensus_features.max():.4f}]')

# Classification
predictions = model.classifier(consensus_features)
print(f'\n7. Final predictions: shape={predictions.shape}')
print(f'   Range: [{predictions.min():.4f}, {predictions.max():.4f}]')
print(f'   NaN: {torch.isnan(predictions).any()}')
