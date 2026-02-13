#!/usr/bin/env python3
"""
Debug script to trace NaN through network layers
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import torch.nn as nn
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
if total_blocks == 8:
    backbone = 'resnet18'
elif total_blocks == 16:
    conv1_key = 'backbone.layer1.0.conv1.weight'
    kernel_size = state_dict[conv1_key].shape[-1]
    if kernel_size == 1:
        backbone = 'resnet50'
    elif kernel_size == 3:
        backbone = 'resnet34'
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

print('=== Tracing NaN through network ===\n')

B, T, C, H, W = dummy_input.shape
x = dummy_input.view(-1, C, H, W)
print(f'1. Reshaped input: shape={x.shape}, nan={torch.isnan(x).any()}')

# Layer 1
x = model.backbone.conv1(x)
print(f'2. After conv1: shape={x.shape}, nan={torch.isnan(x).any()}')

x = model.backbone.bn1(x)
print(f'3. After bn1: shape={x.shape}, nan={torch.isnan(x).any()}')

x = model.backbone.relu(x)
print(f'4. After relu: shape={x.shape}, nan={torch.isnan(x).any()}')

x = model.backbone.maxpool(x)
print(f'5. After maxpool: shape={x.shape}, nan={torch.isnan(x).any()}')

# Layer 1 blocks
x = model.backbone.layer1(x)
print(f'6. After layer1: shape={x.shape}, nan={torch.isnan(x).any()}')

# Layer 2
x = model.backbone.layer2(x)
print(f'7. After layer2: shape={x.shape}, nan={torch.isnan(x).any()}')

# Layer 3
x = model.backbone.layer3(x)
print(f'8. After layer3: shape={x.shape}, nan={torch.isnan(x).any()}')

# Layer 4
x = model.backbone.layer4(x)
print(f'9. After layer4: shape={x.shape}, nan={torch.isnan(x).any()}')

# Global average pooling
x = model.backbone.avgpool(x)
print(f'10. After avgpool: shape={x.shape}, nan={torch.isnan(x).any()}')

# Flatten
x = x.view(x.size(0), -1)
print(f'11. After flatten: shape={x.shape}, nan={torch.isnan(x).any()}')

# Classifier
x = model.classifier[0](x)  # Dropout (should be identity in eval mode)
print(f'12. After dropout: shape={x.shape}, nan={torch.isnan(x).any()}')

x = model.classifier[1](x)  # Linear
print(f'13. After classifier: shape={x.shape}, nan={torch.isnan(x).any()}')
print(f'   Output range: [{x.min():.4f}, {x.max():.4f}]')
