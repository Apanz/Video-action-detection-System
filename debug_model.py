#!/usr/bin/env python3
"""
Debug script to check model forward pass
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
print(f'Blocks per layer: {dict(sorted(blocks.items()))}')
print(f'Total blocks: {total_blocks}')

if total_blocks == 8:
    backbone = 'resnet18'
elif total_blocks == 16:
    conv1_key = 'backbone.layer1.0.conv1.weight'
    kernel_size = state_dict[conv1_key].shape[-1]
    print(f'Layer1 conv1 kernel size: {kernel_size}x{kernel_size}')
    if kernel_size == 1:
        backbone = 'resnet50'
    elif kernel_size == 3:
        backbone = 'resnet34'
else:
    backbone = 'resnet18'

print(f'Detected backbone: {backbone}')

# Create model
model = create_model(
    dataset='ucf101',
    backbone=backbone,
    pretrained=False,
    num_segments=3,
    frames_per_segment=5
)

# Load weights
model.load_state_dict(state_dict, strict=False)
model.eval()

# Check BatchNorm layers
print('\n=== Checking BatchNorm running stats ===')
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        print(f'{name}:')
        print(f'  running_mean: range=[{module.running_mean.min():.4f}, {module.running_mean.max():.4f}], nan={torch.isnan(module.running_mean).any()}')
        print(f'  running_var: range=[{module.running_var.min():.4f}, {module.running_var.max():.4f}], nan={torch.isnan(module.running_var).any()}')

# Test with normalized input (ImageNet)
print('\n=== Testing with ImageNet normalized input ===')
dummy_input = torch.randn(1, 3, 3, 224, 224)
print(f'Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]')

# Test layer by layer
x = dummy_input.view(-1, 3, 224, 224)
print(f'\nAfter reshape: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]')

x = model.backbone.conv1(x)
print(f'After conv1: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}], nan={torch.isnan(x).any()}')

x = model.backbone.bn1(x)
print(f'After bn1: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}], nan={torch.isnan(x).any()}')

x = model.backbone.relu(x)
print(f'After relu1: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}], nan={torch.isnan(x).any()}')

x = model.backbone.maxpool(x)
print(f'After maxpool: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}], nan={torch.isnan(x).any()}')

# Test full forward pass
print('\n=== Testing full forward pass ===')
with torch.no_grad():
    output = model(dummy_input)

print(f'Output shape: {output.shape}')
print(f'Output range: [{output.min():.4f}, {output.max():.4f}]')
print(f'Output contains NaN: {torch.isnan(output).any()}')
print(f'Output contains Inf: {torch.isinf(output).any()}')

# Get prediction
probs = torch.softmax(output, dim=1)
top_idx = torch.argmax(probs).item()
confidence = probs[0, top_idx].item()
print(f'\nPredicted class: {top_idx}, confidence: {confidence:.4f}')
