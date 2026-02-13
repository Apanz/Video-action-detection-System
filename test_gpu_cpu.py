#!/usr/bin/env python3
"""
Test GPU vs CPU performance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
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
kernel_size = state_dict['backbone.layer1.0.conv1.weight'].shape[-1]
backbone = 'resnet34' if kernel_size == 3 else 'resnet50'

print("="*60)
print("Testing GPU vs CPU Performance")
print("="*60)

# Test on CPU
print("\n1. Testing on CPU...")
model_cpu = create_model(
    dataset='ucf101',
    backbone=backbone,
    pretrained=False,
    num_segments=3,
    frames_per_segment=5
)
model_cpu.load_state_dict(state_dict, strict=False)
model_cpu.eval()

# Test with ImageNet normalized input
test_input = torch.randn(10, 3, 3, 224, 224)

with torch.no_grad():
    output_cpu = model_cpu(test_input)

probs_cpu = torch.softmax(output_cpu, dim=1)
top_pred_cpu = torch.argmax(probs_cpu, dim=1)
confidence_cpu = torch.gather(probs_cpu, 1, top_pred_cpu.unsqueeze(1))

print(f"   Top prediction distribution: {torch.bincount(top_pred_cpu.squeeze())}")
print(f"   Average confidence: {confidence_cpu.mean().item():.4f}")
print(f"   Output range: [{output_cpu.min():.4f}, {output_cpu.max():.4f}]")

# Test on GPU if available
if torch.cuda.is_available():
    print("\n2. Testing on GPU...")
    model_gpu = create_model(
        dataset='ucf101',
        backbone=backbone,
        pretrained=False,
        num_segments=3,
        frames_per_segment=5
    )
    model_gpu.load_state_dict(state_dict, strict=False)
    model_gpu.eval()
    model_gpu.cuda()

    test_input_gpu = test_input.cuda()

    with torch.no_grad():
        output_gpu = model_gpu(test_input_gpu)

    probs_gpu = torch.softmax(output_gpu, dim=1)
    top_pred_gpu = torch.argmax(probs_gpu, dim=1)
    confidence_gpu = torch.gather(probs_gpu, 1, top_pred_gpu.unsqueeze(1))

    # Copy back to CPU for comparison
    top_pred_gpu_cpu = top_pred_gpu.cpu()
    confidence_gpu_cpu = confidence_gpu.cpu()

    print(f"   Top prediction distribution: {torch.bincount(top_pred_gpu_cpu.squeeze())}")
    print(f"   Average confidence: {confidence_gpu_cpu.mean().item():.4f}")
    print(f"   Output range: [{output_gpu.min():.4f}, {output_gpu.max():.4f}]")

    # Compare
    diff = torch.abs(output_cpu - output_gpu.cpu())
    print(f"\n3. CPU vs GPU comparison:")
    print(f"   Max difference: {diff.max().item():.6f}")
    print(f"   Mean difference: {diff.mean().item():.6f}")
    print(f"   Predictions match: {torch.equal(top_pred_cpu.squeeze(), top_pred_gpu_cpu.squeeze())}")

else:
    print("\nGPU not available, skipping GPU test")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("If CPU and GPU predictions differ significantly,")
print("check if:")
print("  1. Model has random weights (not loaded from checkpoint)")
print("  2. Checkpoint was saved during training (inconsistent state)")
print("  3. BatchNorm running_stats need different handling for CPU/GPU")
