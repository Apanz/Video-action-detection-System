#!/usr/bin/env python3
"""
Test script to check if model loads correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.core import create_model

# Load checkpoint
checkpoint_path = 'outputs/checkpoints/ucf101_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print("CHECKPOINT ANALYSIS")
print("=" * 60)
print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Number of keys in model_state_dict: {len(checkpoint['model_state_dict'])}")

# Print first 15 checkpoint keys
print("\nFirst 15 checkpoint keys:")
for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:15]):
    param = checkpoint['model_state_dict'][key]
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"  {i}: {key} | shape={param.shape} | nan={has_nan} | inf={has_inf}")

print("\n" + "=" * 60)
print("MODEL CREATION")
print("=" * 60)

# Create model
model = create_model(
    dataset='ucf101',
    backbone='resnet18',
    pretrained=False,
    num_segments=3,
    frames_per_segment=5
)

model.eval()

print(f"\nModel state dict keys: {len(model.state_dict())}")

# Print first 15 model keys
print("\nFirst 15 model keys:")
for i, key in enumerate(list(model.state_dict().keys())[:15]):
    param = model.state_dict()[key]
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"  {i}: {key} | shape={param.shape} | nan={has_nan} | inf={has_inf}")

print("\n" + "=" * 60)
print("WEIGHT LOADING")
print("=" * 60)

# Load weights
load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print(f"\nLoad result: {load_result}")
print(f"Missing keys: {len(load_result.missing_keys)}")
if load_result.missing_keys:
    print("\nMissing keys:")
    for i, key in enumerate(load_result.missing_keys):
        print(f"  {i}: {key}")

print(f"\nUnexpected keys: {len(load_result.unexpected_keys)}")
if load_result.unexpected_keys:
    print("\nUnexpected keys:")
    for i, key in enumerate(load_result.unexpected_keys):
        print(f"  {i}: {key}")

print("\n" + "=" * 60)
print("MODEL TEST")
print("=" * 60)

# Test forward pass
dummy_input = torch.randn(1, 3, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"Output contains NaN: {torch.isnan(output).any().item()}")
print(f"Output contains Inf: {torch.isinf(output).any().item()}")

# Check model parameters after loading
print("\n" + "=" * 60)
print("MODEL PARAMETERS AFTER LOADING")
print("=" * 60)

for name, param in model.named_parameters():
    has_nan = torch.isnan(param.data).any().item()
    has_inf = torch.isinf(param.data).any().item()
    if has_nan or has_inf:
        print(f"  {name} | shape={param.shape} | nan={has_nan} | inf={has_inf}")

print("\nNo NaN or Inf found in loaded model parameters!" if not any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()) else "\nFound NaN or Inf in loaded model parameters!")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
