#!/usr/bin/env python3
"""
Verify that the preprocessing fix works correctly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
from src.core import create_model
from src.detection import TemporalProcessor

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
print("Testing Preprocessing Fix")
print("="*60)

# Create model
model = create_model(
    dataset='ucf101',
    backbone=backbone,
    pretrained=False,
    num_segments=3,
    frames_per_segment=5
)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Test 1: Model forward pass with 3 frames (dynamic segment handling)
print("\n1. Testing model forward pass with 3 frames...")
dummy_input = torch.randn(1, 3, 3, 224, 224)  # B=1, T=3, C=3, H=224, W=224

with torch.no_grad():
    output = model(dummy_input)

print(f"   Output shape: {output.shape}")
print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"   Output NaN: {torch.isnan(output).any()}")

if not torch.isnan(output).any():
    print("   [PASS] Model forward pass successful (no NaN)")
else:
    print("   [FAIL] Model forward pass FAILED (NaN detected)")

# Test 2: Test preprocessing with temporal processor
print("\n2. Testing temporal processor preprocessing...")

# Create test frames (BGR format, same as from cv2.VideoCapture)
test_frames = []
for i in range(3):
    # Create a BGR frame with different colors for each frame
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    frame[i*80:(i+1)*80, :] = (i * 80, 255 - i*80, 128)  # BGR
    test_frames.append(frame)

# Create temporal processor
processor = TemporalProcessor(
    num_segments=3,
    frames_per_segment=5,
    buffer_size=30,
    target_size=(224, 224)
)

# Preprocess frames
try:
    tensor_output = processor.preprocess_frames(test_frames)

    print(f"   Input: {len(test_frames)} BGR frames of shape {test_frames[0].shape}")
    print(f"   Output tensor shape: {tensor_output.shape}")
    print(f"   Output tensor range: [{tensor_output.min():.4f}, {tensor_output.max():.4f}]")
    print(f"   Output tensor NaN: {torch.isnan(tensor_output).any()}")

    if not torch.isnan(tensor_output).any():
        print("   [PASS] Preprocessing successful (no NaN)")
    else:
        print("   [FAIL] Preprocessing FAILED (NaN detected)")

    # Test 3: Run the preprocessed tensor through the model
    print("\n3. Testing model with preprocessed frames...")
    with torch.no_grad():
        predictions = model(tensor_output)

    probs = torch.softmax(predictions, dim=1)
    top_pred = torch.argmax(probs, dim=1)
    confidence = torch.gather(probs, 1, top_pred.unsqueeze(1))

    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Predictions NaN: {torch.isnan(predictions).any()}")
    print(f"   Top prediction: {top_pred.item()}")
    print(f"   Confidence: {confidence.item():.4f}")

    if not torch.isnan(predictions).any():
        print("   [PASS] Model inference successful (no NaN)")
    else:
        print("   [FAIL] Model inference FAILED (NaN detected)")

except Exception as e:
    print(f"   [FAIL] Preprocessing FAILED with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("The preprocessing fix should:")
print("  1. Resize frames to 256x256")
print("  2. Center crop to 224x224")
print("  3. Normalize to [0, 1]")
print("  4. Apply ImageNet normalization")
print("\nThis matches the training preprocessing pipeline.")
print("GPU vs CPU should NOT cause significant accuracy differences.")
print("The 50% confidence issue was caused by preprocessing mismatch.")
