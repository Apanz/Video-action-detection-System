"""
Model Metadata Extractor
Extracts and validates model information from checkpoint files
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple


class ModelMetadata:
    """
    Extracts metadata from model checkpoint files

    Features:
    - Extracts model architecture information
    - Validates model files
    - Generates human-readable model descriptions
    """

    # Known model signatures
    UCF101_CLASSES = 101
    HMDB51_CLASSES = 51

    @staticmethod
    def extract_metadata(checkpoint_path: str) -> Dict:
        """
        Extract model metadata from checkpoint file

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary containing model metadata
        """
        if not os.path.exists(checkpoint_path):
            return {
                'path': checkpoint_path,
                'is_valid': False,
                'error': 'File does not exist'
            }

        try:
            # Get file info
            file_path = Path(checkpoint_path)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Extract metadata
            metadata = {
                'path': str(file_path.absolute()),
                'filename': file_path.name,
                'file_size_mb': round(file_size_mb, 2),
                'is_valid': True,
            }

            # Try to extract model state dict
            if isinstance(checkpoint, dict):
                # Check for different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    ModelMetadata._extract_from_state_dict(state_dict, metadata, checkpoint)
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    ModelMetadata._extract_from_state_dict(state_dict, metadata, checkpoint)
                else:
                    # Assume checkpoint is the state dict itself
                    ModelMetadata._extract_from_state_dict(checkpoint, metadata, {})
            else:
                # Checkpoint is a model object
                metadata['error'] = 'Unknown checkpoint format'
                metadata['is_valid'] = False

            # Infer dataset from num_classes
            if 'num_classes' in metadata:
                # Ensure num_classes is an integer for comparison
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

            # Set defaults for missing fields
            metadata.setdefault('backbone', 'unknown')
            metadata.setdefault('num_classes', 'unknown')
            metadata.setdefault('num_segments', 'unknown')
            metadata.setdefault('frames_per_segment', 'unknown')
            metadata.setdefault('dataset', 'unknown')

            # Debug: Print extracted metadata for troubleshooting
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
        Extract information from model state dictionary

        Args:
            state_dict: Model state dictionary
            metadata: Metadata dictionary to update
            checkpoint: Full checkpoint (may contain training info)
        """

        # Check for config section first (newer checkpoints)
        if 'config' in checkpoint:
            config = checkpoint['config']
            metadata['backbone'] = config.get('backbone', 'unknown')
            metadata['num_classes'] = config.get('num_classes', 'unknown')
            metadata['num_segments'] = config.get('num_segments', 'unknown')
            metadata['frames_per_segment'] = config.get('frames_per_segment', 'unknown')
            metadata['dataset'] = config.get('dataset', 'unknown')
            # Still extract training info below
        else:
            # Detect backbone from layer names (legacy checkpoints)
            # Use the same detection logic as action_classifier.py
            state_keys = list(state_dict.keys())

            # Try to detect backbone architecture
            if any('backbone.layer' in k for k in state_keys):
                # Count blocks in each layer to determine ResNet variant
                blocks = {}
                for key in state_dict.keys():
                    if 'backbone.layer' in key and '.conv1.weight' in key:
                        layer_name = key.split('.')[1]
                        block_idx = int(key.split('.')[2])
                        blocks[layer_name] = max(blocks.get(layer_name, 0), block_idx + 1)

                # Determine backbone based on block counts
                total_blocks = sum(blocks.values())

                if total_blocks == 8:
                    metadata['backbone'] = 'resnet18'
                elif total_blocks == 16:
                    # ResNet-34 and ResNet-50 both have 16 blocks
                    # Check kernel size to determine block type:
                    # BasicBlock (ResNet-34): conv1 uses 3x3 kernels
                    # Bottleneck (ResNet-50): conv1 uses 1x1 kernels
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

            # Try to extract num_classes from final layer
            # TSN models may use different classifier key patterns:
            # - 'fc.weight' (original TSN)
            # - 'classifier.1.weight' (newer TSN with MLP head)
            # - 'classifier.weight' (generic)
            num_classes_key = None
            for key in state_keys:
                # Check for various classifier patterns
                if 'fc.weight' in key:
                    num_classes_key = key
                    break
                elif 'classifier' in key and 'weight' in key:
                    # Use the last layer in classifier (usually the final linear layer)
                    # e.g., 'classifier.1.weight' or 'classifier.3.weight'
                    if not num_classes_key or key > num_classes_key:
                        num_classes_key = key

            if num_classes_key:
                metadata['num_classes'] = int(state_dict[num_classes_key].shape[0])
                print(f"[ModelMetadata] Found num_classes={metadata['num_classes']} from key '{num_classes_key}'")

            # Try to extract num_segments from new_weights shape
            # (in TSN, new_weights has shape [num_segments, C, 1, 1])
            for key in state_keys:
                if 'new_weights' in key:
                    shape = state_dict[key].shape
                    if len(shape) >= 1:
                        metadata['num_segments'] = int(shape[0])
                    break

            # If num_segments still unknown, try to infer from model structure
            if metadata.get('num_segments', 'unknown') == 'unknown':
                # Some checkpoints store this in different formats
                # Check if we can infer from cons_weight or other TSN-specific layers
                for key in state_keys:
                    if 'cons_weight' in key or 'new_weights' in key:
                        shape = state_dict[key].shape
                        if len(shape) >= 1 and shape[0] < 10:  # Reasonable segment count
                            metadata['num_segments'] = int(shape[0])
                            break

            # If still unknown, use common default for UCF101 models
            if metadata.get('num_segments', 'unknown') == 'unknown':
                # Most UCF101 TSN models use 3 segments by default
                # This matches the default in action_classifier.py
                metadata['num_segments'] = 3

        # Extract training info if available
        if 'epoch' in checkpoint:
            metadata['trained_epochs'] = checkpoint['epoch']

        if 'best_acc' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc'])
        elif 'best_acc1' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc1'])

        # Set default frames_per_segment if not already set
        if 'frames_per_segment' not in metadata or metadata['frames_per_segment'] == 'unknown':
            # Common TSN configurations use 1 or 5 frames per segment
            # Since we can't detect this from the checkpoint structure,
            # use a reasonable default based on num_segments
            if metadata.get('num_segments', 'unknown') != 'unknown':
                # Common practice: fewer segments -> more frames per segment
                num_seg = metadata['num_segments']
                if num_seg <= 3:
                    metadata['frames_per_segment'] = 5  # 3x5 = 15 frames
                elif num_seg <= 5:
                    metadata['frames_per_segment'] = 5  # 5x5 = 25 frames
                else:
                    metadata['frames_per_segment'] = 1  # More segments, 1 frame each
            else:
                # If we can't determine num_segments, use the most common config
                metadata['frames_per_segment'] = 5  # Most TSN models use 5

    @staticmethod
    def validate_model(checkpoint_path: str) -> Tuple[bool, str]:
        """
        Validate model checkpoint file

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(checkpoint_path):
            return False, "File does not exist"

        try:
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Check if it's a valid format
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                    return True, ""
                # Check if it has model-like keys
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
        Generate human-readable model description

        Args:
            metadata: Model metadata dictionary

        Returns:
            Description string
        """
        if not metadata.get('is_valid', False):
            return f"Invalid model: {metadata.get('error', 'Unknown error')}"

        parts = [
            f"{metadata.get('backbone', 'Unknown').upper()}",
            f"{metadata.get('num_classes', '?')} classes",
        ]

        # Add temporal info if available
        num_segments = metadata.get('num_segments')
        frames_per_segment = metadata.get('frames_per_segment')

        if num_segments != 'unknown' and frames_per_segment != 'unknown':
            total_frames = num_segments * frames_per_segment
            parts.append(f"{num_segments}×{frames_per_segment}={total_frames} frames")
        elif num_segments != 'unknown':
            parts.append(f"{num_segments} segments")

        # Add training info if available
        if 'best_accuracy' in metadata:
            parts.append(f"acc: {metadata['best_accuracy']:.2f}%")

        return " | ".join(map(str, parts))

    @staticmethod
    def scan_models_directory(models_dir: str) -> Dict[str, list]:
        """
        Scan directory for model files

        Args:
            models_dir: Directory to scan

        Returns:
            Dictionary with categories as keys and lists of model metadata as values
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            return {}

        categories = {
            'ucf101': [],
            'custom': []
        }

        # Scan subdirectories
        for category_dir in models_path.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name.lower()

            # Determine category
            if category_name == 'ucf101':
                category = 'ucf101'
            else:
                category = 'custom'

            # Scan for .pth and .pt files
            for model_file in category_dir.glob('*.pth'):
                metadata = ModelMetadata.extract_metadata(str(model_file))
                metadata['category'] = category
                categories[category].append(metadata)

            for model_file in category_dir.glob('*.pt'):
                metadata = ModelMetadata.extract_metadata(str(model_file))
                metadata['category'] = category
                categories[category].append(metadata)

        # Sort each category by filename
        for category in categories:
            categories[category].sort(key=lambda x: x['filename'])

        return categories
