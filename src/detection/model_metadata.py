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
                if metadata['num_classes'] == ModelMetadata.UCF101_CLASSES:
                    metadata['dataset'] = 'ucf101'
                elif metadata['num_classes'] == ModelMetadata.HMDB51_CLASSES:
                    metadata['dataset'] = 'hmdb51'
                else:
                    metadata['dataset'] = f'custom({metadata["num_classes"]}_classes)'

            # Set defaults for missing fields
            metadata.setdefault('backbone', 'unknown')
            metadata.setdefault('num_classes', 'unknown')
            metadata.setdefault('num_segments', 'unknown')
            metadata.setdefault('frames_per_segment', 'unknown')
            metadata.setdefault('dataset', 'unknown')

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
            state_keys = list(state_dict.keys())

            # Try to detect backbone architecture
            if any('resnet' in k.lower() for k in state_keys):
                # Detect ResNet version
                if any('layer4' in k for k in state_keys):
                    metadata['backbone'] = 'resnet50' if any('bn3' in k for k in state_keys) else 'resnet18'
                else:
                    metadata['backbone'] = 'resnet18'
            elif any('inception' in k.lower() for k in state_keys):
                metadata['backbone'] = 'inception_v3'
            elif any('vgg' in k.lower() for k in state_keys):
                metadata['backbone'] = 'vgg16'
            elif any('mobilenet' in k.lower() for k in state_keys):
                metadata['backbone'] = 'mobilenet_v2'
            else:
                metadata['backbone'] = 'unknown'

            # Try to extract num_classes from final layer
            for key in state_keys:
                if 'fc.weight' in key or 'classifier.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    metadata['num_classes'] = num_classes
                    break

            # Try to extract num_segments from new_weights shape
            # (in TSN, new_weights has shape [num_segments, C, 1, 1])
            for key in state_keys:
                if 'new_weights' in key:
                    shape = state_dict[key].shape
                    if len(shape) >= 1:
                        metadata['num_segments'] = shape[0]
                    break

        # Extract training info if available
        if 'epoch' in checkpoint:
            metadata['trained_epochs'] = checkpoint['epoch']

        if 'best_acc' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc'])
        elif 'best_acc1' in checkpoint:
            metadata['best_accuracy'] = float(checkpoint['best_acc1'])

        # Set default frames_per_segment if not already set
        if 'frames_per_segment' not in metadata:
            metadata['frames_per_segment'] = 'unknown'

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
