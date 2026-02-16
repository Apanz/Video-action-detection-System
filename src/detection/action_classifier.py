"""
Action Classifier Module
Integrates trained TSN model for action classification
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple


class ActionClassifier:
    """
    Action classifier using trained TSN models
    """

    def __init__(self, checkpoint_path: str, device: str = 'auto',
                 num_segments: int = 3, frames_per_segment: int = 5,
                 backbone: str = 'resnet18'):
        """
        Initialize action classifier

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cpu', 'cuda', or 'auto'
            num_segments: Number of temporal segments
            frames_per_segment: Frames per segment
            backbone: Backbone architecture
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading action classifier on {self.device}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get model configuration
        if 'model_state_dict' in checkpoint:
            # Extract info from checkpoint
            # For UCF101
            num_classes = 101
            dataset_name = 'ucf101'

            # Try to get from checkpoint metadata
            if 'config' in checkpoint:
                config = checkpoint['config']
                num_classes = config.get('num_classes', num_classes)
                backbone = config.get('backbone', backbone)
                num_segments = config.get('num_segments', num_segments)
                frames_per_segment = config.get('frames_per_segment', frames_per_segment)
            else:
                # CRITICAL: Infer backbone from checkpoint structure
                # Count blocks in each layer
                state_dict = checkpoint['model_state_dict']
                blocks = {}
                for key in state_dict.keys():
                    if 'backbone.layer' in key and '.conv1.weight' in key:
                        layer_name = key.split('.')[1]
                        block_idx = int(key.split('.')[2])
                        blocks[layer_name] = max(blocks.get(layer_name, 0), block_idx + 1)

                # Determine backbone based on block counts
                total_blocks = sum(blocks.values())
                print(f"[DEBUG classifier] Inferred blocks per layer: {dict(sorted(blocks.items()))}")
                print(f"[DEBUG classifier] Total blocks: {total_blocks}")

                if total_blocks == 8:
                    backbone = 'resnet18'
                    print("[DEBUG classifier] Detected backbone: ResNet-18")
                elif total_blocks == 16:
                    # ResNet-34 and ResNet-50 both have 16 blocks
                    # Check kernel size to determine block type:
                    # BasicBlock (ResNet-34): conv1 uses 3x3 kernels
                    # Bottleneck (ResNet-50): conv1 uses 1x1 kernels
                    # Find a conv1.weight in layer1 and check its shape
                    conv1_key = None
                    for key in state_dict.keys():
                        if 'backbone.layer1.0.conv1.weight' in key:
                            conv1_key = key
                            break

                    if conv1_key is not None:
                        kernel_size = state_dict[conv1_key].shape[-1]
                        print(f"[DEBUG classifier] Layer1 conv1 kernel size: {kernel_size}x{kernel_size}")

                        if kernel_size == 1:
                            backbone = 'resnet50'
                            print("[DEBUG classifier] Detected backbone: ResNet-50 (Bottleneck)")
                        elif kernel_size == 3:
                            backbone = 'resnet34'
                            print("[DEBUG classifier] Detected backbone: ResNet-34 (BasicBlock)")
                        else:
                            print(f"[WARNING classifier] Unknown kernel size {kernel_size}, using default: {backbone}")
                    else:
                        print(f"[WARNING classifier] Cannot determine block type, using default: {backbone}")
                elif total_blocks == 36:
                    backbone = 'resnet101'
                    print("[DEBUG classifier] Detected backbone: ResNet-101")
                else:
                    print(f"[WARNING classifier] Unknown architecture with {total_blocks} blocks, using default: {backbone}")

        else:
            # If checkpoint doesn't have metadata, use defaults
            num_classes = 101
            dataset_name = 'ucf101'

        # Create model
        from core import create_model
        self.model = create_model(
            dataset=dataset_name,
            backbone=backbone,
            pretrained=False,  # We're loading trained weights
            num_segments=num_segments,
            frames_per_segment=frames_per_segment
        )

        # DEBUG: Print checkpoint keys
        print(f"[DEBUG classifier] Checkpoint keys: {checkpoint.keys()}")
        if 'model_state_dict' in checkpoint:
            print(f"[DEBUG classifier] Checkpoint has model_state_dict with {len(checkpoint['model_state_dict'])} keys")
            # Print first 10 keys
            for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")
        elif 'state_dict' in checkpoint:
            print(f"[DEBUG classifier] Checkpoint has state_dict with {len(checkpoint['state_dict'])} keys")
            # Print first 10 keys
            for i, key in enumerate(list(checkpoint['state_dict'].keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")
        else:
            print(f"[DEBUG classifier] Checkpoint has {len(checkpoint)} keys (direct)")
            # Print first 10 keys
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                print(f"  {i}: {key}")
            print("  ...")

        # DEBUG: Print model keys
        model_state_dict = self.model.state_dict()
        print(f"[DEBUG classifier] Model has {len(model_state_dict)} keys")
        # Print first 10 keys
        for i, key in enumerate(list(model_state_dict.keys())[:10]):
            print(f"  {i}: {key}")
        print("  ...")

        # Load weights with strict=False to see what's missing
        if 'model_state_dict' in checkpoint:
            load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            load_result = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # Direct loading
            load_result = self.model.load_state_dict(checkpoint, strict=False)

        # DEBUG: Print load result
        print(f"[DEBUG classifier] Load result: {load_result}")
        print(f"[DEBUG classifier] Missing keys: {len(load_result.missing_keys)}")
        if load_result.missing_keys:
            print(f"[DEBUG classifier] Missing keys (first 10): {load_result.missing_keys[:10]}")
        print(f"[DEBUG classifier] Unexpected keys: {len(load_result.unexpected_keys)}")
        if load_result.unexpected_keys:
            print(f"[DEBUG classifier] Unexpected keys (first 10): {load_result.unexpected_keys[:10]}")

        # CRITICAL: Check if any weights contain NaN
        has_nan = False
        has_inf = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"[ERROR classifier] Parameter '{name}' contains NaN!")
                has_nan = True
            if torch.isinf(param.data).any():
                print(f"[ERROR classifier] Parameter '{name}' contains Inf!")
                has_inf = True

        if has_nan:
            print("[ERROR classifier] Model contains NaN weights! Checkpoint may be corrupted.")
        if has_inf:
            print("[ERROR classifier] Model contains Inf weights! Checkpoint may be corrupted.")

        # Test forward pass with dummy input
        print("[DEBUG classifier] Testing forward pass with dummy input...")
        dummy_input = torch.randn(1, 3, 3, 224, 224)
        dummy_input = dummy_input.to(self.device)
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
            print(f"[DEBUG classifier] Dummy output shape: {dummy_output.shape}")
            print(f"[DEBUG classifier] Dummy output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
            print(f"[DEBUG classifier] Dummy output contains NaN: {torch.isnan(dummy_output).any()}")
            print(f"[DEBUG classifier] Dummy output contains Inf: {torch.isinf(dummy_output).any()}")

        # Verify weights are loaded - check first layer
        first_conv_weight = None
        first_bn_weight = None
        classifier_weight = None

        for name, param in self.model.named_parameters():
            if 'backbone.conv1' in name and first_conv_weight is None:
                first_conv_weight = param.data
            elif 'backbone.bn1' in name and first_bn_weight is None:
                first_bn_weight = param.data
            elif 'classifier.1' in name:
                classifier_weight = param.data

        if first_conv_weight is not None:
            conv_msg = f"[VERIFY] First conv weight mean: {first_conv_weight.mean():.6f}"
            print(conv_msg)
        else:
            print("[VERIFY] First conv weight: None")

        if first_bn_weight is not None:
            bn_msg = f"[VERIFY] First bn weight mean: {first_bn_weight.mean():.6f}"
            print(bn_msg)
        else:
            print("[VERIFY] First bn weight: None")

        if classifier_weight is not None:
            cls_msg = f"[VERIFY] Classifier weight mean: {classifier_weight.mean():.6f}"
            print(cls_msg)
        else:
            print("[VERIFY] Classifier weight: None")

        # Check if weights look random (untrained)
        # NOTE: Trained models with BatchNorm typically have weights with mean ≈ 0, which is normal
        # This check was overly strict and caused false alarms. Removed to avoid confusion.
        # The real validation is: (1) load_result shows all keys matched, (2) forward pass produces valid outputs
        if first_conv_weight is not None and abs(first_conv_weight.mean()) < 0.001:
            # Only warn if mean is EXTREMELY close to 0 (which might indicate zero initialization)
            print("[WARNING] First layer weights have very small mean. Verify checkpoint is trained.")

        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # Class names for UCF101
        self.ucf101_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BoxingPunchingBag',
            'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleaningAndFlossingTeeth',
            'CliffDiving', 'Diving', 'Drumming', 'Fencing', 'FrontCrawl',
            'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HeadMassage',
            'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
            'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
            'Knitting', 'Lunges', 'MagicTrick', 'Mixing', 'MoppingFloor',
            'Nunchucks', 'ParallelBars', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
            'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla',
            'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
            'PushUps', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
            'SkateBoarding', 'Skiing', 'SkiJet', 'Skydiving', 'SoccerJuggling',
            'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
            'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping',
            'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
        ]

        # Prediction smoothing
        self.prediction_history = []
        self.smoothing_window = 5

        # Model parameters
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.total_frames = num_segments * frames_per_segment

        print(f"Action classifier loaded successfully!")
        print(f"Model: {backbone}")
        print(f"Segments: {num_segments}, Frames per segment: {frames_per_segment}")
        print(f"Classes: {num_classes}")

    def classify(self, frames: torch.Tensor) -> Tuple[str, float]:
        """
        Classify action from temporal frames

        Args:
            frames: Input tensor (1, T, C, H, W)

        Returns:
            Tuple of (action_label, confidence)
        """
        # DEBUG: Check model parameters once for NaN
        if not hasattr(self, '_checked_params'):
            print("[DEBUG classifier] Checking model parameters for NaN...")
            for name, param in self.model.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"[ERROR classifier] Parameter '{name}' contains NaN!")
                if torch.isinf(param.data).any():
                    print(f"[ERROR classifier] Parameter '{name}' contains Inf!")
            self._checked_params = True

        with torch.no_grad():
            # Move to device
            frames = frames.to(self.device)

            # DEBUG: Log input info
            print(f"[DEBUG classifier] Input tensor shape: {frames.shape}")
            print(f"[DEBUG classifier] Input tensor range: [{frames.min():.4f}, {frames.max():.4f}]")

            # Forward pass
            outputs = self.model(frames)

            # DEBUG: Log model output
            print(f"[DEBUG classifier] Model output shape: {outputs.shape}")
            print(f"[DEBUG classifier] Model output range: min={outputs.min():.4f}, max={outputs.max():.4f}")

            # VALIDATION: Check for NaN or Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("[WARNING classifier] Model outputs contain NaN or Inf values!")
                print(f"[WARNING classifier] NaN count: {torch.isnan(outputs).sum()}")
                print(f"[WARNING classifier] Inf count: {torch.isinf(outputs).sum()}")
                # Return safe fallback
                return "Unknown", 0.0

            # Get probabilities
            try:
                probs = torch.softmax(outputs, dim=1)
                probs = probs.cpu().numpy()[0]  # Remove batch dim

                # DEBUG: Log probabilities
                print(f"[DEBUG classifier] Probabilities shape: {probs.shape}")
                print(f"[DEBUG classifier] Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
                print(f"[DEBUG classifier] Probabilities sum: {probs.sum():.4f}")
            except Exception as e:
                print(f"[ERROR classifier] Softmax failed: {e}")
                return "Unknown", 0.0

            # Get top prediction
            top_idx = np.argmax(probs)
            confidence = float(probs[top_idx])
            action_label = self.ucf101_classes[top_idx]

            # DEBUG: Log prediction
            print(f"[DEBUG classifier] Predicted action: {action_label} (index: {top_idx})")
            print(f"[DEBUG classifier] Confidence: {confidence}")

            # Apply smoothing
            self.prediction_history.append((action_label, confidence))
            if len(self.prediction_history) > self.smoothing_window:
                self.prediction_history.pop(0)

            # Smooth prediction
            if len(self.prediction_history) > 1:
                # Use most common action with average confidence
                action_counts = {}
                total_confidence = {}

                for action, conf in self.prediction_history:
                    action_counts[action] = action_counts.get(action, 0) + 1
                    total_confidence[action] = total_confidence.get(action, 0) + conf

                # Get action with highest count
                best_action = max(action_counts, key=action_counts.get)
                avg_confidence = total_confidence[best_action] / action_counts[best_action]

                return best_action, avg_confidence

            return action_label, confidence

    def clear_prediction_history(self):
        """Clear prediction history for fresh start"""
        self.prediction_history.clear()

    def predict_proba(self, frames: torch.Tensor) -> np.ndarray:
        """
        Return probability distribution for all classes

        Args:
            frames: Input tensor (1, T, C, H, W)

        Returns:
            Probability array of shape (num_classes,)
        """
        with torch.no_grad():
            # Move to device
            frames = frames.to(self.device)

            # Forward pass
            outputs = self.model(frames)

            # VALIDATION: Check for NaN or Inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("[WARNING classifier] Model outputs contain NaN or Inf values!")
                # Return uniform distribution
                num_classes = len(self.ucf101_classes)
                return np.ones(num_classes) / num_classes

            # Get probabilities
            try:
                probs = torch.softmax(outputs, dim=1)
                probs = probs.cpu().numpy()[0]  # Remove batch dim
                return probs
            except Exception as e:
                print(f"[ERROR classifier] Softmax failed: {e}")
                num_classes = len(self.ucf101_classes)
                return np.ones(num_classes) / num_classes


class SimpleClassifier:
    """
    Fallback classifier for when checkpoint is not available
    """

    def __init__(self):
        """Initialize simple classifier"""
        self.action_names = ["Unknown"]
        self.is_dummy = True
        # UCF101 class names for compatibility
        self.ucf101_classes = [
            'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
            'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
            'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BoxingPunchingBag',
            'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleaningAndFlossingTeeth',
            'CliffDiving', 'Diving', 'Drumming', 'Fencing', 'FrontCrawl',
            'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HeadMassage',
            'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
            'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking',
            'Knitting', 'Lunges', 'MagicTrick', 'Mixing', 'MoppingFloor',
            'Nunchucks', 'ParallelBars', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
            'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla',
            'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch',
            'PushUps', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
            'SkateBoarding', 'Skiing', 'SkiJet', 'Skydiving', 'SoccerJuggling',
            'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
            'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping',
            'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'
        ]

    def classify(self, frames: torch.Tensor) -> Tuple[str, float]:
        """Dummy classification - returns unknown"""
        return "Unknown", 0.0

    def predict_proba(self, frames: torch.Tensor) -> np.ndarray:
        """
        Return probability distribution for all classes

        Args:
            frames: Input tensor (1, T, C, H, W)

        Returns:
            Probability array of shape (num_classes,)
        """
        # Return uniform distribution over all classes
        num_classes = len(self.ucf101_classes)
        probs = np.ones(num_classes) / num_classes
        return probs


def load_classifier(checkpoint_path: str, device: str = 'auto',
                   num_segments: int = 3, frames_per_segment: int = 5,
                   backbone: str = 'resnet18') -> ActionClassifier:
    """
    Load action classifier from checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        num_segments: Number of temporal segments
        frames_per_segment: Frames per segment
        backbone: Backbone architecture

    Returns:
        ActionClassifier instance
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return SimpleClassifier()

    try:
        return ActionClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            backbone=backbone
        )
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return SimpleClassifier()
