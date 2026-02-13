"""
Configuration file for TSN-based Video Action Recognition
Simple and clear configuration for undergraduate thesis
"""

import os
from pathlib import Path

# =============================================================================
# ROOT DIRECTORY
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
MODELS_DIR = ROOT_DIR / "models"  # Directory for YOLO models


# =============================================================================
# DATA CONFIGURATION
# =============================================================================
class DataConfig:
    # Root directories
    UCF101_ROOT = str(DATA_DIR / "ucf101" / "UCF101" / "UCF-101")
    UCF101_SPLITS = str(DATA_DIR / "ucf101" / "UCF101TrainTestSplits-RecognitionTask" / "ucfTrainTestlist")
    HMDB51_ROOT = str(DATA_DIR / "hmdb51" / "HMDB51")
    HMDB51_SPLITS = str(DATA_DIR / "hmdb51" / "splits")  # Will create if needed

    # Dataset parameters
    NUM_SEGMENTS = 5           # Number of temporal segments for TSN (increased for UCF101)
    FRAMES_PER_SEGMENT = 5      # Frames per segment (total = 5 * 5 = 25 frames)
    INPUT_SIZE = 224            # Input image size (224x224)
    NUM_WORKERS = 4             # Number of data loading workers

    # UCF101 specific
    UCF101_NUM_CLASSES = 101
    UCF101_NUM_FRAMES = 25      # Target frames per video for UCF101

    # HMDB51 specific
    HMDB51_NUM_CLASSES = 51
    HMDB51_NUM_FRAMES = 16      # Target frames per video for HMDB51


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
class ModelConfig:
    # Model type
    MODEL_TYPE = "TSN"          # Temporal Segment Networks

    # Backbone options: 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'
    BACKBONE = "resnet50"      # Changed to ResNet50 for maximum accuracy
    PRETRAINED = True           # Use ImageNet pretrained weights
    DROPOUT = 0.5               # Dropout rate

    # Number of classes
    UCF101_NUM_CLASSES = 101
    HMDB51_NUM_CLASSES = 51

    # Checkpoint paths
    DEFAULT_UCF101_CHECKPOINT = str(CHECKPOINTS_DIR / "ucf101_best.pth")
    DEFAULT_HMDB51_CHECKPOINT = str(CHECKPOINTS_DIR / "hmdb51_best.pth")


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
class TrainConfig:
    # Training parameters (optimized for cloud GPU with maximum accuracy)
    BATCH_SIZE = 32             # Batch size (reduce if OOM)
    NUM_EPOCHS = 120           # Increased epochs for better convergence
    LEARNING_RATE = 0.001       # Initial learning rate
    MOMENTUM = 0.9              # SGD momentum
    WEIGHT_DECAY = 0.0005      # L2 regularization (reduced for better generalization)

    # Regularization parameters (for preventing overfitting)
    DROPOUT_RATE = 0.5          # Dropout rate
    LABEL_SMOOTHING = 0.1       # Label smoothing (0.1 is recommended for overfitting)
    GRAD_CLIP = 1.0             # Gradient clipping max norm (0.0 to disable)

    # Mixup/CutMix augmentation
    MIXUP_ALPHA = 0.2           # Mixup alpha (0.0 to disable)
    CUTMIX_BETA = 1.0           # CutMix beta (0.0 to disable)
    AUG_TYPE = 'mixup'           # 'mixup' or 'cutmix'

    # Learning rate scheduler (cosine annealing with longer training)
    SCHEDULER_TYPE = 'cosine'    # 'step' or 'cosine'
    STEP_SIZE = 15              # Decay LR every N epochs (for step scheduler)
    GAMMA = 0.1                 # LR decay factor (for step scheduler)
    T_MAX = 120                # Max epochs for cosine annealing (increased to 120)
    ETA_MIN = 1e-5              # Minimum LR for cosine annealing (increased)

    # Data augmentation
    AGGRESSIVE_AUG = True       # Use aggressive augmentation for better regularization

    # Checkpoint settings
    SAVE_DIR = str(CHECKPOINTS_DIR)
    LOG_DIR = str(LOGS_DIR)
    SAVE_FREQ = 5               # Save checkpoint every N epochs
    SAVE_BEST = True            # Save only best model

    # Early stopping (increased patience for longer training)
    EARLY_STOPPING_PATIENCE = 20  # Stop if no improvement for N epochs

    # Device
    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

    # PyTorch cache directory (for downloading pretrained models)
    TORCH_CACHE_DIR = "E:/torch_models"  # Change to your preferred location


# =============================================================================
# DETECTION CONFIGURATION
# =============================================================================
class DetectionConfig:
    # YOLO settings
    YOLO_MODEL = "yolov5s.pt"
    DETECTION_CONFIDENCE = 0.5
    DEVICE = "auto"

    # YOLO model paths
    YOLO_MODELS_DIR = str(MODELS_DIR)
    DEFAULT_YOLO_MODELS = {
        "yolov5s": str(MODELS_DIR / "yolov5s.pt"),
        "yolov8n": str(MODELS_DIR / "yolov8n.pt"),
        "yolov8s": str(MODELS_DIR / "yolov8s.pt"),
        "yolov8m": str(MODELS_DIR / "yolov8m.pt")
    }

    # Temporal processing (updated to match dataset config)
    NUM_SEGMENTS = 5           # Updated to 5 segments for UCF101
    FRAMES_PER_SEGMENT = 5      # Updated to 5 frames per segment (25 total)
    TEMPORAL_BUFFER_SIZE = 30
    SMOOTHING_WINDOW = 5

    # Output settings
    OUTPUT_FPS = 30.0
    OUTPUT_CODEC = "mp4v"
    OUTPUT_QUALITY = 95
    VIDEOS_DIR = str(VIDEOS_DIR)

    # Display settings
    SHOW_DISPLAY = True
    SHOW_INFO_PANEL = True
    SHOW_TIMESTAMP = True

    # Checkpoint paths
    DEFAULT_UCF101_CHECKPOINT = str(CHECKPOINTS_DIR / "ucf101_best.pth")
    DEFAULT_HMDB51_CHECKPOINT = str(CHECKPOINTS_DIR / "hmdb51_best.pth")


# =============================================================================
# GUI CONFIGURATION
# =============================================================================
class GuiConfig:
    WINDOW_TITLE = "Video Action Detection System"
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_MIN_WIDTH = 1024
    WINDOW_MIN_HEIGHT = 600

    # Detection tab settings
    VIDEO_DISPLAY_WIDTH = 640
    VIDEO_DISPLAY_HEIGHT = 480

    # Training tab settings
    PLOT_UPDATE_INTERVAL = 10  # Update plots every N epochs


# =============================================================================
# CONFIG DICTIONARY FOR EASY ACCESS
# =============================================================================
CONFIG = {
    'data': DataConfig,
    'model': ModelConfig,
    'train': TrainConfig,
    'detection': DetectionConfig,
    'gui': GuiConfig,
}
