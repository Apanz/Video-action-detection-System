# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video action detection project implementing the TSN (Temporal Segment Networks) model for human action recognition. The project supports UCF101 (101 action classes) and HMDB51 (51 action classes) datasets using PyTorch.

**Status**: The project is fully implemented with training and evaluation scripts, real-time video behavior detection, and a complete PyQt GUI.

## Project Structure

```
video_action_detection/
├── src/                    # Source code (new organized structure)
│   ├── core/              # Core modules
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration management
│   │   ├── model.py       # TSN model architecture
│   │   └── utils.py       # Utility functions
│   ├── data/              # Dataset modules
│   │   ├── __init__.py
│   │   └── datasets.py    # UCF101 and HMDB51 datasets
│   ├── detection/          # Real-time detection modules
│   │   ├── __init__.py
│   │   ├── pipeline.py     # Main processing pipeline
│   │   ├── human_detector.py    # YOLO person detection
│   │   ├── temporal_processor.py # Frame buffering
│   │   ├── action_classifier.py  # TSN classification
│   │   └── video_writer.py     # Video output
│   ├── training/          # Training modules
│   │   ├── __init__.py
│   │   └── trainer.py     # Trainer class
│   └── gui/               # GUI modules (new)
│       ├── __init__.py
│       ├── main_window.py  # Main application window
│       ├── detection_tab.py # Real-time detection UI
│       ├── training_tab.py  # Training UI
│       ├── video_thread.py  # Video processing worker thread
│       └── training_thread.py # Training worker thread
├── scripts/               # Standalone scripts
│   ├── train.py          # Training script (refactored)
│   ├── eval.py           # Evaluation script (refactored)
│   ├── app.py            # GUI entry point (new)
│   └── realtime_detection.py # CLI detection (refactored)
├── configs/              # Configuration files
│   └── default.yaml     # Default configuration
├── data/                # Dataset directories
│   ├── ucf101/          # UCF101 dataset
│   └── hmdb51/          # HMDB51 dataset
├── outputs/              # Output directories
│   ├── checkpoints/      # Model checkpoints
│   ├── logs/             # TensorBoard logs
│   └── videos/           # Output videos
├── config.py             # Legacy config (kept for compatibility)
├── dataset.py           # Legacy dataset (kept for compatibility)
├── model.py             # Legacy model (kept for compatibility)
├── train.py             # Legacy train script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── CLAUDE.md           # This file
```

## New Organized Structure

The project has been reorganized into a clean, modular structure:

### Core Module (`src/core/`)
- **config.py**: Centralized configuration with path handling
  - Uses `Path` for cross-platform path management
  - Includes `ROOT_DIR` for project root reference
  - Separate configs: `DataConfig`, `ModelConfig`, `TrainConfig`, `DetectionConfig`, `GuiConfig`

- **model.py**: TSN and TSNWithConsensus models
  - Clean import structure
  - Updated torchvision API usage

- **utils.py**: Utility functions
  - `set_seed()`: Reproducibility
  - `get_device()`: Device selection
  - `count_parameters()`: Model statistics
  - `format_time()`: Time formatting

### Data Module (`src/data/`)
- **datasets.py**: Combined dataset classes
  - `VideoDataset`: Base class
  - `UCF101Dataset`: UCF101-specific implementation
  - `HMDB51Dataset`: HMDB51-specific implementation
  - Transform functions: `get_train_transform()`, `get_test_transform()`

### Detection Module (`src/detection/`)
- **pipeline.py**: Main detection pipeline
- **human_detector.py**: YOLO-based person detection
- **temporal_processor.py**: Frame buffering and TSN sampling
- **action_classifier.py**: TSN model integration
- **video_writer.py**: Video output with overlays

### Training Module (`src/training/`)
- **trainer.py**: Trainer class with training loop

### GUI Module (`src/gui/`)
- **main_window.py**: Main application window
- **detection_tab.py**: Real-time detection interface
- **training_tab.py**: Training interface with progress plots
- **video_thread.py**: Worker thread for video processing
- **training_thread.py**: Worker thread for training

## Usage

### GUI Application (Recommended)

```bash
# Start the GUI
python scripts/app.py
```

**GUI Features:**
- **Detection Tab**: Real-time webcam/video detection with visual controls
- **Training Tab**: Model training with live progress plots
- **Real-time Updates**: Frame-by-frame video display
- **Progress Tracking**: Training loss/accuracy plots
- **Configurable Settings**: All parameters adjustable via UI

### Command Line Interface

**Training:**
```bash
# Basic training
python train.py --dataset ucf101 --epochs 50 --batch_size 32

# Full command with all options
python train.py --dataset ucf101 --epochs 50 --batch_size 32 \
    --backbone resnet18 --num_segments 3 --frames_per_segment 5 \
    --lr 0.001 --step_size 15 --gamma 0.1 \
    --save_freq 5 --num_workers 4
```

**Real-time Detection:**
```bash
# Webcam detection
python realtime_detection.py --mode webcam

# Process video file
python realtime_detection.py --mode video --input video.mp4 --output result.mp4
```

## Configuration System

The new configuration system in `src/core/config.py` provides:

### Path Management
```python
from src.core.config import ROOT_DIR, DATA_DIR, OUTPUTS_DIR
# All paths use pathlib.Path for cross-platform compatibility
```

### Configuration Classes
- **DataConfig**: Dataset paths and parameters
- **ModelConfig**: Model architecture and checkpoint paths
- **TrainConfig**: Training hyperparameters
- **DetectionConfig**: Detection parameters
- **GuiConfig**: GUI settings

## Import Patterns

When importing from the new structure:

```python
# Core modules
from src.core import DataConfig, ModelConfig, TrainConfig
from src.core import create_model, TSN
from src.core import set_seed, get_device

# Data modules
from src.data import UCF101Dataset, HMDB51Dataset
from src.data import get_train_transform, get_test_transform

# Detection modules
from src.detection import DetectionPipeline
from src.detection import HumanDetector
from src.detection import ActionClassifier, load_classifier
from src.detection import TemporalProcessor
from src.detection import VideoWriter, FrameOverlay

# Training modules
from src.training import Trainer

# GUI modules
from src.gui import MainWindow, DetectionTab, TrainingTab
```

## Legacy Compatibility

The old file structure is still available for backward compatibility:
- `config.py` at project root
- `dataset.py` at project root
- `model.py` at project root
- `train.py` at project root

These can still be used but the new `src/` structure is recommended.

## Dependencies

### Core Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
tqdm>=4.65.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
```

### GUI Dependencies
```
PyQt5>=5.15.0
matplotlib>=3.7.0
```

### Detection Dependencies
```
ultralytics>=8.0.0  # YOLOv5/YOLOv8
```

## Recent Updates (2026-01-22)

### v2.0 Major Refactoring
1. **Reorganized Project Structure**: Moved to `src/` layout
2. **Added PyQt GUI**: Complete graphical interface with:
   - Real-time detection tab
   - Training tab with progress plots
   - Worker threads for non-blocking operations
3. **Updated Configuration**: Centralized config with path management
4. **Refactored Imports**: Clean, modular import structure
5. **Created New Scripts**: `scripts/app.py` for GUI entry point

### v1.1 (Previous Session)
- Real-time detection system implementation
- YOLOv5 integration
- Video output functionality

### v1.0 (Original)
- TSN training system
- Model evaluation
- Dataset loading

## Key Architectural Decisions

1. **Modular Structure**: Clear separation of concerns (core, data, detection, training, gui)
2. **Path Management**: Using `pathlib.Path` for cross-platform compatibility
3. **Threading**: Worker threads for GUI to prevent blocking
4. **Configuration**: Centralized config with type hints
5. **Backward Compatibility**: Legacy files still available

## Development Guidelines

When adding new features:
1. Add to appropriate `src/` subdirectory
2. Update corresponding `__init__.py` for exports
3. Follow existing import patterns
4. Update this documentation
5. Test both GUI and CLI interfaces

## Troubleshooting

### Import Errors
```python
# If you get import errors, try:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Path Issues
- The new structure uses `pathlib.Path` for cross-platform paths
- Always use `ROOT_DIR` for project root reference
- Never hardcode paths with `\\` or `/` - use `os.path.join()` or `/` operator

### GUI Not Starting
- Ensure PyQt5 is installed: `pip install PyQt5`
- Check matplotlib backend is compatible
- Run from project root directory

## Future Enhancements

Potential areas for improvement:
1. Add dataset management tab
2. Model versioning and comparison
3. Batch video processing
4. Export predictions to CSV/JSON
5. Video trimming/editing features
6. Model inference visualization
