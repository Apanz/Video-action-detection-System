"""
YOLO Model Downloader Script
Downloads YOLO models to the models directory
"""

import os
import sys
from pathlib import Path

# Add src directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from core.config import MODELS_DIR


def download_yolo_models():
    """Download YOLO models to models directory"""
    import torch
    from ultralytics import YOLO

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Models to download
    models = [
        "yolov5s.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt"
    ]

    print(f"Models will be downloaded to: {MODELS_DIR}")
    print("=" * 60)

    for model_name in models:
        model_path = MODELS_DIR / model_name

        # Skip if already exists
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            continue

        print(f"Downloading {model_name}...")
        try:
            # Download model using ultralytics
            model = YOLO(model_name)

            # Save to models directory
            model.save(str(model_path))

            print(f"✓ Successfully downloaded {model_name} to {model_path}")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {str(e)}")
            continue

    print("" + "=" * 60)
    print("Download process completed!")
    print(f"Models are saved in: {MODELS_DIR}")
    print("Available models:")
    for model in MODELS_DIR.glob("*.pt"):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  - {model.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("YOLO Model Downloader")
    print("=" * 60)
    print("This script will download YOLO models for human detection.")
    print("Models: yolov5s.pt, yolov8n.pt, yolov8s.pt, yolov8m.pt")
    print("=" * 60)

    try:
        download_yolo_models()
    except KeyboardInterrupt:
        print("Download interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
