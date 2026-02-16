"""
YOLO Model Downloader Script
Downloads YOLO models to the models directory
"""

import os
import sys
from pathlib import Path

# 将src目录添加到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from core.config import MODELS_DIR


def download_yolo_models():
    """下载YOLO模型到模型目录"""
    import torch
    from ultralytics import YOLO

    # 确保模型目录存在
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 要下载的模型
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

        # 如果已存在则跳过
        if model_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            continue

        print(f"Downloading {model_name}...")
        try:
            # 使用ultralytics下载模型
            model = YOLO(model_name)

            # 保存到模型目录
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
