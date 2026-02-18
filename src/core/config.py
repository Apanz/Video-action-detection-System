"""
配置文件
"""

import os
from pathlib import Path

# =============================================================================
# 根目录
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
MODELS_DIR = ROOT_DIR / "models"  # 统一的模型存储目录
TORCH_CACHE_DIR = MODELS_DIR / "torch_cache"  # PyTorch缓存子目录
YOLO_MODELS_DIR = MODELS_DIR / "yolo_models"  # YOLO检测模型子目录
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"  # 训练模型子目录


# =============================================================================
# 数据配置
# =============================================================================
class DataConfig:
    # 根目录
    UCF101_ROOT = str(DATA_DIR / "ucf101" / "UCF101" / "UCF-101")
    UCF101_SPLITS = str(DATA_DIR / "ucf101" / "UCF101TrainTestSplits-RecognitionTask" / "ucfTrainTestlist")
    HMDB51_ROOT = str(DATA_DIR / "hmdb51" / "HMDB51")
    HMDB51_SPLITS = str(DATA_DIR / "hmdb51" / "splits")  # 需要时创建

    # 数据集参数
    NUM_SEGMENTS = 5  # TSN的时间分段数量（为UCF101数据集增加）
    FRAMES_PER_SEGMENT = 5  # 每个分段的帧数（总计 = 5 * 5 = 25帧）
    INPUT_SIZE = 224  # 输入图像尺寸（224x224）
    NUM_WORKERS = 4  # 数据加载工作线程数

    # UCF101特定参数
    UCF101_NUM_CLASSES = 101

    # HMDB51特定参数
    HMDB51_NUM_CLASSES = 51


# =============================================================================
# 模型配置
# =============================================================================
class ModelConfig:
    # 模型类型
    MODEL_TYPE = "TSN"  # 时序片段网络

    # 骨干网络选项: 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'
    BACKBONE = "resnet34"
    PRETRAINED = True  # 使用ImageNet预训练权重
    DROPOUT = 0.5  # Dropout比率

    # 类别数量
    UCF101_NUM_CLASSES = 101
    HMDB51_NUM_CLASSES = 51

    # 检查点路径
    DEFAULT_UCF101_CHECKPOINT = str(CHECKPOINTS_DIR / "ucf101_best.pth")
    DEFAULT_HMDB51_CHECKPOINT = str(CHECKPOINTS_DIR / "hmdb51_best.pth")


# =============================================================================
# 训练配置
# =============================================================================
class TrainConfig:
    # 训练参数
    BATCH_SIZE = 32  # 批量大小（内存不足时减小）
    NUM_EPOCHS = 120  # 增加训练轮次以获得更好的收敛
    LEARNING_RATE = 0.001  # 初始学习率
    MOMENTUM = 0.9  # SGD动量
    WEIGHT_DECAY = 0.0005  # L2正则化（减小以提高泛化能力）

    # 正则化参数（用于防止过拟合）
    DROPOUT_RATE = 0.5  # Dropout比率
    LABEL_SMOOTHING = 0.1  # 标签平滑（0.1推荐用于过拟合情况）
    GRAD_CLIP = 1.0  # 梯度裁剪最大范数（0.0表示禁用）

    # Mixup/CutMix数据增强
    MIXUP_ALPHA = 0.2  # Mixup alpha参数（0.0表示禁用）
    CUTMIX_BETA = 1.0  # CutMix beta参数（0.0表示禁用）
    AUG_TYPE = 'mixup'  # 'mixup' 或 'cutmix'

    # 学习率调度器（余弦退火，延长训练时间）
    SCHEDULER_TYPE = 'cosine'  # 'step' 或 'cosine'
    STEP_SIZE = 15  # 每N个轮次衰减学习率（用于步进调度器）
    GAMMA = 0.1  # 学习率衰减因子（用于步进调度器）
    T_MAX = 120  # 余弦退火的最大轮次
    ETA_MIN = 1e-5  # 余弦退火的最小学习率

    # 数据增强
    AGGRESSIVE_AUG = True  # 使用激进增强以提高正则化效果

    # 检查点设置
    SAVE_DIR = str(CHECKPOINTS_DIR)
    LOG_DIR = str(LOGS_DIR)
    SAVE_FREQ = 5  # 每N个轮次保存检查点
    SAVE_BEST = True  # 仅保存最佳模型

    # 早停机制（增加耐心值以适应更长训练）
    EARLY_STOPPING_PATIENCE = 20  # N个轮次无改善时停止

    # 设备
    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

    # PyTorch缓存目录（用于下载预训练模型）
    TORCH_CACHE_DIR = str(TORCH_CACHE_DIR)  # 使用models/torch_cache目录


# =============================================================================
# 检测配置
# =============================================================================
class DetectionConfig:
    # YOLO设置
    YOLO_MODEL = "yolov5s.pt"
    DETECTION_CONFIDENCE = 0.5
    DEVICE = "auto"

    # YOLO模型路径
    YOLO_MODELS_DIR = str(YOLO_MODELS_DIR)

    # 训练模型目录
    TRAINED_MODELS_DIR = str(TRAINED_MODELS_DIR)

    DEFAULT_YOLO_MODELS = {
        "yolov5s": str(MODELS_DIR / "yolo_models" / "yolov5s.pt"),
        "yolov8n": str(MODELS_DIR / "yolo_models" / "yolov8n.pt"),
        "yolov8s": str(MODELS_DIR / "yolo_models" / "yolov8s.pt"),
        "yolov8m": str(MODELS_DIR / "yolo_models" / "yolov8m.pt")
    }

    # 时序处理（已更新以匹配数据集配置）
    NUM_SEGMENTS = 5  # 更新为5个片段以适应UCF101
    FRAMES_PER_SEGMENT = 5  # 更新为每个片段5帧（共25帧）
    TEMPORAL_BUFFER_SIZE = 30  # 存储最近30帧的预测结果
    SMOOTHING_WINDOW = 5  # 滑动窗口大小（用于平滑预测结果）

    # 输出设置
    OUTPUT_FPS = 30.0
    OUTPUT_CODEC = "mp4v"
    OUTPUT_QUALITY = 95
    VIDEOS_DIR = str(VIDEOS_DIR)

    # 显示设置
    SHOW_DISPLAY = True
    SHOW_INFO_PANEL = True
    SHOW_TIMESTAMP = True

    # 检查点路径
    DEFAULT_UCF101_CHECKPOINT = str(CHECKPOINTS_DIR / "ucf101_best.pth")
    DEFAULT_HMDB51_CHECKPOINT = str(CHECKPOINTS_DIR / "hmdb51_best.pth")


# =============================================================================
# GUI配置
# =============================================================================
class GuiConfig:
    WINDOW_TITLE = "Video Action Detection System"
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_MIN_WIDTH = 1024
    WINDOW_MIN_HEIGHT = 600

    # 检测标签页设置
    VIDEO_DISPLAY_WIDTH = 640
    VIDEO_DISPLAY_HEIGHT = 480


# =============================================================================
# 配置字典便于访问
# =============================================================================
CONFIG = {
    'data': DataConfig,
    'model': ModelConfig,
    'train': TrainConfig,
    'detection': DetectionConfig,
    'gui': GuiConfig,
}
