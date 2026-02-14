# 视频行为识别系统

基于深度学习的视频行为识别系统

本项目实现了基于 TSN（Temporal Segment Networks，时序分段网络）的视频行为识别系统，支持 UCF101 和 HMDB51 数据集。系统包含完整的训练模块、实时视频检测功能和图形用户界面。

---

## 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [命令行使用](#命令行使用)
- [数据集准备](#数据集准备)
- [使用指南](#使用指南)
- [技术架构](#技术架构)


---

## 项目简介

TSN 是一种经典的视频动作识别网络架构，它通过将视频划分为多个时序段，从每个段中采样帧，并对帧级特征进行聚合，从而有效捕捉视频的时序信息。

本系统包含三个核心模块：
1. **训练系统**：完整的 TSN 模型训练流程
2. **实时检测系统**：YOLO 人体检测 + TSN 动作分类
3. **图形界面**：PyQt5 实现的可视化操作界面

---

## 主要特性

### 核心功能
- 支持预训练 CNN 骨干网络（ResNet 系列、MobileNet 等）
- 支持 UCF101（101 类动作）和 HMDB51（51 类动作）数据集
- 完整的训练和评估流程（命令行）
- 支持模型微调（跨数据集迁移学习）

### 实时检测
- YOLOv5/YOLOv8 人体检测集成
- 支持摄像头实时检测和视频文件处理
- 带检测框和标签的视频输出
- **时间平滑处理**：指数移动平均，提升预测稳定性
- **置信度阈值过滤**：仅显示高置信度预测，避免错误分类

### 图形界面 (GUI)
- 现代化的 PyQt5 界面设计
- 实时检测可视化
- **检测结果管理**：自动收集并归类检测结果，支持导出
- **模型管理**：上传、查看、删除和管理检测模型
- 非阻塞的多线程处理
- 直观的参数配置面板

---

## 项目结构

```
video_action_detection/
├── src/                          # 源代码目录
│   ├── core/                      # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py              # 配置管理
│   │   ├── model.py               # TSN 模型架构
│   │   └── utils.py               # 工具函数
│   ├── data/                      # 数据集模块
│   │   ├── __init__.py
│   │   └── datasets.py            # UCF101 和 HMDB51 数据集
│   ├── detection/                  # 实时检测模块
│   │   ├── __init__.py
│   │   ├── pipeline.py             # 主处理流程
│   │   ├── human_detector.py       # YOLO 人体检测
│   │   ├── temporal_processor.py    # 时序处理和帧缓冲
│   │   ├── action_classifier.py    # 动作分类器
│   │   ├── video_writer.py         # 视频输出和覆盖层
│   │   ├── result_collector.py     # 检测结果收集器
│   │   └── model_metadata.py      # 模型元数据提取
│   ├── training/                  # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py             # 训练器类
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   └── logger.py              # 日志记录
│   └── gui/                       # GUI 模块
│       ├── __init__.py
│       ├── main_window.py          # 主窗口
│       ├── detection_tab.py         # 实时检测标签页
│       ├── results_tab.py          # 检测结果标签页
│       ├── model_management_tab.py  # 模型管理标签页
│       ├── tensorboard_reader.py   # TensorBoard 日志读取
│       ├── training_curves_widget.py # 训练曲线组件
│       ├── video_thread.py         # 视频处理线程
│       └── resources/             # GUI 资源文件
├── scripts/                      # 独立脚本
│   ├── app.py                    # GUI 入口
│   ├── train.py                  # 训练脚本
│   ├── eval.py                   # 评估脚本
│   ├── realtime_detection.py      # 实时检测脚本
│   └── download_yolo_models.py    # YOLO 模型下载
├── data/                         # 数据目录
│   ├── ucf101/                   # UCF101 数据集
│   │   ├── UCF101/               # 视频文件
│   │   └── UCF101TrainTestSplits-RecognitionTask/ # 训练/测试划分
│   └── hmdb51/                   # HMDB51 数据集
│       └── HMDB51/               # 帧图像目录
├── outputs/                      # 输出目录
│   ├── checkpoints/               # 模型检查点
│   ├── models/                    # 模型管理目录
│   │   ├── ucf101/               # UCF101 模型
│   │   ├── hmdb51/               # HMDB51 模型
│   │   └── custom/               # 自定义模型
│   ├── results/                  # 检测结果
│   │   └── frames/               # 保存的帧图像
│   ├── logs/                     # 训练日志 (TensorBoard)
│   └── videos/                   # 输出视频
├── models/                       # 预训练模型目录 (YOLO等)
├── icon/                        # 图标资源
│   └── camara.svg
├── requirements.txt              # Python 依赖
├── CLAUDE.md                   # Claude Code 项目指南
└── README.md                   # 本文件
```

---

## 环境安装

### 系统要求
- Python 3.8 或更高版本
- CUDA 11.0+（推荐用于 GPU 加速）
- Windows/Linux/macOS

### 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 核心依赖
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

### GUI 依赖
```
PyQt5>=5.15.0
matplotlib>=3.7.0
```

### 检测依赖
```
ultralytics>=8.0.0  # YOLOv5/YOLOv8
```

---
## 快速开始

### 启动 GUI 应用

```bash
python scripts/app.py
```

GUI 界面包含：
- **实时检测标签页**：实时摄像头/视频检测
- **检测结果标签页**：查看和管理检测结果
- **模型管理标签页**：上传、查看和管理检测模型

---

## 命令行使用

除了 GUI 界面，项目还提供了完整的命令行接口，适用于自动化测试和服务器环境。

### 训练模型

```bash
# 基础训练（使用默认参数）
python scripts/train.py --dataset ucf101 --backbone resnet34 --epochs 100


# 完整训练命令（所有参数）
python scripts/train.py \
    --dataset ucf101 \
    --split_id 1 \
    --backbone resnet50 \
    --num_segments 5 \
    --frames_per_segment 5 \
    --epochs 120 \
    --batch_size 32 \
    --lr 0.001 \
    --step_size 20 \
    --gamma 0.1 \
    --num_workers 4 \
    --save_freq 5
```

**训练参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集选择 (ucf101/hmdb51) | ucf101 |
| `--split_id` | UCF101 划分 ID (1/2/3) | 1 |
| `--backbone` | 骨干网络 (resnet18/resnet34/resnet50/mobilenet_v2) | resnet50 |
| `--num_segments` | 时序段数 | 5 |
| `--frames_per_segment` | 每段帧数 | 5 |
| `--epochs` | 训练轮数 | 120 |
| `--batch_size` | 批次大小 | 32 |
| `--lr` | 学习率 | 0.001 |
| `--step_size` | 学习率衰减步长 | 15 |
| `--gamma` | 学习率衰减因子 | 0.1 |
| `--num_workers` | 数据加载线程数 | 4 |
| `--save_freq` | 保存检查点频率 | 5 |
| `--label_smoothing` | 标签平滑 (0.0-0.2，防过拟合) | 0.1 |
| `--mixup_alpha` | Mixup混合系数 (0.0禁用，0.1-0.3推荐) | 0.2 |
| `--cutmix_beta` | CutMix混合系数 (0.0禁用) | 1.0 |
| `--weight_decay` | 权重衰减/L2正则化 | 0.0005 |
| `--grad_clip` | 梯度裁剪阈值 (0.0禁用) | 1.0 |
| `--aug_type` | 增强类型 (mixup/cutmix) | mixup |
| `--aggressive_aug` | 使用激进图像增强 | True |
| `--scheduler` | 学习率调度器 (step/cosine) | cosine（推荐）|
| `--t_max` | 余弦退火最大轮数 | 120 |
| `--eta_min` | 余弦退火最小学习率 | 1e-5 |
| `--patience` | 早停耐心值 | 20 |

**微调训练：**

```bash
# 从 UCF101 模型微调到 HMDB51
python scripts/train.py \
    --dataset hmdb51 \
    --backbone resnet50 \
    --finetune outputs/checkpoints/ucf101_best.pth \
    --freeze_backbone \
    --freeze_epochs 5 \
    --finetune_lr 0.0001 \
    --epochs 30
```

### 评估模型

```bash
# 评估训练好的模型
python scripts/eval.py \
    --checkpoint outputs/checkpoints/ucf101_best.pth \
    --dataset ucf101 \
    --split_id 1 \
    --backbone resnet50 \
    --num_segments 5 \
    --frames_per_segment 5 \
    --batch_size 32
```

**评估参数说明：**

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 模型检查点路径（必需） |
| `--dataset` | 数据集选择 (ucf101/hmdb51) |
| `--split_id` | UCF101 划分 ID |
| `--backbone` | 骨干网络 |
| `--num_segments` | 时序段数 |
| `--frames_per_segment` | 每段帧数 |
| `--batch_size` | 批次大小 |

### 实时检测

```bash
# 摄像头检测
python scripts/realtime_detection.py \
    --mode webcam \
    --camera 0 \
    --checkpoint outputs/checkpoints/ucf101_best.pth \
    --yolo_model yolov8s \
    --confidence 0.5

# 视频文件检测
python scripts/realtime_detection.py \
    --mode video \
    --input test_video.mp4 \
    --checkpoint outputs/checkpoints/ucf101_best.pth \
    --yolo_model yolov8s \
    --confidence 0.5 \
    --output output_video.mp4 \
    --fps 30
```

**检测参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 检测模式 (webcam/video) | webcam |
| `--camera` | 摄像头索引 | 0 |
| `--input` | 视频文件路径（video 模式必需） | - |
| `--checkpoint` | TSN 模型检查点路径（必需） | - |
| `--yolo_model` | YOLO 模型 (yolov5s/yolov8n/yolov8s/yolov8m) | yolov5s |
| `--confidence` | 检测置信度阈值 | 0.5 |
| `--output` | 输出视频路径 | - |
| `--fps` | 输出视频帧率 | 30.0 |
| `--no_display` | 禁用显示窗口（无头模式） | False |

### 下载 YOLO 模型

```bash
# 下载所有预定义的 YOLO 模型
python scripts/download_yolo_models.py
```

下载以下模型到 `models/` 目录：
- yolov5s.pt
- yolov8n.pt
- yolov8s.pt
- yolov8m.pt

---

## 数据集准备

### UCF101 数据集

UCF101 数据集应位于 `data/ucf101/` 目录下：

```
data/ucf101/
├── UCF101/                          # 视频文件目录
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   └── ...
│   └── ...
└── UCF101TrainTestSplits-RecognitionTask/  # 训练/测试划分
    └── ucfTrainTestlist/
        ├── classInd.txt
        ├── trainlist01.txt
        ├── trainlist02.txt
        ├── trainlist03.txt
        ├── testlist01.txt
        ├── testlist02.txt
        └── testlist03.txt
```

### HMDB51 数据集

HMDB51 数据集应位于 `data/hmdb51/` 目录下，使用预提取的帧图像：

```
data/hmdb51/
└── HMDB51/                          # 帧图像目录
    ├── brush_hair/
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0/
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1/
    │   └── ... (每个视频一个目录，包含提取的帧)
    └── ... (其他动作类别目录)
```

---

## 使用指南

**模型目录结构：**
```
outputs/models/
├── ucf101/
│   └── ucf101_best.pth (默认模型)
└── custom/
    └── your_model.pth (自定义模型)
```

---

## 技术架构

### 检测流程

```
视频输入 → YOLO 人体检测 → 时序缓冲 → TSN 动作分类 → 视觉覆盖
```

### GUI 架构

```
主窗口
├── 菜单栏（文件、帮助）
├── 标题栏（设备信息）
└── 标签页
    ├── 实时检测标签页
    │   ├── 控制面板（模式、模型、参数）
    │   ├── 视频显示
    │   └── 状态显示（FPS、动作、置信度）
    ├── 检测结果标签页
    │   ├── 会话信息
    │   ├── 动作统计表
    │   ├── 动作详情和帧预览
    │   └── 导出功能
    └── 模型管理标签页
        ├── 模型列表
        ├── 模型详情
        └── 操作按钮（上传、删除、设为默认）
```

### 线程管理

- **视频线程**：在后台处理视频，通过信号更新 UI
- **结果收集**：自动收集检测结果并传递给结果标签页
- **互斥锁**：保护共享数据
- **信号机制**：线程间通信

---





