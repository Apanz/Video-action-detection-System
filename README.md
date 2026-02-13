# 视频行为识别系统

基于深度学习的视频行为识别系统 - 本科毕业设计项目

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
- [模型微调](#模型微调)
- [性能对比](#性能对比)
- [常见问题](#常见问题)
- [技术架构](#技术架构)
- [更新日志](#更新日志)
- [参考文献](#参考文献)
- [作者与许可](#作者与许可)

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
├── src/                    # 源代码目录
│   ├── core/              # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py       # 配置管理
│   │   ├── model.py       # TSN 模型架构
│   │   └── utils.py       # 工具函数
│   ├── data/              # 数据集模块
│   │   ├── __init__.py
│   │   └── datasets.py    # UCF101 和 HMDB51 数据集
│   ├── detection/          # 实时检测模块
│   │   ├── __init__.py
│   │   ├── pipeline.py     # 主处理流程
│   │   ├── human_detector.py    # YOLO 人体检测
│   │   ├── temporal_processor.py # 时序处理
│   │   ├── action_classifier.py  # 动作分类
│   │   ├── video_writer.py     # 视频输出
│   │   ├── result_collector.py # 检测结果收集器
│   │   └── model_metadata.py   # 模型元数据提取器
│   ├── training/          # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py     # 训练器类
│   └── gui/               # GUI 模块
│       ├── __init__.py
│       ├── main_window.py  # 主窗口
│       ├── detection_tab.py # 检测标签页
│       ├── results_tab.py  # 检测结果标签页
│       ├── model_management_tab.py  # 模型管理标签页
│       └── video_thread.py  # 视频处理线程
├── scripts/               # 独立脚本
│   ├── app.py            # GUI 入口
│   ├── train.py          # 训练脚本
│   ├── eval.py           # 评估脚本
│   ├── realtime_detection.py # 实时检测脚本
│   └── download_yolo_models.py # YOLO 模型下载
├── configs/              # 配置文件目录
├── data/                # 数据目录
│   ├── ucf101/          # UCF101 数据集
│   ├── hmdb51/          # HMDB51 数据集
│   └── COCO/            # COCO 数据集
├── outputs/              # 输出目录
│   ├── checkpoints/      # 模型检查点
│   ├── models/           # 模型管理目录
│   │   ├── ucf101/      # UCF101 模型
│   │   ├── hmdb51/      # HMDB51 模型
│   │   └── custom/      # 自定义模型
│   ├── results/          # 检测结果
│   │   └── frames/      # 保存的帧图像
│   ├── logs/             # 训练日志
│   └── videos/           # 输出视频
├── models/               # 模型目录
├── icon/                # 图标资源
│   └── camara.svg
├── requirements.txt      # Python 依赖
└── README.md            # 本文件
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

### 训练参数调优指南

#### 欠拟合（训练准确率 << 验证准确率）

**症状**：训练准确率低于50%，验证准确率反而更高
**原因**：正则化过强或学习率不合适

**解决方案**：
```bash
# 1. 关闭所有正则化（保守配置）
python scripts/train.py --dataset ucf101 --backbone resnet34 --epochs 120 \
--lr 0.001 --mixup_alpha 0.0 --label_smoothing 0.0 \
--weight_decay 0.0001 --aggressive_aug false

# 2. 适度正则化（平衡配置）
python scripts/train.py --dataset ucf101 --backbone resnet34 --epochs 120 \
--lr 0.001 --mixup_alpha 0.2 --label_smoothing 0.1 \
--weight_decay 0.0005 --aggressive_aug false

# 3. 使用余弦退火调度器（更平滑的学习率衰减）
python scripts/train.py --dataset ucf101 --backbone resnet34 --epochs 120 \
--lr 0.001 --scheduler cosine --t_max 50 \
--mixup_alpha 0.0 --label_smoothing 0.0 --aggressive_aug false

# 4. 增加学习率
python scripts/train.py --dataset ucf101 --backbone resnet34 --epochs 120 \
--lr 0.002 --scheduler cosine --t_max 50 \
--mixup_alpha 0.0 --label_smoothing 0.0 --aggressive_aug false
```

#### 参数调优建议

| 参数 | 欠拟合时 | 过拟合时 | 最佳平衡 |
|------|----------|----------|----------|
| `--lr` | 0.002 | 0.0005 | 0.001 |
| `--label_smoothing` | 0.0 | 0.15 | 0.05 |
| `--mixup_alpha` | 0.0 | 0.3 | 0.1-0.2 |
| `--weight_decay` | 0.0001 | 0.001 | 0.0005 |
| `--aggressive_aug` | false | true | false |
| `--step_size` | 30 | 10 | 20 或使用cosine |
| `--scheduler` | cosine | step | cosine（推荐） |
| `--dropout` | 0.3 | 0.7 | 0.5 |

#### 常见训练问题诊断

**问题1：训练准确率低于50%，验证准确率反而更高**
- **原因**：数据增强过于激进 + 正则化参数过大
- **解决**：`--aggressive_aug false --mixup_alpha 0.0 --label_smoothing 0.0`

**问题2：训练准确率很高(>95%)，验证准确率很低(<60%)**
- **原因**：严重过拟合
- **解决**：`--mixup_alpha 0.3 --label_smoothing 0.15 --weight_decay 0.001`

**问题3：训练损失不下降**
- **原因**：学习率过小或过大
- **解决**：尝试 `--lr 0.01` 或 `--lr 0.0005`

**问题4：训练后期准确率波动大**
- **原因**：学习率衰减过快
- **解决**：`--scheduler cosine --t_max 50` 或增加 `--step_size`

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

这将下载以下模型到 `models/` 目录：
- yolov5s.pt
- yolov8n.pt
- yolov8s.pt
- yolov8m.pt

### 使用 TensorBoard 监控训练

```bash
# 启动 TensorBoard（默认端口 6006）
tensorboard --logdir outputs/logs

# 指定端口
tensorboard --logdir outputs/logs --port 6007
```

然后在浏览器中访问 `http://localhost:6006` 查看训练曲线。

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

### COCO 数据集

COCO 数据集目录 `data/COCO/` 用于 YOLO 人体检测模型的预训练。

---

## 使用指南

### GUI 使用

#### 实时检测标签页

**功能特性：**
- 实时视频显示（摄像头或视频文件）
- 模式选择：网页摄像头 / 视频文件
- 模型配置：检查点选择、YOLO 模型选择、置信度调整
- 输出选项：保存视频、显示检测覆盖层、**记录检测结果**
- 实时统计：FPS、检测到的动作、置信度、处理进度

**使用流程：**
1. 选择检测模式（摄像头或视频文件）
2. 选择训练好的模型检查点
3. 配置检测参数（YOLO 模型、检测置信度、输出帧率）
4. （可选）勾选"记录检测结果"以自动收集检测结果
5. 点击"开始检测"
6. 查看实时检测结果

#### 检测结果标签页

**功能特性：**
- **会话信息**：显示检测会话的详细信息（ID、视频源、时间范围、总帧数）
- **动作统计表**：按帧数排序显示所有检测到的动作类别
  - 帧数、占比、平均置信度
  - 点击行查看动作详情
- **动作详情**：查看特定动作的保存帧预览
  - 每个动作最多保存 10 个代表性帧
  - 显示帧索引、时间戳、置信度
  - 支持查看完整帧图像
- **导出功能**：
  - 导出 JSON：完整的检测结果和统计信息
  - 导出 CSV：表格格式的动作统计
  - 导出特定动作的帧图像
- **清除结果**：清除当前显示的所有结果

**数据存储策略：**
- 每个动作类别最多保存 10 个代表性帧（节省磁盘空间）
- 采用置信度优先的采样策略
- 帧图像保存至 `outputs/results/frames/{session_id}/`
- 统计数据包含总帧数、检测到的帧数、每个动作的详细信息

#### 模型管理标签页

**功能特性：**
- **模型列表**：显示所有可用模型，按类别分组
  - UCF101 模型
  - HMDB51 模型
  - 自定义模型
- **模型详情**：查看模型的完整信息
  - 文件名、路径、大小、修改时间
  - 骨干网络、数据集、时序参数
  - 训练信息（轮数、最佳精度）
  - 模型验证状态
- **上传模型**：
  - 选择模型文件（.pth/.pt）
  - 选择目标类别（UCF101/HMDB51/自定义）
  - 自动复制到相应目录
- **删除模型**：删除不需要的模型文件（带确认对话框）
- **设为默认**：设置特定数据集的默认模型
- **加载模型**：直接加载选中的模型到检测标签页

**模型目录结构：**
```
outputs/models/
├── ucf101/
│   └── ucf101_best.pth (默认模型)
├── hmdb51/
│   └── hmdb51_best.pth (默认模型)
└── custom/
    └── your_model.pth (自定义模型)
```

---

## 模型微调

### 使用方法

使用命令行进行微调训练：
1. 选择目标数据集（如 HMDB51）
2. 使用 `--finetune` 参数指定预训练模型
3. 配置微调参数（冻结骨干网络、冻结轮数等）
4. 运行训练脚本

### 微调参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 预训练检查点 | 微调源模型路径 | 无 |
| 冻结骨干网络 | 初始阶段冻结骨干网络 | False |
| 冻结轮数 | 冻结训练的轮数 | 5 |
| 解冻学习率 | 解冻后的学习率 | 0.0001 |

### 训练策略

- **阶段 1（第 1-5 轮）**：冻结骨干网络，仅训练分类层，使用较大学习率
- **阶段 2（第 6+ 轮）**：解冻所有层，使用较小学习率进行全网络微调

> **注意**：微调后的模型只能用于新数据集，不能用于原数据集评估。

---

## 性能对比

### YOLO 模型选择

| 模型 | mAP | FPS (CPU) | FPS (GPU) | 推荐场景 |
|------|-----|-----------|-----------|----------|
| YOLOv5s | 37.3% | 6.3 | 98.0 | 基准 |
| YOLOv8n | 28.4% | 4.5 | 80.0 | 超实时 |
| YOLOv8s | 44.9% | 10.0 | 173.0 | ⭐ 推荐 |
| YOLOv8m | 50.2% | 7.5 | 120.0 | 高精度 |

### 骨干网络对比

| 骨干网络 | 参数量 | 速度 | 精度 | 推荐场景 |
|----------|--------|------|------|----------|
| ResNet18 | ~11M | 快 | 中等 | 快速实验 |
| ResNet34 | ~21M | 中 | 较好 | 平衡 |
| ResNet50 | ~25M | 慢 | 高 | 精度优先 |
| MobileNetV2 | ~3.5M | 很快 | 中等 | 移动端 |

### 实验结果参考

- **UCF101 数据集**（ResNet18 骨干）：
  - 单个 split: 约 70-80%
  - 三个 split 平均: 约 75-82%

- **HMDB51 数据集**（ResNet18 骨干）：
  - 从零开始训练：约 40-50%
  - 使用 UCF101 微调：约 45-55%（提升约 5%）

---

## 常见问题

### GUI 相关问题

**GUI 无法启动**
```bash
# 确保安装了 PyQt5
pip install PyQt5

# 检查 matplotlib 后端
python -c "import matplotlib; print(matplotlib.get_backend())"
```

**视频显示异常**
- 降低视频分辨率
- 使用 GPU 加速
- 减小检测缓冲区大小

### 检测结果相关问题

**记录检测结果功能在哪里？**
- 在"实时检测"标签页的"输出选项"中
- 勾选"记录检测结果"复选框
- 检测完成后，切换到"检测结果"标签页查看

**检测结果保存位置？**
- 帧图像：`outputs/results/frames/{session_id}/`
- 可导出 JSON/CSV 到任意位置
- 每个检测会话有唯一的 session_id

**为什么某些动作的帧数很少？**
- 每个动作类别最多保存 10 个代表性帧
- 采用置信度优先的采样策略
- 这是设计特性，用于节省磁盘空间

**如何导出检测结果？**
- 在"检测结果"标签页
- 点击"导出JSON"或"导出CSV"按钮
- 选择保存位置

### 模型管理相关问题

**如何添加自己的模型？**
- 切换到"模型管理"标签页
- 点击"上传模型"按钮
- 选择模型文件（.pth/.pt）
- 选择目标类别（UCF101/HMDB51/自定义）

**如何删除不需要的模型？**
- 在模型列表中选择模型
- 点击"删除"按钮
- 确认删除操作

**如何设置默认模型？**
- 选择要设为默认的模型
- 点击"设为默认"按钮
- 模型会被复制为 `{dataset}_best.pth`

**模型详情显示"无效"？**
- 模型文件可能损坏
- 模型格式不被支持
- 尝试重新训练或获取模型

### 训练相关问题

**训练功能在哪里？**
- GUI 中已移除训练标签页
- 使用命令行训练：`python scripts/train.py --help`
- 详细的训练参数请参考"命令行使用"部分

**GPU 内存不足**
- 减小批次大小：`--batch_size 16`
- 选择更小的骨干网络（如 ResNet18）

**训练太慢**
- 减少训练轮数进行快速测试：`--epochs 50`
- 使用更小的骨干网络：`--backbone resnet18`

### 检测相关问题

**YOLO 模型未找到**
```bash
# 安装 ultralytics（推荐）
pip install ultralytics
```

**检测效果不佳**
- 调整检测置信度阈值
- 确保光照条件良好
- 尝试不同的 YOLO 模型
- 检查视频分辨率是否合适

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

## 更新日志

### v2.2 (2026-02-08)

- ✨ **新增检测结果管理模块**：自动收集和归类检测结果
  - 按动作类别组织检测结果
  - 每类最多保存 10 个代表性帧（节省磁盘空间）
  - 支持 JSON/CSV 导出
  - 帧预览和详情查看
- ✨ **新增模型管理模块**：完整的模型管理系统
  - 查看所有可用模型（UCF101、HMDB51、自定义）
  - 上传新模型文件
  - 删除不需要的模型
  - 查看模型详细信息（架构、参数、训练信息）
  - 设置默认模型
  - 直接加载模型到检测标签页
- 🔧 **GUI 重构**：移除训练标签页，专注于检测功能
  - 训练功能仍可通过命令行使用（`scripts/train.py`）
  - 优化标签页布局和用户体验
- 📁 **新增目录结构**：
  - `outputs/models/ucf101/` - UCF101 模型目录
  - `outputs/models/hmdb51/` - HMDB51 模型目录
  - `outputs/models/custom/` - 自定义模型目录
  - `outputs/results/frames/` - 检测结果帧图像

### v2.1 (2026-01-29)
- 🔥 **修复TSN时间采样问题** - 实现标准的时间分段采样（而非随机采样）
- 📊 **提升至25帧输入** - 使用完整的UCF101视频帧数（5段×5帧=25帧）
- 🏗️ **升级至ResNet34** - 使用更强的骨干网络追求最高准确率
- 📈 **优化训练参数** - 增加训练轮数至120、使用余弦退火调度器
- 🔇 **添加时间平滑处理** - 指数移动平均，提升预测稳定性
- 🎯 **添加置信度阈值过滤** - 仅显示高置信度预测
- 🧹 **移除调试日志** - 提升推理性能

### v2.0 (2026-01-22)

- ✨ 完整的 PyQt5 图形界面
- 🏗️ 项目结构重组（`src/` 目录）
- 🔧 统一的配置系统
- 📊 训练实时图表
- 🎥 实时视频检测 UI
- 🧵 多线程处理
- 📝 更新文档

### v1.1 (2026-01-22)

- ✨ 实时视频检测功能
- 🔧 集成 YOLOv5 人体检测
- 📹 支持摄像头和视频文件输入
- 💾 添加视频输出功能

### v1.0

- 🎯 完成 TSN 训练系统
- 📊 实现模型评估流程
- 🔧 添加微调功能
- 📝 完成基础文档



