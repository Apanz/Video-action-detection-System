"""
Dataset classes for UCF101 and HMDB51
Implements TSN-style temporal segment sampling
"""

import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from core.config import DataConfig


class VideoDataset(Dataset):
    """
    支持视频文件（UCF101）和帧图像（HMDB51）
    """

    def __init__(self, video_paths, labels, num_segments=None,
                 frames_per_segment=None, transform=None,
                 mode='train', target_frames=25):
        """
        Args:
            video_paths: List of video paths (either .avi files or frame directories)
            labels: List of corresponding labels
            num_segments: Number of temporal segments for TSN (default: from DataConfig)
            frames_per_segment: Number of frames to sample per segment (default: from DataConfig)
            transform: Image transformations
            mode: 'train' or 'test' - affects sampling strategy
            target_frames: Target number of frames in video
        """
        # Use config defaults if not specified
        if num_segments is None:
            num_segments = DataConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

        self.video_paths = video_paths
        self.labels = labels
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.mode = mode
        self.target_frames = target_frames

        # 采样的总帧数
        self.num_frames = num_segments * frames_per_segment

    def _load_video_frames(self, video_path):
        """从视频文件（ucf101）加载帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 采样帧索引
        if self.mode == 'train':
            # 训练时随机采样
            frame_indices = sorted(random.sample(range(total_frames),
                                                 min(self.num_frames, total_frames)))
        else:
            # 测试时均匀采样
            frame_indices = np.linspace(0, total_frames - 1,
                                        min(self.num_frames, total_frames),
                                        dtype=int).tolist()

        # 加载帧
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()     # ret(是否成功提取): bool, frame: np.ndarray
            if ret:
                # 将BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 回退：读取失败时重复最后一帧
                if frames:
                    frames.append(frames[-1])
                else:
                    # 创建空白帧
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        # 需要时用最后一帧填充
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def _load_frame_images(self, frame_dir):
        """从预提取的帧图像加载（HMDB51）"""
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

        # 采样帧索引
        total_frames = len(frame_files)
        if total_frames == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        if self.mode == 'train':
            # 训练时随机采样
            frame_indices = sorted(random.sample(range(total_frames),
                                                 min(self.num_frames, total_frames)))
        else:
            # 测试时均匀采样
            frame_indices = np.linspace(0, total_frames - 1,
                                        min(self.num_frames, total_frames),
                                        dtype=int).tolist()

        # 加载帧
        frames = []
        for idx in frame_indices:
            frame_path = os.path.join(frame_dir, frame_files[idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))

        # 需要时用最后一帧填充
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def _sample_frames_tsn(self, frames):
        """
        TSN风格的时序片段采样
        将视频分成num_segments个片段，从每个片段中采样frames_per_segment帧
        """
        num_available = len(frames)

        # 计算片段大小
        segment_size = num_available // self.num_segments

        sampled_indices = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * segment_size
            end_idx = start_idx + segment_size

            # 对于最后一个片段，包含所有剩余帧
            if seg_idx == self.num_segments - 1:
                end_idx = num_available

            # 在此片段内采样frames_per_segment帧
            seg_frame_count = self.frames_per_segment
            seg_frame_count = min(seg_frame_count, end_idx - start_idx)

            if self.mode == 'train':
                # 带时序抖动的随机采样（训练模式）
                if end_idx - start_idx > seg_frame_count:
                    seg_indices = random.sample(
                        range(start_idx, end_idx),
                        seg_frame_count
                    )
                else:
                    # 片段中没有足够的帧，使用所有帧并重复
                    seg_indices = list(range(start_idx, end_idx))
                    while len(seg_indices) < seg_frame_count:
                        seg_indices.append(seg_indices[-1])
            else:
                # 片段内均匀采样（验证/测试模式）
                if end_idx - start_idx <= seg_frame_count:
                    # 使用片段中的所有帧
                    seg_indices = list(range(start_idx, end_idx))
                else:
                    # 在片段内均匀采样
                    seg_indices = [
                        int(start_idx + (end_idx - start_idx) * i / seg_frame_count)
                        for i in range(seg_frame_count)
                    ]

            sampled_indices.extend(seg_indices)

        # 按索引排序并返回帧
        sampled_indices = sorted(sampled_indices)
        return [frames[i] for i in sampled_indices]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 检查是视频文件还是帧目录
        if video_path.endswith('.avi'):
            frames = self._load_video_frames(video_path)
        else:
            frames = self._load_frame_images(video_path)

        # TSN风格采样
        frames = self._sample_frames_tsn(frames)

        # 转换帧
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # 需要时转换为PIL图像(torchvision.transforms 仅接受 PIL Image 或 Tensor)
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                transformed = self.transform(frame)
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)
        else:
            # 默认转换
            default_transform = transforms.Compose([
                transforms.ToPILImage() if isinstance(frames[0], np.ndarray) else lambda x: x,
                transforms.Resize(DataConfig.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transformed_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                transformed = default_transform(frame)
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)

        return {
            'video': frames,  # Shape: (T, C, H, W)
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': video_path
        }


class UCF101Dataset(Dataset):
    """
    UCF101 Dataset for video action recognition
    Uses video files (.avi) directly
    """

    def __init__(self, root_dir, split_dir, split_id=1, mode='train',
                 num_segments=None, frames_per_segment=None, transform=None):
        """
        Args:
            root_dir: Directory containing UCF101 video files
            split_dir: Directory containing split files
            split_id: Which split to use (1, 2, or 3)
            mode: 'train' or 'test'
            num_segments: Number of temporal segments (default: from DataConfig)
            frames_per_segment: Frames per segment (default: from DataConfig)
            transform: Image transformations
        """
        # Use config defaults if not specified
        if num_segments is None:
            num_segments = DataConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

        # 标准化路径以兼容Windows
        self.root_dir = os.path.normpath(root_dir)
        split_dir = os.path.normpath(split_dir)
        self.mode = mode
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform

        # 加载类别索引
        class_ind_file = os.path.join(split_dir, 'classInd.txt')
        self.class_names = []
        with open(class_ind_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ')
                self.class_names.append(class_name)

        # 加载分割文件
        if mode == 'train':
            split_file = os.path.join(split_dir, f'trainlist{split_id:02d}.txt')
        else:
            split_file = os.path.join(split_dir, f'testlist{split_id:02d}.txt')

        self.video_paths = []
        self.labels = []

        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if mode == 'train':
                    # 格式: v_Basketball_g01_c01.avi 1
                    video_name = parts[0]
                    label = int(parts[1]) - 1  # 转换为从0开始
                else:
                    # 格式: v_Basketball_g01_c01.avi
                    video_name = parts[0]
                    # 提取类别名称并查找标签
                    class_name = video_name.split('/')[0]
                    try:
                        label = self.class_names.index(class_name)
                    except ValueError:
                        continue

                video_path = os.path.join(root_dir, video_name)
                if os.path.exists(video_path):
                    self.video_paths.append(video_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.video_paths)} {mode} samples from UCF101")

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames_tsn(self, total_frames: int, num_frames: int):
        """
        使用正确的TSN时序片段策略采样帧。
        TSN将视频分成num_segments个片段，然后从每个片段中采样frames_per_segment帧。

        Args:
            total_frames: 视频中的总帧数
            num_frames: 要采样的总帧数（num_segments * frames_per_segment）

        Returns:
            按升序排列的帧索引列表
        """
        frame_indices = []

        # 计算片段大小
        segment_size = total_frames // self.num_segments

        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * segment_size
            end_idx = start_idx + segment_size

            # 对于最后一个片段，包含所有剩余帧
            if seg_idx == self.num_segments - 1:
                end_idx = total_frames

            # 在此片段内采样帧
            seg_frame_count = self.frames_per_segment
            seg_frame_count = min(seg_frame_count, end_idx - start_idx)

            if self.mode == 'train':
                # 带时序抖动的随机采样（如TSN论文中所述）
                if end_idx - start_idx > seg_frame_count:
                    seg_indices = np.random.choice(
                        range(start_idx, end_idx),
                        size=seg_frame_count,
                        replace=False
                    )
                else:
                    # 此片段中没有足够的帧，使用所有帧并重复
                    seg_indices = list(range(start_idx, end_idx))
                    while len(seg_indices) < seg_frame_count:
                        seg_indices.append(seg_indices[-1])
            else:
                # 片段内均匀采样（测试/验证模式）
                if end_idx - start_idx <= seg_frame_count:
                    # 使用片段中的所有帧
                    seg_indices = list(range(start_idx, end_idx))
                else:
                    # 在片段内均匀采样
                    seg_indices = np.linspace(
                        start_idx, end_idx - 1,
                        num=seg_frame_count, dtype=int
                    ).tolist()

            frame_indices.extend(seg_indices)

        return sorted(frame_indices)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 加载视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = self.num_segments * self.frames_per_segment

        # 使用正确的TSN时序片段策略采样帧索引
        frame_indices = self._sample_frames_tsn(total_frames, num_frames)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            elif frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (DataConfig.INPUT_SIZE, DataConfig.INPUT_SIZE)))

        cap.release()

        # 需要时填充
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # 转换帧
        if self.transform:
            transformed_frames = [self.transform(f) for f in frames]
        else:
            default_transform = transforms.Compose([
                transforms.Resize(DataConfig.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transformed_frames = [default_transform(f) for f in frames]

        return {
            'video': torch.stack(transformed_frames),
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': video_path
        }


class HMDB51Dataset(Dataset):
    """
    HMDB51 Dataset for video action recognition
    Uses pre-extracted frame images
    """

    def __init__(self, root_dir, split_dir=None, mode='train',
                 num_segments=None, frames_per_segment=None, transform=None):
        """
        Args:
            root_dir: Directory containing HMDB51 frame images
            split_dir: Directory containing split files (optional)
            mode: 'train' or 'test'
            num_segments: Number of temporal segments (default: from DataConfig)
            frames_per_segment: Frames per segment (default: from DataConfig)
            transform: Image transformations
        """
        # Use config defaults if not specified
        if num_segments is None:
            num_segments = DataConfig.NUM_SEGMENTS
        if frames_per_segment is None:
            frames_per_segment = DataConfig.FRAMES_PER_SEGMENT

        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform

        # 获取所有动作类别
        self.class_names = sorted([d for d in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, d))])

        # 收集视频路径和标签
        self.video_paths = []
        self.labels = []

        if split_dir and os.path.exists(split_dir):
            # 如果存在分割文件则从其加载
            split_file = os.path.join(split_dir, f'{mode}.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        video_path, label = line.strip().split(' ')
                        full_path = os.path.join(root_dir, video_path)
                        if os.path.exists(full_path):
                            self.video_paths.append(full_path)
                            self.labels.append(int(label))
            else:
                # 手动创建分割
                self._create_splits()
        else:
            # 未提供分割，创建训练/验证分割
            self._create_splits()

        print(f"Loaded {len(self.video_paths)} {mode} samples from HMDB51")

    def _create_splits(self, train_ratio=0.8):
        """从可用数据创建训练/验证分割"""
        all_videos = []
        all_labels = []

        for class_id, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for video_name in os.listdir(class_dir):
                    video_path = os.path.join(class_dir, video_name)
                    if os.path.isdir(video_path):
                        all_videos.append(video_path)
                        all_labels.append(class_id)

        # 打乱并分割
        indices = list(range(len(all_videos)))
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if self.mode == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        self.video_paths = [all_videos[i] for i in selected_indices]
        self.labels = [all_labels[i] for i in selected_indices]

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames_tsn(self, total_frames: int, num_frames: int):
        """
        使用正确的TSN时序片段策略采样帧。
        TSN将视频分成num_segments个片段，然后从每个片段中采样frames_per_segment帧。

        Args:
            total_frames: 视频中的总帧数
            num_frames: 要采样的总帧数（num_segments * frames_per_segment）

        Returns:
            按升序排列的帧索引列表
        """
        frame_indices = []

        # 计算片段大小
        segment_size = total_frames // self.num_segments

        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * segment_size
            end_idx = start_idx + segment_size

            # 对于最后一个片段，包含所有剩余帧
            if seg_idx == self.num_segments - 1:
                end_idx = total_frames

            # 在此片段内采样帧
            seg_frame_count = self.frames_per_segment
            seg_frame_count = min(seg_frame_count, end_idx - start_idx)

            if self.mode == 'train':
                # 带时序抖动的随机采样（如TSN论文中所述）
                if end_idx - start_idx > seg_frame_count:
                    seg_indices = random.sample(
                        range(start_idx, end_idx),
                        seg_frame_count
                    )
                else:
                    # 此片段中没有足够的帧，使用所有帧并重复
                    seg_indices = list(range(start_idx, end_idx))
                    while len(seg_indices) < seg_frame_count:
                        seg_indices.append(seg_indices[-1])
            else:
                # 片段内均匀采样（测试/验证模式）
                if end_idx - start_idx <= seg_frame_count:
                    # 使用片段中的所有帧
                    seg_indices = list(range(start_idx, end_idx))
                else:
                    # 在片段内均匀采样
                    seg_indices = [
                        int(start_idx + (end_idx - start_idx) * i / seg_frame_count)
                        for i in range(seg_frame_count)
                    ]

            frame_indices.extend(seg_indices)

        return sorted(frame_indices)

    def __getitem__(self, idx):
        video_dir = self.video_paths[idx]
        label = self.labels[idx]

        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(video_dir)
                              if f.endswith('.jpg') or f.endswith('.png')])

        num_frames = self.num_segments * self.frames_per_segment
        total_frames = len(frame_files)

        # 使用TSN时序片段策略采样帧索引
        frame_indices = self._sample_frames_tsn(total_frames, num_frames)

        # 加载帧
        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(video_dir, frame_files[frame_idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        # 需要时填充
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # 转换帧
        if self.transform:
            transformed_frames = [self.transform(f) for f in frames]
        else:
            default_transform = transforms.Compose([
                transforms.Resize(DataConfig.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transformed_frames = [default_transform(f) for f in frames]

        return {
            'video': torch.stack(transformed_frames),
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': video_dir
        }


def get_train_transform(aggressive=True):
    """
    获取带数据增强的训练转换

    Args:
        aggressive: 如果为True，使用更激进的增强以获得更好的正则化效果

    Returns:
        转换流水线
    """
    if aggressive:
        # 更激进的增强以获得更好的正则化效果
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 更宽的缩放范围
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # 偶尔垂直翻转
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 更强的颜色抖动
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移和缩放
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # 随机擦除
        ])
    else:
        # 原始适度增强
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_test_transform():
    """获取测试/验证转换"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class MixupAugmentation:
    """
    用于视频动作识别的Mixup数据增强
    使用随机混合系数混合两个视频及其标签

    参考: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """

    def __init__(self, alpha=0.2, num_classes=101):
        """
        Args:
            alpha: 混合系数的Beta分布参数
            num_classes: 独热编码的类别数
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        """
        对一批视频和标签应用mixup

        Args:
            batch: 包含'video'和'label'键的字典
                   video形状: (B, T, C, H, W)
                   label形状: (B,)

        Returns:
            具有相同结构的混合批次
        """
        videos = batch['video']
        labels = batch['label']
        batch_size = videos.size(0)

        if self.alpha > 0:
            # 从Beta分布采样混合系数
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 随机打乱第二个样本的索引
        rand_index = torch.randperm(batch_size).to(videos.device)

        # 混合视频
        mixed_videos = lam * videos + (1 - lam) * videos[rand_index]

        # 使用独热编码混合标签
        labels_one_hot = torch.zeros(batch_size, self.num_classes).to(videos.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_one_hot_rand = torch.zeros(batch_size, self.num_classes).to(videos.device)
        labels_one_hot_rand.scatter_(1, labels[rand_index].unsqueeze(1), 1.0)

        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot_rand

        return {
            'video': mixed_videos,
            'label': mixed_labels,
            'lam': lam,
            'rand_index': rand_index,
            'original_labels': labels
        }


class CutMixAugmentation:
    """
    用于视频动作识别的CutMix数据增强
    从一个视频中剪切一个框并将其粘贴到另一个视频上

    参考: "CutMix: Regularization Strategy to Train Strong Classifiers"
    """

    def __init__(self, beta=1.0, num_classes=101, prob=0.5):
        """
        Args:
            beta: Beta分布参数
            num_classes: 类别数
            prob: 应用CutMix的概率
        """
        self.beta = beta
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, batch):
        """
        对一批视频和标签应用CutMix

        Args:
            batch: 包含'video'和'label'键的字典
                   video形状: (B, T, C, H, W)
                   label形状: (B,)

        Returns:
            具有相同结构的CutMix批次
        """
        videos = batch['video']
        labels = batch['label']
        batch_size = videos.size(0)

        # 仅以特定概率应用CutMix
        if torch.rand(1).item() > self.prob:
            return {
                'video': videos,
                'label': labels,
                'original_labels': labels
            }

        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(batch_size).to(videos.device)

        # 获取图像维度
        _, _, _, H, W = videos.shape

        # 采样边界框
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 采样中心位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 获取框坐标
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 创建修改后的视频
        cutmix_videos = videos.clone()
        cutmix_videos[:, :, :, bby1:bby2, bbx1:bbx2] = videos[rand_index, :, :, bby1:bby2, bbx1:bbx2]

        # 根据实际框面积计算调整后的lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # 混合标签
        labels_one_hot = torch.zeros(batch_size, self.num_classes).to(videos.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_one_hot_rand = torch.zeros(batch_size, self.num_classes).to(videos.device)
        labels_one_hot_rand.scatter_(1, labels[rand_index].unsqueeze(1), 1.0)

        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot_rand

        return {
            'video': cutmix_videos,
            'label': mixed_labels,
            'lam': lam,
            'rand_index': rand_index,
            'original_labels': labels
        }
