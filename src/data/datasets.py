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


class VideoDataset(Dataset):
    """
    Base class for video action recognition datasets
    Supports both video files (UCF101) and frame images (HMDB51)
    """

    def __init__(self, video_paths, labels, num_segments=3,
                 frames_per_segment=5, transform=None,
                 mode='train', target_frames=25):
        """
        Args:
            video_paths: List of video paths (either .avi files or frame directories)
            labels: List of corresponding labels
            num_segments: Number of temporal segments for TSN
            frames_per_segment: Number of frames to sample per segment
            transform: Image transformations
            mode: 'train' or 'test' - affects sampling strategy
            target_frames: Target number of frames in video
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.mode = mode
        self.target_frames = target_frames

        # Total frames to sample
        self.num_frames = num_segments * frames_per_segment

    def _load_video_frames(self, video_path):
        """Load frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frame indices
        if self.mode == 'train':
            # Random sampling for training
            frame_indices = sorted(random.sample(range(total_frames),
                                                 min(self.num_frames, total_frames)))
        else:
            # Uniform sampling for testing
            frame_indices = np.linspace(0, total_frames - 1,
                                        min(self.num_frames, total_frames),
                                        dtype=int).tolist()

        # Load frames
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback: repeat last frame if read fails
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create blank frame
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def _load_frame_images(self, frame_dir):
        """Load frames from pre-extracted frame images"""
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

        # Sample frame indices
        total_frames = len(frame_files)
        if total_frames == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        if self.mode == 'train':
            # Random sampling for training
            frame_indices = sorted(random.sample(range(total_frames),
                                                 min(self.num_frames, total_frames)))
        else:
            # Uniform sampling for testing
            frame_indices = np.linspace(0, total_frames - 1,
                                        min(self.num_frames, total_frames),
                                        dtype=int).tolist()

        # Load frames
        frames = []
        for idx in frame_indices:
            frame_path = os.path.join(frame_dir, frame_files[idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(np.array(frame))

        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def _sample_frames_tsn(self, frames):
        """
        TSN-style temporal segment sampling
        Divides video into segments and samples frames from each
        """
        num_available = len(frames)

        # Calculate segment boundaries
        segment_size = num_available // self.num_segments
        if segment_size < 1:
            segment_size = 1

        sampled_indices = []
        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * segment_size
            end_idx = min(start_idx + segment_size, num_available)

            if self.mode == 'train':
                # Random position within segment
                frame_idx = random.randint(start_idx, max(start_idx, end_idx - 1))
            else:
                # Center of segment
                frame_idx = (start_idx + end_idx) // 2

            sampled_indices.append(frame_idx)

        # Sample frames_per_segment frames around each segment center
        result_frames = []
        for idx in sampled_indices:
            for _ in range(self.frames_per_segment):
                # Add some randomness for training
                offset = random.randint(-2, 2) if self.mode == 'train' else 0
                actual_idx = max(0, min(num_available - 1, idx + offset))
                result_frames.append(frames[actual_idx])

        return result_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Check if video file or frame directory
        if video_path.endswith('.avi'):
            frames = self._load_video_frames(video_path)
        else:
            frames = self._load_frame_images(video_path)

        # TSN-style sampling
        frames = self._sample_frames_tsn(frames)

        # Transform frames
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # Convert to PIL Image if needed
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                transformed = self.transform(frame)
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)
        else:
            # Default transform
            default_transform = transforms.Compose([
                transforms.ToPILImage() if isinstance(frames[0], np.ndarray) else lambda x: x,
                transforms.Resize(224),
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
                 num_segments=3, frames_per_segment=5, transform=None):
        """
        Args:
            root_dir: Directory containing UCF101 video files
            split_dir: Directory containing split files
            split_id: Which split to use (1, 2, or 3)
            mode: 'train' or 'test'
            num_segments: Number of temporal segments
            frames_per_segment: Frames per segment
            transform: Image transformations
        """
        # Normalize paths for Windows compatibility
        self.root_dir = os.path.normpath(root_dir)
        split_dir = os.path.normpath(split_dir)
        self.mode = mode
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform

        # Load class index
        class_ind_file = os.path.join(split_dir, 'classInd.txt')
        self.class_names = []
        with open(class_ind_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split(' ')
                self.class_names.append(class_name)

        # Load split file
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
                    # Format: v_Basketball_g01_c01.avi 1
                    video_name = parts[0]
                    label = int(parts[1]) - 1  # Convert to 0-based
                else:
                    # Format: v_Basketball_g01_c01.avi
                    video_name = parts[0]
                    # Extract class name and find label
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
        Sample frames using proper TSN temporal segment strategy.

        TSN divides video into num_segments, then samples frames_per_segment
        from each segment. This is the core principle of Temporal Segment Networks.

        Args:
            total_frames: Total number of frames in the video
            num_frames: Total frames to sample (num_segments * frames_per_segment)

        Returns:
            List of frame indices sorted in ascending order
        """
        frame_indices = []

        # Calculate segment size
        segment_size = total_frames // self.num_segments

        for seg_idx in range(self.num_segments):
            start_idx = seg_idx * segment_size
            end_idx = start_idx + segment_size

            # For last segment, include all remaining frames
            if seg_idx == self.num_segments - 1:
                end_idx = total_frames

            # Sample frames within this segment
            seg_frame_count = self.frames_per_segment
            seg_frame_count = min(seg_frame_count, end_idx - start_idx)

            if self.mode == 'train':
                # Random sampling with temporal jitter (as in TSN paper)
                if end_idx - start_idx > seg_frame_count:
                    seg_indices = np.random.choice(
                        range(start_idx, end_idx),
                        size=seg_frame_count,
                        replace=False
                    )
                else:
                    # Not enough frames in this segment, use all with repetition
                    seg_indices = list(range(start_idx, end_idx))
                    while len(seg_indices) < seg_frame_count:
                        seg_indices.append(seg_indices[-1])
            else:
                # Uniform sampling within segment (test/validation mode)
                if end_idx - start_idx <= seg_frame_count:
                    # Use all frames in segment
                    seg_indices = list(range(start_idx, end_idx))
                else:
                    # Sample uniformly within segment
                    seg_indices = np.linspace(
                        start_idx, end_idx - 1,
                        num=seg_frame_count, dtype=int
                    ).tolist()

            frame_indices.extend(seg_indices)

        return sorted(frame_indices)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = self.num_segments * self.frames_per_segment

        # Sample frame indices using proper TSN temporal segment strategy
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
                frames.append(Image.new('RGB', (224, 224)))

        cap.release()

        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Transform frames
        if self.transform:
            transformed_frames = [self.transform(f) for f in frames]
        else:
            default_transform = transforms.Compose([
                transforms.Resize(224),
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
                 num_segments=3, frames_per_segment=5, transform=None):
        """
        Args:
            root_dir: Directory containing HMDB51 frame images
            split_dir: Directory containing split files (optional)
            mode: 'train' or 'test'
            num_segments: Number of temporal segments
            frames_per_segment: Frames per segment
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform

        # Get all action classes
        self.class_names = sorted([d for d in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, d))])

        # Collect video paths and labels
        self.video_paths = []
        self.labels = []

        if split_dir and os.path.exists(split_dir):
            # Load from split file if available
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
                # Create splits manually
                self._create_splits()
        else:
            # No splits provided, create train/val split
            self._create_splits()

        print(f"Loaded {len(self.video_paths)} {mode} samples from HMDB51")

    def _create_splits(self, train_ratio=0.8):
        """Create train/val splits from available data"""
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

        # Shuffle and split
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

    def __getitem__(self, idx):
        video_dir = self.video_paths[idx]
        label = self.labels[idx]

        # Get all frame files
        frame_files = sorted([f for f in os.listdir(video_dir)
                              if f.endswith('.jpg') or f.endswith('.png')])

        num_frames = self.num_segments * self.frames_per_segment

        # Sample frame indices (TSN style)
        if self.mode == 'train':
            # Random sampling
            frame_indices = sorted(random.sample(range(len(frame_files)),
                                                 min(num_frames, len(frame_files))))
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, len(frame_files) - 1,
                                        min(num_frames, len(frame_files)),
                                        dtype=int).tolist()

        # Load frames
        frames = []
        for idx in frame_indices:
            frame_path = os.path.join(video_dir, frame_files[idx])
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Transform frames
        if self.transform:
            transformed_frames = [self.transform(f) for f in frames]
        else:
            default_transform = transforms.Compose([
                transforms.Resize(224),
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
    Get training transformations with data augmentation

    Args:
        aggressive: If True, uses more aggressive augmentation for better regularization

    Returns:
        Transform pipeline
    """
    if aggressive:
        # More aggressive augmentation for better regularization
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Wider scale range
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Occasionally flip vertically
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger color jitter
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translation and scale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # Random erasing
        ])
    else:
        # Original moderate augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_test_transform():
    """Get test/val transformations"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class MixupAugmentation:
    """
    Mixup data augmentation for video action recognition
    Blends two videos and their labels with a random mixing coefficient

    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """

    def __init__(self, alpha=0.2, num_classes=101):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
            num_classes: Number of classes for one-hot encoding
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        """
        Apply mixup to a batch of videos and labels

        Args:
            batch: Dictionary with 'video' and 'label' keys
                   video shape: (B, T, C, H, W)
                   label shape: (B,)

        Returns:
            Mixed batch with same structure
        """
        videos = batch['video']
        labels = batch['label']
        batch_size = videos.size(0)

        if self.alpha > 0:
            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Randomly shuffle indices for second sample
        rand_index = torch.randperm(batch_size).to(videos.device)

        # Mix videos
        mixed_videos = lam * videos + (1 - lam) * videos[rand_index]

        # Mix labels using one-hot encoding
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
    CutMix data augmentation for video action recognition
    Cuts a box from one video and pastes it onto another

    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers"
    """

    def __init__(self, beta=1.0, num_classes=101, prob=0.5):
        """
        Args:
            beta: Beta distribution parameter
            num_classes: Number of classes
            prob: Probability of applying CutMix
        """
        self.beta = beta
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, batch):
        """
        Apply CutMix to a batch of videos and labels

        Args:
            batch: Dictionary with 'video' and 'label' keys
                   video shape: (B, T, C, H, W)
                   label shape: (B,)

        Returns:
            CutMix batch with same structure
        """
        videos = batch['video']
        labels = batch['label']
        batch_size = videos.size(0)

        # Only apply CutMix with certain probability
        if torch.rand(1).item() > self.prob:
            return {
                'video': videos,
                'label': labels,
                'original_labels': labels
            }

        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(batch_size).to(videos.device)

        # Get image dimensions
        _, _, _, H, W = videos.shape

        # Sample bounding box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Sample center position
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Get box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Create modified videos
        cutmix_videos = videos.clone()
        cutmix_videos[:, :, :, bby1:bby2, bbx1:bbx2] = videos[rand_index, :, :, bby1:bby2, bbx1:bbx2]

        # Calculate adjusted lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Mix labels
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
