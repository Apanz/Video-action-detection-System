"""
基于TSN的视频动作识别训练模块
"""

import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.config import DataConfig, ModelConfig, TrainConfig, ROOT_DIR
from data import UCF101Dataset, HMDB51Dataset, get_train_transform, get_test_transform, MixupAugmentation, CutMixAugmentation
from core import create_model
from utils import TrainingLogger
from training.loss import SoftCrossEntropyLoss


# 设置PyTorch缓存目录
# PyTorch hub会在TORCH_CACHE_DIR下自动创建hub/和checkpoints/子目录
torch.hub.set_dir(TrainConfig.TORCH_CACHE_DIR)


class Trainer:
    """TSN模型的训练器类"""

    def __init__(self, args):
        self.args = args

        # 设置设备
        self.device = torch.device(TrainConfig.DEVICE)
        print(f"Using device: {self.device}")

        # 创建输出目录
        os.makedirs(TrainConfig.SAVE_DIR, exist_ok=True)
        os.makedirs(TrainConfig.LOG_DIR, exist_ok=True)

        # 设置增强日志记录器
        log_name = f"{args.dataset}_{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = TrainingLogger(
            log_dir=TrainConfig.LOG_DIR,
            experiment_name=log_name,
            config=self._get_config_dict(args)
        )
        self.log_dir = self.logger.get_log_dir()
        print(f"TensorBoard logs: {self.log_dir}")

        # 获取类别数
        num_classes = DataConfig.UCF101_NUM_CLASSES if args.dataset.lower() == 'ucf101' else DataConfig.HMDB51_NUM_CLASSES

        # 创建模型
        self.model = create_model(
            dataset=args.dataset,
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout=args.dropout,
            num_segments=args.num_segments,
            frames_per_segment=args.frames_per_segment
        ).to(self.device)

        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # 创建数据集
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        # 支持标签平滑的损失函数（支持软标签用于Mixup/CutMix）
        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.criterion = SoftCrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"Using label smoothing: {label_smoothing}")
        print(f"Using SoftCrossEntropyLoss (supports Mixup/CutMix soft labels)")

        # 可配置权重衰减的优化器
        weight_decay = getattr(args, 'weight_decay', TrainConfig.WEIGHT_DECAY)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=TrainConfig.MOMENTUM,
            weight_decay=weight_decay
        )
        print(f"Weight decay: {weight_decay}")

        # 学习率调度器（同时支持步进和余弦）
        scheduler_type = getattr(args, 'scheduler', 'step')
        # 保存调度器配置以备后用
        self.scheduler_config = {
            'type': scheduler_type,
            't_max': getattr(args, 't_max', args.epochs),
            'eta_min': getattr(args, 'eta_min', 1e-6),
            'step_size': args.step_size,
            'gamma': args.gamma,
            'epochs': args.epochs
        }

        if scheduler_type == 'cosine':
            t_max = getattr(args, 't_max', args.epochs)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=getattr(args, 'eta_min', 1e-6)
            )
            print(f"Using cosine annealing scheduler (T_max={t_max})")
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.step_size,
                gamma=args.gamma
            )
            print(f"Using step LR scheduler (step_size={args.step_size}, gamma={args.gamma})")

        # 训练状态
        self.current_epoch = 0
        self.best_acc = 0.0
        self.early_stopping_counter = 0
        self.save_dir = TrainConfig.SAVE_DIR

        # 梯度裁剪
        self.grad_clip = getattr(args, 'grad_clip', 0.0)
        if self.grad_clip > 0:
            print(f"Gradient clipping enabled: {self.grad_clip}")

        # Mixup/CutMix数据增强
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.0)
        self.cutmix_beta = getattr(args, 'cutmix_beta', 0.0)
        self.aug_type = getattr(args, 'aug_type', 'mixup')

        if self.mixup_alpha > 0 or self.cutmix_beta > 0:
            if self.aug_type == 'mixup':
                self.augmenter = MixupAugmentation(alpha=self.mixup_alpha, num_classes=num_classes)
                print(f"Using Mixup augmentation (alpha={self.mixup_alpha})")
            elif self.aug_type == 'cutmix':
                self.augmenter = CutMixAugmentation(beta=self.cutmix_beta, num_classes=num_classes)
                print(f"Using CutMix augmentation (beta={self.cutmix_beta})")
            self.use_augmentation = True
        else:
            self.use_augmentation = False

        # 从检查点恢复（用于评估或继续训练）
        if args.resume:
            self.load_resume_checkpoint(args.resume)
        # 微调设置
        elif args.finetune:
            self.load_finetune_checkpoint(args.finetune, args.freeze_backbone)
            self.frozen_epochs = args.freeze_epochs if args.freeze_backbone else 0
            self.initial_lr = args.lr
            self.finetune_lr = args.finetune_lr
        else:
            self.frozen_epochs = 0
            self.finetune_lr = None

        # 打印层冻结状态
        frozen_layers = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        if frozen_layers:
            print(f"Frozen layers: {len(frozen_layers)}")
        else:
            print("All layers are trainable")

    def load_finetune_checkpoint(self, checkpoint_path, freeze_backbone):
        """
        加载预训练检查点进行微调。
        """
        print(f"\n{'='*50}")
        print(f"Fine-tuning from: {checkpoint_path}")
        print(f"Freeze backbone: {freeze_backbone}")
        print(f"{'='*50}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = checkpoint['model_state_dict']

        model_state_dict = self.model.state_dict()

        # 加载权重，处理分类头的大小不匹配
        loaded_keys = []
        skipped_keys = []

        for key, pretrained_param in pretrained_state_dict.items():
            if key not in model_state_dict:
                skipped_keys.append(f"{key} (not in current model)")
                continue

            current_param = model_state_dict[key]

            # 如果大小不匹配则跳过分类头
            if 'classifier' in key and pretrained_param.shape != current_param.shape:
                print(f"  Skip {key}: pretrained {pretrained_param.shape} -> current {current_param.shape}")
                skipped_keys.append(f"{key} (size mismatch)")
                continue

            model_state_dict[key].copy_(pretrained_param)
            loaded_keys.append(key)

        self.model.load_state_dict(model_state_dict)

        print(f"Loaded {len(loaded_keys)} layers from checkpoint")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} layers:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")

        # 如果需要则冻结骨干网络
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for name, param in self.model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        print(f"Pre-trained epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Pre-trained best acc: {checkpoint.get('best_acc', 'unknown'):.2f}%")
        print()

    def load_resume_checkpoint(self, checkpoint_path):
        """
        加载检查点以恢复训练或评估。
        加载模型权重以及可选的优化器/调度器状态。
        """
        print(f"\n{'='*50}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*50}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_state_dict = checkpoint['model_state_dict']

        # 加载模型权重
        self.model.load_state_dict(checkpoint_state_dict)

        # 加载优化器和调度器（如果可用）
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")

        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")

        # 恢复调度器配置（如果可用）
        if 'scheduler_config' in checkpoint and checkpoint['scheduler_config']:
            self.scheduler_config = checkpoint['scheduler_config']

        # 如果可用则恢复训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_acc = checkpoint.get('best_acc', 0.0)

        print(f"Resumed from epoch: {self.current_epoch}")
        print(f"Checkpoint best accuracy: {self.best_acc:.2f}%")
        print()

    def unfreeze_backbone(self, new_lr=None):
        """解冻骨干，并可选择调整学习率"""
        print(f"\n{'='*50}")
        print("Unfreezing backbone for full fine-tuning")
        print(f"{'='*50}")

        for param in self.model.parameters():
            param.requires_grad = True

        if new_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate changed to {new_lr}")

            # 重新创建调度器以使用新的基准学习率
            cfg = self.scheduler_config
            remaining_epochs = cfg['epochs'] - self.current_epoch

            if cfg['type'] == 'cosine':
                # 使用剩余的epoch数作为新的T_max
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining_epochs,
                    eta_min=cfg['eta_min']
                )
                print(f"Scheduler reset: CosineAnnealingLR(T_max={remaining_epochs}, eta_min={cfg['eta_min']})")
            else:
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=cfg['step_size'],
                    gamma=cfg['gamma']
                )
                print(f"Scheduler reset: StepLR(step_size={cfg['step_size']}, gamma={cfg['gamma']})")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def create_datasets(self):
        """创建训练和验证数据集"""
        # 获取aggressive_aug配置
        aggressive_aug = getattr(self.args, 'aggressive_aug', TrainConfig.AGGRESSIVE_AUG)
        train_transform = get_train_transform(aggressive=aggressive_aug)
        val_transform = get_test_transform()

        if aggressive_aug:
            print(f"Using aggressive data augmentation")

        if self.args.dataset.lower() == 'ucf101':
            train_dataset = UCF101Dataset(
                root_dir=DataConfig.UCF101_ROOT,
                split_dir=DataConfig.UCF101_SPLITS,
                split_id=self.args.split_id,
                mode='train',
                num_segments=self.args.num_segments,
                frames_per_segment=self.args.frames_per_segment,
                transform=train_transform
            )
            val_dataset = UCF101Dataset(
                root_dir=DataConfig.UCF101_ROOT,
                split_dir=DataConfig.UCF101_SPLITS,
                split_id=self.args.split_id,
                mode='test',
                num_segments=self.args.num_segments,
                frames_per_segment=self.args.frames_per_segment,
                transform=val_transform
            )
        elif self.args.dataset.lower() == 'hmdb51':
            train_dataset = HMDB51Dataset(
                root_dir=DataConfig.HMDB51_ROOT,
                mode='train',
                num_segments=self.args.num_segments,
                frames_per_segment=self.args.frames_per_segment,
                transform=train_transform
            )
            val_dataset = HMDB51Dataset(
                root_dir=DataConfig.HMDB51_ROOT,
                mode='test',
                num_segments=self.args.num_segments,
                frames_per_segment=self.args.frames_per_segment,
                transform=val_transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        return train_dataset, val_dataset

    def _get_config_dict(self, args) -> dict:
        """获取用于记录的配置字典。"""
        return {
            'dataset': args.dataset,
            'backbone': args.backbone,
            'pretrained': args.pretrained,
            'dropout': args.dropout,
            'num_segments': args.num_segments,
            'frames_per_segment': args.frames_per_segment,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'momentum': TrainConfig.MOMENTUM,
            'weight_decay': getattr(args, 'weight_decay', TrainConfig.WEIGHT_DECAY),
            'scheduler': getattr(args, 'scheduler', 'step'),
            'step_size': args.step_size,
            'gamma': args.gamma,
            'label_smoothing': getattr(args, 'label_smoothing', 0.0),
            'grad_clip': getattr(args, 'grad_clip', 0.0),
            'mixup_alpha': getattr(args, 'mixup_alpha', 0.0),
            'cutmix_beta': getattr(args, 'cutmix_beta', 0.0),
        }

    def train_epoch(self):
        """训练一个轮次"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            # Apply Mixup/CutMix augmentation
            if self.use_augmentation:
                batch = self.augmenter(batch)
                videos = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)  # Mixed labels (one-hot)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(videos)

            # Handle mixed labels or regular labels
            if self.use_augmentation:
                loss = self.criterion(outputs, labels)
                # For mixup/cutmix, accuracy is computed on original labels
                original_labels = batch['original_labels'].to(self.device)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(original_labels).sum().item()
            else:
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

            # Log to tensorboard
            if batch_idx % 50 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx       # 累计批次
                self.logger.log_batch_metrics(
                    loss=loss.item(),
                    acc=100. * correct / total,
                    step=global_step,
                    lr=self.optimizer.param_groups[0]['lr']
                )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                videos = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, filename, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scheduler_config': getattr(self, 'scheduler_config', None),
            'best_acc': self.best_acc,
            # 添加模型配置元数据
            'config': {
                'backbone': self.model.backbone_name,
                'num_classes': self.model.num_classes,
                'num_segments': self.model.num_segments,
                'frames_per_segment': self.model.frames_per_segment,
                'dataset': getattr(self.args, 'dataset', 'ucf101'),
                'input_size': getattr(self.args, 'input_size', [224, 224])
            }
        }

        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

        if is_best:
            best_path = os.path.join(self.save_dir, f"{self.args.dataset}_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved to {best_path}")

    def log_gradients(self, step):
        """记录梯度直方图和范数。"""
        self.logger.log_gradients(self.model, step)

    def log_parameters(self, step):
        """记录参数直方图和范数。"""
        self.logger.log_parameters(self.model, step)

    def log_model_graph(self, sample_input):
        """记录模型计算图。"""
        self.logger.log_model_graph(self.model, sample_input)

    def log_epoch_metrics(self, train_loss, train_acc, val_loss, val_acc, epoch):
        """记录轮次结束时的指标。"""
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_epoch_metrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch=epoch,
            lr=lr
        )
        # 轮次后强制刷新
        self.logger.flush()

    def log_hyperparams(self, config):
        """记录超参数。"""
        self.logger.log_hyperparams(config)

    def close_logger(self):
        """关闭日志记录器并确保所有数据已写入。"""
        self.logger.close()

    def verify_logs(self):
        """验证日志文件完整性。"""
        return self.logger.verify_logs()
