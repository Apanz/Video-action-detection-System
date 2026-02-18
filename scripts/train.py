#!/usr/bin/env python3
"""
基于TSN的视频动作识别训练脚本
用于在UCF101/HMDB51数据集上训练模型的命令行界面
"""

import sys
import argparse
from pathlib import Path

# 将src目录添加到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from training import Trainer
from core.config import TrainConfig, DataConfig, ModelConfig


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train TSN model for video action recognition')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='ucf101',
                        choices=['ucf101', 'hmdb51'],
                        help='Dataset to use (ucf101 or hmdb51)')
    parser.add_argument('--split_id', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Split ID for UCF101 (1, 2, or 3)')

    # 模型参数
    parser.add_argument('--backbone', type=str, default=ModelConfig.BACKBONE,
                        choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='CNN backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=ModelConfig.PRETRAINED,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--dropout', type=float, default=ModelConfig.DROPOUT,
                        help='Dropout rate')
    parser.add_argument('--num_segments', type=int, default=DataConfig.NUM_SEGMENTS,
                        help='Number of temporal segments')
    parser.add_argument('--frames_per_segment', type=int, default=DataConfig.FRAMES_PER_SEGMENT,
                        help='Frames per segment')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=TrainConfig.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=TrainConfig.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=TrainConfig.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--step_size', type=int, default=TrainConfig.STEP_SIZE,
                        help='Learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=TrainConfig.GAMMA,
                        help='Learning rate decay factor')
    parser.add_argument('--num_workers', type=int, default=DataConfig.NUM_WORKERS,
                        help='Number of data loading workers')

    # 检查点参数
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    # 微调参数
    parser.add_argument('--finetune', type=str, default=None,
                        help='Path to checkpoint for fine-tuning')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone during initial fine-tuning')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Number of epochs to freeze backbone')
    parser.add_argument('--finetune_lr', type=float, default=0.0001,
                        help='Learning rate for fine-tuning')

    # 早停机制
    parser.add_argument('--patience', type=int, default=TrainConfig.EARLY_STOPPING_PATIENCE,
                        help='Early stopping patience')

    # 正则化参数（用于防止过拟合）
    parser.add_argument('--weight_decay', type=float, default=TrainConfig.WEIGHT_DECAY,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=TrainConfig.LABEL_SMOOTHING,
                        help='Label smoothing (0.0 to disable)')
    parser.add_argument('--grad_clip', type=float, default=TrainConfig.GRAD_CLIP,
                        help='Gradient clipping max norm (0.0 to disable)')
    parser.add_argument('--mixup_alpha', type=float, default=TrainConfig.MIXUP_ALPHA,
                        help='Mixup alpha (0.0 to disable)')
    parser.add_argument('--cutmix_beta', type=float, default=TrainConfig.CUTMIX_BETA,
                        help='CutMix beta (0.0 to disable)')
    parser.add_argument('--aug_type', type=str, default=TrainConfig.AUG_TYPE,
                        choices=['mixup', 'cutmix'],
                        help='Augmentation type (mixup or cutmix)')
    parser.add_argument('--aggressive_aug', type=lambda x: x.lower() == 'true',
                        default=TrainConfig.AGGRESSIVE_AUG,
                        help='Use aggressive data augmentation (true/false)')
    parser.add_argument('--scheduler', type=str, default=TrainConfig.SCHEDULER_TYPE,
                        choices=['step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=TrainConfig.T_MAX,
                        help='T_max for cosine annealing scheduler')
    parser.add_argument('--eta_min', type=float, default=TrainConfig.ETA_MIN,
                        help='Minimum LR for cosine annealing scheduler')

    args = parser.parse_args()

    # 打印配置
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Split ID: {args.split_id}")
    print(f"Backbone: {args.backbone}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Dropout: {args.dropout}")
    print(f"Num Segments: {args.num_segments}")
    print(f"Frames per Segment: {args.frames_per_segment}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Scheduler: {args.scheduler}")
    if args.scheduler == 'step':
        print(f"Step Size: {args.step_size}, Gamma: {args.gamma}")
    else:
        print(f"Cosine Annealing: T_max={args.t_max}, eta_min={args.eta_min}")
    print(f"Num Workers: {args.num_workers}")
    print(f"")
    print("Regularization:")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Gradient Clipping: {args.grad_clip}")
    print(f"  Mixup Alpha: {args.mixup_alpha}")
    print(f"  CutMix Beta: {args.cutmix_beta}")
    print(f"  Aggressive Augmentation: {args.aggressive_aug}")
    if args.finetune:
        print(f"")
        print(f"Fine-tuning from: {args.finetune}")
        print(f"Freeze Backbone: {args.freeze_backbone}")
        print(f"Freeze Epochs: {args.freeze_epochs}")
        print(f"Fine-tune LR: {args.finetune_lr}")
    print("="*60)

    try:
        # 创建训练器
        trainer = Trainer(args)

        # 在开始时记录超参数
        print("\nLogging hyperparameters...")
        trainer.log_hyperparams(trainer._get_config_dict(args))

        # 记录模型图（可选，如果引起问题可以注释掉）
        try:
            # 获取一个样本批次用于记录模型图
            sample_batch = next(iter(trainer.train_loader))
            sample_input = sample_batch['video'][:1].to(trainer.device)  # 单个样本
            trainer.log_model_graph(sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        # 训练循环
        for epoch in range(trainer.current_epoch, args.epochs):
            trainer.current_epoch = epoch

            # 在frozen_epochs之后解冻骨干网络
            if epoch == trainer.frozen_epochs and trainer.frozen_epochs > 0:
                trainer.unfreeze_backbone(new_lr=trainer.finetune_lr)

            # 训练
            train_loss, train_acc = trainer.train_epoch()

            # 验证
            val_loss, val_acc = trainer.validate()

            # 使用增强的日志记录器记录轮次指标
            trainer.log_epoch_metrics(train_loss, train_acc, val_loss, val_acc, epoch)

            # 记录梯度和参数（每5个轮次一次以减少开销）
            if epoch % 5 == 0:
                trainer.log_gradients(epoch)
                trainer.log_parameters(epoch)

            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 检查是否为最佳模型
            is_best = val_acc > trainer.best_acc
            if is_best:
                trainer.best_acc = val_acc
                trainer.early_stopping_counter = 0
                print(f"*** Best model saved with acc: {val_acc:.2f}% ***")
            else:
                trainer.early_stopping_counter += 1

            # 保存检查点
            if (epoch + 1) % args.save_freq == 0:
                filename = f"{args.dataset}_epoch{epoch+1}.pth"
                trainer.save_checkpoint(filename, is_best)

            if is_best:
                trainer.save_checkpoint(f"{args.dataset}_best.pth", True)

            # 早停机制
            if trainer.early_stopping_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation accuracy: {trainer.best_acc:.2f}%")
        print("="*60)

        # 关闭前验证日志
        print("\nVerifying log files...")
        verification = trainer.verify_logs()
        print(f"Log directory: {verification['experiment_dir']}")
        print(f"Log status: {verification.get('status', 'unknown')}")

        # 关闭日志记录器
        trainer.close_logger()

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user")
        print("="*60)
        # 退出前刷新并关闭日志记录器
        try:
            trainer.logger.flush()
            verification = trainer.verify_logs()
            print(f"Log directory: {verification['experiment_dir']}")
            print(f"Log status: {verification.get('status', 'unknown')}")
            trainer.close_logger()
        except Exception as e:
            print(f"Error closing logger: {e}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error during training: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()

        # 尝试在崩溃前保存日志
        try:
            if 'trainer' in locals():
                trainer.logger.flush()
                trainer.close_logger()
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
