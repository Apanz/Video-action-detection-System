#!/usr/bin/env python3
"""
基于TSN的视频动作识别评估脚本
用于评估训练模型的命令行界面
"""

import sys
import argparse
from pathlib import Path

# 将src目录添加到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from training import Trainer
from core.config import TrainConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(trainer):
    """
    在测试集上评估模型

    Args:
        trainer: 包含已加载模型的训练器实例

    Returns:
        包含评估指标的字典
    """
    trainer.model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # 每类别准确率跟踪
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch in tqdm(trainer.val_loader, desc="Evaluating"):
            videos = batch['video'].to(trainer.device)
            labels = batch['label'].to(trainer.device)

            # 前向传播
            outputs = trainer.model(videos)
            loss = trainer.criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 每类别跟踪
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()

                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    # 计算总体指标
    avg_loss = total_loss / len(trainer.val_loader)
    avg_acc = 100. * correct / total

    # 计算每类别准确率
    per_class_acc = {}
    for class_id in class_total:
        per_class_acc[class_id] = 100. * class_correct[class_id] / class_total[class_id]

    results = {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'correct': correct,
        'total': total,
        'per_class_accuracy': per_class_acc
    }

    return results


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='Evaluate TSN model for video action recognition')

    # 检查点参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='ucf101',
                        choices=['ucf101', 'hmdb51'],
                        help='Dataset to evaluate on')
    parser.add_argument('--split_id', type=int, default=1,
                        choices=[1, 2, 3],
                        help='Split ID for UCF101 (1, 2, or 3)')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='CNN backbone architecture')
    parser.add_argument('--num_segments', type=int, default=3,
                        help='Number of temporal segments')
    parser.add_argument('--frames_per_segment', type=int, default=5,
                        help='Frames per segment')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # 验证检查点是否存在
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # 打印配置
    print("="*60)
    print("Evaluation Configuration")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Split ID: {args.split_id}")
    print(f"Backbone: {args.backbone}")
    print(f"Num Segments: {args.num_segments}")
    print(f"Frames per Segment: {args.frames_per_segment}")
    print("="*60)

    # 加载检查点以获取配置
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=TrainConfig.DEVICE)

    # 如果可用则提取数据集信息
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        best_acc = checkpoint.get('best_acc', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        print(f"Checkpoint epoch: {epoch}")
        print(f"Checkpoint best acc: {best_acc:.2f}%" if best_acc != 'Unknown' else "Checkpoint best acc: Unknown")

    # 创建用于评估的训练器
    print("\nCreating trainer...")

    # 为评估创建最小参数
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        split_id=args.split_id,
        backbone=args.backbone,
        pretrained=False,  # 评估时不需要预训练
        dropout=args.dropout,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        epochs=1,  # 评估时不使用
        batch_size=args.batch_size,
        lr=0.001,  # 不使用
        step_size=15,  # 不使用
        gamma=0.1,  # 不使用
        num_workers=args.num_workers,
        save_freq=1,  # 不使用
        resume=args.checkpoint,  # 从检查点恢复
        finetune=None,
        freeze_backbone=False,
        freeze_epochs=0,
        finetune_lr=0.0001,
        patience=100  # 不使用
    )

    try:
        trainer = Trainer(eval_args)

        # 评估模型
        print("\nEvaluating model...")
        results = evaluate_model(trainer)

        # 打印结果
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.2f}%")
        print(f"Correct: {results['correct']}/{results['total']}")
        print("="*60)

        # 打印每类别准确率
        print("\nPer-Class Accuracy:")
        print("-"*60)
        for class_id in sorted(results['per_class_accuracy'].keys()):
            acc = results['per_class_accuracy'][class_id]
            print(f"Class {class_id}: {acc:.2f}%")
        print("-"*60)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
