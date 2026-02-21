"""
增强的评估指标模块
支持精度、召回率、F1分数、Top-K准确率、混淆矩阵等
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sklearn.metrics as sk_metrics


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                    num_classes: int, top_k: List[int] = [1, 5]) -> Dict[str, float]:
    """
    计算全面的分类评估指标

    Args:
        predictions: 模型输出logits，形状(B, num_classes)
        targets: 真实标签，形状(B,)，类别索引
        num_classes: 类别总数
        top_k: 要计算的top-k值列表，例如[1, 5]

    Returns:
        包含各种指标的字典
    """
    # 获取预测类别
    _, predicted = predictions.max(dim=1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()

    # 获取预测概率（用于软指标）
    probs = torch.softmax(predictions, dim=1).cpu().numpy()

    metrics = {}

    # 1. Top-K准确率
    for k in top_k:
        if k <= num_classes:
            top_k_acc = top_k_accuracy(predictions, targets, k)
            metrics[f'top_{k}_accuracy'] = top_k_acc

    # 2. 总体准确率
    metrics['accuracy'] = (predicted == targets).mean() * 100

    # 3. Precision, Recall, F1 (macro, micro, weighted)
    precision_macro, recall_macro, f1_macro, _ = sk_metrics.precision_recall_fscore_support(
        targets, predicted, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = sk_metrics.precision_recall_fscore_support(
        targets, predicted, average='micro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = sk_metrics.precision_recall_fscore_support(
        targets, predicted, average='weighted', zero_division=0
    )

    metrics['precision_macro'] = precision_macro * 100
    metrics['recall_macro'] = recall_macro * 100
    metrics['f1_macro'] = f1_macro * 100

    metrics['precision_micro'] = precision_micro * 100
    metrics['recall_micro'] = recall_micro * 100
    metrics['f1_micro'] = f1_micro * 100

    metrics['precision_weighted'] = precision_weighted * 100
    metrics['recall_weighted'] = recall_weighted * 100
    metrics['f1_weighted'] = f1_weighted * 100

    # 4. 每类别指标
    per_class_precision, per_class_recall, per_class_f1, _ = sk_metrics.precision_recall_fscore_support(
        targets, predicted, average=None, zero_division=0
    )

    per_class_metrics = {}
    for class_id in range(num_classes):
        class_mask = targets == class_id
        if class_mask.sum() > 0:
            per_class_metrics[class_id] = {
                'precision': per_class_precision[class_id] * 100,
                'recall': per_class_recall[class_id] * 100,
                'f1': per_class_f1[class_id] * 100,
                'support': class_mask.sum()
            }

    metrics['per_class_metrics'] = per_class_metrics

    # 5. 混淆矩阵
    cm = sk_metrics.confusion_matrix(targets, predicted, labels=list(range(num_classes)))
    metrics['confusion_matrix'] = cm

    return metrics


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    计算Top-K准确率

    Args:
        predictions: 模型输出logits，形状(B, num_classes)
        targets: 真实标签，形状(B,)
        k: 考虑的前k个预测

    Returns:
        Top-K准确率（百分比）
    """
    # 获取top-k预测
    _, top_k_preds = predictions.topk(k, dim=1)

    # 检查真实标签是否在top-k预测中
    targets_reshaped = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = top_k_preds.eq(targets_reshaped).any(dim=1).sum().item()

    return (correct / targets.size(0)) * 100


def compute_class_statistics(targets: torch.Tensor, num_classes: int) -> Dict[int, Dict[str, int]]:
    """
    计算每个类别的样本统计

    Args:
        targets: 真实标签，形状(B,)
        num_classes: 类别总数

    Returns:
        每个类别的样本数统计
    """
    targets_np = targets.cpu().numpy()
    class_counts = {}

    for class_id in range(num_classes):
        count = (targets_np == class_id).sum()
        class_counts[class_id] = {'count': count}

    return class_counts


def format_metrics_report(metrics: Dict, class_names: Optional[List[str]] = None) -> str:
    """
    格式化评估指标为可读的报告

    Args:
        metrics: compute_metrics()返回的指标字典
        class_names: 可选的类别名称列表

    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EVALUATION METRICS REPORT")
    lines.append("=" * 80)

    # 总体指标
    lines.append("\n[Overall Metrics]")
    lines.append("-" * 80)

    # Top-K准确率
    for key in sorted(metrics.keys()):
        if key.startswith('top_') and key.endswith('accuracy'):
            k = key.split('_')[1]
            lines.append(f"Top-{k} Accuracy: {metrics[key]:.2f}%")

    lines.append(f"\nAccuracy: {metrics['accuracy']:.2f}%")

    # Precision/Recall/F1
    lines.append(f"\nPrecision (Macro): {metrics['precision_macro']:.2f}%")
    lines.append(f"Recall (Macro): {metrics['recall_macro']:.2f}%")
    lines.append(f"F1-Score (Macro): {metrics['f1_macro']:.2f}%")

    lines.append(f"\nPrecision (Weighted): {metrics['precision_weighted']:.2f}%")
    lines.append(f"Recall (Weighted): {metrics['recall_weighted']:.2f}%")
    lines.append(f"F1-Score (Weighted): {metrics['f1_weighted']:.2f}%")

    # 每类别指标
    if 'per_class_metrics' in metrics:
        lines.append("\n[Per-Class Metrics]")
        lines.append("-" * 80)
        lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 80)

        per_class = metrics['per_class_metrics']
        for class_id in sorted(per_class.keys()):
            class_metrics = per_class[class_id]
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"

            lines.append(
                f"{class_name:<20} "
                f"{class_metrics['precision']:>10.2f} "
                f"{class_metrics['recall']:>10.2f} "
                f"{class_metrics['f1']:>10.2f} "
                f"{class_metrics['support']:>10}"
            )

    # 混淆矩阵摘要
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        lines.append("\n[Confusion Matrix Summary]")
        lines.append("-" * 80)
        lines.append(f"Most confused pairs (top 5):")

        # 获取最常混淆的类别对（排除对角线）
        num_classes = cm.shape[0]
        confused_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    confused_pairs.append(((i, j), cm[i, j]))

        confused_pairs.sort(key=lambda x: x[1], reverse=True)

        for (true_cls, pred_cls), count in confused_pairs[:5]:
            if count > 0:
                true_name = class_names[true_cls] if class_names and true_cls < len(class_names) else f"Class_{true_cls}"
                pred_name = class_names[pred_cls] if class_names and pred_cls < len(class_names) else f"Class_{pred_cls}"
                lines.append(f"  {true_name} -> {pred_name}: {count} times")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def calculate_mean_class_accuracy(metrics: Dict) -> float:
    """
    计算平均类别准确率（忽略类别大小）

    Args:
        metrics: compute_metrics()返回的指标字典

    Returns:
        平均类别准确率（百分比）
    """
    if 'per_class_metrics' not in metrics:
        return 0.0

    per_class = metrics['per_class_metrics']
    recalls = [m['recall'] for m in per_class.values()]

    return np.mean(recalls) if recalls else 0.0


def find_best_and_worst_classes(metrics: Dict, class_names: Optional[List[str]] = None,
                                 top_n: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    找出表现最好和最差的类别

    Args:
        metrics: compute_metrics()返回的指标字典
        class_names: 可选的类别名称列表
        top_n: 返回的top N类别数

    Returns:
        (best_classes, worst_classes) 每个是[(class_name, f1_score), ...]的列表
    """
    if 'per_class_metrics' not in metrics:
        return [], []

    per_class = metrics['per_class_metrics']
    class_scores = []

    for class_id, class_metrics in per_class.items():
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
        class_scores.append((class_name, class_metrics['f1'], class_metrics['support']))

    # 按F1分数排序
    class_scores.sort(key=lambda x: x[1], reverse=True)

    best = [(name, score) for name, score, _ in class_scores[:top_n]]
    worst = [(name, score) for name, score, _ in class_scores[-top_n:]]

    return best, worst
