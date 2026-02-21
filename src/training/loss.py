"""
损失函数模块，支持软标签（用于Mixup/CutMix增强）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """
    支持软标签和硬标签的交叉熵损失

    用于Mixup/CutMix数据增强，这些增强会产生软标签（混合的one-hot向量）
    而不是硬标签（类别索引）。

    软标签格式：(B, C) 其中C是类别数，每个元素是[0,1]范围内的概率
    硬标签格式：(B,) 每个元素是类别索引

    损失计算：
        L = -sum(soft_labels * log_softmax(logits))

    对于硬标签，该损失与nn.CrossEntropyLoss等价
    """

    def __init__(self, label_smoothing=0.0, reduction='mean'):
        """
        Args:
            label_smoothing: 标签平滑参数（0.0表示不使用）
            reduction: 'none', 'mean', 或 'sum'
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        计算软交叉熵损失

        Args:
            logits: 模型输出，形状(B, C)，未归一化的对数概率
            targets: 目标标签
                     - 软标签: (B, C) one-hot向量或概率分布
                     - 硬标签: (B,) 类别索引

        Returns:
            损失值
        """
        # 检测是软标签还是硬标签
        if targets.dim() == 1:
            # 硬标签: 转换为one-hot
            num_classes = logits.size(1)
            soft_labels = F.one_hot(targets, num_classes=num_classes).float()

            # 应用标签平滑
            if self.label_smoothing > 0:
                soft_labels = soft_labels * (1 - self.label_smoothing) + \
                              self.label_smoothing / num_classes
        elif targets.dim() == 2:
            # 软标签: 已经是one-hot或概率分布
            soft_labels = targets.float()

            # 应用标签平滑（如果尚未应用）
            if self.label_smoothing > 0:
                num_classes = logits.size(1)
                uniform_prior = self.label_smoothing / num_classes
                soft_labels = soft_labels * (1 - self.label_smoothing) + uniform_prior
        else:
            raise ValueError(f"Unsupported targets shape: {targets.shape}. "
                           f"Expected (B,) for hard labels or (B, C) for soft labels.")

        # 计算log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # 计算损失: -(soft_labels * log_probs).sum(dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1)

        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class CrossEntropyLossWithSoftLabels(nn.Module):
    """
    兼容nn.CrossEntropyLoss的包装器，自动检测并处理软标签

    这是一个便捷包装器，可以无缝替换现有的nn.CrossEntropyLoss。
    它会自动检测输入是软标签还是硬标签，并相应地计算损失。
    """

    def __init__(self, label_smoothing=0.0, reduction='mean'):
        """
        Args:
            label_smoothing: 标签平滑参数
            reduction: 'none', 'mean', 或 'sum'
        """
        super(CrossEntropyLossWithSoftLabels, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.soft_loss = SoftCrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction
        )
        self.hard_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction=reduction
        )

    def forward(self, logits, targets):
        """
        自动检测并处理软/硬标签

        Args:
            logits: 模型输出，形状(B, C)
            targets: 目标标签，可以是软标签或硬标签

        Returns:
            损失值
        """
        # 检测是软标签还是硬标签
        if targets.dim() == 2:
            # 软标签
            return self.soft_loss(logits, targets)
        else:
            # 硬标签
            return self.hard_loss(logits, targets)
