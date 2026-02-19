"""
TensorBoard 日志增强训练记录器模块
提供可靠的日志记录功能，确保日志刷新，支持全面的指标追踪，
以及日志文件完整性检查。

"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from core.config import ROOT_DIR


class TrainingLogger:
    """
    增强型训练日志记录器，支持TensorBoard和文件日志。
    确保数据刷新以可靠地可视化训练曲线。
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化训练日志记录器。

        Args:
            log_dir: 日志的基础目录
            experiment_name: 实验/运行的名称
            config: 要保存的配置字典
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name

        # 创建实验目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(
            str(self.experiment_dir),
            flush_secs=10,  # 每10秒自动刷新
            filename_suffix='.log'
        )

        # 设置文件日志
        self._setup_file_logger()

        # 初始化计数器
        self._write_count = 0
        self._last_flush_time = time.time()
        self._flush_interval = 10  # 每10次写入刷新

        # 保存配置
        if config is not None:
            self.save_config(config)

        # 记录初始化
        self.info(f"TrainingLogger initialized for {experiment_name}")
        self.info(f"Log directory: {self.experiment_dir}")

    def _setup_file_logger(self):
        """为文本日志文件设置Python日志记录。"""
        log_file = self.experiment_dir / "training.log"

        # 配置日志记录器
        self.logger = logging.getLogger(f"TrainingLogger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # 移除现有的处理器
        self.logger.handlers.clear()

        # 文件处理器
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def info(self, message: str):
        """记录信息消息。"""
        self.logger.info(message)
        print(f"[INFO] {message}")

    def warning(self, message: str):
        """记录警告消息。"""
        self.logger.warning(message)
        print(f"[WARNING] {message}")

    def error(self, message: str):
        """记录错误消息。"""
        self.logger.error(message)
        print(f"[ERROR] {message}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int] = None,
        prefix: str = ''
    ):
        """
        将多个标量指标记录到TensorBoard。

        Args:
            metrics: 指标名称和值的字典
            step: 全局步数
            epoch: 当前轮次（可选）
            prefix: 指标名称的前缀（例如，'train/', 'val/'）
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            try:
                self.writer.add_scalar(full_name, value, step)
            except Exception as e:
                self.error(f"Failed to log metric {full_name}: {e}")

        if epoch is not None:
            self._write_count += 1
            self._auto_flush()

    def log_batch_metrics(
        self,
        loss: float,
        acc: float,
        step: int,
        lr: Optional[float] = None
    ):
        """
        记录批次级别的指标。

        Args:
            loss: 批次损失
            acc: 批次准确率
            step: 全局步数
            lr: 学习率（可选）
        """
        metrics = {
            'batch_loss': loss,
            'batch_acc': acc
        }
        if lr is not None:
            metrics['learning_rate'] = lr

        self.log_metrics(metrics, step, prefix='train/')

    def log_epoch_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch: int,
        lr: float
    ):
        """
        记录轮次结束时的指标。

        Args:
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            epoch: 当前轮次
            lr: 学习率
        """
        # 训练指标
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train/acc', train_acc, epoch)

        # 验证指标
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/acc', val_acc, epoch)

        # 学习率
        self.writer.add_scalar('lr', lr, epoch)

        self._write_count += 1
        self._auto_flush()

        # 记录到文件
        self.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={lr:.6f}"
        )

    def log_gradients(self, model: nn.Module, step: int):
        """
        记录梯度直方图和范数。

        Args:
            model: PyTorch模型
            step: 全局步数
        """
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 梯度直方图
                    self.writer.add_histogram(
                        f'gradients/{name}',
                        param.grad.data,
                        step
                    )

                    # 梯度范数
                    grad_norm = param.grad.data.norm(2).item()
                    self.writer.add_scalar(
                        f'grad_norm/{name}',
                        grad_norm,
                        step
                    )
        except Exception as e:
            self.error(f"Failed to log gradients: {e}")

        self._write_count += 1
        self._auto_flush()

    def log_parameters(self, model: nn.Module, step: int):
        """
        记录参数直方图（权重和偏置）。

        Args:
            model: PyTorch模型
            step: 全局步数
        """
        try:
            for name, param in model.named_parameters():
                self.writer.add_histogram(
                    f'parameters/{name}',
                    param.data,
                    step
                )

                # 参数范数
                param_norm = param.data.norm(2).item()
                self.writer.add_scalar(
                    f'param_norm/{name}',
                    param_norm,
                    step
                )
        except Exception as e:
            self.error(f"Failed to log parameters: {e}")

        self._write_count += 1
        self._auto_flush()

    def log_model_graph(self, model: nn.Module, inputs):
        """
        记录模型计算图。

        Args:
            model: PyTorch模型
            inputs: 示例输入张量
        """
        try:
            self.writer.add_graph(model, inputs)
            self.info("Model graph logged successfully")
            self._auto_flush()
        except Exception as e:
            self.error(f"Failed to log model graph: {e}")

    def log_hyperparams(self, config: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """
        为TensorBoard HParams插件记录超参数。

        Args:
            config: 配置字典
            metrics: 可选的初始指标
        """
        try:
            # 为tensorboard展平配置
            hparams = {}
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    hparams[key] = value
                elif isinstance(value, (list, tuple)):
                    hparams[key] = str(value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float, str, bool)):
                            hparams[f"{key}_{k}"] = v
                        else:
                            hparams[f"{key}_{k}"] = str(v)
                else:
                    hparams[key] = str(value)

            # 如果未提供则定义默认指标
            if metrics is None:
                metrics = {
                    'hparam/accuracy': 0,
                    'hparam/loss': float('inf')
                }

            self.writer.add_hparams(hparams, metrics)
            self.info("Hyperparameters logged successfully")
            self._auto_flush()

        except Exception as e:
            self.error(f"Failed to log hyperparameters: {e}")

    def log_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        epoch: int
    ):
        """
        记录混淆矩阵可视化。

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            epoch: 当前轮次
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import io

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)

            # 创建图形
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.colorbar()

            # 添加标签
            tick_marks = range(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names)

            # 添加文本
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            # 转换为图像并记录
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = plt.imread(buf)

            self.writer.add_image(f'confusion_matrix/epoch_{epoch}', image, epoch, dataformats='HWC')
            plt.close(fig)

            self._write_count += 1
            self._auto_flush()

        except Exception as e:
            self.error(f"Failed to log confusion matrix: {e}")

    def save_config(self, config: Dict[str, Any]):
        """
        将配置保存到JSON文件。

        Args:
            config: 配置字典
        """
        config_file = self.experiment_dir / "config.json"

        # 将Path对象转换为字符串
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_serializable = convert_paths(config)

        try:
            with open(config_file, 'w') as f:
                json.dump(config_serializable, f, indent=2)
            self.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.error(f"Failed to save configuration: {e}")

    def _auto_flush(self):
        """在特定次数写入后自动刷新。"""
        if self._write_count >= self._flush_interval:
            self.flush()

    def flush(self):
        """强制刷新所有缓冲区到磁盘。"""
        try:
            self.writer.flush()
            for handler in self.logger.handlers:
                handler.flush()
            self._write_count = 0
            self._last_flush_time = time.time()
        except Exception as e:
            self.error(f"Failed to flush logs: {e}")

    def close(self):
        """关闭日志记录器并确保所有数据已写入。"""
        self.info("Closing TrainingLogger...")
        self.flush()

        try:
            self.writer.close()
        except Exception as e:
            self.error(f"Error closing TensorBoard writer: {e}")

        # 关闭文件处理器
        for handler in self.logger.handlers:
            try:
                handler.close()
            except Exception:
                pass

        self.info("TrainingLogger closed")

    def verify_logs(self) -> Dict[str, Any]:
        """
        验证日志文件完整性并返回统计信息。

        Returns:
            包含验证结果的字典
        """
        result = {
            'experiment_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'status': 'unknown'
        }

        try:
            # 检查事件文件
            event_files = list(self.experiment_dir.glob('events.out.tfevents.*'))
            result['files']['event_files'] = [
                {
                    'name': f.name,
                    'size_bytes': f.stat().st_size,
                    'size_kb': f.stat().st_size / 1024
                }
                for f in event_files
            ]

            # 检查日志文件
            log_file = self.experiment_dir / 'training.log'
            if log_file.exists():
                result['files']['training_log'] = {
                    'exists': True,
                    'size_bytes': log_file.stat().st_size,
                    'size_kb': log_file.stat().st_size / 1024,
                    'line_count': len(log_file.read_text().splitlines())
                }
            else:
                result['files']['training_log'] = {'exists': False}

            # 检查配置文件
            config_file = self.experiment_dir / 'config.json'
            result['files']['config'] = {'exists': config_file.exists()}

            # 确定状态
            if event_files:
                total_event_size = sum(f['size_bytes'] for f in result['files']['event_files'])
                if total_event_size > 1000:  # 超过1KB
                    result['status'] = 'ok'
                else:
                    result['status'] = 'warning: small event files'
            else:
                result['status'] = 'error: no event files'

        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'error'

        return result

    def get_log_dir(self) -> Path:
        """获取实验日志目录。"""
        return self.experiment_dir

    def __enter__(self):
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 - 确保日志记录器正确关闭。"""
        self.close()
        return False
