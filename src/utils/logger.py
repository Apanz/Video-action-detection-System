"""
Enhanced Training Logger Module for TensorBoard Logging

Provides robust logging with guaranteed flush, comprehensive metrics tracking,
and log file integrity checking.
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
    Enhanced training logger with TensorBoard and file logging.
    Ensures data flush for reliable training curve visualization.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment/run
            config: Configuration dictionary to save
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name

        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            str(self.experiment_dir),
            flush_secs=10,  # Auto-flush every 10 seconds
            filename_suffix='.log'
        )

        # Set up file logging
        self._setup_file_logger()

        # Initialize counters
        self._write_count = 0
        self._last_flush_time = time.time()
        self._flush_interval = 10  # Flush every 10 writes

        # Save configuration
        if config is not None:
            self.save_config(config)

        # Log initialization
        self.info(f"TrainingLogger initialized for {experiment_name}")
        self.info(f"Log directory: {self.experiment_dir}")

    def _setup_file_logger(self):
        """Set up Python logging for text log files."""
        log_file = self.experiment_dir / "training.log"

        # Configure logger
        self.logger = logging.getLogger(f"TrainingLogger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        print(f"[INFO] {message}")

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        print(f"[WARNING] {message}")

    def error(self, message: str):
        """Log error message."""
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
        Log multiple scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            epoch: Current epoch (optional)
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
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
        Log batch-level metrics.

        Args:
            loss: Batch loss
            acc: Batch accuracy
            step: Global step number
            lr: Learning rate (optional)
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
        Log end-of-epoch metrics.

        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            epoch: Current epoch
            lr: Learning rate
        """
        # Training metrics
        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train/acc', train_acc, epoch)

        # Validation metrics
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/acc', val_acc, epoch)

        # Learning rate
        self.writer.add_scalar('lr', lr, epoch)

        self._write_count += 1
        self._auto_flush()

        # Log to file
        self.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={lr:.6f}"
        )

    def log_gradients(self, model: nn.Module, step: int):
        """
        Log gradient histograms and norms.

        Args:
            model: PyTorch model
            step: Global step number
        """
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Gradient histogram
                    self.writer.add_histogram(
                        f'gradients/{name}',
                        param.grad.data,
                        step
                    )

                    # Gradient norm
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
        Log parameter histograms (weights and biases).

        Args:
            model: PyTorch model
            step: Global step number
        """
        try:
            for name, param in model.named_parameters():
                self.writer.add_histogram(
                    f'parameters/{name}',
                    param.data,
                    step
                )

                # Parameter norm
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
        Log model computation graph.

        Args:
            model: PyTorch model
            inputs: Sample input tensor
        """
        try:
            self.writer.add_graph(model, inputs)
            self.info("Model graph logged successfully")
            self._auto_flush()
        except Exception as e:
            self.error(f"Failed to log model graph: {e}")

    def log_hyperparams(self, config: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """
        Log hyperparameters for TensorBoard HParams plugin.

        Args:
            config: Configuration dictionary
            metrics: Optional initial metrics
        """
        try:
            # Flatten config for tensorboard
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

            # Define default metrics if not provided
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
        Log confusion matrix visualization.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            epoch: Current epoch
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import io

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Create figure
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.colorbar()

            # Add labels
            tick_marks = range(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=90)
            plt.yticks(tick_marks, class_names)

            # Add text
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

            # Convert to image and log
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
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / "config.json"

        # Convert Path objects to strings
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
        """Automatically flush after a certain number of writes."""
        if self._write_count >= self._flush_interval:
            self.flush()

    def flush(self):
        """Force flush all buffers to disk."""
        try:
            self.writer.flush()
            for handler in self.logger.handlers:
                handler.flush()
            self._write_count = 0
            self._last_flush_time = time.time()
        except Exception as e:
            self.error(f"Failed to flush logs: {e}")

    def close(self):
        """Close logger and ensure all data is written."""
        self.info("Closing TrainingLogger...")
        self.flush()

        try:
            self.writer.close()
        except Exception as e:
            self.error(f"Error closing TensorBoard writer: {e}")

        # Close file handlers
        for handler in self.logger.handlers:
            try:
                handler.close()
            except Exception:
                pass

        self.info("TrainingLogger closed")

    def verify_logs(self) -> Dict[str, Any]:
        """
        Verify log file integrity and return statistics.

        Returns:
            Dictionary with verification results
        """
        result = {
            'experiment_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'status': 'unknown'
        }

        try:
            # Check for event files
            event_files = list(self.experiment_dir.glob('events.out.tfevents.*'))
            result['files']['event_files'] = [
                {
                    'name': f.name,
                    'size_bytes': f.stat().st_size,
                    'size_kb': f.stat().st_size / 1024
                }
                for f in event_files
            ]

            # Check for log file
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

            # Check for config file
            config_file = self.experiment_dir / 'config.json'
            result['files']['config'] = {'exists': config_file.exists()}

            # Determine status
            if event_files:
                total_event_size = sum(f['size_bytes'] for f in result['files']['event_files'])
                if total_event_size > 1000:  # More than 1KB
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
        """Get the experiment log directory."""
        return self.experiment_dir

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures logger is closed properly."""
        self.close()
        return False
