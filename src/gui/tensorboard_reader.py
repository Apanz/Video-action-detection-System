"""
TensorBoard log reader for GUI visualization
Uses PyTorch-native TensorBoard reader (no TensorFlow dependency)
"""

import os
import sys
import glob
import importlib
from collections import defaultdict
from typing import Dict, Optional, Tuple

# 尝试导入TensorBoard并进行更好的错误处理
HAS_TORCH_TB = False
TORCH_TB_ERROR = None
_event_accumulator = None

try:
    from tensorboard.backend.event_processing import event_accumulator as _ea
    _event_accumulator = _ea
    HAS_TORCH_TB = True
except ImportError as e:
    TORCH_TB_ERROR = str(e)


def check_tensorboard_available() -> Tuple[bool, str]:
    """
    检查TensorBoard是否可用并返回诊断信息。

    Returns:
        (is_available, diagnostic_message)元组
    """
    if HAS_TORCH_TB and _event_accumulator is not None:
        try:
            import tensorboard
            tb_version = getattr(tensorboard, '__version__', 'unknown')
            return True, f"TensorBoard is available (version {tb_version})"
        except ImportError:
            return True, "TensorBoard is available"

    # 尝试获取更多诊断信息
    tb_spec = importlib.util.find_spec("tensorboard")
    if tb_spec is None:
        return False, (
            "TensorBoard is not installed.\n\n"
            "Install with: pip install tensorboard\n\n"
            "Current Python executable:\n"
            f"  {sys.executable}\n\n"
            "Current sys.path:\n" + "\n".join(f"  {p}" for p in sys.path[:5])
        )

    return False, (
        f"TensorBoard found at:\n  {tb_spec.origin}\n\n"
        f"But import failed with error:\n  {TORCH_TB_ERROR}\n\n"
        "This may indicate a corrupted installation.\n"
        "Try reinstalling: pip install --force-reinstall tensorboard"
    )


def extract_scalars_from_log_dir(log_dir: str) -> Dict[str, Tuple[list, list]]:
    """
    从TensorBoard日志目录提取标量数据。

    Args:
        log_dir: 包含events.out.tfevents文件的TensorBoard日志目录路径

    Returns:
        以指标名称为键的字典，每个包含(steps, values)元组
        示例: {
            'train/loss': ([0, 1, 2, ...], [2.5, 2.1, 1.8, ...]),
            'val/acc': ([0, 1, 2, ...], [0.45, 0.52, 0.58, ...])
        }
    """
    if not HAS_TORCH_TB:
        available, message = check_tensorboard_available()
        raise ImportError(
            f"TensorBoard package is required but not available.\n\n"
            f"{message}\n\n"
            f"Please install TensorBoard:\n"
            f"  pip install tensorboard"
        )

    scalar_data = defaultdict(lambda: ([], []))

    # 查找目录中的所有事件文件
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))

    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")

    # 使用EventAccumulator加载事件数据
    ea = _event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 提取标量数据
    tags = ea.Tags()['scalars']

    for tag in tags:
        scalar_events = ea.Scalars(tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        scalar_data[tag] = (steps, values)

    return dict(scalar_data)


def get_training_curves(log_dir: str) -> Optional[Dict]:
    """
    从日志目录提取训练和验证的损失/精度曲线。

    Args:
        log_dir: TensorBoard日志目录路径

    Returns:
        包含提取曲线的字典，失败时为None
        {
            'train_loss': (steps, values),
            'train_acc': (steps, values),
            'val_loss': (steps, values),
            'val_acc': (steps, values)
        }
    """
    try:
        all_scalars = extract_scalars_from_log_dir(log_dir)

        result = {}
        for key, tag in [('train_loss', 'train/loss'),
                         ('train_acc', 'train/acc'),
                         ('val_loss', 'val/loss'),
                         ('val_acc', 'val/acc')]:
            if tag in all_scalars:
                result[key] = all_scalars[tag]

        return result if result else None

    except ImportError as e:
        # 重新抛出ImportError以便调用者可以处理TensorBoard可用性
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error extracting training curves:\n{error_details}")
        # 返回None以指示失败，但也记录完整错误
        return None


def validate_log_directory(log_dir: str) -> Tuple[bool, str]:
    """
    验证目录是否包含TensorBoard日志文件。

    Args:
        log_dir: 要检查的路径

    Returns:
        (is_valid, error_message)元组
    """
    if not os.path.exists(log_dir):
        return False, "Directory does not exist"

    if not os.path.isdir(log_dir):
        return False, "Path is not a directory"

    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))

    if not event_files:
        return False, "No TensorBoard event files found"

    return True, ""
