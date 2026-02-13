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

# Try importing TensorBoard with better error handling
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
    Check if TensorBoard is available and return diagnostic info.

    Returns:
        Tuple of (is_available, diagnostic_message)
    """
    if HAS_TORCH_TB and _event_accumulator is not None:
        try:
            import tensorboard
            tb_version = getattr(tensorboard, '__version__', 'unknown')
            return True, f"TensorBoard is available (version {tb_version})"
        except ImportError:
            return True, "TensorBoard is available"

    # Try to get more diagnostic info
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
    Extract scalar data from TensorBoard log directory.

    Args:
        log_dir: Path to TensorBoard log directory containing events.out.tfevents files

    Returns:
        Dictionary with metric names as keys, each containing (steps, values) tuple
        Example: {
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

    # Find all event files in directory
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))

    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")

    # Load event data using EventAccumulator
    ea = _event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Extract scalar data
    tags = ea.Tags()['scalars']

    for tag in tags:
        scalar_events = ea.Scalars(tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        scalar_data[tag] = (steps, values)

    return dict(scalar_data)


def get_training_curves(log_dir: str) -> Optional[Dict]:
    """
    Extract training and validation loss/accuracy curves from log directory.

    Args:
        log_dir: Path to TensorBoard log directory

    Returns:
        Dictionary with extracted curves, or None if failed
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
        # Re-raise ImportError so caller can handle TensorBoard availability
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error extracting training curves:\n{error_details}")
        # Return None to indicate failure, but also log the full error
        return None


def validate_log_directory(log_dir: str) -> Tuple[bool, str]:
    """
    Validate that a directory contains TensorBoard log files.

    Args:
        log_dir: Path to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(log_dir):
        return False, "Directory does not exist"

    if not os.path.isdir(log_dir):
        return False, "Path is not a directory"

    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents*'))

    if not event_files:
        return False, "No TensorBoard event files found"

    return True, ""
