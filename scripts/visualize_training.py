#!/usr/bin/env python3
"""
Training Curve Visualization Utility

Loads and plots training curves from TensorBoard logs.
Supports multiple runs comparison and report generation.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib.pyplot as plt
    import numpy as np
    import json
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install tensorboard matplotlib numpy")
    sys.exit(1)


def load_tensorboard_logs(log_dir: str) -> Dict[str, Any]:
    """
    Load TensorBoard logs from a directory.

    Args:
        log_dir: Path to the log directory

    Returns:
        Dictionary with loaded data
    """
    log_path = Path(log_dir)

    # Find event file
    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")

    # Load event data
    event_acc = EventAccumulator(str(log_path))
    event_acc.Reload()

    # Get available scalar tags
    tags = event_acc.Tags()['scalars']
    print(f"Found {len(tags)} scalar tags in logs")

    # Load all scalar data
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }

    # Load config if available
    config_file = log_path / 'config.json'
    config = None
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    return {
        'scalars': data,
        'config': config,
        'log_dir': str(log_path)
    }


def plot_training_curves(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves from loaded data.

    Args:
        data: Loaded log data
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    scalars = data['scalars']
    config = data.get('config', {})

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Training Curves - {Path(data['log_dir']).name}",
        fontsize=16,
        fontweight='bold'
    )

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if 'train/loss' in scalars:
        ax1.plot(
            scalars['train/loss']['steps'],
            scalars['train/loss']['values'],
            label='Train Loss',
            linewidth=2
        )
    if 'val/loss' in scalars:
        ax1.plot(
            scalars['val/loss']['steps'],
            scalars['val/loss']['values'],
            label='Val Loss',
            linewidth=2
        )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    if 'train/acc' in scalars:
        ax2.plot(
            scalars['train/acc']['steps'],
            scalars['train/acc']['values'],
            label='Train Acc',
            linewidth=2
        )
    if 'val/acc' in scalars:
        ax2.plot(
            scalars['val/acc']['steps'],
            scalars['val/acc']['values'],
            label='Val Acc',
            linewidth=2
        )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning rate
    ax3 = axes[1, 0]
    if 'lr' in scalars:
        ax3.plot(
            scalars['lr']['steps'],
            scalars['lr']['values'],
            label='Learning Rate',
            linewidth=2,
            color='orange'
        )
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Batch loss (if available)
    ax4 = axes[1, 1]
    if 'train/batch_loss' in scalars:
        # Downsample for plotting if too many points
        steps = scalars['train/batch_loss']['steps']
        values = scalars['train/batch_loss']['values']
        if len(steps) > 1000:
            indices = np.linspace(0, len(steps) - 1, 1000, dtype=int)
            steps = [steps[i] for i in indices]
            values = [values[i] for i in indices]

        ax4.plot(steps, values, label='Batch Loss', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Batch Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No batch loss data', ha='center', va='center')
        ax4.set_title('Batch Loss (Not Available)')

    # Add config info as text
    if config:
        config_text = f"Dataset: {config.get('dataset', 'N/A')}\n"
        config_text += f"Backbone: {config.get('backbone', 'N/A')}\n"
        config_text += f"Batch Size: {config.get('batch_size', 'N/A')}\n"
        config_text += f"LR: {config.get('lr', 'N/A')}"
        fig.text(0.02, 0.02, config_text, fontsize=9, family='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compare_runs(
    log_dirs: List[str],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare multiple training runs.

    Args:
        log_dirs: List of log directory paths
        save_path: Path to save the comparison plot
        show: Whether to display the plot
    """
    # Load all runs
    runs = []
    for log_dir in log_dirs:
        try:
            data = load_tensorboard_logs(log_dir)
            runs.append(data)
        except Exception as e:
            print(f"Warning: Could not load {log_dir}: {e}")

    if not runs:
        print("No valid runs to compare")
        return

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Run Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    for i, run in enumerate(runs):
        scalars = run['scalars']
        name = Path(run['log_dir']).name
        if 'val/acc' in scalars:
            ax1.plot(
                scalars['val/acc']['steps'],
                scalars['val/acc']['values'],
                label=name,
                linewidth=2,
                color=colors[i]
            )
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss comparison
    ax2 = axes[1]
    for i, run in enumerate(runs):
        scalars = run['scalars']
        name = Path(run['log_dir']).name
        if 'val/loss' in scalars:
            ax2.plot(
                scalars['val/loss']['steps'],
                scalars['val/loss']['values'],
                label=name,
                linewidth=2,
                color=colors[i]
            )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_statistics(data: Dict[str, Any]):
    """Print training statistics."""
    scalars = data['scalars']
    config = data.get('config', {})

    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)

    # Config info
    if config:
        print("\nConfiguration:")
        print(f"  Dataset: {config.get('dataset', 'N/A')}")
        print(f"  Backbone: {config.get('backbone', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {config.get('lr', 'N/A')}")
        print(f"  Epochs: {config.get('epochs', 'N/A')}")

    # Final metrics
    print("\nFinal Metrics:")
    if 'val/acc' in scalars:
        final_val_acc = scalars['val/acc']['values'][-1]
        best_val_acc = max(scalars['val/acc']['values'])
        print(f"  Final Val Acc: {final_val_acc:.2f}%")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")

    if 'val/loss' in scalars:
        final_val_loss = scalars['val/loss']['values'][-1]
        best_val_loss = min(scalars['val/loss']['values'])
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")

    if 'train/acc' in scalars:
        final_train_acc = scalars['train/acc']['values'][-1]
        print(f"  Final Train Acc: {final_train_acc:.2f}%")

    # Training duration (if available)
    if 'train/loss' in scalars:
        epochs = len(scalars['train/loss']['steps'])
        print(f"  Total Epochs: {epochs}")

    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize training curves from TensorBoard logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single run
  python visualize_training.py --log-dir ../outputs/logs/ucf101_resnet50_20260212_143022

  # Plot without showing (just save)
  python visualize_training.py --log-dir ../outputs/logs/exp_name --save plot.png --no-show

  # Compare multiple runs
  python visualize_training.py --compare exp1 exp2 exp3

  # Print statistics only
  python visualize_training.py --log-dir ../outputs/logs/exp_name --stats-only

  # Scan and list all available experiments
  python visualize_training.py --list
        """
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Path to specific log directory'
    )

    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple experiments (provide experiment names or paths)'
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save plot to file (e.g., plot.png, plot.pdf)'
    )

    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot'
    )

    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, no plots'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments in logs directory'
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        script_dir = Path(__file__).parent.absolute()
        logs_root = script_dir.parent / "outputs" / "logs"

        if not logs_root.exists():
            print(f"Logs directory not found: {logs_root}")
            return

        experiments = [d for d in logs_root.iterdir() if d.is_dir()]
        if not experiments:
            print("No experiment directories found")
        else:
            print(f"Available experiments in {logs_root}:")
            for exp in sorted(experiments):
                event_files = list(exp.glob('events.out.tfevents.*'))
                size = sum(f.stat().st_size for f in event_files) / 1024  # KB
                print(f"  - {exp.name} ({size:.1f} KB)")
        return

    # Compare mode
    if args.compare:
        compare_runs(args.compare, save_path=args.save, show=not args.no_show)
        return

    # Single run mode
    if not args.log_dir:
        parser.error("--log-dir is required (or use --list to see available experiments)")

    # Load data
    try:
        data = load_tensorboard_logs(args.log_dir)
    except Exception as e:
        print(f"Error loading logs: {e}")
        sys.exit(1)

    # Print statistics
    print_statistics(data)

    # Plot (unless stats-only)
    if not args.stats_only:
        plot_training_curves(data, save_path=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
