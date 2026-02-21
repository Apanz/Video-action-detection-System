#!/usr/bin/env python3
"""
Ablation Study Runner

Simple script to run ablation experiments sequentially for TSN video action detection.
Records experiment configurations and results to a CSV file for easy comparison.

Usage:
    # Run all experiments (100 epochs each)
    python scripts/run_ablation.py

    # Test mode (5 epochs each, for quick verification)
    python scripts/run_ablation.py --test

    # View results
    cat outputs/ablation_results.csv
"""

import subprocess
import csv
import time
import argparse
import re
from datetime import datetime
from pathlib import Path


# Base configuration (shared by all experiments)
BASE_ARGS = [
    "python", "scripts/train.py",
    "--dataset", "ucf101",
    "--split_id", "1",
    "--pretrained",
    "--epochs", "100",
    "--batch_size", "32",
    "--lr", "0.001",
    "--scheduler", "cosine",
    "--t_max", "100",
    "--eta_min", "1e-5",
    "--dropout", "0.5",
    "--weight_decay", "0.0005",
    "--label_smoothing", "0.1",
    "--grad_clip", "1.0",
    "--aggressive_aug", "true",
    "--num_workers", "0",  # Windows compatibility
    "--save_freq", "5",
    "--patience", "20"
]


# Experiment configurations
# 11 ablation experiments testing backbone, temporal settings, and augmentation
EXPERIMENTS = [
    # Group 1: Backbone (4 experiments)
    {
        "id": 1,
        "name": "Exp1_baseline",
        "group": "backbone",
        "backbone": "resnet34",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 2,
        "name": "Exp2_backbone_resnet18",
        "group": "backbone",
        "backbone": "resnet18",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 3,
        "name": "Exp3_backbone_resnet50",
        "group": "backbone",
        "backbone": "resnet50",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 4,
        "name": "Exp4_backbone_mobilenet",
        "group": "backbone",
        "backbone": "mobilenet_v2",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    # Group 2: Temporal (4 experiments)
    {
        "id": 5,
        "name": "Exp5_temporal_3x3",
        "group": "temporal",
        "backbone": "resnet34",
        "num_segments": 3,
        "frames_per_segment": 3,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 6,
        "name": "Exp6_temporal_5x5",
        "group": "temporal",
        "backbone": "resnet34",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 7,
        "name": "Exp7_temporal_7x3",
        "group": "temporal",
        "backbone": "resnet34",
        "num_segments": 7,
        "frames_per_segment": 3,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 8,
        "name": "Exp8_temporal_3x7",
        "group": "temporal",
        "backbone": "resnet34",
        "num_segments": 3,
        "frames_per_segment": 7,
        "mixup_alpha": 0.2,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    # Group 3: Augmentation (3 experiments)
    {
        "id": 9,
        "name": "Exp9_no_mixup",
        "group": "augmentation",
        "backbone": "resnet34",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.0,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
    {
        "id": 10,
        "name": "Exp10_cutmix",
        "group": "augmentation",
        "backbone": "resnet34",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.0,
        "cutmix_beta": 1.0,
        "aug_type": "cutmix"
    },
    {
        "id": 11,
        "name": "Exp11_mixup_strong",
        "group": "augmentation",
        "backbone": "resnet34",
        "num_segments": 5,
        "frames_per_segment": 5,
        "mixup_alpha": 0.4,
        "cutmix_beta": 0.0,
        "aug_type": "mixup"
    },
]


# CSV column headers
CSV_HEADERS = [
    "exp_id",
    "name",
    "group",
    "backbone",
    "num_segments",
    "frames_per_segment",
    "mixup_alpha",
    "cutmix_beta",
    "aug_type",
    "status",
    "start_time",
    "end_time",
    "duration_hours",
    "best_val_acc",
    "best_epoch",
    "error_msg"
]


def init_csv(csv_path: Path) -> None:
    """Initialize CSV file with headers if it doesn't exist."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        print(f"[INFO] Created CSV file: {csv_path}")
    else:
        print(f"[INFO] Using existing CSV file: {csv_path}")


def update_csv_record(csv_path: Path, record: dict) -> None:
    """Update a single experiment record in the CSV file."""
    records = []

    # Read existing records
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)

    # Find and update the matching record
    updated = False
    for i, r in enumerate(records):
        if r['exp_id'] == record['exp_id']:
            records[i] = record
            updated = True
            break

    # If not found, append
    if not updated:
        records.append(record)

    # Write back
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(records)


def create_record(exp_config: dict) -> dict:
    """Create a CSV record dictionary from experiment config."""
    return {
        "exp_id": exp_config["id"],
        "name": exp_config["name"],
        "group": exp_config["group"],
        "backbone": exp_config["backbone"],
        "num_segments": exp_config["num_segments"],
        "frames_per_segment": exp_config["frames_per_segment"],
        "mixup_alpha": exp_config["mixup_alpha"],
        "cutmix_beta": exp_config["cutmix_beta"],
        "aug_type": exp_config["aug_type"],
        "status": "pending",
        "start_time": "",
        "end_time": "",
        "duration_hours": "",
        "best_val_acc": "",
        "best_epoch": "",
        "error_msg": ""
    }


def build_command(exp_config: dict, test_mode: bool = False) -> list:
    """Build the command list for running an experiment."""
    cmd = BASE_ARGS.copy()

    # Update epochs for test mode
    if test_mode:
        # Find and replace the epochs value
        for i, arg in enumerate(cmd):
            if arg == "--epochs" and i + 1 < len(cmd):
                cmd[i + 1] = "5"
                break

    # Add experiment-specific arguments
    cmd.extend([
        "--backbone", exp_config["backbone"],
        "--num_segments", str(exp_config["num_segments"]),
        "--frames_per_segment", str(exp_config["frames_per_segment"]),
        "--mixup_alpha", str(exp_config["mixup_alpha"]),
        "--cutmix_beta", str(exp_config["cutmix_beta"]),
        "--aug_type", exp_config["aug_type"]
    ])

    return cmd


def parse_training_output(output: str) -> tuple:
    """Parse training output to extract best accuracy and epoch."""
    best_acc = None
    best_epoch = None

    # Look for "Best validation accuracy" line
    match = re.search(r'Best validation accuracy: ([\d.]+)%', output)
    if match:
        best_acc = float(match.group(1))

    # Look for best model save messages to get best epoch
    # Format: "*** Best model saved with acc: XX.XX% ***"
    best_acc_lines = re.findall(r'\*\*\* Best model saved with acc: ([\d.]+)% \*\*\*', output)
    if best_acc_lines:
        # Get the last occurrence (final best)
        pass  # We already have best_acc from the final summary

    # Try to find epoch info from early stopping or completion
    match_epoch = re.search(r'Early stopping triggered after (\d+) epochs', output)
    if match_epoch:
        best_epoch = int(match_epoch.group(1))

    match_epoch2 = re.search(r'Training completed!', output)
    if match_epoch2:
        # Look for the last epoch number printed
        epochs = re.findall(r'Epoch (\d+)/', output)
        if epochs:
            best_epoch = int(epochs[-1])

    return best_acc, best_epoch


def run_experiment(exp_config: dict, csv_path: Path, test_mode: bool = False, verbose: bool = False) -> dict:
    """Run a single experiment and update CSV record."""
    exp_id = exp_config["id"]
    name = exp_config["name"]

    print("\n" + "=" * 80)
    print(f"[EXPERIMENT {exp_id}] {name}")
    print("=" * 80)
    print(f"Group: {exp_config['group']}")
    print(f"Backbone: {exp_config['backbone']}")
    print(f"Segments: {exp_config['num_segments']} x {exp_config['frames_per_segment']} "
          f"= {exp_config['num_segments'] * exp_config['frames_per_segment']} frames")
    print(f"Augmentation: {exp_config['aug_type']} (α={exp_config['mixup_alpha']}, β={exp_config['cutmix_beta']})")
    print(f"Test Mode: {test_mode}")
    print("=" * 80)

    # Initialize record
    record = create_record(exp_config)
    record["status"] = "running"
    record["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_csv_record(csv_path, record)

    # Build command
    cmd = build_command(exp_config, test_mode)
    print(f"\n[CMD] {' '.join(cmd)}\n")

    # Run experiment
    start_time = time.time()
    output = ""
    returncode = 0
    error_msg = ""

    try:
        if verbose:
            # Stream output to console
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            output = ""  # Output was streamed directly to console
            returncode = result.returncode
        else:
            # Capture output
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            output = result.stdout + result.stderr
            returncode = result.returncode

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Exception occurred: {error_msg}")
        returncode = 1

    # Calculate duration
    duration_seconds = time.time() - start_time
    duration_hours = duration_seconds / 3600

    # Parse results
    best_acc, best_epoch = parse_training_output(output)

    # Update record
    record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["duration_hours"] = f"{duration_hours:.2f}"

    if returncode == 0 and best_acc is not None:
        record["status"] = "completed"
        record["best_val_acc"] = f"{best_acc:.2f}"
        record["best_epoch"] = str(best_epoch) if best_epoch else ""
        print(f"\n[SUCCESS] Experiment {exp_id} completed!")
        print(f"[RESULT] Best Val Acc: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"[TIME] Duration: {duration_hours:.2f} hours")
    else:
        record["status"] = "failed"
        if best_acc is not None:
            record["best_val_acc"] = f"{best_acc:.2f}"
            record["best_epoch"] = str(best_epoch) if best_epoch else ""
            record["error_msg"] = f"Partial completion (returncode={returncode})"
        else:
            record["error_msg"] = error_msg or f"returncode={returncode}"
        print(f"\n[FAILED] Experiment {exp_id} failed!")
        print(f"[ERROR] {record['error_msg']}")

        # Show last 50 lines of output on failure (if not verbose)
        if not verbose and output:
            lines = output.strip().split('\n')
            if len(lines) > 0:
                print(f"\n[LAST 50 LINES OF OUTPUT]")
                print("-" * 80)
                for line in lines[-50:]:
                    print(line)
                print("-" * 80)

    update_csv_record(csv_path, record)

    return record


def print_summary(csv_path: Path) -> None:
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    if not csv_path.exists():
        print("[INFO] No results file found yet.")
        return

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        print("[INFO] No experiment records found.")
        return

    # Group by status
    completed = [r for r in records if r['status'] == 'completed']
    failed = [r for r in records if r['status'] == 'failed']
    running = [r for r in records if r['status'] == 'running']
    pending = [r for r in records if r['status'] == 'pending']

    print(f"Total Experiments: {len(records)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Running: {len(running)}")
    print(f"  Pending: {len(pending)}")

    if completed:
        print(f"\n{'ID':<4} {'Name':<25} {'Group':<15} {'Best Acc':<10} {'Epoch':<6}")
        print("-" * 80)
        for r in completed:
            acc = r['best_val_acc'] or 'N/A'
            epoch = r['best_epoch'] or 'N/A'
            print(f"{r['exp_id']:<4} {r['name']:<25} {r['group']:<15} {acc:<10} {epoch:<6}")

    if failed:
        print(f"\n[FAILED EXPERIMENTS]")
        for r in failed:
            print(f"  {r['exp_id']}: {r['name']} - {r['error_msg']}")

    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run ablation study for TSN video action detection'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: run 5 epochs per experiment for quick verification'
    )
    parser.add_argument(
        '--exp-id',
        type=int,
        default=None,
        help='Run only a specific experiment (1-11)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Custom CSV path (default: outputs/ablation_results.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full training output during experiments'
    )

    args = parser.parse_args()

    # Set CSV path
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(__file__).parent.parent / "outputs" / "ablation_results.csv"

    # Initialize CSV
    init_csv(csv_path)

    # Filter experiments if --exp-id is specified
    experiments_to_run = EXPERIMENTS
    if args.exp_id is not None:
        experiments_to_run = [e for e in EXPERIMENTS if e['id'] == args.exp_id]
        if not experiments_to_run:
            print(f"[ERROR] Invalid experiment ID: {args.exp_id}")
            print(f"[INFO] Valid IDs: 1-{len(EXPERIMENTS)}")
            return

    # Print mode info
    mode = "TEST MODE (5 epochs)" if args.test else "FULL MODE (100 epochs)"
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY RUNNER - {mode}")
    print(f"{'='*80}")
    print(f"Experiments to run: {len(experiments_to_run)}")
    print(f"Results CSV: {csv_path}")
    print(f"{'='*80}")

    # Run experiments
    for i, exp_config in enumerate(experiments_to_run, 1):
        print(f"\n[PROGRESS] Experiment {i}/{len(experiments_to_run)}")
        run_experiment(exp_config, csv_path, test_mode=args.test, verbose=args.verbose)

    # Print summary
    print_summary(csv_path)

    print(f"\n[INFO] Results saved to: {csv_path}")


if __name__ == '__main__':
    main()
