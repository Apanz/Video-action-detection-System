#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorBoard Log Verification Utility

Scans log directories and checks file integrity.
Reports empty/corrupted log files and displays statistics.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def scan_log_directory(log_dir: Path) -> dict:
    """
    Scan a log directory and analyze its contents.

    Args:
        log_dir: Path to log directory

    Returns:
        Dictionary with scan results
    """
    result = {
        'path': str(log_dir),
        'exists': log_dir.exists(),
        'is_dir': log_dir.is_dir() if log_dir.exists() else False,
        'files': [],
        'total_size_bytes': 0,
        'issues': []
    }

    if not result['exists']:
        result['issues'].append("Directory does not exist")
        return result

    if not result['is_dir']:
        result['issues'].append("Path is not a directory")
        return result

    # Scan for files
    event_files = []
    other_files = []

    for item in log_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size
            result['total_size_bytes'] += size

            file_info = {
                'name': item.name,
                'size_bytes': size,
                'size_formatted': format_size(size)
            }

            if item.name.startswith('events.out.tfevents.'):
                event_files.append(file_info)
            else:
                other_files.append(file_info)

    result['event_files'] = event_files
    result['other_files'] = other_files

    # Check for issues
    if not event_files:
        result['issues'].append("No TensorBoard event files found")
    else:
        # Check for suspiciously small event files
        for event_file in event_files:
            if event_file['size_bytes'] < 1000:  # Less than 1KB
                result['issues'].append(
                    f"Event file '{event_file['name']}' is suspiciously small "
                    f"({event_file['size_formatted']}) - may be empty"
                )
            elif event_file['size_bytes'] < 100:  # Less than 100 bytes
                result['issues'].append(
                    f"Event file '{event_file['name']}' is likely empty "
                    f"({event_file['size_formatted']}) - only headers written"
                )

    return result


def print_scan_result(result: dict, verbose: bool = False):
    """Print scan result in a formatted way."""
    print("\n" + "="*70)
    print(f"Log Directory: {result['path']}")
    print("="*70)

    if not result['exists']:
        print(f"❌ Directory does not exist")
        return

    if not result['is_dir']:
        print(f"❌ Not a directory")
        return

    # Print event files
    event_files = result.get('event_files', [])
    print(f"\n📊 TensorBoard Event Files: {len(event_files)}")

    if event_files:
        for i, event_file in enumerate(event_files, 1):
            size_indicator = "✓" if event_file['size_bytes'] >= 1000 else "⚠"
            print(f"  {size_indicator} {i}. {event_file['name']}")
            print(f"     Size: {event_file['size_formatted']}")
    else:
        print("  ⚠ No event files found")

    # Print other files
    other_files = result.get('other_files', [])
    if other_files and verbose:
        print(f"\n📄 Other Files: {len(other_files)}")
        for i, file in enumerate(other_files, 1):
            print(f"  {i}. {file['name']} ({file['size_formatted']})")

    # Print total size
    print(f"\n💾 Total Size: {format_size(result['total_size_bytes'])}")

    # Print issues
    issues = result.get('issues', [])
    if issues:
        print(f"\n⚠️  Issues Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✓ No issues detected")

    print("="*70)


def scan_all_logs(logs_root: Path, verbose: bool = False):
    """Scan all experiment directories in the logs root."""
    print(f"\nScanning all logs in: {logs_root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not logs_root.exists():
        print(f"❌ Logs directory does not exist: {logs_root}")
        return

    # Get all subdirectories (experiments)
    experiments = [d for d in logs_root.iterdir() if d.is_dir()]

    if not experiments:
        print("⚠️  No experiment directories found")
        return

    print(f"Found {len(experiments)} experiment(s)\n")

    # Scan each experiment
    results = []
    for exp_dir in sorted(experiments):
        result = scan_log_directory(exp_dir)
        results.append(result)
        print_scan_result(result, verbose=verbose)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_issues = sum(len(r.get('issues', [])) for r in results)
    total_size = sum(r.get('total_size_bytes', 0) for r in results)

    print(f"Total experiments: {len(results)}")
    print(f"Total size: {format_size(total_size)}")
    print(f"Total issues: {total_issues}")

    if total_issues == 0:
        print("\n✓ All logs appear healthy!")
    else:
        print(f"\n⚠️  {total_issues} issue(s) found across all experiments")

    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify TensorBoard log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all logs in default location
  python verify_logs.py

  # Scan specific experiment
  python verify_logs.py --experiment ucf101_resnet50_20260212_143022

  # Verbose output with file details
  python verify_logs.py --verbose

  # Custom logs directory
  python verify_logs.py --log-dir /path/to/logs
        """
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Path to logs directory (default: ../outputs/logs)'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Specific experiment name to check'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output with file details'
    )

    args = parser.parse_args()

    # Determine logs directory
    if args.log_dir:
        logs_root = Path(args.log_dir).absolute()
    else:
        # Default to ../outputs/logs relative to script location
        script_dir = Path(__file__).parent.absolute()
        logs_root = script_dir.parent / "outputs" / "logs"

    # Scan
    if args.experiment:
        # Scan specific experiment
        exp_dir = logs_root / args.experiment
        result = scan_log_directory(exp_dir)
        print_scan_result(result, verbose=args.verbose)
    else:
        # Scan all experiments
        scan_all_logs(logs_root, verbose=args.verbose)


if __name__ == '__main__':
    main()
