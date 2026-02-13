#!/usr/bin/env python3
"""
Diagnostic script to check TensorBoard availability in GUI environment.
Run this to diagnose TensorBoard import issues.
"""

import sys
from pathlib import Path

# Setup sys.path exactly like app.py does
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

print("=" * 60)
print("TensorBoard Diagnostic Report")
print("=" * 60)

print("\n1. Python Environment:")
print(f"   Executable: {sys.executable}")
print(f"   Version: {sys.version}")

print("\n2. sys.path (first 5 entries):")
for i, p in enumerate(sys.path[:5]):
    print(f"   [{i}] {p}")

print("\n3. TensorBoard Import Test:")

# Test 1: Direct import
print("\n   [Test 1] Direct import of tensorboard package:")
try:
    import tensorboard
    print(f"   SUCCESS: tensorboard version {tensorboard.__version__}")
    print(f"   Location: {tensorboard.__file__}")
except ImportError as e:
    print(f"   FAILED: {e}")

# Test 2: Import event_accumulator (what GUI uses)
print("\n   [Test 2] Import event_accumulator (GUI method):")
try:
    from tensorboard.backend.event_processing import event_accumulator
    print("   SUCCESS: event_accumulator imported")
except ImportError as e:
    print(f"   FAILED: {e}")

# Test 3: Import using alias (like tensorboard_reader.py)
print("\n   [Test 3] Import with alias (tensorboard_reader.py method):")
try:
    from tensorboard.backend.event_processing import event_accumulator as _ea
    print("   SUCCESS: event_accumulator imported with alias")
except ImportError as e:
    print(f"   FAILED: {e}")

# Test 4: Use importlib to find tensorboard
print("\n   [Test 4] Using importlib to locate tensorboard:")
import importlib.util
spec = importlib.util.find_spec("tensorboard")
if spec:
    print(f"   Found at: {spec.origin}")
else:
    print("   NOT FOUND")

# Test 5: Check if GUI module can import
print("\n   [Test 5] GUI tensorboard_reader module check:")
try:
    from gui.tensorboard_reader import check_tensorboard_available, HAS_TORCH_TB, TORCH_TB_ERROR
    available, message = check_tensorboard_available()
    print(f"   HAS_TORCH_TB: {HAS_TORCH_TB}")
    print(f"   TORCH_TB_ERROR: {TORCH_TB_ERROR}")
    print(f"   Available: {available}")
    print(f"   Message: {message}")
except Exception as e:
    print(f"   FAILED to import: {e}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)

# Provide recommendations
print("\nRecommendations:")
if "deeplearning" not in sys.executable:
    print("  [!] You are NOT running in the 'deeplearning' conda environment!")
    print("  [!] Activate it first: conda activate deeplearning")
    print("  [!] Then run: python scripts/app.py")
else:
    print("  [OK] Running in correct conda environment")
    print("  If TensorBoard import fails, try:")
    print("    pip install --force-reinstall tensorboard")
