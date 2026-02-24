#!/usr/bin/env python
"""Check which dependencies are available for encoder training"""

import sys
import os

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CARLA-RL-Agents'))

print("=" * 70)
print("CHECKING DEPENDENCIES FOR ENCODER TRAINING")
print("=" * 70)
print()

# Check core packages
packages = [
    ('torch', 'PyTorch'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('tensorboard', 'TensorBoard'),
    ('carla', 'CARLA'),
]

print("EXTERNAL PACKAGES:")
for pkg, name in packages:
    try:
        __import__(pkg)
        print(f"  [✓] {name:20} ({pkg})")
    except ImportError:
        print(f"  [✗] {name:20} ({pkg}) - MISSING")

print()
print("LOCAL MODULES:")

# Check local modules
local_modules = [
    ('commons.spatioTemporal_transformer', 'SpatioTemporalTransformer'),
    ('commons.feature_extractor', 'FeatureExtractor'),
    ('commons.cbfQP_layer', 'CBF QP Layer'),
    ('models.pipeline', 'Pipeline'),
    ('stable_baselines3', 'Stable Baselines3'),
]

for module_path, name in local_modules:
    try:
        __import__(module_path)
        print(f"  [✓] {name:30} ({module_path})")
    except ImportError as e:
        print(f"  [✗] {name:30} ({module_path}) - {str(e)[:40]}")

# Try importing the encoder training script components
print()
print("ENCODER TRAINING SCRIPT COMPONENTS:")
try:
    from training.encoder_training import EncoderTrainer
    print(f"  [✓] EncoderTrainer class")
except ImportError as e:
    print(f"  [✗] EncoderTrainer class - {str(e)[:60]}")

try:
    from training.encoder_training import ContrastiveLoss
    print(f"  [✓] ContrastiveLoss class")
except ImportError as e:
    print(f"  [✗] ContrastiveLoss class - {str(e)[:60]}")

print()
print("=" * 70)
print("REQUIRED INSTALLATIONS:")
print("=" * 70)
print()
print("Run these commands to install missing dependencies:")
print()
print("pip install torch torchvision torchaudio")
print("pip install opencv-python")
print("pip install tensorboard")
print("pip install carla")
print()
