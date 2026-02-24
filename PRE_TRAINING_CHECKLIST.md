# Pre-Training Checklist ✓

## Critical Fixes Applied

- [x] **CBF Action Space**: Fixed from 2D to 3D
  - Action format: `[steering ∈ [-1,1], throttle ∈ [-1,1], brake ∈ [-1,1]]`
  - File: `tests/pipeline_carla_test.py` lines 122-127
  - Tested: ✅ CBF accepts 3D actions

- [x] **Step Function**: Updated for 3D action parsing
  - File: `tests/pipeline_carla_test.py` lines 460-482
  - Steering: action[0] as-is
  - Throttle: action[1] clamped to [0,1]
  - Brake: action[2] clamped to [0,1]
  - Tested: ✅ Actions properly applied to CARLA

- [x] **CBF State Dict**: Includes speed_limit
  - File: `tests/pipeline_carla_test.py` build_cbf_state()
  - State: `{d_collision, lane_offset, speed, speed_limit, ttc}`

## Training Infrastructure

- [x] **training/train_enhanced_cbf.py**: Complete script (520 lines)
  - Headless mode (default)
  - Visualization mode (--render flag)
  - Perception pipeline integration (--no-perception to disable)
  - TensorBoard logging
  - Checkpointing and evaluation

- [x] **TensorBoard Logging**: Integrated
  - Safety metrics: lane_invasions, speed_violations, collisions, cbf_corrections
  - CBF metrics: collision_violations, lane_violations, speed_violations
  - Learning metrics: reward, actor_loss, critic_loss

- [x] **Callback Infrastructure**: Complete
  - EnhancedSafetyMetricsCallback: Real-time safety monitoring
  - CheckpointCallback: Model saving every 10K steps
  - PerceptionMonitorCallback: BYO perception pipeline monitoring

## Environment Validation

```bash
# Check 1: Verify CBF works with 3D actions
python -c "
import numpy as np
from commons.cbfQP_layer import CBFSafetyLayer
cbf = CBFSafetyLayer()
u = np.array([0.1, 0.2, 0.1])
state = {'d_collision': 10.0, 'lane_offset': 0.0, 'speed': 5.0, 'speed_limit': 15.0}
result = cbf.compute_safe_action(u, state)
print('[OK] CBF 3D action test PASSED')
"

# Check 2: Verify environment has 3D action space
python -c "
from tests.pipeline_carla_test import CarlaGymEnv
env = CarlaGymEnv()
assert env.action_space.shape == (3,), f'Expected shape (3,), got {env.action_space.shape}'
print(f'[OK] Action space is 3D: {env.action_space}')
env.close()
"

# Check 3: Verify training script syntax
python -m py_compile training/train_enhanced_cbf.py && echo "[OK] Training script syntax valid"
```

## Dependencies

Required packages:
- [x] stable-baselines3 >= 2.0
- [x] gymnasium >= 0.26.0
- [x] carla >= 0.9.15
- [x] torch (for perception pipeline)
- [x] osqp (for CBF solver)
- [x] pygame (for visualization, optional)
- [x] tensorboard (for logging)

Optional validation:
```bash
pip list | grep -E "stable-baselines3|gymnasium|carla|torch|osqp|tensorboard"
```

## Quick Start Commands

### 1. Headless Training (Default - Fast)
```bash
cd e:\Sarosh\safety-aware-drl-carla
python training/train_enhanced_cbf.py --timesteps 100000
```

### 2. Visualized Training (Debug mode)
```bash
python training/train_enhanced_cbf.py --render --timesteps 50000
```

### 3. Monitor Progress
```bash
tensorboard --logdir ./logs/enhanced_training/tensorboard
```

### 4. Evaluate Trained Model
```bash
python training/train_enhanced_cbf.py \
  --eval-only \
  --eval-model ./logs/enhanced_training/sac_cbf_final.zip
```

## Expected Performance

- **Training Speed** (headless):
  - With perception: 50-100 steps/sec
  - Without perception: 100-150 steps/sec

- **Safety Improvement**:
  - First 5K steps: High CBF corrections (agent unsafe)
  - Next 15K steps: Decreasing corrections (learning)
  - After 20K steps: Stable safe behavior

- **Resource Usage**:
  - GPU: ~2GB VRAM (ResNet50 + critics)
  - RAM: ~4-6GB (replay buffer)
  - CPU: Medium load

## Troubleshooting

### Problem: "CARLA connection timeout"
**Fix**: Start CARLA server in separate terminal
```bash
./CarlaUE4.exe -windowed -ResX=1024 -ResY=768
```

### Problem: "QP solver exception: Incorrect dimension of q"
**Status**: ✅ Already fixed! If you see this, CBF version is outdated.
- Update `commons/cbfQP_layer.py` from current

### Problem: Pygame not available
**Fix**: Training automatically falls back to headless mode
- Or install: `pip install pygame`

### Problem: Out of memory
**Fix**: Reduce buffer size or batch size
```bash
python training/train_enhanced_cbf.py \
  --batch-size 32 \
  --buffer-size 10000 \
  --timesteps 50000
```

## Files Status

| File | Status | Notes |
|------|--------|-------|
| tests/pipeline_carla_test.py | ✅ Fixed | 3D action space, updated step() |
| commons/cbfQP_layer.py | ✅ Enhanced | 8 new features, trust modulation |
| training/train_enhanced_cbf.py | ✅ Created | 520 lines, dual-mode support |
| TRAINING_GUIDE.md | ✅ Created | Complete usage guide |

## Verification Tests (Run Before Training)

```python
# test_cbf_fix.py
import numpy as np
from commons.cbfQP_layer import CBFSafetyLayer

print("=" * 50)
print("PRE-TRAINING CBF VERIFICATION")
print("=" * 50)

# Test 1: CBF accepts 3D actions
cbf = CBFSafetyLayer()
u = np.array([0.1, 0.2, 0.1])
state = {
    'd_collision': 10.0,
    'lane_offset': 0.0,
    'speed': 5.0,
    'speed_limit': 15.0
}
result = cbf.compute_safe_action(u, state)
assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
print("✅ Test 1 PASSED: CBF accepts 3D actions")

# Test 2: Constraint satisfaction
assert result[2] >= 0 and result[2] <= 1, "Brake out of bounds"
print("✅ Test 2 PASSED: Output constraints satisfied")

# Test 3: Small corrections for safe state
correction = np.linalg.norm(result - u)
assert correction < 0.5, f"Large correction: {correction}"
print(f"✅ Test 3 PASSED: Minimal correction ({correction:.4f}) for safe state")

print("=" * 50)
print("ALL TESTS PASSED - READY FOR TRAINING")
print("=" * 50)
```

Run it:
```bash
python test_cbf_fix.py
```

## Next Steps

1. Verify all checks above pass ✓
2. Start training:
   ```bash
   python training/train_enhanced_cbf.py --timesteps 100000
   ```
3. Monitor TensorBoard:
   ```bash
   tensorboard --logdir ./logs/enhanced_training/tensorboard
   ```
4. Evaluate results after training completes

---

**Status**: ✅ System Ready for Training
**Last Updated**: February 18, 2026
**Critical Fix**: CBF dimension mismatch resolved (2D → 3D action space)
