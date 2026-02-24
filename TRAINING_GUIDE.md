# Enhanced SAC Training Guide - CBF Safety Layer with Transformer Perception

## Overview

The `training/train_enhanced_cbf.py` script provides a complete training pipeline for SAC agents with:
- **Enhanced CBF Safety Layer** with dynamic constraints
- **Transformer Perception Pipeline** (ResNet50 + SpatioTemporal Encoder)
- **Dual Training Modes**: Headless (fast) and Visualized (slow with pygame)
- **Complete TensorBoard Logging** including CBF metrics
- **Trust-Aware Corrections** via ensemble critics

---

## Quick Start

### 1. Fixed CBF Layer Issue ✓

**Problem:** Action dimension mismatch (2D action space vs 3D CBF expectations)
- Previous: `action_space = Box(2,)` with `[steering, throttle/brake]`
- Fixed: `action_space = Box(3,)` with `[steering, throttle, brake]`

**Verification:**
```bash
python -c "
import numpy as np
from commons.cbfQP_layer import CBFSafetyLayer
cbf = CBFSafetyLayer()
u = np.array([0.1, 0.2, 0.1])  # 3D action
state = {'d_collision': 10.0, 'lane_offset': 0.0, 'speed': 5.0, 'speed_limit': 15.0}
result = cbf.compute_safe_action(u, state)
print('[OK] CBF working with 3D actions')
"
```

---

## Training Modes

### Mode 1: Headless Training (Default - FAST ⚡)
Best for long training runs on servers or machines without display.

```bash
cd e:\Sarosh\safety-aware-drl-carla

# Basic training
python training/train_enhanced_cbf.py --timesteps 100000

# Faster training without perception
python training/train_enhanced_cbf.py --timesteps 100000 --no-perception
```

**Output:**
- Logs to `./logs/enhanced_training/tensorboard/`
- Checkpoints to `./logs/enhanced_training/checkpoints/`
- Training info to `./logs/enhanced_training/training_info.txt`

### Mode 2: Visualized Training (--render flag - SLOWER 📺)
Enable pygame visualization for debugging and understanding agent behavior.

```bash
# Train with visualization
python training/train_enhanced_cbf.py --render --timesteps 50000

# Train with visualization, fewer NPC vehicles
python training/train_enhanced_cbf.py --render --num-npc 5 --timesteps 50000
```

**Visualization Includes:**
- ✅ **Top-left**: RGB camera view (640×480)
- ✅ **Top-right**: Depth map visualization (640×480)
- ✅ **Bottom**: Real-time metrics (step, distance, reward, CBF status)

**WARNING:** Visualization is ~5-10× slower than headless mode.

---

## Command-Line Arguments

```
positional arguments:
  none

optional arguments:
  --timesteps TS          Total training timesteps [default: 100000]
  --log-dir DIR           Logging directory [default: ./logs/enhanced_training]
  --lr RATE               Learning rate [default: 3e-4]
  --batch-size BS         Batch size [default: 64]
  --buffer-size BS        Replay buffer size [default: 50000]
  --num-npc N             Number of NPC vehicles [default: 20]
  
  --render                Enable pygame visualization (slower)
  --no-perception         Disable transformer perception pipeline
  
  --eval-only             Evaluate trained model (no training)
  --eval-model PATH       Path to model for evaluation
  --eval-episodes N       Number of eval episodes [default: 5]
```

---

## Training Examples

### Example 1: Quick Test (Headless, No Perception)
Fast training for testing:
```bash
python training/train_enhanced_cbf.py \
  --timesteps 10000 \
  --num-npc 5 \
  --batch-size 32 \
  --no-perception
```

### Example 2: Standard Training (Headless)
Recommended for production training:
```bash
python training/train_enhanced_cbf.py \
  --timesteps 100000 \
  --log-dir ./logs/standard_training \
  --lr 3e-4 \
  --batch-size 64 \
  --num-npc 20
```

### Example 3: Debug Training (Visualized)
For understanding agent behavior:
```bash
python training/train_enhanced_cbf.py \
  --render \
  --timesteps 5000 \
  --log-dir ./logs/debug_training \
  --num-npc 3
```

### Example 4: Evaluate Trained Model
```bash
python training/train_enhanced_cbf.py \
  --eval-only \
  --eval-model ./logs/standard_training/sac_cbf_final.zip \
  --eval-episodes 10
```

---

## TensorBoard Monitoring

View training progress in real-time:

```bash
tensorboard --logdir ./logs/enhanced_training/tensorboard
```

Then open browser to `http://localhost:6006`

### Key Metrics to Monitor

#### Safety Metrics
- **safety/lane_invasions**: Should decrease to near 0
- **safety/speed_violations**: Should stay low
- **safety/collisions**: Should NEVER increase
- **safety/cbf_corrections**: Should decrease as agent learns

#### CBF Metrics
- **cbf/total_corrections**: Cumulative correction count
- **cbf/episode_corrections**: Corrections per episode
- **cbf/avg_correction_mag**: Average correction magnitude
- **cbf/collision_violations**: Constraint violations
- **cbf/lane_violations**: Lane constraint breaches
- **cbf/speed_violations**: Speed constraint breaches

#### Learning Metrics
- **rollout/ep_rew_mean**: Episode reward (should improve)
- **train/actor_loss**: Actor loss (typically decreasing)
- **train/critic_loss**: Critic loss
- **train/ent_coef_loss**: Entropy coefficient

---

## What Each Component Does

### 1. Action Space Fix (3D)
```python
# Before: [steering, throttle/brake] → 2D
# After:  [steering, throttle, brake] → 3D

# Step function handles:
- steering ∈ [-1, 1]  (steering angle)
- throttle ∈ [0, 1]   (acceleration)
- brake ∈ [0, 1]      (deceleration)
```

### 2. CBF Safety Layer
- **Dynamic Speed Limits**: Reads from CARLA waypoints
- **Proactive Lane Keeping**: Predicts future offset
- **Speed-Dependent Lie Derivatives**: Realistic constraints
- **Rate Limiting**: Smooth steering/throttle changes
- **Trust Score Modulation**: Conservative under uncertainty
- **Reward Penalty**: -0.01 × correction magnitude

### 3. Transformer Perception
- **ResNet50 Backbone**: Pretrained feature extraction
- **SpatioTemporal Encoder**: Captures temporal dynamics
- **Optional**: Can disable with `--no-perception` for faster training

### 4. Visualization (--render mode)
- **Top-left RGB**: Camera sensor view
- **Top-right Depth**: Depth map for collision detection
- **Bottom Info**: Real-time metrics and CBF status
- **Updated at 20 FPS**: Low overhead visualization

---

## Performance Expectations

### Training Speed
- **Headless (no perception)**: ~100-150 timesteps/second
- **Headless (with perception)**: ~50-100 timesteps/second
- **Visualized (any mode)**: ~5-10 timesteps/second

### Safety Improvements Over Training
- **Episode 0-5K**: High CBF corrections (agent is unsafe)
- **Episode 5K-20K**: Decreasing corrections (agent learning)
- **Episode 20K+**: Stable safe behavior (few corrections)

### Resource Usage
- **GPU**: ResNet50 + SAC critic networks (~2GB VRAM)
- **CPU**: CARLA client, SAC training, CBF solver
- **RAM**: ~4-6GB for replay buffer (50K transitions)

---

## Common Issues & Solutions

### Issue: "Pygame initialization failed"
**Cause**: No display available or pygame not installed
**Solution**: 
- Use headless mode (default): `python training/train_enhanced_cbf.py`
- Or install pygame: `pip install pygame`

### Issue: "CARLA connection timeout"
**Cause**: CARLA server not running on localhost:2000
**Solution**:
```bash
# In separate terminal, start CARLA
./CarlaUE4.exe -windowed -ResX=1024 -ResY=768
```

### Issue: "QP solver exception"
**Cause**: CBF constraints conflicting or infeasible
**Solution**:
- Loosen constraints: edit `CBFSafetyLayerWrapper(alpha=0.5, ...)`
- Check that d_min < 10m and y_max < 2.0m

### Issue: "Out of memory"
**Cause**: Replay buffer too large or batch size too big
**Solution**:
```bash
python training/train_enhanced_cbf.py \
  --batch-size 32 \
  --buffer-size 10000
```

### Issue: Low reward despite training
**Cause**: Correction penalty too high or environment too hard
**Solution**:
- Reduce correction penalty in code: `correction_penalty=0.001`
- Reduce speed limits or collision distance in CBF
- Increase num_npc gradually: start with `--num-npc 5`

---

## Files Modified

1. **tests/pipeline_carla_test.py**
   - ✅ Fixed action space from 2D to 3D
   - ✅ Updated step() to handle 3D actions
   - ✅ Added speed_limit to CBF state dict
   - ✅ Enhanced CBFSafetyLayerWrapper with reward penalty

2. **tests/cnn_policy_test.py**
   - ✅ Compatible with new 3D action space
   - ✅ Works with enhanced CBF layer

3. **commons/cbfQP_layer.py**
   - ✅ Speed-dependent Lie derivatives
   - ✅ Trust score modulation
   - ✅ Proactive lane constraint
   - ✅ Dynamic speed limits
   - ✅ Rate limiting

4. **training/train_enhanced_cbf.py** (NEW)
   - ✅ Complete training script
   - ✅ Headless + visualization modes
   - ✅ TensorBoard logging
   - ✅ CBF metrics integration

---

## Next Steps

1. **Start Training**:
   ```bash
   python training/train_enhanced_cbf.py --timesteps 100000
   ```

2. **Monitor Progress**:
   ```bash
   tensorboard --logdir ./logs/enhanced_training/tensorboard
   ```

3. **Evaluate Results**:
   ```bash
   python training/train_enhanced_cbf.py \
     --eval-only \
     --eval-model ./logs/enhanced_training/sac_cbf_final.zip
   ```

4. **Visualize Behavior** (optional):
   ```bash
   python training/train_enhanced_cbf.py \
     --render \
     --eval-only \
     --eval-model ./logs/enhanced_training/sac_cbf_final.zip
   ```

---

## Troubleshooting Commands

```bash
# Check CBF is working
python -c "
from commons.cbfQP_layer import CBFSafetyLayer
import numpy as np
cbf = CBFSafetyLayer()
action = np.array([0.1, 0.5, 0.1])
state = {'d_collision': 2.0, 'lane_offset': 0.0, 'speed': 10.0, 'speed_limit': 15.0}
result = cbf.compute_safe_action(action, state)
print('[OK] CBF: action corrected from', action, 'to', result)
"

# Check action space
python -c "
from tests.pipeline_carla_test import CarlaGymEnv
env = CarlaGymEnv()
print('[OK] Action space:', env.action_space)
print('[OK] Observation space:', env.observation_space)
env.close()
"

# Syntax check training script
python -m py_compile training/train_enhanced_cbf.py && echo "[OK] Training script syntax valid"
```

---

## Advanced Configuration

Edit these values in `training/train_enhanced_cbf.py`:

```python
# CBF parameters (line ~50)
create_training_env(
    headless=True,          # Set False for visualization
    time_limit=1000,        # Episode length
    num_npc=20,             # Traffic vehicles
    perception_enabled=True # Transformer pipeline
)

# SAC hyperparameters (line ~400)
policy_kwargs = {
    "net_arch": {"pi": [256, 256], "qf": [256, 256]},
    "n_critics": 5,         # Ensemble size
    "trust_lambda": 0.01,   # Uncertainty sensitivity
}

# CBF parameters (line ~420)
CBFSafetyLayerWrapper(
    env,
    alpha=1.0,              # Constraint decay rate
    correction_penalty=0.01 # Reward penalty scale
)
```

---

**Status**: ✅ Ready for training
**Last Updated**: February 18, 2026
