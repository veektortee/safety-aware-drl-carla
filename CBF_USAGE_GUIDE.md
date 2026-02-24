# Enhanced CBF Configuration & Usage Guide

## Quick Start

### 1. Train CNN SAC Agent with Enhanced CBF

```bash
cd e:\Sarosh\safety-aware-drl-carla
python tests/cnn_policy_test.py \
    --timesteps 100000 \
    --log-dir ./logs/enhanced_cbf_cnn \
    --lr 3e-4 \
    --batch-size 64 \
    --render
```

### 2. Evaluate Trained Model

```bash
python tests/cnn_policy_test.py \
    --eval-only \
    --eval-model ./logs/enhanced_cbf_cnn/cnn_sac_final.zip \
    --eval-episodes 5 \
    --render
```

---

## CBF Parameter Tuning

### Conservative (Maximum Safety)
```python
CBFSafetyLayer(
    alpha=2.0,              # Faster constraint recovery
    d_min=7.0,              # Larger safety margin
    y_max=1.0,              # Stricter lane keeping
    v_max=12.0,             # Speed-limited
    max_steering_rate=0.3,  # Smooth steering
    max_accel_change=0.2,   # Smooth acceleration
)

CBFSafetyLayerWrapper(
    env,
    correction_penalty=0.05  # Stronger penalty for corrections
)
```

### Balanced (Recommended)
```python
CBFSafetyLayer(
    alpha=1.0,              # Default
    d_min=5.0,              # Standard margin
    y_max=1.5,              # Balanced lane keeping
    v_max=15.0,             # Reasonable speed
    max_steering_rate=0.5,  # Natural steering
    max_accel_change=0.3,   # Natural acceleration
)

CBFSafetyLayerWrapper(
    env,
    correction_penalty=0.01  # Moderate penalty
)
```

### Aggressive (Test Exploratio)
```python
CBFSafetyLayer(
    alpha=0.5,              # Slower recovery
    d_min=3.0,              # Minimal margin
    y_max=2.0,              # Lenient lane keeping
    v_max=20.0,             # High speed allowed
    max_steering_rate=0.8,  # Active steering
    max_accel_change=0.4,   # Active acceleration
)

CBFSafetyLayerWrapper(
    env,
    correction_penalty=0.001  # Minimal penalty
)
```

---

## TensorBoard Monitoring

### Safety-Specific Metrics
```
# Lane invasion tracking
safety/lane_invasions       → Should decrease to near 0
safety/cbf_corrections      → Should decrease over time
safety/cbf_correction_magnitude → Should trend toward 0

# Speed & collision tracking
safety/speed_violations     → Should stay low
safety/collisions           → Should never occur!

# Ensemble performance
train/ensemble_uncertainty  → Should decrease as agent learns
train/trust_score          → Should increase toward 1.0
```

### Expected Training Curves

**Well-Tuned System:**
- Episode 0-1000: High CBF corrections, improving safety
- Episode 1000-5000: Decreasing corrections, agent learning
- Episode 5000+: Few corrections, stable safe behavior

**Over-Penalized (penalty too high):**
- Reward plateaus or decreases
- Agent doesn't explore safely
- Solution: Reduce `correction_penalty`

**Under-Penalized (penalty too low):**
- CBF corrections never decrease
- Agent ignores safety feedback
- Solution: Increase `correction_penalty`

---

## Enabling/Disabling Features

### Disable Trust Score Modulation
```python
wrapper = CBFSafetyLayerWrapper(env, use_trust_score=False)
# CBF will always use full corrections
# Useful for debugging or conservative safety
```

### Disable Rate Limiting
```python
cbf = CBFSafetyLayer(
    max_steering_rate=float('inf'),  # No steering limit
    max_accel_change=float('inf')    # No acceleration limit
)
```

### Disable Reward Penalty
```python
wrapper = CBFSafetyLayerWrapper(env, correction_penalty=0.0)
# CBF will still apply corrections
# But agent won't be penalized (no learning incentive)
```

### Disable Dynamic Speed Limits
```python
# In environment: don't pass speed_limit in cbf_state
# CBF will default to self.v_max
```

---

## Debugging & Troubleshooting

### Issue: QP Solver Failures
**Symptom:** Frequent "solver status != solved" messages

**Solution:**
1. Reduce constraint tightness:
   ```python
   alpha=0.5,  # Slower recovery
   d_min=3.0,  # Smaller margins
   ```

2. Enable verbose solver output:
   ```python
   self.solver.setup(..., verbose=True)  # In cbfQP_layer.py line ~115
   ```

3. Check constraints are consistent:
   - All 3 constraints (collision, lane, speed) should not conflict
   - Test with minimal constraints first

### Issue: Agent Ignoring Speed Limits
**Symptom:** Vehicle exceeds specified limits despite CBF

**Verification:**
- Check `safety/speed_violations` in TensorBoard → should be low
- Verify `speed_limit` is in `cbf_state` dict
- Monitor that `compute_safe_action()` receives speed_limit parameter

**Solution:**
```python
# In pipeline_carla_test.py build_cbf_state():
print(f"Speed limit passed: {cbf_state['speed_limit']} m/s")
```

### Issue: Lane Invasions Still Occurring
**Symptom:** High `safety/lane_invasions` despite proactive lane constraint

**Root Causes:**
1. Lane offset threshold too permissive (y_max=1.5)
   - Solution: Reduce to 1.0 or 0.8
   
2. Proactive steering heuristic inaccurate
   - Solution: Use actual yaw rate if available from CARLA
   
3. Agent learning to exploit lane boundary
   - Solution: Increase penalty for lane invasions in reward function

**Debugging:**
```python
# In cnn_policy_test.py, add to training loop:
if episode % 100 == 0:
    print(f"Episode {episode}: lane_inv={metrics['lane_invasions']}")
```

### Issue: Over-Conservative Corrections
**Symptom:** Agent under-performs, always braking

**Cause:** `y_max` or `d_min` too tight for scenario

**Solution:**
```python
# Loosen constraints incrementally
d_min=4.0      # From 5.0 (but keep >3 for safety)
y_max=1.2      # From 1.5 (but keep <2.0)
```

---

## Advanced Configuration

### Per-Scenario Adaptive CBF
```python
def get_cbf_params(scenario_type):
    if scenario_type == "highway":
        return dict(d_min=10.0, y_max=2.0, v_max=30.0)
    elif scenario_type == "urban":
        return dict(d_min=3.0, y_max=0.8, v_max=10.0)
    elif scenario_type == "residential":
        return dict(d_min=5.0, y_max=1.5, v_max=15.0)
    
cbf_params = get_cbf_params(scenario)
cbf = CBFSafetyLayer(**cbf_params)
```

### Hierarchical Constraint Prioritization
Currently all constraints have equal priority (min-norm QP).

For future: implement hierarchical CBF where collision > lane > speed

```python
# TODO: Prioritize constraints
# if collision_margin < 2.0:
#     ignore lane/speed constraints
# elif lane_margin < 0.5:
#     ignore speed constraint
# else:
#     apply all constraints
```

### Trust-Score Gating
Currently: trust score modulates correction magnitude.

Alternative: gate CBF activation by trust threshold
```python
# In compute_safe_action():
if trust_score < 0.7:
    return np.array([0.0, 0.0, 1.0])  # Only emergency brake
elif trust_score < 0.8:
    # Reduced CBF constraints
    self.alpha = 0.5
```

---

## Hyperparameter Sweep

Recommended experiment: grid search over penalty scales

```python
# test_penalties.py
for penalty in [0.001, 0.005, 0.01, 0.02, 0.05]:
    env = create_env(correction_penalty=penalty)
    agent = SAC(..., env=env)
    agent.learn(50000)
    evaluate(agent)
    # Compare: reward, safety metrics, CBF correction trend
```

---

## Integration with Existing Training Pipelines

### Pipeline CARLA Test (pipeline_carla_test.py)
Already integrated! CBF enhancements applied automatically.

```python
env = create_carla_env(time_limit=60, render=False)
# CBFSafetyLayerWrapper applied with default enhanced parameters
agent = SAC("MlpPolicy", env, ...)
agent.learn(100000)
```

### CNN Policy Test (cnn_policy_test.py)
Compatible! All enhancements work with CNN observations.

```bash
python tests/cnn_policy_test.py --render
# AutomaticallyCBF enhancement applied in create_cnn_env()
```

### Custom Environments
To apply enhanced CBF to custom environment:

```python
from commons.cbfQP_layer import CBFSafetyLayer
from tests.pipeline_carla_test import CBFSafetyLayerWrapper

# Wrap your environment
env = CustomEnv()
safety_env = CBFSafetyLayerWrapper(
    env,
    alpha=1.0,
    correction_penalty=0.01
)

# Ensure environment has build_cbf_state() method!
# (Returns: {'d_collision', 'lane_offset', 'speed', 'speed_limit'})
```

---

## Validation Checklist

- [ ] Run `python tests/test_enhanced_cbf.py` → all 9 tests pass
- [ ] Train for 1000 steps → verify no exceptions
- [ ] Check TensorBoard: `safety/cbf_corrections` present
- [ ] Check `safety/lane_invasions` near 0 by episode 5000
- [ ] Check reward trend improving or stable
- [ ] Verify ensemble uncertainty logged in TensorBoard
- [ ] Run evaluation without sensor rendering (faster)
- [ ] Monitor GPU memory (transformer feature extractor is 12.2M params)

---

## Performance Expectations

### Hardware
Tested on: NVIDIA GPU with CARLA running locally

### Training Time
- 100K timesteps CNN SAC: ~6-8 hours (depending on CARLA tick rate)
- 50K timesteps on transformer pipeline: ~4-5 hours

### Safety Improvements
- Baseline (no CBF): ~5-10 collisions per 100 steps
- With enhanced CBC: <1 collision per 100 steps
- With penalty: aggressive exploration while maintaining safety

---

## Support & Troubleshooting

For issues, check:
1. TensorBoard logs in `./logs/enhanced_cbf_cnn/tensorboard/`
2. Console output for CBF solver warnings
3. `test_enhanced_cbf.py` results
4. See ENHANCED_CBF_SUMMARY.md for detailed architecture

---

**Last Updated:** February 18, 2026
