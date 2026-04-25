# SAC Model Inference Demo

This directory contains scripts to visualize and run inference on trained SAC agent models.

## Supported Models

| Model | Description | File |
|-------|-------------|------|
| **5CNN** | 5-layer CNN feature extractor | `5cnn.zip` |
| **2CNN** | 2-layer CNN feature extractor | `2cnn.zip` |
| **5STT** | 5-block Spatiotemporal Transformer | `5stt.zip` |
| **2STT** | 2-block Spatiotemporal Transformer | `2stt.zip` |

## Requirements

- CARLA server running on `localhost:2000`
- Python 3.8+
- Dependencies: See `../requirements.txt`

## Quick Start

### 1. List Available Models

```bash
python run_model_inference.py --list-models
```

### 2. Run Inference with Default Model (5CNN)

```bash
python run_model_inference.py --episodes 3
```

### 3. Run Inference with Specific Model

```bash
# Test 2-layer CNN model
python run_model_inference.py --model 2cnn --episodes 5

# Test 5-block Spatiotemporal Transformer
python run_model_inference.py --model 5stt --episodes 2

# Test 2-block Spatiotemporal Transformer
python run_model_inference.py --model 2stt --episodes 1
```

### 4. Test Without CBF Safety Layer

```bash
python run_model_inference.py --model 5cnn --episodes 3 --no-cbf
```

### 5. Use CPU Instead of GPU

```bash
python run_model_inference.py --model 2stt --episodes 1 --cpu
```

## Options

```
--model {5cnn, 2cnn, 5stt, 2stt}     Model to test (default: 5cnn)
--episodes N                          Number of episodes to run (default: 3)
--checkpoint-dir PATH                Directory with model checkpoints
--no-cbf                             Disable CBF safety layer
--cpu                                Use CPU instead of GPU
--list-models                        List available models and exit
--quiet                              Suppress verbose output
--help                               Show help message
```

## Output

Each inference run generates:

1. **Console Output**
   - Episode-by-episode progress
   - Real-time metrics (reward, waypoints, collisions)
   - Summary statistics

2. **Results File**
   - Saved as `results_{MODEL}_{N}eps.json`
   - Contains detailed statistics for each episode

Example results:

```json
[
  {
    "episode_num": 1,
    "model": "5cnn",
    "total_reward": 245.67,
    "episode_length": 287,
    "collisions": 0,
    "waypoints_crossed": 12,
    "cbf_corrections": 34,
    "cbf_enabled": true
  }
]
```

## Model Comparison Workflow

Compare all 4 models:

```bash
# Run each model for 2 episodes
python run_model_inference.py --model 5cnn --episodes 2
python run_model_inference.py --model 2cnn --episodes 2
python run_model_inference.py --model 5stt --episodes 2
python run_model_inference.py --model 2stt --episodes 2

# Compare results using the comparison script
python compare_models.py
```

## Metrics Tracked

### Performance
- **Total Reward**: Cumulative episode reward
- **Episode Length**: Number of steps completed
- **Success Rate**: Episodes reaching endpoint

### Safety
- **Collisions**: Number of collisions detected
- **CBF Corrections**: Safety layer interventions
- **Collision Distance**: Minimum distance to obstacles

### Navigation
- **Waypoints Crossed**: Milestones reached
- **Progress %**: Route completion percentage
- **Distance to Next Waypoint**: Navigation accuracy

## CARLA Environment Settings

Default configuration during inference:
- **Time Limit**: 300 steps (15 seconds at 20 FPS)
- **NPC Vehicles**: 50
- **Pedestrians**: 15
- **CBF Safety**: Enabled by default
- **Render Mode**: Human (visual feedback)

## Troubleshooting

### "Model not found" Error
- Check model file exists: `models/checkpoints/{MODEL}.zip`
- Verify checkpoint directory path with `--checkpoint-dir`

### "CARLA module not found"
- Install CARLA PythonAPI
- Ensure CARLA server is running

### "Connection refused" Error
- Start CARLA server: `./CarlaUE4.sh -carla-rpc-port=2000`
- Wait for server to initialize

### Low GPU Memory
- Use `--cpu` flag to run on CPU
- Reduce `--episodes` count

## Advanced Usage

### Run with Custom Checkpoint Directory

```bash
python run_model_inference.py --model 5cnn \
  --checkpoint-dir /path/to/checkpoints \
  --episodes 5
```

### Batch Testing Script

Create `test_all_models.sh`:

```bash
#!/bin/bash

for model in 5cnn 2cnn 5stt 2stt; do
    echo "Testing $model..."
    python run_model_inference.py --model $model --episodes 3
    sleep 5
done

echo "All models tested. Results saved."
python compare_models.py
```

Run with:

```bash
bash test_all_models.sh
```

## Output Examples

### Episode Progress (Live)

```
======================================================================
EPISODE 1 - Inference
======================================================================
Model: SAC-5CNN
CBF Safety: Enabled
======================================================================

[Step  50] Reward:   1.2345 | Total:      31.4567
           | Waypoints: 2/40 | Collision Distance: 45.23m
[Step 100] Reward:   1.0123 | Total:      52.3210
           | Waypoints: 4/40 | Collision Distance: 38.15m
```

### Episode Summary

```
======================================================================
EPISODE SUMMARY
======================================================================
Total Reward:         245.6789
Episode Length:            287 steps
Collisions:                  0
Waypoints Crossed:          12
CBF Corrections:            34
======================================================================
```

### Inference Summary (Multiple Episodes)

```
======================================================================
INFERENCE SUMMARY
======================================================================
Model:              SAC-5CNN
Episodes Completed: 3/3

Reward Statistics:
  Mean:             256.1234
  Std:               12.3456
  Min:              241.2345
  Max:              271.0123

Length Statistics:
  Mean:             298.33 steps
  Std:               15.67 steps
  Min:               287 steps
  Max:               318 steps

Safety Statistics:
  Total Collisions: 0
  Mean per Ep:        0.00

Navigation Statistics:
  Mean Waypoints:     14.33
  Total Waypoints:    43
======================================================================
```

## Model Files

Expected checkpoint structure:

```
models/
└── checkpoints/
    ├── 5cnn.zip
    ├── 2cnn.zip
    ├── 5stt.zip
    └── 2stt.zip
```

Each `.zip` file contains:
- SAC policy network weights
- Value/Q-networks
- Hyperparameters
- Training metadata

## Related Files

- **Training Script**: `../tests/pipeline_carla_test.py`
- **Pipeline Module**: `../models/pipeline.py`
- **CBF Safety Layer**: `../commons/cbfQP_layer.py`
- **Environment**: `../tests/pipeline_carla_test.py` (CarlaGymEnv)

## Notes

- First run may take longer as CARLA initializes actors
- Visual rendering helps understand model behavior
- Use `--quiet` flag for batch testing without console spam
- Results are saved in JSON format for analysis/plotting

## Next Steps

1. Run a model: `python run_model_inference.py --model 5cnn --episodes 3`
2. Analyze results: Check `results_5cnn_3eps.json`
3. Compare models: Run other models and use `compare_models.py`
4. Fine-tune: Modify hyperparameters or retrain if needed
