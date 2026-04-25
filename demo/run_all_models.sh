#!/bin/bash
# Bash script to test all SAC model variants
# Usage: ./run_all_models.sh [num_episodes]

set -e

# Get number of episodes from command line (default: 2)
NUM_EPISODES=${1:-2}

echo ""
echo "=============================================================================="
echo "SAC MODEL INFERENCE - AUTOMATIC TEST ALL VARIANTS"
echo "=============================================================================="
echo "Episodes per model: $NUM_EPISODES"
echo "=============================================================================="
echo ""

# Get script directory
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMO_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test each model
MODELS=("5cnn" "2cnn" "5stt" "2stt")
MODEL_COUNT=0
COMPLETED_COUNT=0
FAILED_COUNT=0

for model in "${MODELS[@]}"; do
    ((MODEL_COUNT++))
    
    echo ""
    echo "=============================================================================="
    echo -e "${GREEN}Model $model - Episode set $MODEL_COUNT of ${#MODELS[@]}${NC}"
    echo "=============================================================================="
    
    if python run_model_inference.py --model "$model" --episodes "$NUM_EPISODES"; then
        ((COMPLETED_COUNT++))
        echo -e "${GREEN}[OK] Model $model completed${NC}"
    else
        ((FAILED_COUNT++))
        echo -e "${RED}[ERROR] Model $model failed${NC}"
    fi
    
    # Wait between models (except last one)
    if [ "$model" != "2stt" ]; then
        echo ""
        echo -e "${YELLOW}Waiting 5 seconds before next model...${NC}"
        sleep 5
    fi
done

# Print summary
echo ""
echo "=============================================================================="
echo "TEST SUMMARY"
echo "=============================================================================="
echo "Total models tested:  $MODEL_COUNT"
echo "Completed:           $COMPLETED_COUNT"
echo "Failed:              $FAILED_COUNT"
echo "=============================================================================="
echo ""

# Compare models if all succeeded
if [ $FAILED_COUNT -eq 0 ]; then
    echo -e "${GREEN}Running model comparison...${NC}"
    python compare_models.py --export model_comparison_results.json
else
    echo -e "${RED}Some tests failed. Skipping comparison.${NC}"
fi

echo ""
echo -e "${GREEN}Results saved. Check results_*.json and model_comparison_results.json${NC}"
echo ""
