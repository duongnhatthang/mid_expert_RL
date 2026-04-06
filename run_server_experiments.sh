#!/bin/bash
# Run full experiment suite on CC-server inside tmux
# Usage: ssh CC-server, then: bash run_server_experiments.sh

set -e

PROJECT_DIR="$HOME/mid_expert_RL"
CONDA_BIN="$HOME/miniconda3/bin/conda"
ENV_NAME="mid_expert_rl"

echo "=== Setting up environment ==="
if ! $CONDA_BIN env list | grep -q "$ENV_NAME"; then
    echo "Creating conda env: $ENV_NAME"
    $CONDA_BIN create -n "$ENV_NAME" python=3.9 -y
fi
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"

# Clone or update repo
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout hypothesis-sweep
    git pull origin hypothesis-sweep
else
    git clone git@github.com:duongnhatthang/mid_expert_RL.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git checkout hypothesis-sweep
fi

pip install -r requirements.txt -q

echo ""
echo "=== 1/4: LR Sweep (calibration) ==="
PYTHONPATH=. python run_lr_sweep.py --budget 500 --n-seeds 10 --output-dir results/lr_sweep

echo ""
echo "=== 2/4: Learning Curves (auto-saturation) ==="
PYTHONPATH=. python run_experiments.py --mode learning_curve --grid-size 9 --n-seeds 10 --lr 0.5

echo ""
echo "=== 3/4: Hypothesis Sweep — capability mode ==="
PYTHONPATH=. python run_hypothesis_sweep.py --mode capability --n-seeds 10 --n-workers 10

echo ""
echo "=== 4/4: Hypothesis Sweep — zeta mode ==="
PYTHONPATH=. python run_hypothesis_sweep.py --mode zeta --n-seeds 10 --n-workers 10

echo ""
echo "=== All experiments complete! ==="
echo "Results in: $PROJECT_DIR/results/"
