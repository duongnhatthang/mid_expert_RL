# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning research project testing the hypothesis: **In sparse-reward MDPs with limited sample budgets, a "mid-capacity" teacher outperforms both the best teacher (full knowledge) and a random teacher.** The core intuition is that the best teacher has a complex policy requiring many samples to learn, while a mid-capacity teacher provides a simpler, more learnable signal within constrained budgets.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pandas`, `pytest`

## Commands

### Run experiments

```bash
# Single quick experiment
python run_experiments.py --mode quick --seed 0 --teacher-capacity 1 --sample-budget 12000

# 2x2 exploration matrix (budget × horizon, recommended for testing the hypothesis)
python run_experiments.py --mode 2x2 --grid-size 8 --n-seeds 10 --n-goals 3 --alpha 0.5

# Full parameter suite
python run_experiments.py --mode suite --grid-size 10

# Learning curves
python run_experiments.py --mode learning_curve --grid-size 8 --n-seeds 5 --sample-budget 12000

# Plot existing results from CSV
python run_experiments.py --mode plot2x2 --output results/exploration_2x2_results.csv
```

### Run tests

```bash
pytest tests/ -v
# Single test file
pytest tests/test_run_experiments.py -v
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teacher-capacity` | varies | -1=no teacher, 0=random, 1..n_goals=partial/full knowledge |
| `--alpha` | 0.5 | Teacher advantage weight in PAV-RL (0=ignore teacher, 1=equal weight) |
| `--sample-budget` | 12000 | Total environment steps per experiment |
| `--grid-size` | 8 | N×N grid dimensions |
| `--n-goals` | 3 | Number of goal states |
| `--horizon` | 10×grid_size | Max episode length |
| `--n-seeds` | 5-10 | Random seeds for averaging |

## Architecture

### Training Loop (PAV-RL)

The core algorithm augments standard policy gradient with a teacher advantage signal:

```
grad J = E[∇_π log π(a|s) * (Q^π(s,a) + α * A^μ(s,a))]
```

Where `Q^π` is the student's Monte Carlo return and `A^μ = Q^μ - V^μ` is the teacher's advantage.

### Module Responsibilities

- **`tabular_prototype/environment.py`** — `GridEnv`: N×N grid with goals (reward=1), traps (reward=0), absorbing states. Student starts at center.
- **`tabular_prototype/teacher.py`** — Computes fixed `Q^μ(s,a)` and `V^μ(s)` via value iteration based on which goals the teacher "knows" (its capacity). Never updated during training.
- **`tabular_prototype/student.py`** — `TabularSoftmaxPolicy`: trainable `θ[state, action]` parameters, trajectory collection.
- **`tabular_prototype/training.py`** — PAV-RL gradient computation and policy updates.
- **`tabular_prototype/experiments.py`** — Experiment orchestration: `run_experiment()`, `run_2x2_exploration_experiment()`, `run_learning_curve_experiment()`, `run_experiment_suite()`.
- **`tabular_prototype/visualization.py`** — Policy grids, advantage heatmaps, learning curves, result bar charts.
- **`tabular_prototype/config.py`** — Discount factor: `γ = 1 - 1/H` (makes value functions time-invariant).
- **`run_experiments.py`** — CLI entry point dispatching to experiment runners.

### Teacher Capacity Semantics

- `-1`: No teacher signal (pure RL baseline)
- `0`: Random teacher (imagined goals/traps, uniform policy)
- `1..n_goals-1`: Mid-capacity (knows subset of goals)
- `n_goals`: Best teacher (knows all goals)

### Output

Results saved to `results/` as CSV files and figures under `results/figures/`.
