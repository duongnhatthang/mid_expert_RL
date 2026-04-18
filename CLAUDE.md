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
# Single quick experiment (exact NPG, 50 update steps)
python run_experiments.py --mode quick --seed 0 --teacher-capacity 1 --sample-budget 50

# 2x2 exploration matrix (budget × horizon, recommended for testing the hypothesis)
python run_experiments.py --mode 2x2 --grid-size 9 --n-seeds 10 --n-goals 3 --alpha 0.5

# Full parameter suite
python run_experiments.py --mode suite --grid-size 9

# Learning curves (auto-detects saturation if --sample-budget not given)
python run_experiments.py --mode learning_curve --grid-size 9 --n-seeds 5

# LR calibration sweep
python run_lr_sweep.py --budget 500 --n-seeds 5

# Hypothesis sweep (exact NPG, all alpha/budget/horizon/distance combos)
python run_hypothesis_sweep.py --mode capability --n-seeds 10 --n-workers 4

# Replot from cached results (no re-run). Do NOT pass --all-plots in regular
# runs — it enables distance_effect_alpha and heatmap_dist figures that are
# off by default and should stay off unless explicitly requested.
python run_hypothesis_sweep.py --mode capability --skip-run --output-dir results/capability_sweep/<timestamp>

# Plot existing results from CSV
python run_experiments.py --mode plot2x2 --output results/exploration_2x2_results.csv
```

### Run tests

```bash
PYTHONPATH=. pytest tests/ -v
# Single test file
PYTHONPATH=. pytest tests/test_run_experiments.py -v
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teacher-capacity` | varies | -1=no teacher, 0=random, 1..n_goals=partial/full knowledge |
| `--alpha` | 0.5 | Convex mixing weight α ∈ [0,1]: 0=vanilla NPG (Q^π only), 1=teacher only (A^μ) |
| `--sample-budget` | varies | Update steps (exact mode, default) or observations (sample mode) |
| `--grid-size` | 9 | N×N grid dimensions |
| `--n-goals` | 3 | Number of goal states |
| `--horizon` | 10×grid_size | Max episode length |
| `--n-seeds` | 30 | Random seeds for averaging (30 for smooth learning curves) |

## Architecture

### Training Loop (PAV-RL with Exact NPG)

The default training mode uses an **exact NPG (mirror descent) update** for tabular softmax:

```
θ[s,a] += lr · ((1-α) · Q^π(s,a) + α · A^μ(s,a))    ∀ s, a
```

Where `Q^π` is the student's exact action-value (computed via Bellman policy evaluation each update), `A^μ = Q^μ - V^μ` is the teacher's advantage, and α ∈ [0,1] is a convex mixing weight. Softmax normalization is automatic. Derived from Lemma F.2 (Agarwal et al. 2021, extended for PAV-RL).

**Vanilla NPG** is α=0: `θ += lr · Q^π` (no teacher signal).

A legacy **sample-based mode** (`exact_gradient=False`) is available, using trajectory-based policy gradient with the same convex combination: `eff = (1-α)·Q^π(s,a) + α·A^μ(s,a)`, where the gradient is computed over sampled trajectories. Budget = number of observations in this mode.

### Module Responsibilities

- **`tabular_prototype/environment.py`** — `GridEnv`: N×N grid with goals (reward=1), traps (reward=0), absorbing states. Student starts at center.
- **`tabular_prototype/teacher.py`** — Computes fixed `Q^μ(s,a)` and `V^μ(s)` via value iteration based on which goals the teacher "knows" (its capacity). Never updated during training.
- **`tabular_prototype/student.py`** — `TabularSoftmaxPolicy`: trainable `θ[state, action]` parameters, trajectory collection.
- **`tabular_prototype/training.py`** — Exact NPG update (`exact_npg_update`), sample-based PAV-RL gradient (`compute_pav_rl_gradient`), exact Q^π via Bellman policy evaluation (`compute_student_qvalues`), state-action visitation tracking, and policy updates.
- **`tabular_prototype/experiments.py`** — Experiment orchestration: `run_experiment()`, `run_2x2_exploration_experiment()`, `run_learning_curve_experiment()`, `run_experiment_suite()`.
- **`tabular_prototype/visualization.py`** — Policy grids, advantage heatmaps, learning curves, result bar charts.
- **`tabular_prototype/config.py`** — Discount factor: `γ = 1 - 1/H` (makes value functions time-invariant).
- **`run_experiments.py`** — CLI entry point dispatching to experiment runners.

### Teacher Capacity Semantics

- `-1`: No teacher signal (legacy; use α=0 instead for vanilla NPG)
- `0`: Random teacher (uniform policy)
- `1..n_goals-1`: Mid-capacity (knows subset of goals)
- `n_goals`: Best teacher (knows all goals)

Note: With the convex combination, **α=0 is vanilla NPG** regardless of teacher capacity (teacher signal has zero weight). The sweep includes α=0 and skips redundant teacher variations for it.

### Output

Results saved to `results/` as CSV files and figures under `results/figures/`.
