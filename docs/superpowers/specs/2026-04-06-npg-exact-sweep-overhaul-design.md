# Design: Exact NPG, Sweep Overhaul, and Visualization Improvements

**Date:** 2026-04-06
**Scope:** Items 1-9 from the user's request — touching training loop, experiments, visualization, and sweep configuration.

---

## 1. Learning Curve Until Saturation

### Current behavior
`run_learning_curve_experiment` runs every baseline for a fixed `sample_budget`.

### New behavior
1. Run vanilla NPG (α=0) first with a generous upper-bound budget.
2. Detect saturation: sliding window of last `W` eval points (default W=10). If the absolute change in mean reward over the window is below ε=0.005 for 3 consecutive checks, declare saturated. Record saturation step count `T_sat`.
3. Run all other baselines for `T_sat` update steps (exact mode) or equivalent observations (sample mode).
4. **Undiscounted V^π(s_0):** At each eval point, also compute V^π with γ_undiscounted = 1 - 1e-8 (numerically stable substitute for γ=1; convergence guaranteed by absorbing states). Store as `exact_V_start_undiscounted` in history.
5. **Plot conventions:** Solid line = undiscounted V^π(s_0). Dashed line = discounted V^π(s_0). Same color per baseline.

### Files changed
- `tabular_prototype/experiments.py` — `run_learning_curve_experiment`
- `tabular_prototype/training.py` — add `compute_student_qvalues` call with γ_undiscounted
- `tabular_prototype/visualization.py` — learning curve plot function

---

## 2. Vanilla NPG Baseline

### Design
Vanilla NPG is simply PAV-RL with α=0 (no teacher signal). No separate code path needed.

- With the convex combination from item 8, α=0 gives `eff = Q^π(s,a)` — pure student Q-values.
- The α=0 sweep point already covers this. Clean up the old `teacher_capacity=-1` special-casing to use α=0 instead.
- **Naming:** In all plots and tables, α=0 is labeled "Vanilla NPG" (regardless of teacher capacity, since teacher is ignored).
- **Coverage:** Include α=0 in every experiment mode: 2x2, learning curve, hypothesis sweep, and zeta-expert experiments.
- Skip sweeping over teacher capacity/zeta values when α=0 (results are identical since teacher signal has zero weight). Run once and replicate the result row for display.

### Files changed
- `tabular_prototype/experiments.py` — all experiment runners
- `tabular_prototype/visualization.py` — label logic
- `run_hypothesis_sweep.py` — sweep config

---

## 3. Budget Sweep Calibration

### Current values
- `CAP_BUDGET_VALUES = [512, 2048, 5000, 20000]`
- `ZETA_BUDGET_VALUES = [1000, 5000, 20000]`

### New approach
- After implementing item 1, run the learning curve experiment on the new 9x9 grid to find T_sat (the update step count where vanilla NPG saturates).
- Set sweep budget values as fractions of T_sat: `{T_sat//10, T_sat//4, T_sat//2, T_sat}`.
- For exact gradient mode, budget = update steps. For sample mode, budget = observations.
- Exact values will be filled in after the first empirical run. The code will accept budget values as parameters rather than hard-coding.

### Files changed
- `run_hypothesis_sweep.py` — budget constants

---

## 4. Budget Logic — Truncation Fix

### Current issue
The training loop at `experiments.py:69-71` can overshoot `sample_budget` by up to `trajectories_per_update * horizon` steps, because the while-loop checks only at the top.

### Fix (sample-based mode only; exact mode doesn't use trajectories)
After collecting trajectories, compute `remaining = sample_budget - total_steps`. If negative, truncate: drop trailing trajectories (or truncate the last one) so `total_steps` does not exceed `sample_budget`. Specifically:
```python
# After collecting trajectories
step_counts = [len(traj) for traj in trajectories]
cumsum = np.cumsum(step_counts)
remaining = sample_budget - total_steps
if remaining < sum(step_counts):
    # Keep only trajectories that fit within budget
    keep = int(np.searchsorted(cumsum, remaining, side='right'))
    trajectories = trajectories[:max(1, keep)]  # always keep at least 1
```

### Files changed
- `tabular_prototype/experiments.py` — `run_experiment` training loop

---

## 5. Full Sweep Analysis for Vanilla NPG

### Design
Since vanilla NPG is α=0 (item 2), it is automatically included in all sweep configurations. The sweep already iterates over alpha values — adding 0.0 to the alpha list covers this.

Generate the same outputs for vanilla NPG as for all other baselines:
- Visitation distribution grids
- Reward heatmaps
- Distance effect plots
- Learning curves

The vanilla NPG row appears as α=0 in all comparison plots. No separate experiment runner needed.

### Files changed
- `run_hypothesis_sweep.py` — add 0.0 to `ALPHA_VALUES`
- Visualization functions already handle this via the alpha sweep

---

## 6. 9x9 Grid with Increased Distance

### Changes
- `GRID_SIZE`: 8 → 9
- Start position: (4, 4) — center of 9x9 grid
- Default goal distance in `generate_equidistant_goals`: `min(4,4,4,4) = 4` (unchanged numerically, but now the grid is larger)
- Sweep distances: `{1, 3, 5, 7}` → `{2, 4, 6, 7}`
  - Distance 7 from (4,4) on 9x9: cells like (0,1), (1,0), (8,3), (3,8) — valid
  - Distance 8 would reach corners (0,0) — only 4 cells, limiting n_goals options
- Update `GOAL_POSITIONS` dict for new distances
- Update `compute_exploration_thresholds` calls (automatic — depends on grid_size)

### Verification
`generate_equidistant_goals(9, 3, distance=7)` must produce 3 valid equidistant goals. On a 9x9 grid, distance=7 from (4,4) gives cells on the diamond: there are 2×7=14 candidate cells (minus any that fall off-grid). For 9x9, all cells at Manhattan distance 7 from (4,4) that fit: let's enumerate — d7 cells include (0,1),(0,7),(1,0),(1,8),(7,0),(7,8),(8,1),(8,7) and more. At least 8 cells, enough for 3 goals.

### Files changed
- `run_hypothesis_sweep.py` — `GRID_SIZE`, `GOAL_POSITIONS`, `DISTANCES`
- `run_experiments.py` — default `--grid-size`
- `tabular_prototype/experiments.py` — default parameter values

---

## 7. Exact NPG Update (Default Mode)

### Mathematical formulation
From Lemma F.2 (Agarwal et al. 2021 extended for PAV-RL):

```
θ_{t+1}[s,a] = θ_t[s,a] + lr · ((1-α)·Q^π(s,a) + α·A^μ(s,a))    ∀ s, a
```

Softmax normalization handles Z^t(s) automatically. The ν (state-dependent offset from the NPG derivation) cancels in the softmax.

Vanilla NPG (α=0): `θ[s,a] += lr · Q^π(s,a)`

### Implementation

New function `exact_npg_update` in `training.py`:
```python
def exact_npg_update(
    policy: TabularSoftmaxPolicy,
    Q_pi: np.ndarray,
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    lr: float,
):
    """Exact NPG update: θ += lr · ((1-α)·Q^π + α·A^μ) for all (s,a)."""
    if Q_mu is not None and V_mu is not None:
        A_mu = Q_mu - V_mu[:, None]
        policy.theta += lr * ((1.0 - alpha) * Q_pi + alpha * A_mu)
    else:
        policy.theta += lr * Q_pi
```

### Training loop (exact mode)
```python
for step in range(budget):  # budget = number of update steps
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha, lr)
    # Eval at intervals (no trajectories needed for training)
```

### Training loop (sample mode — legacy, opt-in)
Current trajectory-based loop with budget = observations. Keep `compute_pav_rl_gradient` for this mode. Also apply item 8 normalization to the sample-based gradient.

### Mode selection
Add `exact_gradient: bool = True` parameter to `run_experiment`. Default is exact (new behavior). Sample mode available via `exact_gradient=False`.

### Figure labeling
- Exact mode: x-axis labeled "Update Steps"
- Sample mode: x-axis labeled "Samples (observations)"

### Files changed
- `tabular_prototype/training.py` — add `exact_npg_update`, update `compute_pav_rl_gradient` for item 8
- `tabular_prototype/experiments.py` — `run_experiment` branching on `exact_gradient`
- `tabular_prototype/visualization.py` — x-axis label logic

---

## 8. Alpha Normalization (Convex Combination)

### Change
From additive:
```
eff = Q^π(s,a) + α · A^μ(s,a)       α ∈ [0, ∞)
```
To convex combination:
```
eff = (1-α) · Q^π(s,a) + α · A^μ(s,a)    α ∈ [0, 1]
```

### Sweep values
Old: `ALPHA_VALUES = [0.5, 2, 5, 10, 50]`
New: `ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`

α=0.0 is vanilla NPG (item 2). This means no separate vanilla NPG code — just α=0 with any teacher.

### Learning rate sweep
Preliminary LR sweep to find optimal range:
`LR_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]`

Run on a representative config (9x9 grid, distance=4, α=0, large horizon) and pick the LR that converges fastest without instability. Then fix LR for the main sweep.

The LR sweep is a one-time calibration step, not part of the main hypothesis sweep.

### Impact on existing code
- `compute_pav_rl_gradient` (sample mode): change line 156 from `G_t + alpha * A_mu` to `(1-alpha) * G_t + alpha * A_mu`
- `exact_npg_update` (exact mode): uses `(1-alpha) * Q_pi + alpha * A_mu` (as specified in item 7)
- All experiment functions: alpha parameter now bounded to [0, 1]

### Files changed
- `tabular_prototype/training.py` — both gradient functions
- `run_hypothesis_sweep.py` — `ALPHA_VALUES`
- `run_experiments.py` — alpha default and help text

---

## 9. Visualization Fixes

### 9a. Colorbar overlap fix
Current: `fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.8)` — colorbar steals space from axes, causing overlap.

Fix: Use `GridSpec` to reserve explicit space for the colorbar:
```python
from matplotlib.gridspec import GridSpec
gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.05])
# Plot heatmaps in gs[:, :-1], colorbar in gs[:, -1]
```

### 9b. More heatmaps per figure (nested grid)
Create a larger composite figure:
- Outer grid: configurations (e.g., budget × horizon = 2×2 or more)
- Inner grid within each outer cell: teacher param rows × alpha columns
- Use `GridSpecFromSubplotSpec` for nesting
- Dynamically size figure based on number of panels

### 9c. Show all baselines
Currently only 3 teacher values shown. Change to show ALL:
- Capability mode: all of {α=0 (vanilla NPG), cap=0 (random), cap=1, ..., cap=n_goals}
- Zeta mode: all of {α=0 (vanilla NPG), ζ=0.0, 0.25, 0.5, 0.75, 1.0}
- Dynamically compute figure size: `figsize = (cell_size * n_cols + 2, cell_size * n_rows + 2)` based on actual number of baselines

### Files changed
- `tabular_prototype/visualization.py` — `visualize_visitation_comparison_grid`, new composite figure function
- `run_hypothesis_sweep.py` — `plot_visitation_grids` (pass all baselines)

---

## Summary of File Changes

| File | Items |
|------|-------|
| `tabular_prototype/training.py` | 7, 8 (exact NPG update, convex combination) |
| `tabular_prototype/experiments.py` | 1, 2, 4, 7 (saturation detection, α=0 baseline, truncation, exact mode) |
| `tabular_prototype/visualization.py` | 1, 9 (undiscounted V line, colorbar fix, nested grids, all baselines) |
| `run_hypothesis_sweep.py` | 2, 3, 5, 6, 8 (grid size, distances, alpha/budget values, vanilla NPG in sweeps) |
| `run_experiments.py` | 6 (default grid size, CLI help text) |
| `tabular_prototype/student.py` | No changes needed |
| `tabular_prototype/teacher.py` | No changes needed |
| `tabular_prototype/config.py` | No changes needed |

## Implementation Order

1. **Item 8** (alpha normalization) — foundational change, affects all other items
2. **Item 7** (exact NPG update) — new default training mode
3. **Item 2** (vanilla NPG = α=0) — labeling and sweep inclusion
4. **Item 6** (9x9 grid) — new environment defaults
5. **Item 4** (budget truncation) — quick fix for sample mode
6. **Item 1** (learning curve saturation) — depends on items 7, 8
7. **Item 3** (budget calibration) — depends on item 1 empirical results
8. **Item 5** (full NPG sweep) — depends on items 2, 7
9. **Item 9** (visualization) — independent, can be parallelized
