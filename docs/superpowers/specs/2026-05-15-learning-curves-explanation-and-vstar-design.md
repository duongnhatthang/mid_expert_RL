# Learning-Curve Figures: Explanation Note + V*(s_0) Skyline — Design

**Date:** 2026-05-15
**Branch:** `learning-curves-by-baseline` (pre-PR follow-up)
**Status:** Draft

## Motivation

The `plot_learning_curves` figures shipped earlier on this branch have three
recurring oddities that confuse readers of the PNGs:

1. **Missing subplots.** At the smallest budgets, no seed completes ≥1
   evaluation tick (`update_count % eval_interval == 0 or is_last`), so the
   group is filtered out and the column never appears. Figures end up with
   `3+4` or `4+3` cells instead of `4+4`.
2. **Sparse points.** Some surviving subplots have only 1–2 evaluation ticks
   per baseline because update count just barely crosses
   `eval_interval=5`.
3. **Different x positions per baseline.** In `hybrid`/`sample` modes,
   trajectory length varies stochastically with policy, so equal env-step
   budgets produce different update counts per baseline. Lines start and end
   at different update-step values within the same subplot.

These quirks are real — not artifacts to "fix" — but the figures don't
explain them. Anyone reading the PNG without the codebase open has no way to
tell. Adding a short note on the figure removes the need to chase a
caption.

The same figures also lack a **performance ceiling**. The Bellman-optimal
`V*(s_0)` for the env is computable from oracle env info and is the
canonical "best any policy could do" line. Without it, readers can't tell
whether a baseline at 0.42 is good (close to ceiling) or bad (far from
ceiling).

## Goals

- Add a one-paragraph explanation note inside every `plot_learning_curves`
  figure covering the three quirks.
- Draw a dashed `V*(s_0)` skyline on every subplot, with the value in the
  legend alongside the existing baseline curves.
- No new sweep runs, no pickle-schema changes, no new CLI flags.

## Non-Goals

- Rendering `V*(s_0)` for `cap_zeta` mode (still no-op, as before).
- Changing the experiment loop, eval cadence, or `history` schema.
- Fixing the "missing subplots" or "sparse points" upstream — these are
  pre-existing edge cases of `experiments.py` with the `is_last` eval logic
  at tiny budgets. Out of scope for this branch; tracked separately.

## Architecture

Single-function change inside `run_hypothesis_sweep.py` →
`plot_learning_curves(...)`. The function already iterates per
`(distance, alpha)` figure and per `(h_type, budget)` subplot, so both
additions slot into the existing loops:

```
plot_learning_curves(all_results, mode, figures_dir):
    [...existing aggregation into groups...]
    v_star_cache = {(dist, h_type): V*(s_0)}            # NEW (inline, no helper)
    for (dist, alpha) in da_pairs:
        fig, axes = plt.subplots(...)
        for h_type in [small, large]:
            for col_idx, budget in ...:
                ax = axes[row_idx][col_idx]
                [...existing per-baseline lines...]
                v_star = v_star_cache[(dist, h_type)]   # NEW
                ax.axhline(v_star, '--', label=...)     # NEW
        fig.suptitle(main_title, y=0.985)               # adjusted y
        fig.text(0.5, 0.945, explanation_note, ...)     # NEW
        fig.tight_layout(rect=[0, 0, 1, 0.92])          # was 0.96
        fig.savefig(...)
```

No new modules. No new public APIs. The V* computation reuses two existing
public helpers in `tabular_prototype/teacher.py`:
`build_optimal_policy(env, goals, gamma)` and
`evaluate_policy_values(env, policy, gamma)`.

## Detail design

### Suptitle + explanation note

```python
fig.suptitle(
    f'Learning curve ({mode} sweep) — '
    f'dist={dist}, ' + rf'$\alpha={alpha}$',
    fontsize=11, y=0.985,
)
fig.text(
    0.5, 0.945,
    "Cells may be missing when no seed completed ≥1 evaluation tick "
    "(budget < eval_interval). Lines for different baselines start at "
    "different update steps because trajectory length varies "
    "stochastically: equal env-step budgets yield different update counts "
    "per baseline.",
    ha='center', va='top', fontsize=7.5, color='dimgray', wrap=True,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
```

The text is fixed wording — it applies to every figure regardless of mode,
distance, or alpha. No conditional rendering: even when a particular figure
happens to have all 8 cells filled, the note still describes the regime
correctly.

### V*(s_0) skyline

**Computation strategy — inline cache in `plot_learning_curves`.** V* depends
only on `(distance, horizon_type)` (γ = 1 − 1/H), not on alpha, budget,
seed, or training mode. So the unique values per call are
`|DISTANCES| × |HORIZON_TYPES| = 6` at most, computed once and looked up
per subplot.

```python
from tabular_prototype.teacher import (
    build_optimal_policy, evaluate_policy_values,
)
from tabular_prototype.config import compute_gamma_from_horizon

horizons = _get_horizons()
goal_pos = _goal_positions(mode)

# Built from keys actually present in `groups` (not from DISTANCES × HORIZON_TYPES)
# so synthetic test data with a subset of distances doesn't trigger lookups
# for missing goal-position keys.
v_star_cache: dict[tuple[int, str], float] = {}
for (dist, _alpha, h_type, _budget, _tv) in groups:
    key = (dist, h_type)
    if key in v_star_cache:
        continue
    h_val = horizons[h_type]
    gamma = compute_gamma_from_horizon(h_val)
    env = GridEnv(grid_size=GRID_SIZE, goals=goal_pos[dist], horizon=h_val)
    pi_star = build_optimal_policy(env, env.goals, gamma)
    _, V_star = evaluate_policy_values(env, pi_star, gamma)
    s0_idx = env.state_to_idx(env.start)
    v_star_cache[key] = float(V_star[s0_idx])
```

**Drawing per subplot:**

```python
v_star = v_star_cache[(dist, h_type)]
ax.axhline(
    v_star, linestyle='--', color='black', linewidth=1.2,
    label=rf'$V^*(s_0)={v_star:.3f}$',
)
```

The `axhline` call lives inside the existing per-subplot block, so the line
participates in `ax.legend(...)` naturally on the rightmost visible
subplot of each row (same convention as the baseline legend).

### Y-axis impact

`axhline` extends the data range, so matplotlib autoscaling may bump the
y-axis upward when V* exceeds the curves. That's the desired effect —
it makes the gap-to-ceiling visible. No code change needed.

### Failure modes

- **`v_star_cache` lookup miss.** Should never happen because the cache is
  built from the same keys we then index. Defensive `cache.get(key)`
  fallback would silently swallow bugs — prefer `cache[key]` and let it
  raise. If a test data row has an exotic distance not in
  `goal_pos`, the loop above raises `KeyError(dist)` on the `goal_pos[dist]`
  lookup, which is the right behavior.
- **`build_optimal_policy` returns degenerate policy.** Already handled
  internally by `_solve_discounted_values` (10000-iteration cap, tolerance
  exit). V* will be 0 in the pathological no-reachable-goal case — the
  dashed line will sit at the x-axis. Fine.

## Tests

Add to `tests/test_run_hypothesis_sweep_plots.py`:

### test_v_star_drawn_on_each_subplot

Monkey-patch `matplotlib.figure.Figure.savefig` to capture the figure
object instead of writing to disk. Call `plot_learning_curves(_zeta_results(), 'zeta', tmp_path)`. For
each captured figure, walk every visible axes and assert at least one
`Line2D` has `get_linestyle() == '--'` and `get_color() == 'black'`. This
catches the case where `axhline` is dropped or guarded behind a flag.

### test_suptitle_has_explanation_note

Same monkey-patch. After calling `plot_learning_curves`, assert at least
one figure-level `Text` artist contains the substring
`"budget < eval_interval"`. Catches accidental removal of the note text.

### test_v_star_finite_and_in_range

Direct unit-style test of the cache logic: import
`build_optimal_policy` and `evaluate_policy_values`, build a `GridEnv` for
`(distance=4, horizon='small')`, compute V*(s_0). Assert `0 < V* <= 1.0`
(must be positive because at least one goal is reachable in ≤ H steps;
must be ≤ 1.0 because the maximum undiscounted return is 1 and discount
shrinks it). Catches sign errors / wrong-state-index bugs.

### Existing tests

- `test_plot_learning_curves_zeta` and `test_plot_learning_curves_capability`:
  no change needed. The synthetic data uses `dist in [4, 6]` which are
  valid keys in `ZETA_GOAL_POSITIONS` / `CAP_GOAL_POSITIONS`.
- `test_plot_learning_curves_ragged_histories`: uses `dist=4`. Also valid.
- `test_plot_learning_curves_cap_zeta_noop`: unchanged. cap_zeta mode still
  early-returns before V* computation.

## Risks

| Risk | Mitigation |
|------|------------|
| Test mode synthetic data uses a distance not in goal-position dicts | Tests already use `[4, 6]` which are valid — caught at code-review of test additions if a new test introduces an exotic distance. |
| `axhline` label clutters the legend on the rightmost subplot | Legend already has 4–5 baseline entries; adding V* makes 5–6 — readable. Acceptable. |
| Suptitle note breaks at narrow figure widths | `wrap=True` in `fig.text` handles narrow widths; `rect=[0,0,1,0.92]` gives the note enough vertical room. |
| User reads V* line as a baseline curve | Black + dashed + explicit `V^*(s_0)=...` label distinguishes it visually and semantically from the colored solid baseline lines. |

## Out of scope (explicit)

- `cap_zeta` learning curves (still no-op).
- Fixing the upstream `is_last` eval bug for very small budgets.
- Adding `V*` to the pickled result schema.
- Per-seed V* averaging (V* is deterministic per `(dist, h_type)` because
  goals are deterministic and there are no traps in the sweep envs).
- Showing V* in `cap_zeta` mode or in any other figure type (visitation
  grids, reward bars, heatmaps).

## Acceptance

- `pytest tests/test_run_hypothesis_sweep_plots.py -v` passes — all
  pre-existing tests plus the three new ones.
- Re-running `python run_hypothesis_sweep.py --mode zeta --skip-run
  --output-dir <existing>` produces updated PNGs that show:
  (a) a black dashed V*(s_0) line on every subplot,
  (b) V*(s_0) entry in the legend on the rightmost subplot of each row,
  (c) a dim-gray multi-line explanation note immediately under the main
      title on every figure.
- No other figure types (visitation, heatmaps, reward bars) are affected.
- `--mode cap_zeta` is unchanged — no V* line, no note, function still
  early-returns.
