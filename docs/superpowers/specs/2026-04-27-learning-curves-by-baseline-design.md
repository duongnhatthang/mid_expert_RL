# Learning Curves by Baseline — Design

**Date:** 2026-04-27
**Branch:** `learning-curves-by-baseline`
**Status:** Draft

## Motivation

The hypothesis sweeps in `run_hypothesis_sweep.py` currently emit only end-of-training summary plots (heatmaps, reward-vs-teacher curves, visitation grids). When debugging "why does ζ=0.67 outperform ζ=1.0 at this budget?" we cannot see the *trajectory* of learning — only the final point. The pickled sweep results already store per-eval-step `history` lists, so this information is sitting unused.

This spec adds an auto-generated set of **learning-curve figures**, one per `(distance, alpha)` combo, that overlay the per-step `V^π(s₀)` evolution of every teacher baseline (ζ values for the zeta sweep, capability values for the capability sweep) across both horizon types and all calibrated budgets.

## Goals

- Visualize learning trajectories of every teacher baseline within a single `(distance, alpha)` slice.
- Reuse existing pickled sweep data — no re-run, no extra training.
- Hook into the same post-run plotting block that already produces visitation grids and heatmaps, so `--skip-run` regenerates the new figures alongside the existing ones.

## Non-Goals

- `cap_zeta` mode learning curves. The 16-line legend reads poorly on a learning curve and the user explicitly excluded it.
- Alternate y-axis metrics. Only `exact_V_start` (V^π(s₀), discounted) is plotted. A future CLI flag could add `mean_reward` or `goal_rate` if needed.
- Re-running experiments at finer evaluation granularity. The existing `eval_interval=5` cadence is sufficient.
- Image-diff regression testing. Tests check file existence and non-zero size only.

## Data source

Each sweep mode pickles its full per-seed results to `<output_dir>/<mode>_sweep_results.pkl` (see `run_hypothesis_sweep.py:561`). Each result dict contains a `history` list with entries like:

```python
{
    'steps': int,                 # update-count at evaluation
    'mean_reward': float,
    'goal_rate': float,
    'exact_V_start': float,       # V^π(s₀) discounted — the metric we plot
    'exact_V_start_undiscounted': float,
    'unique_sa': int,
    'state_entropy': float,
}
```

History entries are recorded every `eval_interval=5` updates plus a final entry at the last step. `exact_V_start` is computed via Bellman policy evaluation (`compute_student_qvalues`) at each evaluation tick — it is the canonical metric the rest of the codebase reports. In `mode="exact"` the x-axis ticks are update steps; in `hybrid`/`sample` they are update counts whose total environment steps are bounded by `sample_budget` (observations).

## Figure layout

**One PNG per `(distance, alpha)` combo.** With the standard sweep that means 3 distances × 4 alphas = **12 PNGs per sweep mode**.

Each figure is a 2×N grid:

- **Rows** (fixed at 2): `horizon_type` ∈ {`small`, `large`}.
- **Columns** (variable): the budgets calibrated for `(distance, horizon_type)`. Budgets are loaded from the calibration JSON during the sweep and stored on every result dict as `sample_budget`. Different `(distance, horizon_type)` cells may have different budget lists, so column count is `max(len(budgets[h_type]) for h_type in ['small','large'])` and missing cells are rendered as hidden axes (set `ax.set_visible(False)`) so each row's columns line up by budget rank.
- **Subplot titles:** `H={h_type}, B={budget}`.
- **Suptitle:** `Learning curve ({mode} sweep) — dist={d}, α={a}` (constructed inside `plot_learning_curves`, where `{mode}` is `zeta` or `capability`).
- **Shared y-label:** `V^π(s₀)`. **Shared x-label:** `update step`.
- `sharex=False, sharey=False` — V-magnitude varies a lot between budgets and horizons; sharing would compress small-budget cells into illegibility. Each subplot autoscales.

### Inside each subplot

- One line per teacher value (ζ ∈ ZETA_VALUES for zeta mode, c ∈ {−1, 0, 1, 2, 3} for capability mode).
- Color/order: reuse `_sort_teacher_vals(...)` (line 606 of `run_hypothesis_sweep.py`) and the existing label helper `_teacher_label(mode, val)` (line 587) so legend formatting matches the rest of the sweep figures (`$\zeta=...$`, `$c=...$`, `no teacher`, `uniform`).
- Each line is the **mean across seeds** at matching `steps`. A ±1σ shaded band is drawn with `ax.fill_between(..., alpha=0.2)`.
- Legend on the **rightmost subplot of each row** only (avoids redundancy in 8-cell figures).

### Capability mode at α=0

`c=−1` ("no teacher") and `c=0` ("uniform") only exist at α=0 in the sweep results — at α>0 the sweep skips redundant teacher variations. The plotting code groups by what is actually present in the pkl, so at α=0 the legend has 5 lines, at α>0 it has 3 (`c=1, c=2, c=3`). No special-casing required.

## Aggregation across seeds

```
for (distance, alpha) in unique_pairs:
    fig = make 2 × N_cols subplots
    for h_type in [small, large]:
        budgets_here = sorted({r['sample_budget'] for r in results if r['distance']==distance and r['horizon_type']==h_type})
        for col, budget in enumerate(budgets_here):
            ax = axes[row(h_type)][col]
            for tv in sorted_teacher_vals:
                rs = [r for r in results if r matches (distance, alpha, h_type, budget, tv)]
                # rs has n_seeds entries, each with a history list
                # All seeds for the same config share the same `steps` schedule
                # because eval_interval and sample_budget are deterministic.
                steps = rs[0]['history'] -> [h['steps'] for h in history]
                values = stack([[h['exact_V_start'] for h in r['history']] for r in rs])  # shape (n_seeds, n_eval_points)
                mean = values.mean(axis=0); std = values.std(axis=0)
                ax.plot(steps, mean, label=teacher_label(mode, tv))
                ax.fill_between(steps, mean-std, mean+std, alpha=0.2)
        # hide trailing unused axes in this row
```

**History-length safety net.** All seeds for the same `(mode, tv, alpha, budget, h_type, dist)` must produce the same number of evaluation points because `eval_interval` and `sample_budget` are deterministic. We assert this in code: if any seed disagrees, raise with a clear message naming the offending config — this would indicate a pkl from a stale run and should not be silently averaged.

## Code placement

- New top-level function `plot_learning_curves(all_results, mode, figures_dir)` in `run_hypothesis_sweep.py`, defined alongside `plot_visitation_grids` (around line 1468).
- Mode gate: returns immediately for `mode == 'cap_zeta'` (no-op).
- Output directory: `<figures_dir>/learning_curves/`. Created if missing. Keeps the new figures from cluttering the flat `figures/` listing already populated by reward/heatmap/visitation plots.
- Output filename: `learning_curve_dist{d}_alpha{a:.2f}.png`.
- Wired into the existing post-run / `--skip-run` plotting block at the call site that currently invokes `plot_visitation_grids` (around `run_hypothesis_sweep.py:2293`). Add one line: `plot_learning_curves(all_results, args.mode, figures_dir)` immediately after the visitation call. Both functions consume the same `all_results` list, so no extra loading.

## Tests

One pytest-style test in `tests/test_run_hypothesis_sweep_plots.py` (new file unless an equivalent already exists; if there is already a sweep-plot test module, append):

- Build a synthetic `all_results` list covering 2 distances × 2 alphas × 2 horizon types × 2 budgets × 2 teacher values × 2 seeds, with hand-crafted `history` lists of length 4 each (so mean/std arithmetic is exercised).
- Call `plot_learning_curves(all_results, mode='zeta', figures_dir=tmp_path)`.
- Assert: `tmp_path/learning_curves/` exists and contains exactly 4 PNGs named `learning_curve_dist{d}_alpha{a:.2f}.png`, each non-empty (`> 1 KB` is a reasonable floor — even a blank matplotlib PNG is several KB, but a truly broken save produces 0 bytes).
- Repeat for `mode='capability'` with capacity values including `-1`.
- Negative test: `mode='cap_zeta'` produces no files.

No image-diff or visual regression. The math is `mean`/`std` of well-defined arrays — if those break, unit tests on the aggregation helper would catch it more reliably than image comparison.

## Risks and edge cases

| Risk | Mitigation |
|------|------------|
| Pkl entries from older runs may lack `exact_V_start` (pre-explicit-policy refactor) | Skip such entries with a one-line warning; if all seeds for a config are skipped, drop the line entirely from that subplot rather than crashing. |
| Different seeds disagree on `history` length for the same config | Assert equal lengths; raise with config tuple in the message. Indicates a corrupted/mixed pkl. |
| One row has more budgets than the other (e.g. 4 small, 3 large) | `n_cols = max(...)`; render unused cells as hidden axes (`set_visible(False)`) so the visible cells stay aligned by budget rank. |
| Capability mode at α=0 has 5 lines; at α>0 has 3 | Group on what's actually in the pkl. No special case needed. |
| `cap_zeta` results passed in by mistake | Early return; logged at INFO level. |

## Out of scope (explicit)

- No new sweep CLI flags (no `--metric`, no `--no-learning-curves`).
- No re-runs at finer eval cadence.
- No `cap_zeta` learning curves.
- No new pickle schema; all data comes from the existing pkl.
- No changes to the experiment loop or `history` schema.

## Acceptance

- A fresh sweep with `--mode zeta` and `--mode capability` produces a `figures/learning_curves/` directory containing 12 PNGs each (3 distances × 4 alphas).
- Re-running with `--skip-run` against an existing output directory regenerates the same figures from the cached pkl.
- `--mode cap_zeta` runs unchanged — no learning-curve PNGs produced, no errors.
- New unit test passes; existing test suite is unaffected.
