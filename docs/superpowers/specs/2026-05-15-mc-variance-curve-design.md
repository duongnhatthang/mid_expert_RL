# MC Variance Curve + α=0 Baseline Overlay — Design

**Date:** 2026-05-15
**Branch:** `advantage-product-and-mc-variance`
**Status:** Draft

## Motivation

The hypothesis sweep already runs `eval_n_episodes=20` Monte Carlo
rollouts at every eval tick (via `evaluate_policy`) and exposes
`mean_reward` + `std_reward` for the undiscounted trajectory return.
But:

1. The **variance** of the trajectory return — a basic diagnostic for
   how stochastic the policy is and how far it is from a deterministic
   optimum — is not plotted anywhere.
2. With sparse rewards, undiscounted return ∈ {0, 1} so its variance
   is just Bernoulli `p(1-p)` (redundant with `goal_rate`). The
   **discounted** return `G = Σ γ^t r_t` is richer: even with sparse
   rewards, first-hit time varies stochastically and produces a
   real-valued G distribution.
3. The item-3 advantage-alignment figure plots per-teacher curves at
   α=1.0 but doesn't show what α=0 (vanilla NPG) looks like on the
   same axes — readers can't tell if teacher baselines help, hurt, or
   are neutral relative to no-teacher.

This spec adds (a) two new per-eval-tick history fields
`mc_var_undiscounted` and `mc_var_discounted`, (b) a new figure
`plot_mc_variance_curve` that emits two PNGs (one per discount choice),
and (c) an α=0 baseline overlay added to BOTH the new variance figure
AND the existing item-3 `plot_advantage_alignment` figure.

## Goals

- Compute the discounted trajectory return per MC rollout inside
  `evaluate_policy`. Add `mean_reward_discounted` and
  `std_reward_discounted` to its return dict.
- Record `mc_var_undiscounted = std_reward**2` and
  `mc_var_discounted = std_reward_discounted**2` in every per-eval
  `history` entry produced by `run_experiment`.
- Emit two single-cell variance PNGs per `(mode, training_mode)` of
  the sweep — one undiscounted, one discounted — alongside the existing
  learning-curve and advantage-alignment outputs.
- Retrofit `plot_advantage_alignment` to overlay an α=0 vanilla-NPG
  baseline line on every figure it emits.
- Filenames and suptitles carry the (dist, H, B, α) config per the
  durable annotation rule.

## Non-Goals

- New CLI script. The metric is recorded universally in the eval loop;
  the figures are emitted by the existing sweep pipeline.
- `cap_zeta` mode variance figure (no-op, mirroring item 3 and learning
  curves).
- Backfill / migration of older pkls that lack the new fields. Plot
  functions skip entries missing the fields.
- Variance against non-zero baseline alphas. `baseline_alpha=0.0` is
  the only baseline value used by default; users can pass a different
  value or `None` via kwarg if needed.

## Architecture

Three-file change.

```
tabular_prototype/training.py
    + evaluate_policy now also tracks discounted return per rollout
    + return dict gains 'mean_reward_discounted', 'std_reward_discounted'

tabular_prototype/experiments.py
    + both eval-tick history.append sites gain two keys:
        'mc_var_undiscounted', 'mc_var_discounted'

run_hypothesis_sweep.py
    + plot_mc_variance_curve(...) — emits TWO PNGs per call
    + plot_advantage_alignment(...) — retrofitted with baseline_alpha kwarg
    + main() invokes plot_mc_variance_curve alongside plot_advantage_alignment
```

A new shared helper `_overlay_baseline_alpha(...)` factored out of the
post-loop baseline-drawing code keeps the retrofit DRY between the two
plot functions.

## Detail design

### `evaluate_policy` extension

In `tabular_prototype/training.py:256`, augment the rollout loop to
track discounted return alongside `episode_reward`. Each rollout
computes `G = Σ γ^t r_t` where `γ = compute_gamma_from_horizon(env.horizon)`
and `t` is the env step within the trajectory. The existing return
dict gains two keys:

```python
return {
    'mean_reward': float(np.mean(total_rewards)),
    'std_reward': float(np.std(total_rewards)),
    'mean_reward_discounted': float(np.mean(discounted_returns)),  # NEW
    'std_reward_discounted': float(np.std(discounted_returns)),    # NEW
    'goal_rate': float(np.mean(goal_reached)),
    'trap_rate': float(np.mean(trap_reached)),
    'mean_episode_length': float(np.mean(episode_lengths)),
}
```

Existing fields (`mean_reward`, `std_reward`, `goal_rate`, …) are
unchanged — purely additive.

### Schema addition (experiments.py)

Both eval-tick `history.append({...})` sites (exact-mode + trajectory-mode)
gain two keys:

```python
'mc_var_undiscounted': eval_results['std_reward'] ** 2,
'mc_var_discounted': eval_results['std_reward_discounted'] ** 2,
```

Same pattern as `adv_product_s0` from item 3.

### Plot function `plot_mc_variance_curve`

Mirrors `plot_advantage_alignment` structure, but the per-teacher
loop runs **twice** — once for each field — and emits two PNGs per
call. Signature:

```python
def plot_mc_variance_curve(
    all_results, mode, figures_dir,
    *,
    distance=6, horizon_type='small',
    alpha=1.0, baseline_alpha=0.0,
    budget_rank=-2,
):
```

The y-axis label and filename depend on the field being plotted:

| Field | y-axis label | Filename |
|-------|--------------|----------|
| `mc_var_undiscounted` | `Var(G) [undiscounted]` | `mc_variance_undiscounted_dist{d}_{h_type}_B{B}_alpha{α:.2f}.png` |
| `mc_var_discounted`   | `Var(G) [discounted]` | `mc_variance_discounted_dist{d}_{h_type}_B{B}_alpha{α:.2f}.png` |

The suptitle is consistent with `plot_advantage_alignment`:
`MC return variance ({mode} sweep) — dist=6, H=8 (small), B={B}, α=1.0`.

### Shared baseline-overlay helper

Both `plot_advantage_alignment` and `plot_mc_variance_curve` need to
overlay one α=0 baseline line on every figure. Factor the logic into
a module-private helper near the existing `_teacher_label` etc.:

```python
def _overlay_baseline_alpha(
    ax, all_results, mode, tcol, field_name, target,
    baseline_alpha,
):
    """Overlay one α=baseline_alpha vanilla-NPG baseline line on ax.

    At α=0 the teacher signal weight is zero so all teacher values
    are mathematically equivalent. We pick the first teacher value in
    canonical sort order as a deterministic representative; in practice
    the sweep code may also dedupe redundant α=0 runs, so this filter
    may match few or many seeds — both are fine.

    No-op when baseline_alpha is None or no matching results are found.
    Renders a black dashed line with light fill_between band, labeled
    `α={baseline_alpha} (vanilla NPG)`.
    """
    if baseline_alpha is None:
        return
    baseline_target = {**target, 'alpha': baseline_alpha}
    matching = [r for r in all_results
                if all(r.get(k) == v for k, v in baseline_target.items())
                and r['history']
                and r['history'][0].get(field_name) is not None]
    if not matching:
        return

    groups = defaultdict(list)
    for r in matching:
        groups[r[tcol]].append(r['history'])
    sorted_tvs = _sort_teacher_vals(pd.Series(list(groups.keys())), mode)
    tv_pick = sorted_tvs[0]  # canonical pick — all equivalent at α=0
    histories = groups[tv_pick]
    min_len = min(len(h) for h in histories)
    steps = np.mean([
        [h['steps'] for h in seed_hist[:min_len]]
        for seed_hist in histories
    ], axis=0)
    values = np.stack([
        [h[field_name] for h in seed_hist[:min_len]]
        for seed_hist in histories
    ], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.plot(steps, mean,
            label=rf'$\alpha={baseline_alpha}$ (vanilla NPG)',
            color='black', linestyle='--', linewidth=1.5,
            marker='s', markersize=3)
    ax.fill_between(steps, mean - std, mean + std,
                    alpha=0.15, color='black')
```

### Plot function structure (variance curve)

```python
def plot_mc_variance_curve(all_results, mode, figures_dir, *,
                            distance=6, horizon_type='small',
                            alpha=1.0, baseline_alpha=0.0,
                            budget_rank=-2):
    if mode == 'cap_zeta':
        return
    target = {'distance': distance, 'horizon_type': horizon_type, 'alpha': alpha}
    # ... resolve training_mode + calibration cell + budget (same as item 3) ...

    for field_name, y_label, file_tag in [
        ('mc_var_undiscounted', r'Var(G) [undiscounted]',
            'mc_variance_undiscounted'),
        ('mc_var_discounted',   r'Var(G) [discounted]',
            'mc_variance_discounted'),
    ]:
        matching = [r for r in all_results
                    if all(r.get(k) == v for k, v in target.items())
                    and r['history']
                    and r['history'][0].get(field_name) is not None]
        if not matching:
            continue

        # Group by teacher, render per-teacher lines (same as item 3)
        # ... draw lines with mean ± std band ...

        # Overlay baseline
        _overlay_baseline_alpha(
            ax, all_results, mode, tcol, field_name, target, baseline_alpha,
        )

        # Labels, suptitle, save
        # ... filename uses file_tag ...
```

### Retrofit to `plot_advantage_alignment`

Signature gains `baseline_alpha=0.0` keyword. Immediately after the
per-teacher rendering loop (after the existing `axhline(0, ...)` call),
invoke `_overlay_baseline_alpha(ax, all_results, mode, tcol,
'adv_product_s0', target, baseline_alpha)`.

The baseline is added BEFORE the legend / final layout calls so it
participates in `ax.legend(...)`.

Filename and suptitle are unchanged from item 3 (parameterized on the
primary `alpha` only — the baseline is implicit).

### Wiring into `main()`

In `run_hypothesis_sweep.py:main()`, after the existing
`plot_advantage_alignment(all_results, args.mode, figures_dir)` call
(added in item 3 Task 10), append one line:

```python
plot_mc_variance_curve(all_results, args.mode, figures_dir)
```

## Failure modes

| Condition | Behavior |
|-----------|----------|
| No primary-α runs in `all_results` | Function returns early, no PNG |
| No baseline-α runs in `all_results` | `_overlay_baseline_alpha` returns silently; primary lines still plotted |
| `baseline_alpha is None` | `_overlay_baseline_alpha` returns immediately; primary lines only (regression-safe) |
| Calibration JSON missing | Skip with INFO log |
| Older pkl entries lack `mc_var_*` fields | Filtered out per field — if all are filtered, skip with INFO log |
| Field present but `None` (older legacy) | Filtered out at the `r['history'][0].get(field_name) is not None` check |

## Tests

New file `tests/test_mc_variance_curve.py`:

### test_evaluate_policy_returns_discounted_fields

```python
def test_evaluate_policy_returns_discounted_fields():
    """evaluate_policy must return mean_reward_discounted and
    std_reward_discounted alongside the existing fields."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    rng = np.random.default_rng(0)
    result = evaluate_policy(env, policy, n_episodes=5, rng=rng)
    for k in ('mean_reward_discounted', 'std_reward_discounted'):
        assert k in result
        assert np.isfinite(result[k])
    # Discounted return must be ≤ undiscounted (rewards are non-negative
    # and γ < 1)
    assert result['mean_reward_discounted'] <= result['mean_reward'] + 1e-9
```

### test_mc_var_recorded_in_exact_history

```python
def test_mc_var_recorded_in_exact_history():
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history']
    for entry in result['history']:
        for k in ('mc_var_undiscounted', 'mc_var_discounted'):
            assert k in entry
            assert entry[k] is not None
            assert np.isfinite(entry[k])
            assert entry[k] >= 0.0
```

### test_mc_var_recorded_in_hybrid_history

Same pattern with `mode='hybrid', sample_budget=20, lr=1.0,
trajectories_per_update=1`.

### test_plot_mc_variance_curve_default_emits_two_pngs

Synthetic `all_results` with both α=1.0 (4 zetas × 3 seeds) and α=0
(1 baseline teacher × 3 seeds) cells filled. Both PNGs (undiscounted
and discounted) must exist with the expected parameterized filenames.

### test_plot_mc_variance_curve_baseline_alpha_none

Call with `baseline_alpha=None`. Monkey-patch `Figure.savefig` to
capture figures. For each captured figure, count `Line2D` artists
with `linestyle='--'` and `color='black'` — assert 0 (no baseline
overlay).

### test_plot_mc_variance_curve_cap_zeta_noop

Mirrors the cap_zeta no-op test from item 3.

### Updates to `tests/test_advantage_alignment.py`

- `_zeta_results_for_advantage_alignment()` — extend to also produce
  α=0 entries (one teacher value, 3 seeds) so the retrofit's baseline
  overlay has data to plot.
- `test_plot_advantage_alignment_default_path` — assert exactly one
  PNG with the expected filename (unchanged) AND assert the captured
  figure has a Line2D with `linestyle='--'` and `color='black'` (the
  baseline overlay).
- New test `test_plot_advantage_alignment_baseline_alpha_none` —
  passing `baseline_alpha=None` produces no dashed black line in the
  figure.

## Risks

| Risk | Mitigation |
|------|------------|
| Item 3's existing tests assume no baseline overlay | Updated synthetic data to include α=0 entries; assertions updated. New `baseline_alpha=None` test pins the no-baseline path. |
| Item 3's existing pkl smoke from Task 10 prints "no field-bearing runs" because older pkls lack `adv_product_s0` | Same behavior for `mc_var_*` — older pkls skip gracefully. Smoke test of the new function on older pkl is expected to return empty. |
| `_overlay_baseline_alpha` picks first sorted teacher value at α=0 — but the sweep may not have α=0 runs at every teacher | The helper iterates `groups` (only teacher values actually present) so the first sorted one is always real data. |
| Discount factor mismatch between training and eval | `evaluate_policy` recomputes γ from `env.horizon` via `compute_gamma_from_horizon`, matching the training loop's γ. |

## Acceptance

- A fresh sweep with `--mode zeta` and `--mode capability` produces:
  - `mc_variance_undiscounted_*.png` and `mc_variance_discounted_*.png`
    (two PNGs per directory).
  - `advantage_alignment_*.png` now has a dashed black α=0 line in
    its legend.
- Re-running with `--skip-run` against pkls produced by THIS branch
  regenerates all three figures from cached data.
- `--mode cap_zeta` runs unchanged — no variance PNGs, no errors.
- Older pkls (predating this branch's schema additions) skip the
  variance figure with an INFO log; no crashes.
- New tests pass; updated item-3 tests still pass.

## Out of scope (explicit)

- Per-teacher pdf / cdf of the trajectory return distribution.
- Higher moments (skew, kurtosis) of the return.
- Variance figure for `cap_zeta` mode.
- Baseline overlays in any figure other than item-3 advantage-alignment
  and item-6 variance curve.
