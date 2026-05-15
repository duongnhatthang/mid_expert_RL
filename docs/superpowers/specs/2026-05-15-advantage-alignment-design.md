# Advantage Alignment Plot — Design

**Date:** 2026-05-15
**Branch:** `advantage-product-and-mc-variance`
**Status:** Draft

## Motivation

At α=1 the student policy is driven entirely by the teacher's advantage
signal `A^μ`. Whether that signal is useful depends on the alignment of
A^μ with the student's own (current) advantage A^π — when the two
agree on action ordering at the start state, the update accelerates
learning; when they disagree, it stalls or harms it. We have no
visualization of this alignment over training. The hypothesis sweep
already records `exact_V_start` (the discounted V^π(s_0)) per eval tick,
but nothing about the geometry of the advantage signal at s_0.

This spec adds an **advantage-alignment scalar** to every per-seed
eval-tick history entry:

```
g^π(t) = E_{a ~ π^t(·|s_0)} [ A^{π^t}(s_0, a) · A^μ(s_0, a) ]
       = Σ_a π^t(a|s_0) · (Q^π(s_0,a) - V^π(s_0)) · (Q^μ(s_0,a) - V^μ(s_0))
```

…and one figure that plots `g^π(t)` over training for a fixed single
cell (`dist=6, h_type='small', alpha=1.0, B=calibrated_budgets[-2]`),
one line per teacher baseline, mean ± 1σ band across seeds.

## Goals

- Record `adv_product_s0` in every per-eval `history` entry produced by
  `tabular_prototype/experiments.py:run_experiment(...)`.
- Emit a single advantage-alignment PNG per `(mode, training_mode)` of
  the sweep, alongside the existing learning-curve and visitation-grid
  outputs.
- The figure visibly states its config (`dist=6, H=8 (small), B=..., α=1.0`)
  in the suptitle per the durable annotation rule.

## Non-Goals

- A separate CLI script. The metric is recorded universally in the eval
  loop; the figure is emitted by the existing sweep pipeline.
- `cap_zeta` mode advantage-alignment figure (no-op, mirroring
  `plot_learning_curves`).
- Backfill / migration of older pkls that lack `adv_product_s0`. The
  plot function skips entries missing the field.
- Per-action plots (the user's original spec called for A^π·A^μ per
  action as subplots; the modification to record only the expectation
  collapses this to a single scalar per step).

## Architecture

Two-file change. No new modules.

```
tabular_prototype/experiments.py
    + helper _compute_adv_product_s0(policy, Q_pi, V_pi, Q_mu, V_mu, start_idx)
    + history dict gains 'adv_product_s0' field at both eval sites
        - exact-mode loop (around line 137)
        - trajectory-mode loop (around line 251)
    + trajectory loop: line 208 change `_, V_pi_new = ...` to
      `Q_pi_new, V_pi_new = ...` so Q_pi_new is in scope at history append

run_hypothesis_sweep.py
    + plot_advantage_alignment(all_results, mode, figures_dir)
    + main() invokes it alongside plot_learning_curves and plot_visitation_grids
```

The new figure is emitted into `<figures_dir>/advantage_alignment.png`
(flat, not under a `learning_curves/` subdir — single PNG per call, not
a series).

## Detail design

### Helper

In `tabular_prototype/experiments.py`, defined near the top of the
module after imports:

```python
def _compute_adv_product_s0(policy, Q_pi, V_pi, Q_mu, V_mu, start_idx):
    """E_{a ~ π}[A^π(s_0,a) · A^μ(s_0,a)] at s_0.

    Returns None when teacher is absent (Q_mu / V_mu is None).
    """
    if Q_mu is None or V_mu is None:
        return None
    probs = policy.get_probs(start_idx)
    A_pi = Q_pi[start_idx] - V_pi[start_idx]
    A_mu = Q_mu[start_idx] - V_mu[start_idx]
    return float(np.sum(probs * A_pi * A_mu))
```

`probs` is a NumPy array of shape `(n_actions,)` per
`TabularSoftmaxPolicy.get_probs`. `A_pi` and `A_mu` are the same shape.
The result is the policy-weighted inner product of the two advantage
vectors at s_0.

### Eval-site additions

**Exact-mode loop** (after `Q_pi_new, V_pi_new = compute_student_qvalues(...)`
at line 116, inside the existing `history.append({...})` at line 137):

```python
history.append({
    'steps': total_steps,
    ...,
    'exact_V_start': float(V_pi_new[start_idx]),
    'exact_V_start_undiscounted': float(V_pi_undiscounted[start_idx]),
    'unique_sa': ...,
    'state_entropy': ...,
    'adv_product_s0': _compute_adv_product_s0(
        policy, Q_pi_new, V_pi_new, Q_mu, V_mu, start_idx,
    ),
})
```

**Trajectory-mode loop**. Currently at line 208:

```python
_, V_pi_new = compute_student_qvalues(env, policy, gamma)
```

Change to:

```python
Q_pi_new, V_pi_new = compute_student_qvalues(env, policy, gamma)
```

Then inside the existing `history.append({...})` at line 251, add the
same `adv_product_s0` entry as the exact-mode loop.

### Plot function

In `run_hypothesis_sweep.py`, near `plot_learning_curves`:

```python
def plot_advantage_alignment(
    all_results: list,
    mode: str,
    figures_dir: str,
    distance: int = 6,
    horizon_type: str = 'small',
    alpha: float = 1.0,
    budget_rank: int = -2,
):
    """Per-(mode, training_mode) figure of g^π(t) over training.

    Default cell: dist=6, h_type='small', alpha=1.0,
    B=calibrated_budgets[budget_rank=-2] (second-largest) for the
    (mode, training_mode) of the sweep. Override any of the four
    cell-defining args to retarget. One line per teacher baseline,
    mean ± 1σ band across seeds. Skipped for cap_zeta mode.

    `budget_rank` indexes into the calibration JSON's `budgets` list
    for the matched (distance, horizon_type, n_goals) cell. Default
    -2 = second-largest. Use -1 for largest, 0 for smallest, etc.
    """
    if mode == 'cap_zeta':
        return
    target = {
        'distance': distance,
        'horizon_type': horizon_type,
        'alpha': alpha,
    }

    # Determine training_mode from the data (every result row carries 'mode')
    training_modes = {r.get('mode', 'exact') for r in all_results
                      if all(r.get(k) == v for k, v in target.items())}
    if not training_modes:
        return
    training_mode = next(iter(training_modes))

    # Resolve budget from calibration JSON by rank
    n_goals = 1 if mode == 'zeta' else 3
    calib = json.load(open(_calibration_path_for(training_mode)))
    cell = _find_calibration_cell(calib, distance, horizon_type, n_goals)
    if not cell:
        return
    budgets = cell.get('budgets', [])
    if not budgets or abs(budget_rank) > len(budgets):
        return
    target['sample_budget'] = budgets[budget_rank]

    # Filter and validate field presence
    matching = [r for r in all_results
                if all(r.get(k) == v for k, v in target.items())
                and r['history']
                and r['history'][0].get('adv_product_s0') is not None]
    if not matching:
        return

    # Group by teacher_val
    tcol = _teacher_col(mode)
    groups = defaultdict(list)
    for r in matching:
        groups[r[tcol]].append(r['history'])

    sorted_teachers = _sort_teacher_vals(
        pd.Series(list(groups.keys())), mode,
    )

    x_label = 'update step' if training_mode == 'exact' else 'env step'

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tv in sorted_teachers:
        histories = groups[tv]
        min_len = min(len(h) for h in histories)
        steps = np.mean([
            [h['steps'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        values = np.stack([
            [h['adv_product_s0'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(steps, mean, label=_teacher_label(mode, tv),
                marker='o', markersize=3, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(
        r'$\mathbb{E}_{a \sim \pi^t}[A^{\pi^t}(s_0,a)\,A^\mu(s_0,a)]$',
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    h_val = cell['horizon']
    fig.suptitle(
        f'Advantage alignment ({mode} sweep) — '
        f"dist={distance}, H={h_val} ({horizon_type}), "
        f"B={target['sample_budget']}, " + rf'$\alpha={alpha}$',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Filename includes the cell so overrides don't collide
    out_path = os.path.join(
        figures_dir,
        f'advantage_alignment_dist{distance}_{horizon_type}'
        f"_B{target['sample_budget']}_alpha{alpha:.2f}.png",
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved {out_path}')
```

**Calibration-key shape varies by training mode.** Inspecting the
existing JSON files:

- `results/calibration.json` (exact): keys look like
  `dist=6_small_ng=1_lr=0.5_grid=9` (includes lr).
- `results/calibration_hybrid.json` and
  `results/calibration_sample.json`: keys look like
  `dist=6_small_ng=1_grid=9` (no lr).

The plot function avoids hardcoding the key by **substring-matching**
on the unambiguous prefix `f'dist={d}_{h_type}_ng={n_goals}_'` and
iterating the JSON's keys. Only one match is expected per
(distance, h_type, n_goals); if zero or multiple match, the function
skips with an INFO log.

```python
def _calibration_path_for(training_mode: str) -> str:
    suffix = '' if training_mode == 'exact' else f'_{training_mode}'
    return f'results/calibration{suffix}.json'

def _find_calibration_cell(calib: dict, distance: int, h_type: str, n_goals: int):
    prefix = f'dist={distance}_{h_type}_ng={n_goals}_'
    matches = [v for k, v in calib.items() if k.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None
```

**Why a horizontal zero reference?** g^π(t) > 0 means student and
teacher agree (signed product positive on weighted actions). g^π(t) < 0
means the policy is putting mass on actions where student and teacher
disagree on advantage sign — i.e. teacher is actively misleading from
the student's own evaluation. The zero line is the alignment threshold.

### Wiring into `main()`

In `run_hypothesis_sweep.py:main()`, find the existing block that calls
`plot_visitation_grids(all_results, args.mode, figures_dir)` and
`plot_learning_curves(all_results, args.mode, figures_dir)`. Append one
line:

```python
plot_advantage_alignment(all_results, args.mode, figures_dir)
```

## Failure modes

| Condition | Behavior |
|-----------|----------|
| No runs in `all_results` match the target cell | Function returns early, no PNG, INFO log |
| Calibration JSON missing for the inferred training_mode | Function returns early, no PNG |
| All matching runs have `adv_product_s0 == None` (legacy pkls or teacher absent throughout) | Function returns early, no PNG |
| Some matching runs have the field, others don't | Function uses only the runs that do; if fewer than `n_seeds_min` (say, 2) remain per baseline, that baseline is skipped silently |
| `n_goals` mismatch between mode and calibration | Documented: mode `'zeta'` → ng=1, mode `'capability'` → ng=3 hardcoded in `plot_advantage_alignment` |

## Tests

New file `tests/test_advantage_alignment.py`. Four tests:

### test_adv_product_s0_zero_for_uniform_at_init

```python
def test_adv_product_s0_zero_for_uniform_at_init():
    """A^π(s_0,·) = 0 under uniform π (V^π = mean Q^π), so the product
    is zero regardless of A^μ. Catches sign / shape bugs."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(8)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    pi_star = build_optimal_policy(env, env.goals, gamma)
    Q_mu, V_mu = evaluate_policy_values(env, pi_star, gamma)
    g = _compute_adv_product_s0(
        policy, Q_pi, V_pi, Q_mu, V_mu,
        start_idx=env.state_to_idx(env.start),
    )
    assert abs(g) < 1e-10, f"expected ~0, got {g}"
```

### test_adv_product_s0_recorded_in_history

```python
def test_adv_product_s0_recorded_in_history():
    """run_experiment must write a finite adv_product_s0 in every
    history entry when a teacher is configured."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4)],
        teacher_capacity=1, zeta=1.0, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        assert 'adv_product_s0' in entry
        assert entry['adv_product_s0'] is not None
        assert np.isfinite(entry['adv_product_s0'])
```

### test_plot_advantage_alignment_emits_png

Build a synthetic `all_results` list with the target cell filled in,
4 zetas, 3 seeds, history with `adv_product_s0` populated linearly
toward a fixed value per zeta. The test reads the budget from
`results/calibration.json` at runtime (rather than hardcoding 34) so
it stays robust to recalibration:

```python
import json
calib = json.load(open('results/calibration.json'))
cell = next(v for k, v in calib.items()
            if k.startswith('dist=6_small_ng=1_'))
budget = cell['budgets'][-2]  # second-largest
# ... build synthetic results with sample_budget=budget ...
```

Call `plot_advantage_alignment(results, mode='zeta', figures_dir=tmp_path)`
(no overrides — exercises the defaults). Assert exactly one PNG with
the parameterized filename
`advantage_alignment_dist6_small_B{budget}_alpha1.00.png` exists with
size > 1 KB.

A second test exercises the override path: same synthetic results
where the data is at a non-default `sample_budget`, called with
`budget_rank=-1` (largest budget). Assert the figure is emitted at
that rank's filename.

### test_plot_advantage_alignment_cap_zeta_noop

```python
def test_plot_advantage_alignment_cap_zeta_noop(tmp_path):
    plot_advantage_alignment([], mode='cap_zeta', figures_dir=str(tmp_path))
    assert not list(tmp_path.glob('*.png'))
```

## Risks

| Risk | Mitigation |
|------|------------|
| Schema growth breaks downstream code that reads pickled history dicts | All readers should be field-getter-safe (`entry.get('adv_product_s0')`); add a one-line note to the history-schema docstring if one exists |
| Computing Q^π/V^π via Bellman policy eval at every eval tick already happens — adding the metric is a free addition. No perf concern. | None needed |
| Cap-mode at α=1 includes c=-1 (no teacher) by mistake → metric is None for those runs | Filter is on field-non-None; the figure simply skips c=-1 lines, no crash |
| `_load_calibration_for_training_mode` doesn't exist yet | Add the thin wrapper as part of this change; reuses existing JSON-loading logic in `_load_calibrated_budgets` |

## Acceptance

- A fresh sweep with `--mode zeta` and `--mode capability` produces
  `<figures_dir>/advantage_alignment.png` (one per directory).
- Re-running with `--skip-run` against an existing output directory
  regenerates the figure from the cached pkl — provided the pkl was
  produced by this branch (older pkls lack the field; figure is
  skipped with an INFO log).
- `--mode cap_zeta` runs unchanged — no advantage-alignment PNG, no
  errors.
- New tests pass; existing tests are unaffected (the schema gain is
  additive — older tests that don't look at `adv_product_s0` are
  blind to it).

## Out of scope (explicit)

- Per-action breakdown plots.
- Different fixed cells beyond `(dist=6, small H, α=1.0,
  B=second-largest)`. If a future analysis wants other cells, a CLI
  flag could be added; not in this spec.
- Variance / higher-moment analysis of g^π(t). Mean and ±1σ band only.
- cap_zeta advantage-alignment figure.
