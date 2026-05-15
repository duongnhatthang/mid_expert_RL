# Advantage Alignment Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record `adv_product_s0 = E_{a~π}[A^π(s_0,a) · A^μ(s_0,a)]` in every per-eval history entry and emit a parameterized advantage-alignment PNG per `(mode, training_mode)` of the sweep.

**Architecture:** Two-file additive change. `tabular_prototype/experiments.py` gains a small pure helper `_compute_adv_product_s0` and writes the new field into both eval-loop sites (exact and trajectory). `run_hypothesis_sweep.py` gains `plot_advantage_alignment(...)` plus two thin calibration-JSON helpers, and `main()` invokes it alongside the existing plot functions.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest. Uses existing project helpers (`compute_student_qvalues`, `build_optimal_policy`, `evaluate_policy_values`); no new dependencies.

---

## File Map

- **Modify:** `tabular_prototype/experiments.py`
  - New helper `_compute_adv_product_s0(...)` (added after the module-level imports)
  - Exact-mode `history.append({...})` (current line 137) gains one new key
  - Trajectory-mode `history.append({...})` (current line 251) gains one new key
  - Trajectory-mode line 208 changes `_, V_pi_new = ...` → `Q_pi_new, V_pi_new = ...`

- **Modify:** `run_hypothesis_sweep.py`
  - Two new helpers `_calibration_path_for(training_mode)` and `_find_calibration_cell(calib, distance, h_type, n_goals)` (added near the existing `_load_calibrated_budgets`)
  - New top-level function `plot_advantage_alignment(all_results, mode, figures_dir, *, distance=6, horizon_type='small', alpha=1.0, budget_rank=-2)` (added near `plot_learning_curves`)
  - `main()` adds one line invoking `plot_advantage_alignment(...)` after the existing `plot_learning_curves(...)` call (around line 2504)

- **Create:** `tests/test_advantage_alignment.py` — four tests covering the helper, schema, default-path figure, override-path figure, and cap_zeta no-op.

## Test Strategy

- Helper test is pure NumPy / GridEnv composition.
- Schema tests call `run_experiment(...)` with tiny budgets and inspect the returned history.
- Plot tests use figure-capture via monkey-patching `matplotlib.figure.Figure.savefig` (same pattern as the existing learning-curve tests).
- Calibration JSON is read at test-time from `results/calibration.json` so the test stays robust to recalibration; the test asserts the produced PNG filename includes the actual second-largest budget.

---

## Task 1: Add `_compute_adv_product_s0` helper + unit test

Pure helper. No imports of `run_experiment` or training loop. Lives in `tabular_prototype/experiments.py` so the schema-additions in Tasks 3 and 5 can call it directly.

**Files:**
- Modify: `tabular_prototype/experiments.py` (add helper after module-level imports, around line 23)
- Create: `tests/test_advantage_alignment.py` (new file)

- [ ] **Step 1: Write the failing test (new file)**

Create `tests/test_advantage_alignment.py` with:

```python
"""Tests for adv_product_s0 metric + plot_advantage_alignment.

Covers:
 - `_compute_adv_product_s0` helper correctness on a uniform policy
   (where A^π is identically zero so the product collapses to zero).
 - run_experiment writes the field in both exact and trajectory modes.
 - plot_advantage_alignment emits the expected PNG via the default path
   and the override path; mode='cap_zeta' is a no-op.
"""

import numpy as np

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy
from tabular_prototype.teacher import (
    build_optimal_policy, evaluate_policy_values,
)
from tabular_prototype.training import compute_student_qvalues
from tabular_prototype.config import compute_gamma_from_horizon
from tabular_prototype.experiments import _compute_adv_product_s0


def test_adv_product_s0_zero_for_uniform_at_init():
    """A^π(s_0,·) = 0 under uniform π (V^π = mean Q^π under uniform), so
    the product collapses to zero regardless of A^μ. Catches sign / shape
    bugs in the helper."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(8)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    pi_star = build_optimal_policy(env, env.goals, gamma)
    Q_mu, V_mu = evaluate_policy_values(env, pi_star, gamma)
    start_idx = env.state_to_idx(env.start)

    g = _compute_adv_product_s0(policy, Q_pi, V_pi, Q_mu, V_mu, start_idx)
    assert abs(g) < 1e-10, f"expected ~0 under uniform π, got {g}"


def test_adv_product_s0_none_when_teacher_absent():
    """When Q_mu/V_mu is None the helper returns None (not 0, to keep the
    'teacher absent' state distinguishable downstream)."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(8)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    start_idx = env.state_to_idx(env.start)

    assert _compute_adv_product_s0(
        policy, Q_pi, V_pi, None, None, start_idx,
    ) is None
```

- [ ] **Step 2: Run tests to verify they fail with ImportError**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py -v`

Expected: FAIL with `ImportError: cannot import name '_compute_adv_product_s0' from 'tabular_prototype.experiments'`.

- [ ] **Step 3: Implement the helper**

In `tabular_prototype/experiments.py`, after the existing module-level imports (after line 22, before `VALID_MODES = ...` on line 25), insert:

```python
def _compute_adv_product_s0(policy, Q_pi, V_pi, Q_mu, V_mu, start_idx):
    """E_{a ~ π}[A^π(s_0,a) · A^μ(s_0,a)] at s_0.

    Returns None when teacher is absent (Q_mu / V_mu is None) so the
    'no-teacher' state is distinguishable from a numeric zero downstream.
    """
    if Q_mu is None or V_mu is None:
        return None
    probs = policy.get_probs(start_idx)
    A_pi = Q_pi[start_idx] - V_pi[start_idx]
    A_mu = Q_mu[start_idx] - V_mu[start_idx]
    return float(np.sum(probs * A_pi * A_mu))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_advantage_alignment.py
git commit -m "feat: add _compute_adv_product_s0 helper and unit tests"
```

---

## Task 2: Failing test that `adv_product_s0` lands in history (exact mode)

**Files:**
- Modify: `tests/test_advantage_alignment.py` (append a new test function)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_advantage_alignment.py`:

```python
def test_adv_product_s0_recorded_in_exact_history():
    """run_experiment(mode='exact') writes a finite adv_product_s0 in
    every history entry when a teacher is configured."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        assert 'adv_product_s0' in entry, \
            f"missing adv_product_s0 in entry {entry}"
        assert entry['adv_product_s0'] is not None
        assert np.isfinite(entry['adv_product_s0'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_adv_product_s0_recorded_in_exact_history -v`

Expected: FAIL with `AssertionError: missing adv_product_s0 in entry ...`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: add failing test for adv_product_s0 in exact history"
```

---

## Task 3: Implement exact-mode schema addition

**Files:**
- Modify: `tabular_prototype/experiments.py:137-147` (the exact-mode `history.append` block)

- [ ] **Step 1: Add the new key to the exact-mode `history.append`**

In `tabular_prototype/experiments.py`, find the block currently at lines 137-147:

```python
                history.append({
                    'steps': update_count,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi_new[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': 0,
                    'state_entropy': 0.0,
                })
```

Replace with:

```python
                history.append({
                    'steps': update_count,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi_new[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': 0,
                    'state_entropy': 0.0,
                    'adv_product_s0': _compute_adv_product_s0(
                        policy, Q_pi_new, V_pi_new, Q_mu, V_mu, start_idx,
                    ),
                })
```

- [ ] **Step 2: Run the exact-mode test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_adv_product_s0_recorded_in_exact_history -v`

Expected: PASS.

- [ ] **Step 3: Run the full experiments test suite to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_training.py tests/test_advantage_alignment.py -v`

Expected: ALL pass.

- [ ] **Step 4: Commit**

```bash
git add tabular_prototype/experiments.py
git commit -m "feat: record adv_product_s0 in exact-mode eval history"
```

---

## Task 4: Failing test that `adv_product_s0` lands in history (trajectory mode)

**Files:**
- Modify: `tests/test_advantage_alignment.py` (append a new test function)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_advantage_alignment.py`:

```python
def test_adv_product_s0_recorded_in_hybrid_history():
    """run_experiment(mode='hybrid') writes a finite adv_product_s0 in
    every history entry. Guards against the trajectory-loop
    history.append site (which is separate from the exact-mode one)."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0, lr=1.0,
        horizon=8, sample_budget=20, mode='hybrid',
        trajectories_per_update=1,
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        assert 'adv_product_s0' in entry
        assert entry['adv_product_s0'] is not None
        assert np.isfinite(entry['adv_product_s0'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_adv_product_s0_recorded_in_hybrid_history -v`

Expected: FAIL with `AssertionError: missing adv_product_s0 ...`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: add failing test for adv_product_s0 in hybrid history"
```

---

## Task 5: Implement trajectory-mode schema addition (with line-208 fix)

**Files:**
- Modify: `tabular_prototype/experiments.py:208` (capture Q_pi_new)
- Modify: `tabular_prototype/experiments.py:251-261` (the trajectory-mode `history.append` block)

- [ ] **Step 1: Capture `Q_pi_new` at line 208**

In `tabular_prototype/experiments.py`, find line 208 (inside the trajectory-mode loop, after the policy update). Current:

```python
            _, V_pi_new = compute_student_qvalues(env, policy, gamma)
```

Replace with:

```python
            Q_pi_new, V_pi_new = compute_student_qvalues(env, policy, gamma)
```

- [ ] **Step 2: Add the new key to the trajectory-mode `history.append`**

Find the block currently at lines 251-261:

```python
                history.append({
                    'steps': total_steps,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi_new[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': vis_m['unique_sa'],
                    'state_entropy': vis_m['state_entropy'],
                })
```

Replace with:

```python
                history.append({
                    'steps': total_steps,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi_new[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': vis_m['unique_sa'],
                    'state_entropy': vis_m['state_entropy'],
                    'adv_product_s0': _compute_adv_product_s0(
                        policy, Q_pi_new, V_pi_new, Q_mu, V_mu, start_idx,
                    ),
                })
```

- [ ] **Step 3: Run the trajectory-mode test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_adv_product_s0_recorded_in_hybrid_history -v`

Expected: PASS.

- [ ] **Step 4: Run the full training + advantage-alignment tests to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_training.py tests/test_advantage_alignment.py -v`

Expected: ALL pass.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/experiments.py
git commit -m "feat: record adv_product_s0 in trajectory-mode eval history"
```

---

## Task 6: Add calibration-lookup helpers + unit test

These two thin helpers wrap `results/calibration*.json` lookups by training_mode and by (distance, horizon_type, n_goals). They go in `run_hypothesis_sweep.py` near the existing `_load_calibrated_budgets` (around line 89).

**Files:**
- Modify: `run_hypothesis_sweep.py` (add two helpers near `_load_calibrated_budgets`)
- Modify: `tests/test_advantage_alignment.py` (append a new test)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_advantage_alignment.py`:

```python
def test_calibration_helpers_find_dist6_small_cell():
    """The two new helpers wrap calibration-JSON path resolution and
    per-cell substring matching. Smoke-test against the checked-in
    results/calibration.json."""
    import json
    from run_hypothesis_sweep import (
        _calibration_path_for, _find_calibration_cell,
    )

    # exact training mode → results/calibration.json
    path = _calibration_path_for('exact')
    assert path.endswith('calibration.json')

    calib = json.load(open(path))
    cell = _find_calibration_cell(calib, distance=6, h_type='small', n_goals=1)
    assert cell is not None, "expected one matching cell for dist=6 small ng=1"
    assert cell['horizon'] == 8
    assert isinstance(cell['budgets'], list) and len(cell['budgets']) >= 2

    # hybrid training mode → results/calibration_hybrid.json
    path_h = _calibration_path_for('hybrid')
    assert path_h.endswith('calibration_hybrid.json')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_calibration_helpers_find_dist6_small_cell -v`

Expected: FAIL with `ImportError: cannot import name '_calibration_path_for' ...`.

- [ ] **Step 3: Implement the two helpers**

In `run_hypothesis_sweep.py`, immediately before the existing `def _load_calibrated_budgets(...)` (currently at line 89), insert:

```python
def _calibration_path_for(training_mode: str) -> str:
    """Map training_mode → calibration JSON path.

    'exact' → results/calibration.json
    'hybrid' → results/calibration_hybrid.json
    'sample' → results/calibration_sample.json
    """
    suffix = '' if training_mode == 'exact' else f'_{training_mode}'
    return f'results/calibration{suffix}.json'


def _find_calibration_cell(calib: dict, distance: int, h_type: str,
                            n_goals: int):
    """Locate the calibration entry for (distance, h_type, n_goals).

    Substring-matches on the prefix `f'dist={d}_{h_type}_ng={n_goals}_'`
    so the call works across both the exact-mode keys (which include lr)
    and hybrid/sample keys (which don't). Returns None if zero or more
    than one match is found.
    """
    prefix = f'dist={distance}_{h_type}_ng={n_goals}_'
    matches = [v for k, v in calib.items() if k.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_calibration_helpers_find_dist6_small_cell -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_hypothesis_sweep.py tests/test_advantage_alignment.py
git commit -m "feat: add calibration-JSON path + per-cell lookup helpers"
```

---

## Task 7: Failing test for `plot_advantage_alignment` (default path)

**Files:**
- Modify: `tests/test_advantage_alignment.py` (append two helper functions and a failing test)

- [ ] **Step 1: Append the test fixture helper and the failing test**

Append to `tests/test_advantage_alignment.py`:

```python
def _adv_history(n_points: int, base: float):
    """Synthetic history list. Mirrors the schema produced by
    run_experiment, including the new adv_product_s0 field."""
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': 0.0,
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
            'adv_product_s0': base + 0.05 * i,
        }
        for i in range(n_points)
    ]


def _zeta_results_for_advantage_alignment():
    """Synthetic all_results filling the default cell
    (distance=6, horizon_type='small', alpha=1.0, B=budgets[-2]).
    4 zetas × 3 seeds = 12 entries."""
    import json
    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    h_val = cell['horizon']

    out = []
    for zeta in [0.0, 0.33, 0.67, 1.0]:
        for seed in [0, 1, 2]:
            out.append({
                'distance': 6,
                'alpha': 1.0,
                'horizon_type': 'small',
                'horizon': h_val,
                'sample_budget': budget,
                'zeta': zeta,
                'seed': seed,
                'mode': 'exact',
                'history': _adv_history(4, base=0.05 * zeta + 0.01 * seed),
            })
    return out


def test_plot_advantage_alignment_default_path(tmp_path, monkeypatch):
    """Default invocation must emit exactly one PNG matching the
    parameterized filename for the default cell."""
    import json
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import run_hypothesis_sweep as sweep

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_advantage_alignment(
        _zeta_results_for_advantage_alignment(),
        mode='zeta',
        figures_dir=str(tmp_path),
    )

    assert captured_figures, "expected at least one figure saved"

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    expected_name = (
        f'advantage_alignment_dist6_small_B{budget}_alpha1.00.png'
    )
    pngs = list(tmp_path.glob('*.png'))
    assert len(pngs) == 1, f"expected 1 PNG, got {[p.name for p in pngs]}"
    assert pngs[0].name == expected_name, \
        f"unexpected filename {pngs[0].name}, expected {expected_name}"
    assert pngs[0].stat().st_size > 1000

    plt.close('all')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_plot_advantage_alignment_default_path -v`

Expected: FAIL with `AttributeError: module 'run_hypothesis_sweep' has no attribute 'plot_advantage_alignment'`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: add failing test for plot_advantage_alignment default path"
```

---

## Task 8: Implement `plot_advantage_alignment`

**Files:**
- Modify: `run_hypothesis_sweep.py` (add new top-level function near `plot_learning_curves` around line 1468)

- [ ] **Step 1: Add the function definition**

In `run_hypothesis_sweep.py`, immediately before the existing `def plot_learning_curves(...)` (currently at line 1468), insert:

```python
def plot_advantage_alignment(
    all_results: list,
    mode: str,
    figures_dir: str,
    *,
    distance: int = 6,
    horizon_type: str = 'small',
    alpha: float = 1.0,
    budget_rank: int = -2,
):
    """Single-cell figure of g^π(t) = E_{a~π}[A^π(s_0,a)·A^μ(s_0,a)].

    Default cell: distance=6, horizon_type='small', alpha=1.0,
    sample_budget = calibrated budgets[budget_rank=-2] (second-largest).
    Override any kwarg to retarget. One line per teacher baseline,
    mean ± 1σ band across seeds. cap_zeta mode is a no-op.
    """
    if mode == 'cap_zeta':
        print("plot_advantage_alignment: cap_zeta mode — skipping")
        return

    import json
    import matplotlib.pyplot as plt
    from collections import defaultdict

    target = {
        'distance': distance,
        'horizon_type': horizon_type,
        'alpha': alpha,
    }

    training_modes = {r.get('mode', 'exact') for r in all_results
                      if all(r.get(k) == v for k, v in target.items())}
    if not training_modes:
        print("plot_advantage_alignment: no matching runs — skipping")
        return
    training_mode = next(iter(training_modes))

    n_goals = 1 if mode == 'zeta' else 3
    try:
        calib = json.load(open(_calibration_path_for(training_mode)))
    except FileNotFoundError:
        print("plot_advantage_alignment: calibration JSON missing — skipping")
        return
    cell = _find_calibration_cell(calib, distance, horizon_type, n_goals)
    if not cell:
        print("plot_advantage_alignment: no calibration cell — skipping")
        return
    budgets = cell.get('budgets', [])
    if not budgets or abs(budget_rank) > len(budgets):
        print("plot_advantage_alignment: budget_rank out of range — skipping")
        return
    target['sample_budget'] = budgets[budget_rank]

    matching = [r for r in all_results
                if all(r.get(k) == v for k, v in target.items())
                and r['history']
                and r['history'][0].get('adv_product_s0') is not None]
    if not matching:
        print("plot_advantage_alignment: no field-bearing runs — skipping")
        return

    tcol = _teacher_col(mode)
    groups = defaultdict(list)
    for r in matching:
        groups[r[tcol]].append(r['history'])

    sorted_teachers = _sort_teacher_vals(
        pd.Series(list(groups.keys())), mode,
    )

    x_label = 'update step' if training_mode == 'exact' else 'env step'

    os.makedirs(figures_dir, exist_ok=True)

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

    out_path = os.path.join(
        figures_dir,
        f'advantage_alignment_dist{distance}_{horizon_type}'
        f"_B{target['sample_budget']}_alpha{alpha:.2f}.png",
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved {out_path}')
```

- [ ] **Step 2: Run the default-path test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_plot_advantage_alignment_default_path -v`

Expected: PASS.

- [ ] **Step 3: Run the full advantage-alignment test file**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py -v`

Expected: ALL pass so far (6 tests from Tasks 1, 2, 4, 6, 7 plus the helper unit test).

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: add plot_advantage_alignment with parameterized cell"
```

---

## Task 9: Override-path test + cap_zeta no-op test

Two additional tests to lock in the override path (changing `budget_rank`) and the cap_zeta no-op.

**Files:**
- Modify: `tests/test_advantage_alignment.py` (append two new tests)

- [ ] **Step 1: Append both tests**

Append to `tests/test_advantage_alignment.py`:

```python
def test_plot_advantage_alignment_override_budget_rank(tmp_path):
    """Passing budget_rank=-1 selects the largest budget; filename
    parameterizes on the resolved budget."""
    import json
    import matplotlib.pyplot as plt
    import run_hypothesis_sweep as sweep

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    largest_budget = cell['budgets'][-1]
    h_val = cell['horizon']

    # Synthetic data at the LARGEST budget so the override matches it.
    results = []
    for zeta in [0.0, 0.33, 0.67, 1.0]:
        for seed in [0, 1, 2]:
            results.append({
                'distance': 6,
                'alpha': 1.0,
                'horizon_type': 'small',
                'horizon': h_val,
                'sample_budget': largest_budget,
                'zeta': zeta,
                'seed': seed,
                'mode': 'exact',
                'history': _adv_history(4, base=0.05 * zeta + 0.01 * seed),
            })

    sweep.plot_advantage_alignment(
        results, mode='zeta', figures_dir=str(tmp_path),
        budget_rank=-1,
    )

    expected = (
        f'advantage_alignment_dist6_small_B{largest_budget}_alpha1.00.png'
    )
    pngs = list(tmp_path.glob('*.png'))
    assert len(pngs) == 1, f"expected 1 PNG, got {[p.name for p in pngs]}"
    assert pngs[0].name == expected

    plt.close('all')


def test_plot_advantage_alignment_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function returns
    immediately and writes nothing."""
    import run_hypothesis_sweep as sweep
    sweep.plot_advantage_alignment([], mode='cap_zeta',
                                    figures_dir=str(tmp_path))
    assert not list(tmp_path.glob('*.png'))
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py -v`

Expected: ALL 9 tests pass (1 helper unit + 1 None-when-teacher-absent + 1 exact schema + 1 hybrid schema + 1 calibration helpers + 1 default-path + 1 override + 1 cap_zeta no-op).

- [ ] **Step 3: Commit**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: cover budget_rank override and cap_zeta no-op"
```

---

## Task 10: Wire `plot_advantage_alignment` into `main()`

**Files:**
- Modify: `run_hypothesis_sweep.py:2501-2505` (the post-run plotting block)

- [ ] **Step 1: Inspect the current call site**

Run: `grep -n "plot_visitation_grids(all_results\|plot_learning_curves(all_results" run_hypothesis_sweep.py`

Confirm the existing call at line 2504 (current branch) is `plot_learning_curves(all_results, args.mode, figures_dir)`.

- [ ] **Step 2: Add the new call**

Immediately AFTER the existing `plot_learning_curves(all_results, args.mode, figures_dir)` line (around line 2504), add:

```python
        plot_advantage_alignment(all_results, args.mode, figures_dir)
```

Match the indentation of the surrounding block (it's inside the post-run / `--skip-run` plotting branch).

- [ ] **Step 3: Run a quick smoke test against a cached pkl**

Run: `python -c "
import pickle
results = pickle.load(open('results/full_suite_20260419_105552/zeta_sample/zeta_sweep_results.pkl', 'rb'))
import run_hypothesis_sweep as sweep
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    sweep.plot_advantage_alignment(results, mode='zeta', figures_dir=d)
    files = os.listdir(d)
    print('files:', files)
"`

Expected: prints `files: []` (older pkls predate the schema change, so `adv_product_s0` is missing and the function skips with an INFO log) OR `files: ['advantage_alignment_dist6_small_B318_alpha1.00.png']` (if the pkl happens to have the new field — won't on existing pkls).

Either outcome is acceptable; the test confirms the wiring doesn't crash. The figure proper will be exercised in Task 12 once a fresh sweep is run.

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: emit advantage-alignment figure in sweep main()"
```

---

## Task 11: Run a tiny live sweep to verify end-to-end

**Files:**
- No code changes.

- [ ] **Step 1: Run a minimal exact-mode zeta sweep**

Run a small sweep that touches the (dist=6, small H, α=1.0) cell. The standard sweep covers it, but a full run is overkill — invoke a single-distance, single-mode run. Tested command (verified to exercise the new code path):

```bash
python run_hypothesis_sweep.py --mode zeta --n-seeds 3 --n-workers 1 \
    --output-dir results/adv_alignment_smoke
```

Expected: standard sweep output, plus `Saved results/adv_alignment_smoke/figures/advantage_alignment_dist6_small_B<N>_alpha1.00.png` printed near the end (the budget `<N>` depends on the calibration JSON's `dist=6_small_ng=1_*` entry — likely 34 with the default `results/calibration.json`).

If the sweep takes more than ~5 minutes on this machine, abort and rerun with `--n-seeds 2`. The smoke test only needs to confirm the figure is emitted from a real pipeline.

- [ ] **Step 2: Open the figure and visually verify**

Run: `open results/adv_alignment_smoke/figures/advantage_alignment_*.png`

Visually verify:
- Suptitle reads `Advantage alignment (zeta sweep) — dist=6, H=8 (small), B=<N>, α=1.0`.
- 4 lines, one per ζ ∈ {0, 0.33, 0.67, 1.0}, with mean and ±1σ band.
- Black dotted horizontal line at y=0 (alignment reference).
- y-axis label is `E_{a ~ π^t}[A^{π^t}(s_0,a) A^μ(s_0,a)]` (LaTeX rendered).

- [ ] **Step 3: No commit needed** — this task only verifies the visual.

---

## Task 12: Final sanity sweep

**Files:**
- No code changes.

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`

Expected: ALL pass. New tests from this plan: 9 in `tests/test_advantage_alignment.py`. Existing tests unaffected.

- [ ] **Step 2: Verify clean working tree**

Run: `git status`

Expected: `nothing to commit, working tree clean` apart from any pre-existing untracked files.

- [ ] **Step 3: Show the commit log delta**

Run: `git log --oneline origin/main..HEAD`

Expected: ~9-10 new commits on top of the spec commit:
- Task 1: feat: add _compute_adv_product_s0 helper and unit tests
- Task 2: test: add failing test for adv_product_s0 in exact history
- Task 3: feat: record adv_product_s0 in exact-mode eval history
- Task 4: test: add failing test for adv_product_s0 in hybrid history
- Task 5: feat: record adv_product_s0 in trajectory-mode eval history
- Task 6: feat: add calibration-JSON path + per-cell lookup helpers
- Task 7: test: add failing test for plot_advantage_alignment default path
- Task 8: feat: add plot_advantage_alignment with parameterized cell
- Task 9: test: cover budget_rank override and cap_zeta no-op
- Task 10: feat: emit advantage-alignment figure in sweep main()

Plus the design spec commit already on the branch.

---

## Spec coverage check

- **adv_product_s0 in exact-mode history** (spec §"Eval-site additions") → Tasks 2, 3.
- **adv_product_s0 in trajectory-mode history + line-208 fix** (spec §"Eval-site additions") → Tasks 4, 5.
- **`_compute_adv_product_s0` helper** (spec §"Helper") → Task 1.
- **None when teacher absent** (spec §"Helper" + Failure modes) → Task 1 (second test in the file).
- **`plot_advantage_alignment` default cell + filename parameterization** (spec §"Plot function") → Tasks 7, 8.
- **Parameterized cell args (distance, horizon_type, alpha, budget_rank)** (spec §"Plot function") → Task 8 implementation + Task 9 override test.
- **cap_zeta no-op** (spec §Failure modes) → Task 9.
- **Calibration JSON substring-matching helpers** (spec §"Calibration-key shape varies by training mode") → Task 6.
- **Wire into `main()`** (spec §"Wiring into main()") → Task 10.
- **Suptitle annotation with (dist, H, B, α)** (durable annotation rule) → Task 8 implementation.

No spec gaps remain.
