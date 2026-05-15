# MC Variance Curve + α=0 Baseline Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record undiscounted and discounted trajectory-return variance in every eval-tick history dict, emit a new two-PNG variance figure per sweep, and retrofit `plot_advantage_alignment` with an α=0 vanilla-NPG baseline overlay.

**Architecture:** Three-file additive change. `tabular_prototype/training.py:evaluate_policy` gains two new return-dict keys for discounted return. `tabular_prototype/experiments.py` writes `mc_var_undiscounted` and `mc_var_discounted` to both eval-tick `history.append` sites. `run_hypothesis_sweep.py` gains a shared `_overlay_baseline_alpha` helper, a new `plot_mc_variance_curve` function (emits two PNGs per call), and is wired into `main()`. `plot_advantage_alignment` is retrofitted to accept `baseline_alpha=0.0` kwarg and call the same helper.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest. Reuses existing project helpers (`compute_gamma_from_horizon`, `_calibration_path_for`, `_find_calibration_cell`, `_teacher_col`, `_sort_teacher_vals`, `_teacher_label`).

---

## File Map

- **Modify:** `tabular_prototype/training.py` (`evaluate_policy` at line 256; augments rollout loop, returns two new keys)
- **Modify:** `tabular_prototype/experiments.py` — exact-mode `history.append` (line 151) and trajectory-mode `history.append` (line 268) each gain two keys
- **Modify:** `run_hypothesis_sweep.py`
  - new `_overlay_baseline_alpha` helper (placed near `_teacher_label`, before line 606)
  - retrofit `plot_advantage_alignment` (line 1493) — add `baseline_alpha=0.0` kwarg, call the helper after the per-teacher loop
  - new `plot_mc_variance_curve` function (placed immediately after `plot_advantage_alignment`)
  - `main()` invokes `plot_mc_variance_curve` after the existing `plot_advantage_alignment` call (line 2651)
- **Create:** `tests/test_mc_variance_curve.py` — 6 new tests for the variance feature
- **Modify:** `tests/test_advantage_alignment.py` — update existing synthetic data + tests to cover the baseline overlay, add 1 new test for `baseline_alpha=None`

## Test Strategy

- `evaluate_policy` extension: pure unit test against the return dict shape.
- Schema additions: tiny `run_experiment(...)` calls assert each history entry has finite non-negative variance fields.
- Plot tests use the figure-capture monkey-patch pattern established in `tests/test_run_hypothesis_sweep_plots.py` and `tests/test_advantage_alignment.py`.
- Baseline overlay assertions look for `Line2D` artists with `linestyle='--'` and `color='black'`.

---

## Task 1: Extend `evaluate_policy` with discounted return

Foundational change. The variance fields downstream are derived from this. TDD red→green inside one task because the API extension is purely additive and the test exercises it directly.

**Files:**
- Modify: `tabular_prototype/training.py:256-302`
- Create: `tests/test_mc_variance_curve.py` (new file)

- [ ] **Step 1: Write the failing test (new file)**

Create `tests/test_mc_variance_curve.py` with:

```python
"""Tests for MC variance curve + α=0 baseline overlay.

Covers:
 - evaluate_policy returns mean_reward_discounted, std_reward_discounted.
 - run_experiment writes mc_var_undiscounted, mc_var_discounted in both
   exact and trajectory mode histories.
 - plot_mc_variance_curve emits two parameterized PNGs.
 - baseline_alpha=None disables the baseline overlay.
 - mode='cap_zeta' is a no-op.
"""

import numpy as np

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy
from tabular_prototype.training import evaluate_policy
from tabular_prototype.config import compute_gamma_from_horizon


def test_evaluate_policy_returns_discounted_fields():
    """evaluate_policy must return mean_reward_discounted and
    std_reward_discounted alongside the existing fields."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    rng = np.random.default_rng(0)
    result = evaluate_policy(env, policy, n_episodes=5, rng=rng)
    for k in ('mean_reward_discounted', 'std_reward_discounted'):
        assert k in result, f"missing {k!r} in evaluate_policy return"
        assert np.isfinite(result[k])
    # Discounted return ≤ undiscounted (rewards non-negative, γ < 1)
    assert (
        result['mean_reward_discounted'] <= result['mean_reward'] + 1e-9
    ), f"discounted={result['mean_reward_discounted']} > undiscounted={result['mean_reward']}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_evaluate_policy_returns_discounted_fields -v`

Expected: FAIL with `AssertionError: missing 'mean_reward_discounted' in evaluate_policy return`.

- [ ] **Step 3: Extend `evaluate_policy`**

In `tabular_prototype/training.py`, find `def evaluate_policy(...)` at line 256. Replace the entire function body (lines 256-302) with:

```python
def evaluate_policy(
    env: GridEnv, policy: TabularSoftmaxPolicy,
    n_episodes: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Evaluate policy performance with absorption state support."""
    from .config import compute_gamma_from_horizon
    gamma = compute_gamma_from_horizon(env.horizon)

    total_rewards = []
    discounted_returns = []
    goal_reached = []
    episode_lengths = []
    trap_reached = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        discounted_return = 0.0
        gamma_t = 1.0
        reached_goal = False
        reached_trap = False
        steps = 0

        for _ in range(env.horizon):
            state_idx = env.state_to_idx(state)
            action = policy.sample_action(state_idx, rng)

            next_state, reward, done = env.step(state, action, rng)

            episode_reward += reward
            discounted_return += gamma_t * reward
            gamma_t *= gamma
            steps += 1

            if next_state in env._goals_set:
                reached_goal = True
            if next_state in env._traps_set:
                reached_trap = True

            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        discounted_returns.append(discounted_return)
        goal_reached.append(reached_goal)
        trap_reached.append(reached_trap)
        episode_lengths.append(steps)

    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_reward_discounted': float(np.mean(discounted_returns)),
        'std_reward_discounted': float(np.std(discounted_returns)),
        'goal_rate': float(np.mean(goal_reached)),
        'trap_rate': float(np.mean(trap_reached)),
        'mean_episode_length': float(np.mean(episode_lengths)),
    }
```

The five new lines added: `from .config import ...`, `gamma = ...`, `discounted_returns = []`, `discounted_return = 0.0`, `gamma_t = 1.0`, `discounted_return += gamma_t * reward`, `gamma_t *= gamma`, `discounted_returns.append(discounted_return)`, plus two new return keys. Existing keys are unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_evaluate_policy_returns_discounted_fields -v`

Expected: PASS.

- [ ] **Step 5: Run the full training test suite to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_training.py tests/test_mc_variance_curve.py -v`

Expected: ALL pass.

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/training.py tests/test_mc_variance_curve.py
git commit -m "feat: evaluate_policy returns discounted return fields"
```

---

## Task 2: Failing tests for mc_var_undiscounted + mc_var_discounted in history

**Files:**
- Modify: `tests/test_mc_variance_curve.py` (append two failing tests)

- [ ] **Step 1: Append two failing tests**

Append to `tests/test_mc_variance_curve.py`:

```python
def test_mc_var_recorded_in_exact_history():
    """run_experiment(mode='exact') writes finite, non-negative
    mc_var_undiscounted and mc_var_discounted in every history entry."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        for k in ('mc_var_undiscounted', 'mc_var_discounted'):
            assert k in entry, f"missing {k!r} in {entry}"
            assert entry[k] is not None
            assert np.isfinite(entry[k])
            assert entry[k] >= 0.0, f"variance must be non-negative, got {entry[k]}"


def test_mc_var_recorded_in_hybrid_history():
    """run_experiment(mode='hybrid') — guards the trajectory-loop
    history.append site."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0, lr=1.0,
        horizon=8, sample_budget=20, mode='hybrid',
        trajectories_per_update=1,
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

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_mc_var_recorded_in_exact_history tests/test_mc_variance_curve.py::test_mc_var_recorded_in_hybrid_history -v`

Expected: BOTH FAIL with `AssertionError: missing 'mc_var_undiscounted' ...`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_mc_variance_curve.py
git commit -m "test: add failing tests for mc_var_* in history (exact + hybrid)"
```

---

## Task 3: Implement schema additions in both eval-loop sites

**Files:**
- Modify: `tabular_prototype/experiments.py:151-165` (exact-mode `history.append`)
- Modify: `tabular_prototype/experiments.py:268-282` (trajectory-mode `history.append`)

- [ ] **Step 1: Add the two keys to exact-mode `history.append`**

In `tabular_prototype/experiments.py`, find the exact-mode `history.append({...})` block (the one with `'unique_sa': 0`, currently around line 151). Currently:

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
                    'mc_var_undiscounted': eval_results['std_reward'] ** 2,
                    'mc_var_discounted': eval_results['std_reward_discounted'] ** 2,
                })
```

- [ ] **Step 2: Add the two keys to trajectory-mode `history.append`**

Find the trajectory-mode `history.append({...})` block (the one with `'unique_sa': vis_m['unique_sa']`, currently around line 268). Currently:

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
                    'mc_var_undiscounted': eval_results['std_reward'] ** 2,
                    'mc_var_discounted': eval_results['std_reward_discounted'] ** 2,
                })
```

- [ ] **Step 3: Run Task 2's tests to verify both pass**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py -v`

Expected: ALL 3 pass.

- [ ] **Step 4: Run the full training + advantage + variance test suites to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_training.py tests/test_advantage_alignment.py tests/test_mc_variance_curve.py -v`

Expected: ALL pass.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/experiments.py
git commit -m "feat: record mc_var_undiscounted + mc_var_discounted in eval history"
```

---

## Task 4: Add shared `_overlay_baseline_alpha` helper

Module-private helper that draws the α=0 vanilla-NPG line on a matplotlib axes. Will be invoked by both `plot_advantage_alignment` (Task 6) and `plot_mc_variance_curve` (Task 8).

**Files:**
- Modify: `run_hypothesis_sweep.py` (insert helper before line 606 — near `_sort_teacher_vals`)

Direct implementation without a separate failing test — the helper is only exercised through the two plot functions whose tests will catch it. This avoids re-asserting the same logic through two test layers.

- [ ] **Step 1: Insert the helper**

In `run_hypothesis_sweep.py`, immediately AFTER `def _sort_teacher_vals(...)` (find by grepping `def _sort_teacher_vals`; insert AFTER the closing `return ...` of that function, before the next function), add:

```python
def _overlay_baseline_alpha(
    ax, all_results, mode, tcol, field_name, target, baseline_alpha,
):
    """Overlay one α=baseline_alpha vanilla-NPG baseline line on `ax`.

    At α=0 the teacher signal weight is zero so all teacher values are
    mathematically equivalent. We pick the first teacher value in
    canonical sort order as a deterministic representative.

    No-op when baseline_alpha is None or no matching results are found.
    Renders a black dashed line with a light fill_between band, labeled
    `α={baseline_alpha} (vanilla NPG)`.
    """
    if baseline_alpha is None:
        return
    from collections import defaultdict
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
    tv_pick = sorted_tvs[0]
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

- [ ] **Step 2: Smoke-import the helper to confirm syntax**

Run: `PYTHONPATH=. python -c "from run_hypothesis_sweep import _overlay_baseline_alpha; print('OK')"`

Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: add _overlay_baseline_alpha helper for vanilla-NPG baseline line"
```

---

## Task 5: Failing test for `plot_advantage_alignment` baseline overlay

**Files:**
- Modify: `tests/test_advantage_alignment.py` — update `_zeta_results_for_advantage_alignment()` and `test_plot_advantage_alignment_default_path`

- [ ] **Step 1: Update `_zeta_results_for_advantage_alignment()` to include α=0 entries**

In `tests/test_advantage_alignment.py`, find the existing `_zeta_results_for_advantage_alignment()` helper. Replace it with:

```python
def _zeta_results_for_advantage_alignment():
    """Synthetic all_results filling the default cell
    (distance=6, horizon_type='small', alpha=1.0, B=budgets[-2]) plus
    an α=0 baseline cell. 4 zetas × 3 seeds for α=1.0 plus 1 zeta × 3
    seeds for α=0.0 = 15 entries."""
    import json
    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    h_val = cell['horizon']

    out = []
    # Primary α=1.0 cells
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
    # α=0 baseline cells (one zeta value — they collapse at α=0)
    for seed in [0, 1, 2]:
        out.append({
            'distance': 6,
            'alpha': 0.0,
            'horizon_type': 'small',
            'horizon': h_val,
            'sample_budget': budget,
            'zeta': 0.0,
            'seed': seed,
            'mode': 'exact',
            'history': _adv_history(4, base=0.02 + 0.005 * seed),
        })
    return out
```

- [ ] **Step 2: Strengthen `test_plot_advantage_alignment_default_path` to assert the baseline overlay**

Find the existing `test_plot_advantage_alignment_default_path(tmp_path, monkeypatch)` test. Replace its body (everything inside the function) with:

```python
    """Default invocation must emit exactly one PNG matching the
    parameterized filename, AND the figure must contain one dashed
    black baseline line (α=0 vanilla NPG overlay)."""
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

    # Baseline overlay assertion: every captured figure must have at
    # least one dashed black line (the α=0 baseline).
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert dashed_black, (
                "expected at least one dashed black baseline line "
                "(α=0 vanilla NPG overlay) on the figure"
            )

    plt.close('all')
```

- [ ] **Step 3: Run the test to verify it FAILS**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_plot_advantage_alignment_default_path -v`

Expected: FAIL with `AssertionError: expected at least one dashed black baseline line ...`.

Reason: `plot_advantage_alignment` does not yet draw a baseline overlay — the existing `axhline(0, ..., linestyle=':')` line is dotted (not dashed), and there is no `axhline` or `plot` with `linestyle='--' and color='black'`.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: assert α=0 baseline overlay in plot_advantage_alignment"
```

---

## Task 6: Retrofit `plot_advantage_alignment` with baseline overlay

**Files:**
- Modify: `run_hypothesis_sweep.py:1493` (the `plot_advantage_alignment` function signature + body)

- [ ] **Step 1: Add `baseline_alpha=0.0` to the function signature**

In `run_hypothesis_sweep.py`, find `def plot_advantage_alignment(` (currently at line 1493). The current signature is:

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
```

Replace with:

```python
def plot_advantage_alignment(
    all_results: list,
    mode: str,
    figures_dir: str,
    *,
    distance: int = 6,
    horizon_type: str = 'small',
    alpha: float = 1.0,
    baseline_alpha: float = 0.0,
    budget_rank: int = -2,
):
```

- [ ] **Step 2: Invoke the baseline-overlay helper after the per-teacher loop**

Inside the same function, find the line `ax.axhline(0, color='black', linewidth=0.8, linestyle=':')`. Immediately BEFORE that line, insert:

```python
    _overlay_baseline_alpha(
        ax, all_results, mode, tcol, 'adv_product_s0', target, baseline_alpha,
    )
```

Match the indentation of the surrounding `ax.plot(...)` / `ax.fill_between(...)` calls (4 spaces inside the function body).

- [ ] **Step 3: Run Task 5's test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_plot_advantage_alignment_default_path -v`

Expected: PASS.

- [ ] **Step 4: Run the full advantage-alignment test file to confirm no regressions**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py -v`

Expected: ALL 8 pass.

- [ ] **Step 5: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: overlay α=0 baseline on plot_advantage_alignment"
```

---

## Task 7: Add `baseline_alpha=None` regression test for `plot_advantage_alignment`

**Files:**
- Modify: `tests/test_advantage_alignment.py` (append one test)

- [ ] **Step 1: Append the test**

Append to `tests/test_advantage_alignment.py`:

```python
def test_plot_advantage_alignment_baseline_alpha_none(tmp_path, monkeypatch):
    """Passing baseline_alpha=None must suppress the α=0 baseline overlay.

    Pins the regression-safety contract: callers can disable the
    overlay if they want only the primary-α curves on the figure."""
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
        baseline_alpha=None,
    )

    assert captured_figures, "expected at least one figure saved"
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert not dashed_black, (
                "expected NO dashed black baseline line when "
                "baseline_alpha=None, got "
                f"{len(dashed_black)} such line(s)"
            )

    plt.close('all')
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `PYTHONPATH=. pytest tests/test_advantage_alignment.py::test_plot_advantage_alignment_baseline_alpha_none -v`

Expected: PASS. The Task-4 helper short-circuits when `baseline_alpha is None`, so no dashed line is drawn.

- [ ] **Step 3: Commit**

```bash
git add tests/test_advantage_alignment.py
git commit -m "test: cover baseline_alpha=None suppression in advantage alignment"
```

---

## Task 8: Failing test for `plot_mc_variance_curve` (default path)

**Files:**
- Modify: `tests/test_mc_variance_curve.py` (append a helper + failing test)

- [ ] **Step 1: Append a synthetic-data helper and a failing test**

Append to `tests/test_mc_variance_curve.py`:

```python
def _var_history(n_points: int, base: float):
    """Synthetic history list with mc_var_* fields populated."""
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': 0.0,
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
            'adv_product_s0': 0.0,
            'mc_var_undiscounted': base + 0.01 * i,
            'mc_var_discounted': 0.5 * (base + 0.01 * i),
        }
        for i in range(n_points)
    ]


def _zeta_results_for_mc_variance():
    """Synthetic all_results for the default variance cell:
    distance=6, horizon_type='small', alpha=1.0, B=budgets[-2] plus
    α=0 baseline. 4 zetas × 3 seeds + 1 zeta × 3 seeds = 15 entries."""
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
                'history': _var_history(4, base=0.05 * zeta + 0.01 * seed),
            })
    for seed in [0, 1, 2]:
        out.append({
            'distance': 6,
            'alpha': 0.0,
            'horizon_type': 'small',
            'horizon': h_val,
            'sample_budget': budget,
            'zeta': 0.0,
            'seed': seed,
            'mode': 'exact',
            'history': _var_history(4, base=0.02 + 0.005 * seed),
        })
    return out


def test_plot_mc_variance_curve_default_emits_two_pngs(tmp_path, monkeypatch):
    """Default invocation must emit exactly two PNGs (undiscounted +
    discounted) with the expected parameterized filenames."""
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

    sweep.plot_mc_variance_curve(
        _zeta_results_for_mc_variance(),
        mode='zeta',
        figures_dir=str(tmp_path),
    )

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    expected = {
        f'mc_variance_undiscounted_dist6_small_B{budget}_alpha1.00.png',
        f'mc_variance_discounted_dist6_small_B{budget}_alpha1.00.png',
    }
    pngs = {p.name for p in tmp_path.glob('*.png')}
    assert pngs == expected, f"expected {expected}, got {pngs}"
    for p in tmp_path.glob('*.png'):
        assert p.stat().st_size > 1000

    # Every captured figure must have a dashed black α=0 baseline line.
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert dashed_black, (
                "expected α=0 baseline overlay (dashed black) on "
                "every captured figure"
            )

    plt.close('all')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_plot_mc_variance_curve_default_emits_two_pngs -v`

Expected: FAIL with `AttributeError: module 'run_hypothesis_sweep' has no attribute 'plot_mc_variance_curve'`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mc_variance_curve.py
git commit -m "test: add failing default-path test for plot_mc_variance_curve"
```

---

## Task 9: Implement `plot_mc_variance_curve`

**Files:**
- Modify: `run_hypothesis_sweep.py` — insert new function immediately AFTER `plot_advantage_alignment`

- [ ] **Step 1: Insert the function definition**

In `run_hypothesis_sweep.py`, find the end of `plot_advantage_alignment` (the existing function body ends with `print(f'Saved {out_path}')` and a closing line). Immediately AFTER the closing of that function (before `def plot_learning_curves`), insert:

```python
def plot_mc_variance_curve(
    all_results: list,
    mode: str,
    figures_dir: str,
    *,
    distance: int = 6,
    horizon_type: str = 'small',
    alpha: float = 1.0,
    baseline_alpha: float = 0.0,
    budget_rank: int = -2,
):
    """Two-PNG single-cell figure of trajectory-return variance.

    Emits one PNG for undiscounted variance and one for discounted
    variance. Default cell: distance=6, horizon_type='small',
    alpha=1.0, sample_budget = calibrated budgets[budget_rank=-2].
    Override any kwarg to retarget. One line per teacher baseline,
    mean ± 1σ band across seeds. α=0 vanilla-NPG baseline overlay is
    added unless baseline_alpha=None. cap_zeta mode is a no-op.
    """
    if mode == 'cap_zeta':
        print("plot_mc_variance_curve: cap_zeta mode — skipping")
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
        print("plot_mc_variance_curve: no matching runs — skipping")
        return
    training_mode = next(iter(training_modes))

    n_goals = 1 if mode == 'zeta' else 3
    try:
        calib = json.load(open(_calibration_path_for(training_mode)))
    except FileNotFoundError:
        print("plot_mc_variance_curve: calibration JSON missing — skipping")
        return
    cell = _find_calibration_cell(calib, distance, horizon_type, n_goals)
    if not cell:
        print("plot_mc_variance_curve: no calibration cell — skipping")
        return
    budgets = cell.get('budgets', [])
    if not budgets or abs(budget_rank) > len(budgets):
        print("plot_mc_variance_curve: budget_rank out of range — skipping")
        return
    target['sample_budget'] = budgets[budget_rank]
    h_val = cell['horizon']

    tcol = _teacher_col(mode)
    x_label = 'update step' if training_mode == 'exact' else 'env step'

    os.makedirs(figures_dir, exist_ok=True)

    for field_name, y_label, file_tag, suptitle_tag in [
        ('mc_var_undiscounted',
         r'$\mathrm{Var}_{\mathrm{MC}}[G]$ [undiscounted]',
         'mc_variance_undiscounted', 'undiscounted'),
        ('mc_var_discounted',
         r'$\mathrm{Var}_{\mathrm{MC}}[G]$ [discounted]',
         'mc_variance_discounted', 'discounted'),
    ]:
        matching = [r for r in all_results
                    if all(r.get(k) == v for k, v in target.items())
                    and r['history']
                    and r['history'][0].get(field_name) is not None]
        if not matching:
            print(f"plot_mc_variance_curve: no field-bearing runs for "
                  f"{field_name} — skipping")
            continue

        groups = defaultdict(list)
        for r in matching:
            groups[r[tcol]].append(r['history'])

        sorted_teachers = _sort_teacher_vals(
            pd.Series(list(groups.keys())), mode,
        )

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for tv in sorted_teachers:
            histories = groups[tv]
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
            ax.plot(steps, mean, label=_teacher_label(mode, tv),
                    marker='o', markersize=3, linewidth=1.5)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

        _overlay_baseline_alpha(
            ax, all_results, mode, tcol, field_name, target, baseline_alpha,
        )

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        fig.suptitle(
            f'MC return variance ({mode} sweep, {suptitle_tag}) — '
            f"dist={distance}, H={h_val} ({horizon_type}), "
            f"B={target['sample_budget']}, " + rf'$\alpha={alpha}$',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(
            figures_dir,
            f'{file_tag}_dist{distance}_{horizon_type}'
            f"_B{target['sample_budget']}_alpha{alpha:.2f}.png",
        )
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'Saved {out_path}')
```

- [ ] **Step 2: Run Task 8's test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_plot_mc_variance_curve_default_emits_two_pngs -v`

Expected: PASS.

- [ ] **Step 3: Run the full variance test file**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py -v`

Expected: ALL 4 pass so far (3 from earlier tasks + the default-path).

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: add plot_mc_variance_curve (two PNGs, parameterized cell)"
```

---

## Task 10: Tests for `baseline_alpha=None` + `cap_zeta` no-op

**Files:**
- Modify: `tests/test_mc_variance_curve.py` (append two tests)

- [ ] **Step 1: Append both tests**

Append to `tests/test_mc_variance_curve.py`:

```python
def test_plot_mc_variance_curve_baseline_alpha_none(tmp_path, monkeypatch):
    """Passing baseline_alpha=None must suppress the α=0 baseline
    overlay in every emitted figure."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import run_hypothesis_sweep as sweep

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_mc_variance_curve(
        _zeta_results_for_mc_variance(),
        mode='zeta',
        figures_dir=str(tmp_path),
        baseline_alpha=None,
    )

    assert captured_figures, "expected at least one figure saved"
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert not dashed_black, (
                "expected NO dashed black baseline line when "
                f"baseline_alpha=None, got {len(dashed_black)}"
            )

    plt.close('all')


def test_plot_mc_variance_curve_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function returns
    immediately and writes nothing."""
    import run_hypothesis_sweep as sweep
    sweep.plot_mc_variance_curve([], mode='cap_zeta',
                                  figures_dir=str(tmp_path))
    assert not list(tmp_path.glob('*.png'))
```

- [ ] **Step 2: Run the two new tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py::test_plot_mc_variance_curve_baseline_alpha_none tests/test_mc_variance_curve.py::test_plot_mc_variance_curve_cap_zeta_noop -v`

Expected: BOTH PASS. The implementation from Task 9 already supports both paths.

- [ ] **Step 3: Run the full variance test file**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py -v`

Expected: ALL 6 pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mc_variance_curve.py
git commit -m "test: cover baseline_alpha=None and cap_zeta no-op for variance curve"
```

---

## Task 11: Wire `plot_mc_variance_curve` into `main()`

**Files:**
- Modify: `run_hypothesis_sweep.py` near line 2651 (the post-run plotting block)

- [ ] **Step 1: Add the new call**

In `run_hypothesis_sweep.py`, find the existing line `plot_advantage_alignment(all_results, args.mode, figures_dir)` (around line 2651 — verify by grepping). Immediately AFTER that line, add:

```python
        plot_mc_variance_curve(all_results, args.mode, figures_dir)
```

Match the surrounding indentation (8 spaces, inside the post-run / `--skip-run` plotting block).

- [ ] **Step 2: Smoke-test the wiring against an older cached pkl**

Run:

```bash
python -c "
import pickle
results = pickle.load(open('results/full_suite_20260419_105552/zeta_sample/zeta_sweep_results.pkl', 'rb'))
import run_hypothesis_sweep as sweep
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    sweep.plot_mc_variance_curve(results, mode='zeta', figures_dir=d)
    files = os.listdir(d)
    print('files:', files)
"
```

Expected: prints `files: []` (older pkls predate the schema additions) with INFO log `plot_mc_variance_curve: no field-bearing runs for mc_var_undiscounted — skipping` (and similar for mc_var_discounted). Function does not crash — confirms graceful handling.

If you get a numpy ImportError, try a different cached pkl. The goal is to confirm the function survives being called with a legacy pkl-loaded list.

- [ ] **Step 3: Run the full variance test file (no regressions)**

Run: `PYTHONPATH=. pytest tests/test_mc_variance_curve.py -v`

Expected: ALL 6 pass.

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: emit MC variance figure in sweep main()"
```

---

## Task 12: Visual smoke + final sanity sweep

**Files:**
- No code changes.

- [ ] **Step 1: Run an inline visual smoke test**

Run:

```bash
python -c "
from tabular_prototype.experiments import run_experiment
from tabular_prototype.environment import generate_equidistant_goals
import run_hypothesis_sweep as sweep
import os

# Fill the default cell (dist=6, small H, α=1.0) plus α=0 baseline.
goals = generate_equidistant_goals(9, 1, distance=6)
results = []
# Primary: α=1.0 with 4 zetas × 3 seeds
for zeta in [0.0, 0.33, 0.67, 1.0]:
    for seed in range(3):
        r = run_experiment(
            grid_size=9, goals=goals, teacher_capacity=1,
            zeta=zeta, alpha=1.0, lr=0.5,
            horizon=8, sample_budget=34, mode='exact',
            seed=seed, eval_interval=5,
        )
        r['distance'] = 6
        r['alpha'] = 1.0
        r['horizon_type'] = 'small'
        r['horizon'] = 8
        r['sample_budget'] = 34
        r['zeta'] = zeta
        r['seed'] = seed
        r['mode'] = 'exact'
        results.append(r)
# Baseline: α=0.0 with one zeta × 3 seeds
for seed in range(3):
    r = run_experiment(
        grid_size=9, goals=goals, teacher_capacity=1,
        zeta=0.0, alpha=0.0, lr=0.5,
        horizon=8, sample_budget=34, mode='exact',
        seed=seed, eval_interval=5,
    )
    r['distance'] = 6
    r['alpha'] = 0.0
    r['horizon_type'] = 'small'
    r['horizon'] = 8
    r['sample_budget'] = 34
    r['zeta'] = 0.0
    r['seed'] = seed
    r['mode'] = 'exact'
    results.append(r)

os.makedirs('/tmp/var_smoke', exist_ok=True)
sweep.plot_advantage_alignment(results, mode='zeta', figures_dir='/tmp/var_smoke')
sweep.plot_mc_variance_curve(results, mode='zeta', figures_dir='/tmp/var_smoke')
print('Files:', sorted(os.listdir('/tmp/var_smoke')))
"
```

Expected: prints three filenames: `advantage_alignment_dist6_small_B34_alpha1.00.png`, `mc_variance_discounted_dist6_small_B34_alpha1.00.png`, `mc_variance_undiscounted_dist6_small_B34_alpha1.00.png`.

- [ ] **Step 2: Open each figure and visually verify**

For each PNG in `/tmp/var_smoke/`, visually verify:
- Suptitle includes `dist=6, H=8 (small), B=34, α=1.0`.
- One dashed black line labeled `α=0.0 (vanilla NPG)` is overlaid.
- Per-teacher lines (4 ζ values for the variance figures, 4 ζ values for the advantage-alignment figure) are present and colored differently from the baseline.
- y-axis label is the expected LaTeX (`Var_MC[G]` for variance figures, `E_{a~π}[A^π A^μ]` for advantage).

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`

Expected: ALL pass. New tests added by this plan:
- `tests/test_mc_variance_curve.py` — 6 tests
- `tests/test_advantage_alignment.py` — 9 tests (1 new + 8 existing, with one strengthened to assert the baseline)

- [ ] **Step 4: Verify clean working tree**

Run: `git status`

Expected: `nothing to commit, working tree clean` apart from any pre-existing untracked files.

- [ ] **Step 5: Show the commit log delta**

Run: `git log --oneline origin/main..HEAD | head -25`

Expected: ~12 new commits on top of the design spec already on the branch, plus the spec and plan commits.

---

## Spec coverage check

- **`evaluate_policy` returns discounted fields** (spec §"`evaluate_policy` extension") → Task 1.
- **`mc_var_undiscounted` + `mc_var_discounted` in exact-mode history** (spec §"Schema addition") → Tasks 2, 3.
- **Same fields in trajectory-mode history** (spec §"Schema addition") → Tasks 2, 3.
- **`_overlay_baseline_alpha` shared helper** (spec §"Shared baseline-overlay helper") → Task 4.
- **`plot_advantage_alignment` retrofit with `baseline_alpha` kwarg + helper invocation** (spec §"Retrofit to `plot_advantage_alignment`") → Tasks 5, 6, 7.
- **`plot_mc_variance_curve` emits two PNGs** (spec §"Plot function `plot_mc_variance_curve`") → Tasks 8, 9.
- **`baseline_alpha=None` regression-safe path** (spec §Failure modes) → Tasks 7, 10.
- **`cap_zeta` no-op** (spec §Failure modes) → Task 10.
- **Wire into `main()`** (spec §"Wiring into `main()`") → Task 11.
- **Filename / suptitle annotations** (durable annotation rule) → Tasks 6, 9 implementation.

No spec gaps remain.
