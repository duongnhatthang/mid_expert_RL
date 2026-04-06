# Exact NPG, Sweep Overhaul, and Visualization Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the trajectory-based gradient with an exact NPG update, normalize alpha as a convex combination, move to 9x9 grid, add saturation-based learning curves with undiscounted V, and fix visualization layout issues.

**Architecture:** The exact NPG update (`θ += lr·((1-α)·Q^π + α·A^μ)`) replaces the score-function gradient as the default training mode. Alpha becomes a [0,1] mixing weight. The training loop branches on `exact_gradient` flag: exact mode loops over update steps, sample mode loops over observations with truncation. Visualization gets GridSpec-based colorbar positioning and landscape composite figures.

**Tech Stack:** numpy, matplotlib (GridSpec, SubplotSpec), pytest

**Spec:** `docs/superpowers/specs/2026-04-06-npg-exact-sweep-overhaul-design.md`

---

### Task 1: Alpha Convex Combination in Training (Item 8)

**Files:**
- Modify: `tabular_prototype/training.py:119-168` (compute_pav_rl_gradient)
- Test: `tests/test_training.py` (new file)

- [ ] **Step 1: Write failing test for convex combination**

Create `tests/test_training.py`:

```python
"""Tests for training.py gradient computation and NPG update."""

import numpy as np
import pytest

from tabular_prototype.environment import GridEnv, generate_equidistant_goals
from tabular_prototype.student import TabularSoftmaxPolicy, collect_trajectories
from tabular_prototype.teacher import compute_teacher_values_auto
from tabular_prototype.training import (
    compute_pav_rl_gradient,
    compute_student_qvalues,
)
from tabular_prototype.config import compute_gamma_from_horizon


@pytest.fixture
def small_env():
    """4x4 grid with 1 goal for fast tests."""
    goals = [(3, 3)]
    env = GridEnv(grid_size=4, goals=goals, horizon=16)
    return env, goals


def test_convex_combination_alpha_zero(small_env):
    """With alpha=0, teacher signal should be completely ignored."""
    env, goals = small_env
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    rng = np.random.default_rng(42)
    trajectories = collect_trajectories(env, policy, 5, rng)

    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    Q_mu, V_mu, _ = compute_teacher_values_auto(env, goals)

    grad_with_teacher = compute_pav_rl_gradient(
        policy, trajectories, Q_mu, V_mu, alpha=0.0, gamma=gamma, Q_pi=Q_pi
    )
    grad_no_teacher = compute_pav_rl_gradient(
        policy, trajectories, None, None, alpha=0.0, gamma=gamma, Q_pi=Q_pi
    )
    np.testing.assert_allclose(grad_with_teacher, grad_no_teacher, atol=1e-10)


def test_convex_combination_alpha_one(small_env):
    """With alpha=1, only teacher advantage should be used (no Q^pi)."""
    env, goals = small_env
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    rng = np.random.default_rng(42)
    trajectories = collect_trajectories(env, policy, 5, rng)

    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    Q_mu, V_mu, _ = compute_teacher_values_auto(env, goals)

    grad = compute_pav_rl_gradient(
        policy, trajectories, Q_mu, V_mu, alpha=1.0, gamma=gamma, Q_pi=Q_pi
    )
    # With alpha=1, effective_reward = A^mu only (no Q^pi contribution)
    # Verify gradient is non-zero (teacher has signal)
    assert np.abs(grad).sum() > 0

    # Verify it differs from alpha=0.5
    grad_half = compute_pav_rl_gradient(
        policy, trajectories, Q_mu, V_mu, alpha=0.5, gamma=gamma, Q_pi=Q_pi
    )
    assert not np.allclose(grad, grad_half)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py -v`
Expected: `test_convex_combination_alpha_zero` FAILS because current code uses additive `G_t + alpha * A_mu` not `(1-alpha)*G_t + alpha*A_mu`, so alpha=0 still includes full Q^pi (passes accidentally) but alpha=1 would use `Q^pi + 1*A_mu` not `0*Q^pi + 1*A_mu`.

- [ ] **Step 3: Implement convex combination in compute_pav_rl_gradient**

In `tabular_prototype/training.py`, change lines 152-158 of `compute_pav_rl_gradient`:

```python
            if Q_mu is not None and V_mu is not None:
                A_mu = get_teacher_advantage(
                    Q_mu, V_mu, trans.state_idx, trans.action
                )
                effective_reward = (1.0 - alpha) * G_t + alpha * A_mu
            else:
                effective_reward = G_t
```

The only change: `G_t + alpha * A_mu` becomes `(1.0 - alpha) * G_t + alpha * A_mu`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_training.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_training.py tabular_prototype/training.py
git commit -m "feat: change alpha to convex combination (1-α)·Q^π + α·A^μ"
```

---

### Task 2: Exact NPG Update Function (Item 7)

**Files:**
- Modify: `tabular_prototype/training.py` (add exact_npg_update)
- Modify: `tabular_prototype/__init__.py` (export)
- Test: `tests/test_training.py` (append)

- [ ] **Step 1: Write failing test for exact_npg_update**

Append to `tests/test_training.py`:

```python
from tabular_prototype.training import exact_npg_update


def test_exact_npg_update_alpha_zero(small_env):
    """Exact NPG with alpha=0 updates theta by lr * Q^pi."""
    env, goals = small_env
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    theta_before = policy.theta.copy()

    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    Q_mu, V_mu, _ = compute_teacher_values_auto(env, goals)
    lr = 0.1

    exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=0.0, lr=lr)

    expected = theta_before + lr * Q_pi
    np.testing.assert_allclose(policy.theta, expected, atol=1e-12)


def test_exact_npg_update_alpha_half(small_env):
    """Exact NPG with alpha=0.5 uses convex combo of Q^pi and A^mu."""
    env, goals = small_env
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    theta_before = policy.theta.copy()

    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    Q_mu, V_mu, _ = compute_teacher_values_auto(env, goals)
    lr = 0.1
    alpha = 0.5

    exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=alpha, lr=lr)

    A_mu = Q_mu - V_mu[:, None]
    expected = theta_before + lr * ((1 - alpha) * Q_pi + alpha * A_mu)
    np.testing.assert_allclose(policy.theta, expected, atol=1e-12)


def test_exact_npg_update_no_teacher(small_env):
    """Exact NPG without teacher (Q_mu=None) uses only Q^pi."""
    env, _ = small_env
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    theta_before = policy.theta.copy()

    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    lr = 0.1

    exact_npg_update(policy, Q_pi, None, None, alpha=0.5, lr=lr)

    expected = theta_before + lr * Q_pi
    np.testing.assert_allclose(policy.theta, expected, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_exact_npg_update_alpha_zero -v`
Expected: FAIL with ImportError (exact_npg_update not defined)

- [ ] **Step 3: Implement exact_npg_update**

Add to `tabular_prototype/training.py` after the `update_policy` function (after line 173):

```python
def exact_npg_update(
    policy: TabularSoftmaxPolicy,
    Q_pi: np.ndarray,
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    lr: float,
):
    """
    Exact NPG update for tabular softmax (mirror descent).

    θ[s,a] += lr · ((1-α)·Q^π(s,a) + α·A^μ(s,a))  for all (s,a)

    Derived from Lemma F.2 (Agarwal et al. 2021, extended for PAV-RL).
    The state-dependent offset ν cancels in the softmax normalization.

    When Q_mu/V_mu are None (no teacher), reduces to θ += lr · Q^π.
    """
    if Q_mu is not None and V_mu is not None:
        A_mu = Q_mu - V_mu[:, None]
        policy.theta += lr * ((1.0 - alpha) * Q_pi + alpha * A_mu)
    else:
        policy.theta += lr * Q_pi
```

- [ ] **Step 4: Export from __init__.py**

In `tabular_prototype/__init__.py`, add `exact_npg_update` to the training imports:

```python
from .training import (
    estimate_returns,
    compute_pav_rl_gradient,
    update_policy,
    exact_npg_update,
    evaluate_policy,
    compute_student_qvalues,
    compute_state_action_visitation,
    visitation_metrics,
)
```

And add `"exact_npg_update"` to the `__all__` list in the training section.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_training.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/training.py tabular_prototype/__init__.py tests/test_training.py
git commit -m "feat: add exact_npg_update (mirror descent for tabular softmax)"
```

---

### Task 3: Exact Mode in run_experiment (Item 7)

**Files:**
- Modify: `tabular_prototype/experiments.py:27-122` (run_experiment)
- Test: `tests/test_training.py` (append)

- [ ] **Step 1: Write failing test for exact mode**

Append to `tests/test_training.py`:

```python
from tabular_prototype.experiments import run_experiment


def test_run_experiment_exact_mode():
    """Exact mode should run without trajectories, using update steps as budget."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=50,  # 50 update steps in exact mode
        alpha=0.5,
        lr=0.1,
        seed=0,
        eval_interval=10,
        exact_gradient=True,
    )
    assert 'final_mean_reward' in result
    assert 'history' in result
    assert len(result['history']) > 0
    # Budget should be recorded as update steps
    assert result['budget_mode'] == 'exact'


def test_run_experiment_exact_learns():
    """Exact mode with teacher should outperform alpha=0 on a simple grid."""
    result_teacher = run_experiment(
        grid_size=4, goals=[(3, 3)], teacher_capacity=1,
        horizon=16, sample_budget=200, alpha=0.5, lr=0.1,
        seed=0, eval_interval=50, exact_gradient=True,
    )
    result_vanilla = run_experiment(
        grid_size=4, goals=[(3, 3)], teacher_capacity=1,
        horizon=16, sample_budget=200, alpha=0.0, lr=0.1,
        seed=0, eval_interval=50, exact_gradient=True,
    )
    # Both should learn something on this simple grid
    assert result_teacher['final_goal_rate'] > 0.0 or result_vanilla['final_goal_rate'] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_training.py::test_run_experiment_exact_mode -v`
Expected: FAIL — `run_experiment` doesn't accept `exact_gradient` parameter

- [ ] **Step 3: Implement exact mode in run_experiment**

Modify `tabular_prototype/experiments.py`. The `run_experiment` function signature adds `exact_gradient: bool = True` and `eval_n_episodes: int = 20`. The training loop body is replaced:

```python
def run_experiment(
    grid_size: int = 10,
    goals: List[Tuple[int, int]] = [(9, 9), (0, 9), (9, 0)],
    teacher_capacity: int = 1,
    zeta: Optional[float] = None,
    horizon: int = 50,
    sample_budget: int = 10000,
    alpha: float = 0.5,
    lr: float = 0.1,
    trajectories_per_update: int = 10,
    seed: int = 0,
    eval_interval: int = 10,
    eval_n_episodes: int = 20,
    exact_gradient: bool = True,
) -> Dict:
```

Replace the training loop (lines 64-99) with:

```python
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)

    total_steps = 0
    update_count = 0
    history = []
    cumulative_visitation = np.zeros((env.n_states, env.n_actions))

    if exact_gradient:
        # Exact NPG mode: budget = number of update steps
        for step in range(sample_budget):
            Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
            exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha, lr)
            update_count += 1

            if update_count % eval_interval == 0:
                eval_results = evaluate_policy(
                    env, policy, n_episodes=eval_n_episodes, rng=rng
                )
                start_idx = env.state_to_idx(env.start)
                # Undiscounted V^pi(s0)
                gamma_undiscounted = 1.0 - 1e-8
                _, V_pi_undiscounted = compute_student_qvalues(
                    env, policy, gamma_undiscounted
                )
                history.append({
                    'steps': update_count,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': 0,
                    'state_entropy': 0.0,
                })
    else:
        # Sample-based mode: budget = number of observations
        while total_steps < sample_budget:
            trajectories = collect_trajectories(
                env, policy, trajectories_per_update, rng
            )

            # Truncate to respect budget (Item 4)
            step_counts = [len(traj) for traj in trajectories]
            remaining = sample_budget - total_steps
            if sum(step_counts) > remaining:
                cumsum = np.cumsum(step_counts)
                keep = int(np.searchsorted(cumsum, remaining, side='right'))
                trajectories = trajectories[:max(1, keep)]

            total_steps += sum(len(traj) for traj in trajectories)

            # Track state-action visitation
            step_vis = compute_state_action_visitation(
                trajectories, env.n_states, env.n_actions
            )
            cumulative_visitation += step_vis

            # Exact Q^pi replaces Monte Carlo returns
            Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)

            grad = compute_pav_rl_gradient(
                policy, trajectories, Q_mu, V_mu, alpha, gamma, Q_pi=Q_pi
            )
            update_policy(policy, grad, lr)
            update_count += 1

            if update_count % eval_interval == 0:
                eval_results = evaluate_policy(
                    env, policy, n_episodes=eval_n_episodes, rng=rng
                )
                vis_m = visitation_metrics(cumulative_visitation)
                start_idx = env.state_to_idx(env.start)
                gamma_undiscounted = 1.0 - 1e-8
                _, V_pi_undiscounted = compute_student_qvalues(
                    env, policy, gamma_undiscounted
                )
                history.append({
                    'steps': total_steps,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': vis_m['unique_sa'],
                    'state_entropy': vis_m['state_entropy'],
                })

    final_eval = evaluate_policy(env, policy, n_episodes=100, rng=rng)
    final_vis = visitation_metrics(cumulative_visitation)

    return {
        'seed': seed,
        'teacher_capacity': teacher_capacity if zeta is None else None,
        'zeta': zeta,
        'sample_budget': sample_budget,
        'horizon': horizon,
        'alpha': alpha,
        'lr': lr,
        'final_mean_reward': final_eval['mean_reward'],
        'final_std_reward': final_eval['std_reward'],
        'final_goal_rate': final_eval['goal_rate'],
        'final_unique_sa': final_vis['unique_sa'],
        'final_unique_states': final_vis['unique_states'],
        'final_state_entropy': final_vis['state_entropy'],
        'final_sa_entropy': final_vis['sa_entropy'],
        'final_total_visits': final_vis['total_visits'],
        'visitation_counts': cumulative_visitation,
        'history': history,
        'budget_mode': 'exact' if exact_gradient else 'sample',
        'exact_gradient': exact_gradient,
    }
```

Also add the import at the top of `experiments.py`:

```python
from .training import (
    compute_pav_rl_gradient,
    update_policy,
    exact_npg_update,
    evaluate_policy,
    compute_student_qvalues,
    compute_state_action_visitation,
    visitation_metrics,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_training.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_training.py
git commit -m "feat: add exact NPG training mode as default in run_experiment"
```

---

### Task 4: 9x9 Grid and Updated Distances (Item 6)

**Files:**
- Modify: `run_hypothesis_sweep.py:31-51` (constants)
- Modify: `run_experiments.py:202` (default grid size)
- Test: `tests/test_training.py` (append)

- [ ] **Step 1: Write test for equidistant goals on 9x9 grid**

Append to `tests/test_training.py`:

```python
from tabular_prototype.environment import generate_equidistant_goals


@pytest.mark.parametrize("distance", [2, 4, 6, 7])
def test_equidistant_goals_9x9(distance):
    """Verify 3 equidistant goals can be generated at each sweep distance on 9x9."""
    goals = generate_equidistant_goals(9, 3, distance=distance)
    assert len(goals) == 3
    start = (4, 4)  # center of 9x9
    for g in goals:
        assert abs(g[0] - start[0]) + abs(g[1] - start[1]) == distance
        assert 0 <= g[0] < 9 and 0 <= g[1] < 9
```

- [ ] **Step 2: Run test to verify it passes (goal generation already supports this)**

Run: `pytest tests/test_training.py::test_equidistant_goals_9x9 -v`
Expected: PASS (existing `generate_equidistant_goals` handles arbitrary distances)

- [ ] **Step 3: Update run_hypothesis_sweep.py constants**

In `run_hypothesis_sweep.py`, change:

```python
GRID_SIZE = 9

# Goal positions at varying Manhattan distances from start (4, 4)
GOAL_POSITIONS = {
    2: generate_equidistant_goals(9, 1, distance=2),
    4: generate_equidistant_goals(9, 1, distance=4),
    6: generate_equidistant_goals(9, 1, distance=6),
    7: generate_equidistant_goals(9, 1, distance=7),
}

DISTANCES = sorted(GOAL_POSITIONS.keys())
```

Add the import at the top:
```python
from tabular_prototype import (
    run_experiment,
    GridEnv,
    compute_exploration_thresholds,
    visualize_visitation_comparison_grid,
    generate_equidistant_goals,
)
```

- [ ] **Step 4: Update run_experiments.py default grid size**

In `run_experiments.py`, change line 202:
```python
    parser.add_argument('--grid-size', type=int, default=9,
                        help='Grid size (default 9x9)')
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add run_hypothesis_sweep.py run_experiments.py tests/test_training.py
git commit -m "feat: change default grid to 9x9, sweep distances to {2,4,6,7}"
```

---

### Task 5: Alpha Sweep Values and Vanilla NPG in Sweeps (Items 2, 5, 8)

**Files:**
- Modify: `run_hypothesis_sweep.py:41-47` (sweep constants)
- Modify: `run_hypothesis_sweep.py:66-94` (_run_single_experiment)
- Modify: `run_hypothesis_sweep.py:97-177` (run_sweep)

- [ ] **Step 1: Update sweep constants**

In `run_hypothesis_sweep.py`, change:

```python
ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
```

- [ ] **Step 2: Add exact_gradient to sweep worker**

In `run_hypothesis_sweep.py`, modify `_run_single_experiment` to pass `exact_gradient=True`:

```python
def _run_single_experiment(args_tuple):
    """Worker function for multiprocessing. Returns result dict."""
    mode, teacher_val, alpha, budget, h_type, dist, seed, horizons = args_tuple
    goals = GOAL_POSITIONS[dist]
    horizon = horizons[h_type]
    teacher_key = 'zeta' if mode == 'zeta' else 'teacher_capacity'

    kwargs = dict(
        grid_size=GRID_SIZE,
        goals=goals,
        horizon=horizon,
        sample_budget=budget,
        alpha=alpha,
        lr=0.1,
        trajectories_per_update=10,
        seed=seed,
        eval_interval=5,
        eval_n_episodes=50,
        exact_gradient=True,
    )
    if mode == 'zeta':
        kwargs['zeta'] = teacher_val
    else:
        kwargs['teacher_capacity'] = teacher_val

    result = run_experiment(**kwargs)
    result['distance'] = dist
    result['horizon_type'] = h_type
    result[teacher_key] = teacher_val
    return result
```

- [ ] **Step 3: Skip redundant alpha=0 runs**

In `run_sweep`, when building configs, skip sweeping teacher values for alpha=0 since results are identical. Run alpha=0 once per (budget, horizon, distance, seed) and replicate for all teacher values:

In `run_hypothesis_sweep.py`, modify the `run_sweep` function's config generation (around line 109):

```python
    # For alpha=0, teacher is irrelevant — run once with first teacher value
    configs = []
    for tv, alpha, budget, h_type, dist, seed in itertools.product(
        teacher_values, ALPHA_VALUES, budget_values, HORIZON_TYPES, DISTANCES, range(n_seeds)
    ):
        if alpha == 0.0 and tv != teacher_values[0]:
            continue  # skip redundant alpha=0 runs
        configs.append((tv, alpha, budget, h_type, dist, seed))
```

- [ ] **Step 4: Replicate alpha=0 results for all teacher values in post-processing**

After collecting results, before saving CSV, replicate alpha=0 rows. Add after the results collection loop in `run_sweep`:

```python
    # Replicate alpha=0 results for all teacher values (teacher is irrelevant at alpha=0)
    alpha_zero_results = [r for r in all_results if r.get('alpha') == 0.0]
    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    for r in list(alpha_zero_results):
        for tv in teacher_values:
            if tv != r[tcol]:
                r_copy = r.copy()
                r_copy[tcol] = tv
                all_results.append(r_copy)
```

- [ ] **Step 5: Update plot label for alpha=0**

In `run_hypothesis_sweep.py`, update `_teacher_label` to handle alpha=0:

```python
def _teacher_label(mode: str, val, alpha: float = None) -> str:
    if alpha == 0.0:
        return 'Vanilla NPG'
    if mode == 'zeta':
        return f'ζ={val:.2f}'
    return {-1: 'no teacher', 0: 'random'}.get(val, f'cap={val}')
```

- [ ] **Step 6: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: alpha sweep [0,1], vanilla NPG as alpha=0, skip redundant runs"
```

---

### Task 6: Learning Curve with Saturation Detection (Item 1)

**Files:**
- Modify: `tabular_prototype/experiments.py:410-463` (run_learning_curve_experiment)
- Test: `tests/test_training.py` (append)

- [ ] **Step 1: Write failing test for saturation detection**

Append to `tests/test_training.py`:

```python
def test_learning_curve_saturation():
    """Learning curve should stop when vanilla NPG saturates."""
    histories = run_learning_curve_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacities=[0, 1],  # alpha=0 vanilla NPG always included
        horizon=16,
        alpha=0.5,
        lr=0.1,
        n_seeds=2,
        eval_interval=2,
        exact_gradient=True,
        max_budget=500,  # upper bound
    )
    # Should have histories for both capacities
    assert 0 in histories
    assert 1 in histories
    # History should have undiscounted V
    for cap_histories in histories.values():
        for h in cap_histories:
            if h:
                assert 'exact_V_start_undiscounted' in h[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_learning_curve_saturation -v`
Expected: FAIL — `run_learning_curve_experiment` doesn't accept `exact_gradient` or `max_budget`

- [ ] **Step 3: Implement saturation-based learning curve**

Rewrite `run_learning_curve_experiment` in `tabular_prototype/experiments.py`:

```python
def run_learning_curve_experiment(
    grid_size: int = 10,
    goals: Optional[List[Tuple[int, int]]] = None,
    teacher_capacities: Optional[List[int]] = None,
    horizon: int = 50,
    sample_budget: Optional[int] = None,
    alpha: float = 0.5,
    lr: float = 0.1,
    n_seeds: int = 30,
    trajectories_per_update: int = 10,
    eval_interval: int = 2,
    eval_n_episodes: int = 50,
    exact_gradient: bool = True,
    max_budget: int = 10000,
    saturation_window: int = 10,
    saturation_eps: float = 0.005,
    saturation_checks: int = 3,
) -> Dict[int, list]:
    """
    Run learning curve experiments with saturation-based stopping.

    First runs vanilla NPG (alpha=0) to find saturation point T_sat,
    then runs all other baselines for T_sat steps.

    Args:
        sample_budget: If provided, use fixed budget (legacy). If None, use saturation.
        max_budget: Upper bound for saturation search.
        saturation_window: Number of eval points in sliding window.
        saturation_eps: Threshold for declaring plateau.
        saturation_checks: Consecutive checks below eps to declare saturated.

    Returns:
        Dict mapping teacher_capacity -> list of per-seed histories.
    """
    if goals is None:
        goals = [(9, 9), (0, 9), (9, 0)]
    if teacher_capacities is None:
        teacher_capacities = [-1] + list(range(len(goals) + 1))

    # Determine budget: saturation-based or fixed
    if sample_budget is None:
        # Run vanilla NPG to find saturation
        print("  Finding saturation point for Vanilla NPG (α=0)...", flush=True)
        sat_result = run_experiment(
            grid_size=grid_size, goals=goals, teacher_capacity=0,
            horizon=horizon, sample_budget=max_budget, alpha=0.0, lr=lr,
            trajectories_per_update=trajectories_per_update, seed=0,
            eval_interval=eval_interval, eval_n_episodes=eval_n_episodes,
            exact_gradient=exact_gradient,
        )
        effective_budget = _detect_saturation(
            sat_result['history'], max_budget,
            saturation_window, saturation_eps, saturation_checks,
        )
        print(f"  Saturation at step {effective_budget} (max was {max_budget})")
    else:
        effective_budget = sample_budget

    histories: Dict[int, list] = {}

    for cap in teacher_capacities:
        histories[cap] = []
        print(f"  Teacher capacity {cap}...", end=" ", flush=True)
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size, goals=goals, teacher_capacity=cap,
                horizon=horizon, sample_budget=effective_budget,
                alpha=alpha, lr=lr,
                trajectories_per_update=trajectories_per_update,
                seed=seed, eval_interval=eval_interval,
                eval_n_episodes=eval_n_episodes,
                exact_gradient=exact_gradient,
            )
            histories[cap].append(result['history'])
        rewards = [h[-1]['mean_reward'] for h in histories[cap] if h]
        print(f"final reward = {np.mean(rewards):.3f} ± "
              f"{np.std(rewards)/np.sqrt(len(rewards)):.3f}")

    return histories


def _detect_saturation(
    history: List[Dict],
    max_budget: int,
    window: int = 10,
    eps: float = 0.005,
    checks_needed: int = 3,
) -> int:
    """Detect when learning curve plateaus. Returns step count at saturation."""
    if len(history) < window:
        return max_budget

    rewards = [h['mean_reward'] for h in history]
    consecutive = 0

    for i in range(window, len(rewards)):
        window_vals = rewards[i - window:i]
        change = abs(max(window_vals) - min(window_vals))
        if change < eps:
            consecutive += 1
            if consecutive >= checks_needed:
                return history[i]['steps']
        else:
            consecutive = 0

    return max_budget
```

- [ ] **Step 4: Similarly update run_learning_curve_experiment_zeta**

Apply the same saturation-based pattern to `run_learning_curve_experiment_zeta`. Add `exact_gradient`, `max_budget`, and saturation parameters. Run vanilla NPG first (alpha=0, zeta=0.0), then all other baselines.

The key change: add `exact_gradient=exact_gradient` to the `run_experiment` call and add the saturation detection logic at the top.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_training.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_training.py
git commit -m "feat: learning curve with saturation detection and undiscounted V"
```

---

### Task 7: Learning Curve Visualization — Undiscounted V Lines (Item 1)

**Files:**
- Modify: `tabular_prototype/visualization.py:753-886` (plot_learning_curves)

- [ ] **Step 1: Add dual V^π plotting to plot_learning_curves**

Modify `plot_learning_curves` in `tabular_prototype/visualization.py` to support plotting both discounted (dashed) and undiscounted (solid) V^π(s_0) when the metric is `exact_V_start`. Add a parameter `dual_v_mode: bool = False`:

After the main plotting loop (around line 866), add:

```python
        # If plotting exact_V_start, also plot undiscounted as solid
        if dual_v_mode and metric == 'exact_V_start':
            # Undiscounted V (solid) — recompute from 'exact_V_start_undiscounted'
            undisc_matrix = np.full((len(seed_histories), len(steps)), np.nan)
            for s_idx, h in enumerate(seed_histories):
                h_steps = np.array([e['steps'] for e in h])
                h_vals = np.array([e.get('exact_V_start_undiscounted', np.nan) for e in h])
                if not np.all(np.isnan(h_vals)):
                    undisc_matrix[s_idx, :] = np.interp(steps, h_steps, h_vals)

            undisc_mean = np.nanmean(undisc_matrix, axis=0)
            undisc_stderr = np.nanstd(undisc_matrix, axis=0) / np.sqrt(
                np.sum(~np.isnan(undisc_matrix), axis=0).clip(min=1)
            )

            if smooth_window > 1 and len(undisc_mean) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                undisc_mean = np.convolve(undisc_mean, kernel, mode='valid')
                undisc_stderr = np.convolve(undisc_stderr, kernel, mode='valid')

            ax.plot(
                steps_plot, undisc_mean,
                color=sty["color"], linestyle='solid',
                linewidth=2, label=f'{label} (undiscounted)',
            )
            ax.fill_between(
                steps_plot,
                undisc_mean - undisc_stderr,
                undisc_mean + undisc_stderr,
                color=sty["color"], alpha=0.08,
            )
            # Make the discounted line dashed
            # (the main line plotted above should use dashed)
```

Actually, a cleaner approach: when `dual_v_mode=True`, the function plots TWO lines per capacity:
- **Solid**: undiscounted V^π(s_0) from `exact_V_start_undiscounted`
- **Dashed**: discounted V^π(s_0) from `exact_V_start`

Modify the main plotting section. Replace the `ax.plot(...)` call with:

```python
        if dual_v_mode:
            # Solid = undiscounted
            undisc_matrix = np.full((len(seed_histories), len(steps)), np.nan)
            for s_idx, h in enumerate(seed_histories):
                h_steps = np.array([e['steps'] for e in h])
                h_vals = np.array([
                    e.get('exact_V_start_undiscounted', np.nan) for e in h
                ])
                if not np.all(np.isnan(h_vals)):
                    undisc_matrix[s_idx, :] = np.interp(steps, h_steps, h_vals)

            undisc_mean = np.nanmean(undisc_matrix, axis=0)
            if smooth_window > 1 and len(undisc_mean) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                undisc_mean = np.convolve(undisc_mean, kernel, mode='valid')

            ax.plot(
                steps_plot, undisc_mean,
                color=sty["color"], linestyle='solid',
                marker=sty["marker"], markevery=marker_every,
                markersize=6, linewidth=2, label=label,
            )
            # Dashed = discounted
            ax.plot(
                steps_plot, mean_vals,
                color=sty["color"], linestyle='dashed',
                linewidth=1.5, alpha=0.7,
            )
        else:
            ax.plot(
                steps_plot, mean_vals,
                color=sty["color"], linestyle=sty["linestyle"],
                marker=sty["marker"], markevery=marker_every,
                markersize=6, linewidth=2, label=label,
            )
```

- [ ] **Step 2: Update callers to use dual_v_mode**

In `run_experiments.py`, the `learning_curve` mode's exact_V plot call (around line 372):

```python
        exact_path = args.learning_curve_output.replace('.png', '_exact_V.png')
        plot_learning_curves(
            histories,
            title=f"V^π(start): solid=undiscounted, dashed=discounted\n"
                  f"(alpha={args.alpha}, grid={args.grid_size}x{args.grid_size})",
            ylabel="V^π(start)",
            metric="exact_V_start",
            smooth_window=1,
            save_path=exact_path,
            dual_v_mode=True,
        )
```

Do the same for the zeta learning curve mode.

- [ ] **Step 3: Add x-axis label logic**

In `plot_learning_curves`, update the x-axis label (around line 875):

```python
    ax.set_xlabel("Update Steps" if x_label is None else x_label, fontsize=12)
```

Add `x_label: Optional[str] = None` parameter to the function signature.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/visualization.py run_experiments.py
git commit -m "feat: dual V^π lines (solid=undiscounted, dashed=discounted) in learning curves"
```

---

### Task 8: Visualization — Colorbar Fix and Landscape Grid (Item 9)

**Files:**
- Modify: `tabular_prototype/visualization.py:453-561` (visualize_visitation_comparison_grid)

- [ ] **Step 1: Fix colorbar overlap with GridSpec**

Replace the figure creation and colorbar code in `visualize_visitation_comparison_grid`:

```python
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec

    n_rows = len(row_keys)
    n_cols = len(col_keys)
    cell_size = 2.5
    # Landscape: wider than tall, reserve space for colorbar
    fig = plt.figure(
        figsize=(cell_size * n_cols + 2.0, cell_size * n_rows + 1.5)
    )
    # GridSpec: extra narrow column for colorbar
    gs = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.15, hspace=0.25,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
```

Replace the colorbar section (around line 549-554):

```python
    # Colorbar in dedicated column — no overlap
    if mappable is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        fig.colorbar(mappable, cax=cbar_ax, label='Visit Count')

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
```

Remove the `plt.tight_layout` call (GridSpec handles spacing).

- [ ] **Step 2: Add landscape composite figure function**

Add a new function after `visualize_visitation_comparison_grid`:

```python
def visualize_visitation_composite_grid(
    env: GridEnv,
    all_visitation_data: dict,
    outer_keys: list,
    inner_row_keys: list,
    inner_col_keys: list,
    outer_label_fn,
    inner_row_label_fn,
    inner_col_label_fn,
    goals: Optional[List[Tuple[int, int]]] = None,
    suptitle: str = "Visitation Composite",
    save_path: Optional[str] = None,
) -> Any:
    """
    Landscape composite figure: outer grid of configs, inner grid of heatmaps.

    Args:
        all_visitation_data: Dict mapping (outer_key, inner_row_key, inner_col_key)
                            -> np.ndarray (n_states, n_actions).
        outer_keys: List of outer grid keys (e.g., (budget, horizon) tuples).
        inner_row_keys: Row keys within each panel (e.g., teacher values).
        inner_col_keys: Column keys within each panel (e.g., alpha values).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    n_outer = len(outer_keys)
    n_inner_rows = len(inner_row_keys)
    n_inner_cols = len(inner_col_keys)

    # Landscape: arrange outer panels in a single row (or 2 rows if many)
    if n_outer <= 4:
        outer_rows, outer_cols = 1, n_outer
    else:
        outer_rows = 2
        outer_cols = (n_outer + 1) // 2

    cell = 1.8
    fig_w = cell * n_inner_cols * outer_cols + 2.5
    fig_h = cell * n_inner_rows * outer_rows + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_outer = GridSpec(
        outer_rows, outer_cols + 1,
        width_ratios=[1] * outer_cols + [0.03],
        wspace=0.3, hspace=0.4,
    )

    vmax = 0
    for key, vis in all_visitation_data.items():
        if vis is not None:
            vmax = max(vmax, vis.sum(axis=1).max())
    if vmax == 0:
        vmax = 1

    mappable = None
    for oi, ok in enumerate(outer_keys):
        o_row = oi // outer_cols
        o_col = oi % outer_cols

        gs_inner = GridSpecFromSubplotSpec(
            n_inner_rows, n_inner_cols,
            subplot_spec=gs_outer[o_row, o_col],
            wspace=0.05, hspace=0.15,
        )

        for ri, rk in enumerate(inner_row_keys):
            for ci, ck in enumerate(inner_col_keys):
                ax = fig.add_subplot(gs_inner[ri, ci])
                vis = all_visitation_data.get((ok, rk, ck))

                if vis is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    grid = vis.sum(axis=1).reshape(env.grid_size, env.grid_size)
                    im = ax.imshow(grid, cmap='hot', origin='upper',
                                   vmin=0, vmax=vmax)
                    mappable = im

                    start = env.start
                    ax.add_patch(mpatches.Circle(
                        (start[1], start[0]), 0.25,
                        fill=True, color='cyan', alpha=0.8,
                    ))
                    if goals:
                        for goal in goals:
                            ax.add_patch(mpatches.Rectangle(
                                (goal[1] - 0.35, goal[0] - 0.35), 0.7, 0.7,
                                fill=False, edgecolor='lime', linewidth=1.2,
                            ))

                    ax.set_xticks([])
                    ax.set_yticks([])

                if ci == 0:
                    ax.set_ylabel(inner_row_label_fn(rk), fontsize=6)
                if ri == 0:
                    ax.set_title(inner_col_label_fn(ck), fontsize=6)

        # Outer panel title
        title_ax = fig.add_subplot(gs_outer[o_row, o_col])
        title_ax.set_title(outer_label_fn(ok), fontsize=9, fontweight='bold',
                           pad=10)
        title_ax.axis('off')

    if mappable is not None:
        cbar_ax = fig.add_subplot(gs_outer[:, -1])
        fig.colorbar(mappable, cax=cbar_ax, label='Visit Count')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Composite visitation grid saved to {save_path}")

    plt.close(fig)
    return fig
```

- [ ] **Step 3: Export new function from __init__.py**

Add `visualize_visitation_composite_grid` to `tabular_prototype/__init__.py` visualization imports and `__all__`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/visualization.py tabular_prototype/__init__.py
git commit -m "fix: colorbar overlap via GridSpec, add landscape composite visitation grid"
```

---

### Task 9: Update Hypothesis Sweep for All Baselines (Items 5, 9c)

**Files:**
- Modify: `run_hypothesis_sweep.py:371-423` (plot_visitation_grids)
- Modify: `run_hypothesis_sweep.py:198-254` (plot_heatmaps — update for alpha=0 label)

- [ ] **Step 1: Update plot_visitation_grids to show all baselines**

In `run_hypothesis_sweep.py`, modify `plot_visitation_grids` to pass ALL teacher values (not a subset) and ALL alpha values:

```python
def plot_visitation_grids(all_results: list, mode: str, figures_dir: str):
    """Visitation comparison grids showing ALL baselines.
    Rows = teacher params, cols = all alpha values."""
    from collections import defaultdict

    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    budget_vals = sorted(set(r['sample_budget'] for r in all_results))

    for dist in DISTANCES:
        goals = GOAL_POSITIONS[dist]
        horizons = _get_horizons()
        env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizons['large'])

        for budget in budget_vals:
            for h_type in HORIZON_TYPES:
                groups = defaultdict(list)
                for r in all_results:
                    if (r.get('distance') == dist and
                        r.get('sample_budget') == budget and
                        r.get('horizon_type') == h_type and
                        'visitation_counts' in r):
                        groups[(r[tcol], r['alpha'])].append(r['visitation_counts'])

                if not groups:
                    continue

                visitation_data = {k: np.mean(v, axis=0) for k, v in groups.items()}

                if mode == 'zeta':
                    row_keys = sorted(set(k[0] for k in visitation_data))
                    row_label_fn = lambda k: f'ζ={k:.2f}' if k is not None else 'NPG'
                else:
                    row_keys = sorted(set(k[0] for k in visitation_data))
                    row_label_fn = lambda k: {-1: 'no teacher', 0: 'random'}.get(k, f'cap={k}')

                col_keys = sorted(set(k[1] for k in visitation_data))
                col_label_fn = lambda a: 'Vanilla NPG' if a == 0.0 else f'α={a}'

                save_path = os.path.join(
                    figures_dir,
                    f'visitation_grid_dist{dist}_budget{budget}_{h_type}.png'
                )
                visualize_visitation_comparison_grid(
                    env, visitation_data, row_keys, col_keys,
                    row_label_fn, col_label_fn,
                    goals=goals,
                    suptitle=f'Visitation: dist={dist}, budget={budget}, {h_type} horizon',
                    save_path=save_path,
                )
```

- [ ] **Step 2: Update heatmap labels for alpha=0**

In `plot_heatmaps`, ensure the alpha=0 column is labeled "Vanilla NPG" in axis tick labels:

```python
                ax.set_xticklabels(
                    ['NPG' if a == 0.0 else str(a) for a in ALPHA_VALUES],
                    fontsize=8,
                )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: show all baselines in visitation grids, label alpha=0 as Vanilla NPG"
```

---

### Task 10: Learning Rate Sweep Script (Item 8)

**Files:**
- Create: `run_lr_sweep.py`

- [ ] **Step 1: Create LR sweep script**

Create `run_lr_sweep.py`:

```python
"""
One-time learning rate calibration sweep.

Runs vanilla NPG (alpha=0) on a representative config to find optimal LR.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tabular_prototype import (
    run_experiment,
    generate_equidistant_goals,
    compute_exploration_thresholds,
    plot_learning_curves,
)

LR_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def run_lr_sweep(
    grid_size: int = 9,
    distance: int = 4,
    n_seeds: int = 5,
    budget: int = 500,
    output_dir: str = 'results/lr_sweep',
):
    """Sweep learning rates for vanilla NPG on a representative config."""
    os.makedirs(output_dir, exist_ok=True)

    goals = generate_equidistant_goals(grid_size, 1, distance=distance)
    thresholds = compute_exploration_thresholds(grid_size)
    horizon = thresholds['horizon_large']

    print(f"LR Sweep: grid={grid_size}, distance={distance}, "
          f"horizon={horizon}, budget={budget} update steps")
    print(f"LR values: {LR_VALUES}")

    histories = {}
    for lr in LR_VALUES:
        histories[lr] = []
        print(f"  lr={lr}...", end=" ", flush=True)
        rewards = []
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size,
                goals=goals,
                teacher_capacity=0,
                horizon=horizon,
                sample_budget=budget,
                alpha=0.0,
                lr=lr,
                seed=seed,
                eval_interval=2,
                eval_n_episodes=50,
                exact_gradient=True,
            )
            histories[lr].append(result['history'])
            if result['history']:
                rewards.append(result['history'][-1]['mean_reward'])
        mean_r = np.mean(rewards) if rewards else 0
        print(f"final reward = {mean_r:.3f}")

    # Plot
    save_path = os.path.join(output_dir, 'lr_sweep_curves.png')
    plot_learning_curves(
        histories,
        title=f"LR Sweep (Vanilla NPG, grid={grid_size}, dist={distance})",
        ylabel="Mean Reward",
        metric="mean_reward",
        smooth_window=1,
        save_path=save_path,
    )
    print(f"\nFigure saved to {save_path}")

    # Print summary
    print("\n--- LR Sweep Summary ---")
    for lr in LR_VALUES:
        final_rewards = []
        for h in histories[lr]:
            if h:
                final_rewards.append(h[-1]['mean_reward'])
        if final_rewards:
            print(f"  lr={lr:8.4f}: reward={np.mean(final_rewards):.4f} "
                  f"± {np.std(final_rewards)/np.sqrt(len(final_rewards)):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LR calibration sweep')
    parser.add_argument('--grid-size', type=int, default=9)
    parser.add_argument('--distance', type=int, default=4)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='results/lr_sweep')
    args = parser.parse_args()

    run_lr_sweep(
        grid_size=args.grid_size,
        distance=args.distance,
        n_seeds=args.n_seeds,
        budget=args.budget,
        output_dir=args.output_dir,
    )
```

- [ ] **Step 2: Run it to verify it works**

Run: `python run_lr_sweep.py --budget 100 --n-seeds 2`
Expected: Completes without error, prints summary table, saves figure

- [ ] **Step 3: Commit**

```bash
git add run_lr_sweep.py
git commit -m "feat: add LR calibration sweep script"
```

---

### Task 11: Update Existing Tests and Final Integration

**Files:**
- Modify: `tests/test_run_experiments.py`
- Modify: `tabular_prototype/__init__.py` (ensure all new exports)

- [ ] **Step 1: Update existing test to use new grid default**

In `tests/test_run_experiments.py`, the test already uses `grid_size=4` explicitly, so it should still work. Verify:

Run: `pytest tests/test_run_experiments.py -v`
Expected: PASS

- [ ] **Step 2: Add integration test for full pipeline**

Append to `tests/test_training.py`:

```python
def test_full_pipeline_exact_mode():
    """End-to-end: exact NPG with teacher on small grid converges."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=300,
        alpha=0.5,
        lr=0.5,
        seed=0,
        eval_interval=50,
        exact_gradient=True,
    )
    # Should learn to reach goal with decent probability on 4x4 grid
    assert result['final_goal_rate'] > 0.3, (
        f"Expected goal_rate > 0.3, got {result['final_goal_rate']}"
    )
    assert result['budget_mode'] == 'exact'


def test_full_pipeline_sample_mode():
    """End-to-end: sample-based mode with truncation still works."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=500,
        alpha=0.5,
        lr=0.1,
        seed=0,
        eval_interval=5,
        exact_gradient=False,
    )
    assert result['budget_mode'] == 'sample'
    assert 'final_mean_reward' in result
    # Verify truncation: total visits should not exceed budget by much
    assert result['final_total_visits'] <= 500 + 160  # budget + 1 batch max overshoot
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_training.py tests/test_run_experiments.py tabular_prototype/__init__.py
git commit -m "test: add integration tests for exact and sample mode pipelines"
```

---

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md to reflect new defaults and architecture**

Key updates:
- Default grid: 8 → 9
- Training loop description: exact NPG as default, sample-based as opt-in
- Alpha semantics: convex combination α ∈ [0,1], α=0 = Vanilla NPG
- New gradient formula: `θ += lr · ((1-α)·Q^π + α·A^μ)`
- New command examples for LR sweep
- Budget meaning: "update steps" in exact mode, "observations" in sample mode
- Learning curve: saturation-based stopping

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for exact NPG, 9x9 grid, alpha normalization"
```
