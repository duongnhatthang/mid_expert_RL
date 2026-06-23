# Coverage / Distribution-Mismatch Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in diagnostic that measures, over training, the mismatch between the student policy's exact discounted state-action occupancy `d^π(s,a)` and two fixed reference occupancies `d^μ(s,a)` (analytic argmax-optimal + converged no-teacher α=0 NPG), reporting concentrability ratios and a panel of divergences with full per-`(s,a)` caching, plus curves and heatmaps.

**Architecture:** A new self-contained module `tabular_prototype/coverage.py` computes exact occupancy via a linear solve `d = (1-γ)ρ₀ᵀ(I-γP_π)⁻¹·π`, the divergence panel (Laplace-smoothed + capped), and the two reference occupancies. `experiments.run_experiment` gains a `track_coverage` flag that builds the references once and injects per-eval-tick scalar metrics into `history` (so existing figure machinery consumes them unchanged) plus final-tick grids into the result. `visualization.py` adds curve and heatmap plotters; `run_npg_diagnostics.py` gains a `--track-coverage` path.

**Tech Stack:** Python 3.9, numpy 2.0.2, matplotlib, pytest. Local conda env `mid_expert`.

## Global Constraints

- Run tests with: `PYTHONPATH=. pytest tests/ -v` (conda env `mid_expert`).
- Occupancy is **exact** (linear solve), never sampled.
- Absorbing states (goals/traps) are **not** self-looping in the env transition (`_apply_action` moves out of them); the occupancy solve MUST self-loop absorbing states explicitly, else discounted mass leaks out of goals.
- Goals are deterministic across seeds (`generate_equidistant_goals` takes no rng), so the env layout — and `d^μ` — is identical across seeds for a fixed cell; `d^π` grids may be averaged across seeds.
- `track_coverage` defaults to `False`; when off, `run_experiment` behavior and output are byte-for-byte unchanged (zero new cost).
- Reference (`learned`) training uses exact NPG steps with saturation early-stop, capped at a fixed step budget (default 2000), decoupled from the student's budget unit.
- Flattened history keys use the form `cov_<ref>_<metric>` with `ref ∈ {analytic, learned}`.
- Two-line figure titles with the `(dist, H, B, α)` cell annotation, matching the existing `_generic_metric_figure` convention.

---

### Task 1: Exact occupancy core (`coverage.py`)

**Files:**
- Create: `tabular_prototype/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `GridEnv` (`env.n_states`, `env.n_actions`, `env._apply_action`, `env.state_to_idx`, `env.get_all_absorbing_states`, `env.start`); `TabularSoftmaxPolicy.theta`.
- Produces:
  - `build_transition_table(env) -> np.ndarray` int `(n_states, n_actions)`, `T[s,a]=next_idx`.
  - `policy_to_matrix(policy) -> np.ndarray` float `(n_states, n_actions)`, row-softmax of `policy.theta`.
  - `compute_occupancy(transition, policy_probs, start_dist, gamma, absorbing_states) -> np.ndarray` float `(n_states, n_actions)`, sums to 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype import coverage


def _tiny_env():
    # 3x3 grid, single goal at a corner, short horizon.
    return GridEnv(grid_size=3, goals=[(0, 0)], horizon=10)


def test_transition_table_shape_and_values():
    env = _tiny_env()
    T = coverage.build_transition_table(env)
    assert T.shape == (env.n_states, env.n_actions)
    # From every state, moving 'up' (action 0) then mapping must match env.
    for s in range(env.n_states):
        state = env.idx_to_state(s)
        for a in range(env.n_actions):
            assert T[s, a] == env.state_to_idx(env._apply_action(state, a))


def test_policy_to_matrix_is_row_softmax():
    env = _tiny_env()
    from tabular_prototype.student import TabularSoftmaxPolicy
    pol = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    pol.theta = np.arange(env.n_states * env.n_actions, dtype=float).reshape(
        env.n_states, env.n_actions
    )
    M = coverage.policy_to_matrix(pol)
    assert M.shape == (env.n_states, env.n_actions)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(M[0], pol.get_probs(0), atol=1e-12)


def test_occupancy_sums_to_one_and_self_loops_absorbing():
    env = _tiny_env()
    T = coverage.build_transition_table(env)
    n_s, n_a = env.n_states, env.n_actions
    probs = np.full((n_s, n_a), 1.0 / n_a)
    start = np.zeros(n_s)
    start[env.state_to_idx(env.start)] = 1.0
    absorbing = {env.state_to_idx(s) for s in env.get_all_absorbing_states()}
    d = coverage.compute_occupancy(T, probs, start, gamma=0.9,
                                   absorbing_states=absorbing)
    assert d.shape == (n_s, n_a)
    assert d.min() >= 0.0
    np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)


def test_occupancy_absorbing_only_chain_closed_form():
    # 2-state chain: state 0 -> state 1 (absorbing). Single action.
    # d(0) = (1-g)*1 ; remaining mass accumulates on absorbing state 1.
    T = np.array([[1], [1]])           # s0 -> s1 ; s1 self (absorbing anyway)
    probs = np.array([[1.0], [1.0]])
    start = np.array([1.0, 0.0])
    g = 0.8
    d = coverage.compute_occupancy(T, probs, start, gamma=g,
                                   absorbing_states={1})
    # d is (2,1); state occupancy = d[:,0].
    np.testing.assert_allclose(d[0, 0], 1.0 - g, atol=1e-9)
    np.testing.assert_allclose(d[1, 0], g, atol=1e-9)
    np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tabular_prototype.coverage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tabular_prototype/coverage.py
"""Coverage / distribution-mismatch diagnostics.

Exact discounted state-action occupancy and divergences between the student
policy's occupancy d^pi and fixed reference occupancies d^mu.
"""

import numpy as np


def build_transition_table(env) -> np.ndarray:
    """Deterministic transition table T[s, a] -> next_state_idx."""
    T = np.zeros((env.n_states, env.n_actions), dtype=int)
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        for a in range(env.n_actions):
            T[s_idx, a] = env.state_to_idx(env._apply_action(state, a))
    return T


def policy_to_matrix(policy) -> np.ndarray:
    """Row-softmax of policy.theta -> (n_states, n_actions) prob matrix."""
    logits = policy.theta - policy.theta.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_occupancy(transition, policy_probs, start_dist, gamma,
                      absorbing_states) -> np.ndarray:
    """Exact discounted state-action occupancy.

    d_state^T = (1 - gamma) * start_dist^T (I - gamma P_pi)^{-1}
    d(s, a)   = d_state(s) * policy_probs(s, a),   sum_{s,a} d = 1.

    Absorbing states self-loop (P[s,s]=1) so terminal mass accumulates there
    instead of leaking, matching episodic termination.
    """
    n_states, n_actions = policy_probs.shape
    absorbing = set(absorbing_states)
    P = np.zeros((n_states, n_states))
    for s in range(n_states):
        if s in absorbing:
            P[s, s] = 1.0
            continue
        for a in range(n_actions):
            P[s, transition[s, a]] += policy_probs[s, a]
    A = np.eye(n_states) - gamma * P
    d_state = (1.0 - gamma) * np.linalg.solve(A.T, np.asarray(start_dist, float))
    return d_state[:, None] * policy_probs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/coverage.py tests/test_coverage.py
git commit -m "feat(coverage): exact discounted state-action occupancy"
```

---

### Task 2: Divergence panel (`coverage.py`)

**Files:**
- Modify: `tabular_prototype/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: occupancy arrays `(n_states, n_actions)` from Task 1.
- Produces:
  - `coverage_divergences(d_pi, d_mu, eps=1e-9, cap=1e3) -> dict` with keys `max_mu_over_pi, max_pi_over_mu, chi2_pi_mu, kl_pi_mu, kl_mu_pi, tv, renyi_inf` (all floats).
  - `coverage_ratio_grids(d_pi, d_mu, eps=1e-9, cap=1e3) -> (ratio_mu_over_pi, ratio_pi_over_mu)` each `(n_states, n_actions)`, smoothed + capped.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_coverage.py

def _rand_dist(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.random(shape)
    return x / x.sum()


def test_divergences_zero_for_identical():
    d = _rand_dist((9, 4), seed=0)
    out = coverage.coverage_divergences(d, d)
    assert out['chi2_pi_mu'] == pytest.approx(0.0, abs=1e-9)
    assert out['kl_pi_mu'] == pytest.approx(0.0, abs=1e-9)
    assert out['kl_mu_pi'] == pytest.approx(0.0, abs=1e-9)
    assert out['tv'] == pytest.approx(0.0, abs=1e-9)
    assert out['max_pi_over_mu'] == pytest.approx(1.0, abs=1e-6)
    assert out['max_mu_over_pi'] == pytest.approx(1.0, abs=1e-6)
    assert out['renyi_inf'] == pytest.approx(0.0, abs=1e-6)


def test_divergences_finite_on_disjoint_support():
    n = (9, 4)
    d_pi = np.zeros(n); d_pi[0, 0] = 1.0
    d_mu = np.zeros(n); d_mu[1, 1] = 1.0   # disjoint support pre-smoothing
    out = coverage.coverage_divergences(d_pi, d_mu, eps=1e-6, cap=1e3)
    for v in out.values():
        assert np.isfinite(v)
    # Capped ratios never exceed cap.
    assert out['max_pi_over_mu'] <= 1e3 + 1e-6
    assert out['max_mu_over_pi'] <= 1e3 + 1e-6


def test_ratio_grids_shape_and_cap():
    d_pi = _rand_dist((9, 4), seed=1)
    d_mu = _rand_dist((9, 4), seed=2)
    r_mp, r_pm = coverage.coverage_ratio_grids(d_pi, d_mu, cap=50.0)
    assert r_mp.shape == (9, 4) and r_pm.shape == (9, 4)
    assert r_mp.max() <= 50.0 and r_pm.max() <= 50.0
    assert np.isfinite(r_mp).all() and np.isfinite(r_pm).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -k "divergences or ratio_grids" -v`
Expected: FAIL — `AttributeError: module 'tabular_prototype.coverage' has no attribute 'coverage_divergences'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to tabular_prototype/coverage.py

def _smooth(d, eps):
    d = np.asarray(d, float) + eps
    return d / d.sum()


def coverage_divergences(d_pi, d_mu, eps=1e-9, cap=1e3) -> dict:
    """Laplace-smoothed divergences between d_pi and d_mu (both summing to 1)."""
    dp = _smooth(d_pi, eps)
    dm = _smooth(d_mu, eps)
    r_mp = np.clip(dm / dp, 0.0, cap)
    r_pm = np.clip(dp / dm, 0.0, cap)
    max_pi_over_mu = float(r_pm.max())
    return {
        'max_mu_over_pi': float(r_mp.max()),
        'max_pi_over_mu': max_pi_over_mu,
        'chi2_pi_mu': float(np.sum((dp - dm) ** 2 / dm)),
        'kl_pi_mu': float(np.sum(dp * np.log(dp / dm))),
        'kl_mu_pi': float(np.sum(dm * np.log(dm / dp))),
        'tv': float(0.5 * np.sum(np.abs(dp - dm))),
        'renyi_inf': float(np.log(max_pi_over_mu)),
    }


def coverage_ratio_grids(d_pi, d_mu, eps=1e-9, cap=1e3):
    """Smoothed + capped per-(s,a) ratio grids (mu/pi, pi/mu)."""
    dp = _smooth(d_pi, eps)
    dm = _smooth(d_mu, eps)
    return np.clip(dm / dp, 0.0, cap), np.clip(dp / dm, 0.0, cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/coverage.py tests/test_coverage.py
git commit -m "feat(coverage): smoothed/capped divergence panel + ratio grids"
```

---

### Task 3: Reference policies + metric orchestrator (`coverage.py`)

**Files:**
- Modify: `tabular_prototype/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `build_optimal_policy` (teacher), `TabularSoftmaxPolicy`, `compute_student_qvalues`, `exact_npg_update` (training); Task 1/2 functions.
- Produces:
  - `build_reference_policy(env, gamma, ref_budget=2000, lr=0.5, tol=1e-5, patience=10) -> np.ndarray` `(n_states, n_actions)` — frozen converged α=0 softmax (the `learned` ref). Deterministic (exact NPG has no rng).
  - `build_reference_occupancies(env, gamma, ref_budget=2000, lr=0.5) -> dict` mapping `'analytic'` and `'learned'` to `d^μ` arrays.
  - `compute_coverage_metrics(env, student_policy, references, gamma, eps=1e-9, cap=1e3, want_grids=False) -> (scalars: dict, grids: dict)`. `scalars` keys are `cov_<ref>_<metric>`. `grids[ref] = {'d_mu','d_pi','ratio_mu_over_pi','ratio_pi_over_mu'}` only when `want_grids` (else `{}`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_coverage.py
from tabular_prototype.config import compute_gamma_from_horizon
from tabular_prototype.student import TabularSoftmaxPolicy


def test_build_reference_policy_is_valid_stochastic_and_deterministic():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    p1 = coverage.build_reference_policy(env, g, ref_budget=200)
    p2 = coverage.build_reference_policy(env, g, ref_budget=200)
    assert p1.shape == (env.n_states, env.n_actions)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-9)
    assert (p1 > 0).all()                      # softmax => full support
    np.testing.assert_allclose(p1, p2, atol=1e-12)  # reproducible


def test_build_reference_occupancies_keys_and_sums():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    refs = coverage.build_reference_occupancies(env, g, ref_budget=200)
    assert set(refs) == {'analytic', 'learned'}
    for d in refs.values():
        np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)


def test_compute_coverage_metrics_flat_keys_and_grids():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    refs = coverage.build_reference_occupancies(env, g, ref_budget=200)
    student = TabularSoftmaxPolicy(env.n_states, env.n_actions)  # uniform
    scalars, grids = coverage.compute_coverage_metrics(
        env, student, refs, g, want_grids=True
    )
    assert 'cov_analytic_max_pi_over_mu' in scalars
    assert 'cov_learned_chi2_pi_mu' in scalars
    for v in scalars.values():
        assert np.isfinite(v)
    assert set(grids) == {'analytic', 'learned'}
    assert grids['analytic']['d_pi'].shape == (env.n_states, env.n_actions)
    # want_grids=False -> empty grids dict.
    _, grids_off = coverage.compute_coverage_metrics(env, student, refs, g)
    assert grids_off == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -k "reference or coverage_metrics" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_reference_policy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to tabular_prototype/coverage.py
from .teacher import build_optimal_policy
from .student import TabularSoftmaxPolicy
from .training import compute_student_qvalues, exact_npg_update


def _absorbing_indices(env):
    return {env.state_to_idx(s) for s in env.get_all_absorbing_states()}


def _start_dist(env):
    d = np.zeros(env.n_states)
    d[env.state_to_idx(env.start)] = 1.0
    return d


def build_reference_policy(env, gamma, ref_budget=2000, lr=0.5,
                           tol=1e-5, patience=10) -> np.ndarray:
    """Train an alpha=0 vanilla-NPG student to saturation; return its softmax.

    Exact NPG has no randomness, so the result is deterministic for a fixed
    (env, gamma). Early-stops when the policy matrix stops changing for
    `patience` consecutive steps, capped at ref_budget steps.
    """
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    prev = None
    stable = 0
    for _ in range(ref_budget):
        Q_pi, _ = compute_student_qvalues(env, policy, gamma)
        exact_npg_update(policy, Q_pi, None, None, alpha=0.0, lr=lr)
        cur = policy_to_matrix(policy)
        if prev is not None and np.abs(cur - prev).max() < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev = cur
    return policy_to_matrix(policy)


def build_reference_occupancies(env, gamma, ref_budget=2000, lr=0.5) -> dict:
    """d^mu for both references ('analytic' argmax-optimal, 'learned' alpha=0)."""
    T = build_transition_table(env)
    absorbing = _absorbing_indices(env)
    start = _start_dist(env)
    pi_analytic = build_optimal_policy(env, env.goals, gamma)
    pi_learned = build_reference_policy(env, gamma, ref_budget, lr)
    return {
        'analytic': compute_occupancy(T, pi_analytic, start, gamma, absorbing),
        'learned': compute_occupancy(T, pi_learned, start, gamma, absorbing),
    }


def compute_coverage_metrics(env, student_policy, references, gamma,
                             eps=1e-9, cap=1e3, want_grids=False):
    """Per-eval-tick coverage scalars (and optional final-tick grids)."""
    T = build_transition_table(env)
    absorbing = _absorbing_indices(env)
    start = _start_dist(env)
    d_pi = compute_occupancy(
        T, policy_to_matrix(student_policy), start, gamma, absorbing
    )
    scalars, grids = {}, {}
    for name, d_mu in references.items():
        for k, v in coverage_divergences(d_pi, d_mu, eps, cap).items():
            scalars[f'cov_{name}_{k}'] = v
        if want_grids:
            r_mp, r_pm = coverage_ratio_grids(d_pi, d_mu, eps, cap)
            grids[name] = {
                'd_mu': d_mu, 'd_pi': d_pi,
                'ratio_mu_over_pi': r_mp, 'ratio_pi_over_mu': r_pm,
            }
    return scalars, grids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_coverage.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/coverage.py tests/test_coverage.py
git commit -m "feat(coverage): reference occupancies + metric orchestrator"
```

---

### Task 4: Wire `track_coverage` into `run_experiment`

**Files:**
- Modify: `tabular_prototype/experiments.py` (signature ~44-59; after teacher/gamma block ~98; exact eval tick ~159-184; trajectory eval tick ~294-320; result dict ~332-353)
- Test: `tests/test_coverage_experiment.py`

**Interfaces:**
- Consumes: `coverage.build_reference_occupancies`, `coverage.compute_coverage_metrics`.
- Produces: `run_experiment(..., track_coverage=False, coverage_ref_budget=2000)`; when `track_coverage`, each `history` eval-tick entry gains `cov_<ref>_<metric>` keys and the result gains `'coverage_grids'` (dict keyed by ref at the final tick, else `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage_experiment.py
import numpy as np
from tabular_prototype.experiments import run_experiment


def _kwargs(**over):
    base = dict(grid_size=5, goals=[(0, 2), (4, 2), (2, 0)],
                teacher_capacity=3, horizon=12, sample_budget=8,
                alpha=1.0, lr=0.5, seed=0, eval_interval=2,
                eval_n_episodes=5, mode='exact')
    base.update(over)
    return base


def test_track_coverage_populates_history_and_grids():
    r = run_experiment(track_coverage=True, **_kwargs())
    last = r['history'][-1]
    for key in ('cov_analytic_max_pi_over_mu', 'cov_analytic_chi2_pi_mu',
                'cov_learned_kl_pi_mu', 'cov_learned_tv',
                'cov_analytic_renyi_inf'):
        assert key in last and np.isfinite(last[key])
    grids = r['coverage_grids']
    assert set(grids) == {'analytic', 'learned'}
    n_sa = (5 * 5, 4)
    assert grids['analytic']['d_pi'].shape == n_sa
    np.testing.assert_allclose(grids['learned']['d_mu'].sum(), 1.0, atol=1e-9)


def test_track_coverage_off_is_unchanged():
    r = run_experiment(track_coverage=False, **_kwargs())
    assert r['coverage_grids'] is None
    assert not any(k.startswith('cov_') for k in r['history'][-1])


def test_track_coverage_sample_mode():
    r = run_experiment(track_coverage=True,
                       **_kwargs(mode='sample', sample_budget=60))
    assert any(k.startswith('cov_') for k in r['history'][-1])
    assert r['coverage_grids'] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_coverage_experiment.py -v`
Expected: FAIL — `TypeError: run_experiment() got an unexpected keyword argument 'track_coverage'`.

- [ ] **Step 3: Write minimal implementation**

In `tabular_prototype/experiments.py`, add the import near the top (after the `.visualization` import on line 24):

```python
from . import coverage as coverage_mod
```

Extend the signature (lines 58-59) — add the two params before the closing `)`:

```python
    mode: str = "exact",
    pg_diag_enabled: bool = False,
    track_coverage: bool = False,
    coverage_ref_budget: int = 2000,
) -> Dict:
```

After the teacher/gamma block (immediately after line 97, before `policy = TabularSoftmaxPolicy(...)` on line 99), build references once and init the grids holder:

```python
    coverage_refs = None
    coverage_grids = None
    if track_coverage:
        coverage_refs = coverage_mod.build_reference_occupancies(
            env, gamma, ref_budget=coverage_ref_budget, lr=lr
        )
```

In **exact mode**, inside the eval-tick block, after `hist_entry` is built and before `history.append(hist_entry)` (between lines 181 and 182, i.e. right after the dict literal and the `if exact_diag is not None:` merge), add:

```python
                if track_coverage:
                    cov_scalars, cov_grids = coverage_mod.compute_coverage_metrics(
                        env, policy, coverage_refs, gamma,
                        want_grids=is_last_step,
                    )
                    hist_entry.update(cov_scalars)
                    if is_last_step:
                        coverage_grids = cov_grids
```

In **trajectory mode** (`hybrid`/`sample`), inside its eval-tick block, after the `if npg_diag is not None:` merge and before `history.append(hist_entry)` (between lines 319 and 320), add:

```python
                if track_coverage:
                    cov_scalars, cov_grids = coverage_mod.compute_coverage_metrics(
                        env, policy, coverage_refs, gamma,
                        want_grids=is_last,
                    )
                    hist_entry.update(cov_scalars)
                    if is_last:
                        coverage_grids = cov_grids
```

In the result dict (after line 350 `'diagnostics': all_diagnostics,`), add:

```python
        'coverage_grids': coverage_grids,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_coverage_experiment.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `PYTHONPATH=. pytest tests/ -q`
Expected: PASS (pre-existing tests unaffected; `coverage_grids` key is additive and `None` by default).

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_coverage_experiment.py
git commit -m "feat(experiments): opt-in track_coverage hook in run_experiment"
```

---

### Task 5: Coverage figures (`visualization.py`)

**Files:**
- Modify: `tabular_prototype/visualization.py` (add after `plot_u_cosine`, ~line 1632)
- Test: `tests/test_coverage_plots.py`

**Interfaces:**
- Consumes: `_generic_metric_figure`; history dicts containing `cov_<ref>_<metric>`; grid dicts `{'d_mu','d_pi','ratio_mu_over_pi','ratio_pi_over_mu'}`.
- Produces:
  - `plot_coverage_curves(histories_by_teacher, baseline_history_alpha_zero, mode, out_dir, cell_info, ref) -> list[str]` — writes one PNG per metric, returns written paths. Filenames `cov_<ref>_<metric>.png`.
  - `plot_coverage_heatmaps(ref, grids, grid_size, out_path, cell_info, student_label) -> None` — 4-panel state-marginal heatmaps (`d^μ(s)`, `d^π(s)`, ratio μ/π, ratio π/μ).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage_plots.py
import os
import numpy as np
from tabular_prototype import visualization as viz

CELL = {'distance': 6, 'horizon': 8, 'horizon_type': 'small',
        'sample_budget': 12, 'alpha': 1.0}

METRICS = ['max_mu_over_pi', 'max_pi_over_mu', 'chi2_pi_mu',
           'kl_pi_mu', 'kl_mu_pi', 'tv', 'renyi_inf']


def _hist(seed_offset):
    out = []
    for tick in range(3):
        entry = {'steps': (tick + 1) * 2}
        for m in METRICS:
            entry[f'cov_analytic_{m}'] = 1.0 + 0.1 * tick + 0.01 * seed_offset
        out.append(entry)
    return out


def test_plot_coverage_curves_writes_all_metrics(tmp_path):
    hbt = {0: [_hist(0), _hist(1)], 3: [_hist(2), _hist(3)]}
    baseline = [_hist(4), _hist(5)]
    paths = viz.plot_coverage_curves(
        histories_by_teacher=hbt,
        baseline_history_alpha_zero=baseline,
        mode='capability', out_dir=str(tmp_path), cell_info=CELL,
        ref='analytic',
    )
    assert len(paths) == len(METRICS)
    for p in paths:
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_plot_coverage_heatmaps_writes_file(tmp_path):
    n_s, n_a, gs = 25, 4, 5
    grids = {
        'd_mu': np.abs(np.random.default_rng(0).random((n_s, n_a))),
        'd_pi': np.abs(np.random.default_rng(1).random((n_s, n_a))),
        'ratio_mu_over_pi': np.abs(np.random.default_rng(2).random((n_s, n_a))),
        'ratio_pi_over_mu': np.abs(np.random.default_rng(3).random((n_s, n_a))),
    }
    out = str(tmp_path / 'cov_analytic_heatmap.png')
    viz.plot_coverage_heatmaps('analytic', grids, gs, out, CELL,
                               student_label='cap=3 (α=1)')
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_coverage_plots.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'plot_coverage_curves'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to tabular_prototype/visualization.py

# Metric -> (ylabel, title fragment) for coverage curves.
_COVERAGE_METRICS = {
    'max_pi_over_mu': (r'$\max_{s,a}\, d^\pi/d^\mu$  (concentrability $C$)',
                       'max ratio dπ/dμ'),
    'max_mu_over_pi': (r'$\max_{s,a}\, d^\mu/d^\pi$',
                       'max ratio dμ/dπ'),
    'chi2_pi_mu': (r'$\chi^2(d^\pi\,\|\,d^\mu)$', 'chi-square'),
    'kl_pi_mu': (r'$\mathrm{KL}(d^\pi\,\|\,d^\mu)$', 'KL(π‖μ)'),
    'kl_mu_pi': (r'$\mathrm{KL}(d^\mu\,\|\,d^\pi)$', 'KL(μ‖π)'),
    'tv': (r'$\mathrm{TV}(d^\pi, d^\mu)$', 'total variation'),
    'renyi_inf': (r'$\log\max_{s,a} d^\pi/d^\mu$', 'Rényi-∞ (log C)'),
}


def plot_coverage_curves(histories_by_teacher, baseline_history_alpha_zero,
                         mode, out_dir, cell_info, ref):
    """One curve figure per coverage metric for reference `ref`.

    Reuses _generic_metric_figure so SEM bands, cell annotation, and the
    α=0 baseline overlay match the other diagnostics. Returns written paths.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for metric, (ylabel, frag) in _COVERAGE_METRICS.items():
        field = f'cov_{ref}_{metric}'
        fig = _generic_metric_figure(
            histories_by_teacher,
            baseline_history_alpha_zero,
            field=field,
            mode=mode,
            cell_info=cell_info,
            title_prefix=f'Coverage {frag} — {ref} reference',
            ylabel=ylabel,
            footer=(f'Reference μ = {ref} policy occupancy. '
                    r'Exact discounted occupancy; Laplace-smoothed, capped.'),
        )
        out_path = os.path.join(out_dir, f'cov_{ref}_{metric}.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
    return written


def plot_coverage_heatmaps(ref, grids, grid_size, out_path, cell_info,
                           student_label):
    """4-panel state-marginal heatmaps: d^μ(s), d^π(s), and both ratio grids.

    State marginals (sum over actions) reshaped to grid_size×grid_size; values
    shown on log10 scale (small floor for zeros).
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    def _marginal(arr):
        return arr.sum(axis=1).reshape(grid_size, grid_size)

    panels = [
        (r'$d^\mu(s)$', _marginal(grids['d_mu'])),
        (r'$d^\pi(s)$', _marginal(grids['d_pi'])),
        (r'$d^\mu/d^\pi$ (state)', _marginal(grids['ratio_mu_over_pi'])),
        (r'$d^\pi/d^\mu$ (state)', _marginal(grids['ratio_pi_over_mu'])),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (label, data) in zip(axes, panels):
        im = ax.imshow(np.log10(data + 1e-12), cmap='viridis')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    h_val = cell_info['horizon']
    fig.suptitle(
        f'Coverage heatmaps — {ref} reference vs {student_label} '
        f'(log10 scale)\n'
        f"dist={cell_info['distance']}, H={h_val} "
        f"({cell_info['horizon_type']}), B={cell_info['sample_budget']}, "
        rf"$\alpha={cell_info['alpha']}$",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_coverage_plots.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/visualization.py tests/test_coverage_plots.py
git commit -m "feat(viz): coverage divergence curves + state-marginal heatmaps"
```

---

### Task 6: `--track-coverage` path in `run_npg_diagnostics.py`

**Files:**
- Modify: `run_npg_diagnostics.py` (imports ~40-42; `_run_one` ~109-126; `main` arg + per-cell loop ~129-243)
- Test: `tests/test_run_npg_diagnostics_coverage.py`

**Interfaces:**
- Consumes: `coverage` grids in `run_experiment` results; `plot_coverage_curves`, `plot_coverage_heatmaps`.
- Produces: a CLI flag `--track-coverage`; an `_emit_coverage_plots(out_dir, mode, cell_info, histories_by_teacher, baseline_histories, grids_by_teacher, baseline_grids, top_tv, grid_size)` helper that writes the coverage figures for both references.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_npg_diagnostics_coverage.py
import os
import numpy as np
import run_npg_diagnostics as rnd

CELL = {'distance': 6, 'horizon': 8, 'horizon_type': 'small',
        'sample_budget': 12, 'alpha': 1.0}
METRICS = ['max_mu_over_pi', 'max_pi_over_mu', 'chi2_pi_mu',
           'kl_pi_mu', 'kl_mu_pi', 'tv', 'renyi_inf']


def _hist():
    out = []
    for tick in range(2):
        e = {'steps': (tick + 1) * 2}
        for ref in ('analytic', 'learned'):
            for m in METRICS:
                e[f'cov_{ref}_{m}'] = 1.0 + 0.1 * tick
        out.append(e)
    return out


def _grids():
    n_s, n_a = 25, 4
    g = lambda s: np.abs(np.random.default_rng(s).random((n_s, n_a)))
    return {ref: {'d_mu': g(0), 'd_pi': g(1),
                  'ratio_mu_over_pi': g(2), 'ratio_pi_over_mu': g(3)}
            for ref in ('analytic', 'learned')}


def test_emit_coverage_plots_writes_both_refs(tmp_path):
    hbt = {0: [_hist()], 3: [_hist()]}
    grids_by_teacher = {0: [_grids()], 3: [_grids()]}
    rnd._emit_coverage_plots(
        out_dir=str(tmp_path), mode='capability', cell_info=CELL,
        histories_by_teacher=hbt, baseline_histories=[_hist()],
        grids_by_teacher=grids_by_teacher, baseline_grids=[_grids()],
        top_tv=3, grid_size=5,
    )
    for ref in ('analytic', 'learned'):
        for m in METRICS:
            assert os.path.exists(os.path.join(tmp_path, f'cov_{ref}_{m}.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_top.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_baseline.png'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_run_npg_diagnostics_coverage.py -v`
Expected: FAIL — `AttributeError: module 'run_npg_diagnostics' has no attribute '_emit_coverage_plots'`.

- [ ] **Step 3: Write minimal implementation**

Update the visualization import (lines 40-42) to add the coverage plotters:

```python
from tabular_prototype.visualization import (
    plot_pg_cosine, plot_pg_variance, plot_u_cosine,
    plot_coverage_curves, plot_coverage_heatmaps,
)
```

Add `import numpy as np` near the other imports (top of file, after `import os`).

Add a `track_coverage` kwarg to `_run_one` so its `run_experiment` call can request coverage. Replace the `kwargs = dict(...)` block (lines 113-120) by appending one line inside the dict:

```python
    kwargs = dict(
        grid_size=args.grid_size, goals=goals, lr=cell['lr'],
        horizon=cell['h_val'], sample_budget=cell['budget'],
        mode=training_mode, seed=seed,
        eval_interval=1, alpha=alpha,
        trajectories_per_update=(cell['tpu'] if args.override_tpu == 0 else args.override_tpu),
        pg_diag_enabled=True,
        track_coverage=args.track_coverage,
    )
```

Add the emitter helper (place it above `main`, after `_run_one`):

```python
def _average_grids(list_of_coverage_grids, ref):
    """Mean d_pi / ratios and (identical) d_mu across seeds for one ref."""
    dicts = [g[ref] for g in list_of_coverage_grids if g is not None]
    keys = ('d_mu', 'd_pi', 'ratio_mu_over_pi', 'ratio_pi_over_mu')
    return {k: np.mean([d[k] for d in dicts], axis=0) for k in keys}


def _emit_coverage_plots(out_dir, mode, cell_info, histories_by_teacher,
                         baseline_histories, grids_by_teacher, baseline_grids,
                         top_tv, grid_size):
    """Write coverage curves (per ref) + heatmaps (top teacher & baseline)."""
    for ref in ('analytic', 'learned'):
        plot_coverage_curves(
            histories_by_teacher=histories_by_teacher,
            baseline_history_alpha_zero=baseline_histories,
            mode=mode, out_dir=out_dir, cell_info=cell_info, ref=ref,
        )
        plot_coverage_heatmaps(
            ref, _average_grids(grids_by_teacher[top_tv], ref), grid_size,
            os.path.join(out_dir, f'cov_{ref}_heatmap_top.png'),
            cell_info, student_label=f'top teacher (α=1)',
        )
        plot_coverage_heatmaps(
            ref, _average_grids(baseline_grids, ref), grid_size,
            os.path.join(out_dir, f'cov_{ref}_heatmap_baseline.png'),
            cell_info, student_label='α=0 (vanilla NPG)',
        )
```

Add the CLI flag in `main` (after the `--override-tpu` arg, ~line 149):

```python
    parser.add_argument(
        '--track-coverage', action='store_true',
        help='Compute and plot coverage / distribution-mismatch diagnostics '
             '(d^pi vs analytic & learned reference occupancies).',
    )
```

Collect grids alongside histories in the α=1 loop (inside the `for tv` / `for seed` loop, after `histories_by_teacher[tv].append(r['history'])` ~line 178), and init the holders before the loop (next to `histories_by_teacher` ~line 172):

```python
            grids_by_teacher: dict = defaultdict(list)
```

and inside the seed loop:

```python
                    grids_by_teacher[tv].append(r.get('coverage_grids'))
```

Collect baseline grids in the α=0 loop (after `baseline_histories.append(r['history'])` ~line 191) — init `baseline_grids = []` next to `baseline_histories = []`:

```python
                baseline_grids.append(r.get('coverage_grids'))
```

Finally, after the existing `if training_mode == 'sample': ... elif training_mode == 'exact': ...` plotting block (after line 240, before `print(f'Wrote {mode_out}/*.png')`), add:

```python
            if args.track_coverage:
                _emit_coverage_plots(
                    out_dir=mode_out, mode=sweep_mode, cell_info=cell_info,
                    histories_by_teacher=dict(histories_by_teacher),
                    baseline_histories=baseline_histories,
                    grids_by_teacher=dict(grids_by_teacher),
                    baseline_grids=baseline_grids,
                    top_tv=tvs_alpha1[-1], grid_size=args.grid_size,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_run_npg_diagnostics_coverage.py -v`
Expected: PASS.

- [ ] **Step 5: End-to-end smoke run (tiny, override budget)**

Run:
```bash
PYTHONPATH=. python run_npg_diagnostics.py \
  --n-seeds 1 --training-modes exact --grid-size 5 \
  --override-budget 6 --override-tpu 0 --track-coverage \
  --output-dir /private/tmp/claude-501/-Users-thangduong-Desktop-mid-expert-RL/f9e698f1-c0ec-46c2-8a74-1efad8ff6696/scratchpad/cov_smoke
```
Expected: terminates with `NPG_DIAGNOSTICS_DONE`; `cov_analytic_*.png`, `cov_learned_*.png`, and both heatmap PNGs exist under `.../cov_smoke/exact/zeta/` and `.../exact/capability/`.

- [ ] **Step 6: Commit**

```bash
git add run_npg_diagnostics.py tests/test_run_npg_diagnostics_coverage.py
git commit -m "feat(diagnostics): --track-coverage path emits curves + heatmaps"
```

---

## Notes / known trade-offs

- **Redundant reference training:** each `run_experiment(track_coverage=True)` call retrains the `learned` reference from scratch even when many calls share the same env (same seed, different teacher). Correct but not optimal; acceptable for the canonical single cell (small grid, ≤2000 cheap exact-NPG steps with early stop). Optimize later only if it dominates runtime (YAGNI).
- **Absorbing-state action split:** occupancy at absorbing states is split across actions by the policy (`d(s,a)=d(s)π(a|s)`), even though no action is truly taken there. Both μ and π use their own policies consistently; state-marginal heatmaps are unaffected, and the effect on `(s,a)` divergences is minor and smoothed. Documented, not corrected.
- **Heatmaps use state marginals** (sum over actions) for legibility; full per-`(s,a)` arrays remain cached in `coverage_grids` for any later recomputation.
```
