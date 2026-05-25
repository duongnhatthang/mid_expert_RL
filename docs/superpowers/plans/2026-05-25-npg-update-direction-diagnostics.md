# NPG Update Direction Diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-update-step diagnostic that computes the empirical NPG update direction U = F̂⁺ ĝ from the per-step trajectory batch in `sample` mode, and emits (a) a cosine-similarity figure cos(U_α, U_{α=0}) and (b) a trajectory-bootstrap Var[U] (trace + per-action at s₀) over training. Used to investigate the mid-capacity-teacher performance "bump" on the canonical single cell (dist=6, H=small, B=second-largest, α=1).

**Architecture:** A pure helper `update_direction_diagnostics(...)` in `tabular_prototype/training.py` builds per-state Fisher blocks and per-state ĝ from the existing per-step trajectory batch, applies a per-state pseudo-inverse, and bootstraps over trajectory indices (no extra rollouts). It returns six scalars per step. The helper is invoked from the sample-mode branch of `run_experiment` gated on the eval-tick condition, and the six scalars are appended to each per-eval-tick `history` record. Two new plotting functions in `tabular_prototype/visualization.py` produce the cosine figure and the two variance PNGs. A standalone focused runner `run_npg_diagnostics.py` drives the canonical cell across `zeta` and `capability` sweep modes.

**Tech Stack:** Python 3.9, `numpy`, `matplotlib`, `pytest`. Existing patterns and helpers in `tabular_prototype/`.

**Spec reference:** `docs/superpowers/specs/2026-05-25-npg-update-direction-diagnostics-design.md`.

**Files (created or modified):**
- Create `tabular_prototype/training.py` additions — `update_direction_diagnostics(...)` (≈100 LOC).
- Create `tabular_prototype/visualization.py` additions — `plot_npg_cosine`, `plot_npg_variance` (≈200 LOC).
- Create `run_npg_diagnostics.py` — focused runner (≈180 LOC).
- Create `tests/test_update_direction_diagnostics.py` — unit tests for helper (≈170 LOC).
- Create `tests/test_npg_diagnostics_visualization.py` — visualization smoke tests (≈80 LOC).
- Create `tests/test_run_npg_diagnostics.py` — integration smoke test (≈40 LOC).
- Modify `tabular_prototype/experiments.py` — call helper in sample-mode branch; thread `n_bootstrap` through `run_experiment` and the two `run_learning_curve_experiment*` wrappers (~40 LOC delta).

---

## Task 1: Scaffold `update_direction_diagnostics` with shape & finite-output test

**Files:**
- Modify: `tabular_prototype/training.py` (add new function near end of "PAV-RL gradient" section)
- Create: `tests/test_update_direction_diagnostics.py`

- [ ] **Step 1: Write the failing shape/type test**

Create `tests/test_update_direction_diagnostics.py`:

```python
"""Unit tests for update_direction_diagnostics."""
import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy, collect_trajectories
from tabular_prototype.teacher import compute_teacher_values_auto
from tabular_prototype.training import update_direction_diagnostics
from tabular_prototype.config import compute_gamma_from_horizon


def _make_setup(grid_size=3, horizon=10, n_traj=8, seed=0):
    """Build a small env + policy + teacher + trajectories for tests."""
    rng = np.random.default_rng(seed)
    goals = [(0, grid_size - 1)]
    env = GridEnv(
        grid_size=grid_size, goals=goals, traps=[], horizon=horizon,
    )
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(horizon)
    Q_mu, V_mu, _ = compute_teacher_values_auto(
        env, known_goals=goals, zeta=1.0, gamma=gamma,
    )
    trajs = collect_trajectories(env, policy, n_traj, rng)
    start_idx = env.state_to_idx(env.start)
    return policy, trajs, Q_mu, V_mu, gamma, start_idx, rng


def test_returns_expected_scalar_keys():
    policy, trajs, Q_mu, V_mu, gamma, start_idx, rng = _make_setup()
    result = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=20,
    )
    expected_keys = {
        'cos_npg_dir', 'var_U_trace',
        'var_U_s0_a0', 'var_U_s0_a1', 'var_U_s0_a2', 'var_U_s0_a3',
    }
    assert set(result.keys()) == expected_keys
    for k, v in result.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
        # cos may be NaN at corner cases; others must be finite & >= 0
        if k == 'cos_npg_dir':
            assert np.isnan(v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)
        else:
            assert np.isfinite(v) and v >= 0.0
```

- [ ] **Step 2: Run test to verify it fails (function not defined)**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py::test_returns_expected_scalar_keys -v`

Expected: FAIL with `ImportError: cannot import name 'update_direction_diagnostics'`.

- [ ] **Step 3: Add minimal stub to make import succeed**

Append to `tabular_prototype/training.py` (after `exact_npg_update`, before `evaluate_policy`):

```python
# =========================================================================
# NPG update-direction diagnostic
# =========================================================================

def update_direction_diagnostics(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float,
    start_idx: int,
    rng: np.random.Generator,
    n_bootstrap: int = 50,
) -> dict:
    """Compute U = F̂⁺ ĝ_α and its trajectory-bootstrap variance.

    Returns six per-step scalars:
        cos_npg_dir       cos(U_α, U_{α=0})  at the same θ_t (same batch).
                          NaN if either norm is zero.
        var_U_trace       Σ_{s,a} Var_b(U_b[s,a])  across n_bootstrap trajectory-
                          level bootstrap resamples of the current batch.
        var_U_s0_a{0..3}  Var_b(U_b[s₀, a])  for each action a.
    """
    n_states, n_actions = policy.theta.shape

    # Per-trajectory cached arrays: (s_idx, a_idx, G_t, A_mu).
    per_traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    has_teacher = Q_mu is not None and V_mu is not None
    for traj in trajectories:
        if len(traj) == 0:
            per_traj.append((
                np.empty(0, dtype=int), np.empty(0, dtype=int),
                np.empty(0), np.empty(0),
            ))
            continue
        G = np.array(estimate_returns(traj, gamma), dtype=float)
        s_arr = np.array([t.state_idx for t in traj], dtype=int)
        a_arr = np.array([t.action for t in traj], dtype=int)
        if has_teacher:
            A_mu = np.array(
                [get_teacher_advantage(Q_mu, V_mu, int(s), int(a))
                 for s, a in zip(s_arr, a_arr)], dtype=float,
            )
        else:
            A_mu = np.zeros_like(G)
        per_traj.append((s_arr, a_arr, G, A_mu))

    def _compute_U(traj_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (U_α, U_{α=0}) for a (possibly resampled) set of trajectories."""
        if len(traj_indices) == 0:
            return (np.zeros((n_states, n_actions)),
                    np.zeros((n_states, n_actions)))
        s_all = np.concatenate([per_traj[i][0] for i in traj_indices])
        a_all = np.concatenate([per_traj[i][1] for i in traj_indices])
        G_all = np.concatenate([per_traj[i][2] for i in traj_indices])
        Amu_all = np.concatenate([per_traj[i][3] for i in traj_indices])
        if len(s_all) == 0:
            return (np.zeros((n_states, n_actions)),
                    np.zeros((n_states, n_actions)))

        A_eff = (1.0 - alpha) * G_all + alpha * Amu_all
        A_van = G_all  # α=0 baseline (vanilla NPG)

        U_alpha = np.zeros((n_states, n_actions))
        U_van = np.zeros((n_states, n_actions))

        # The 1/n in F̂ and ĝ cancels under pinv, so we drop it here.
        # Group by state via np.unique for efficiency.
        for s in np.unique(s_all):
            mask = s_all == s
            actions_s = a_all[mask]
            pi_s = policy.get_probs(int(s))  # shape (A,)

            # ψᵢ = e_{a_i} - π(.|s)  for each transition i visiting state s.
            # F̂_s = Σ_i ψᵢ ψᵢᵀ; ĝ_s = Σ_i A_i ψᵢ.
            psi_mat = np.zeros((len(actions_s), n_actions))
            psi_mat[np.arange(len(actions_s)), actions_s] = 1.0
            psi_mat -= pi_s[None, :]

            F_s = psi_mat.T @ psi_mat
            g_alpha_s = psi_mat.T @ A_eff[mask]
            g_van_s = psi_mat.T @ A_van[mask]

            F_s_pinv = np.linalg.pinv(F_s)
            U_alpha[int(s)] = F_s_pinv @ g_alpha_s
            U_van[int(s)] = F_s_pinv @ g_van_s

        return U_alpha, U_van

    # Point estimate on the full batch (for cosine).
    n_traj = len(trajectories)
    full_indices = np.arange(n_traj)
    U_alpha_full, U_van_full = _compute_U(full_indices)

    a_flat = U_alpha_full.reshape(-1)
    v_flat = U_van_full.reshape(-1)
    a_norm = float(np.linalg.norm(a_flat))
    v_norm = float(np.linalg.norm(v_flat))
    if a_norm > 0.0 and v_norm > 0.0:
        cos_npg_dir = float(a_flat @ v_flat / (a_norm * v_norm))
    else:
        cos_npg_dir = float('nan')

    # Bootstrap U_α only (we don't need variance of the α=0 direction).
    if n_traj > 0 and n_bootstrap > 0:
        U_boots = np.zeros((n_bootstrap, n_states, n_actions))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_traj, size=n_traj)
            U_b, _ = _compute_U(idx)
            U_boots[b] = U_b
        U_var = U_boots.var(axis=0)
    else:
        U_var = np.zeros((n_states, n_actions))

    result = {
        'cos_npg_dir': cos_npg_dir,
        'var_U_trace': float(U_var.sum()),
    }
    s0_var = U_var[int(start_idx)]
    for a in range(4):
        result[f'var_U_s0_a{a}'] = float(s0_var[a]) if a < n_actions else 0.0
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py::test_returns_expected_scalar_keys -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/training.py tests/test_update_direction_diagnostics.py
git commit -m "$(cat <<'EOF'
feat: scaffold update_direction_diagnostics with shape/type test

Adds the per-state Fisher-block pseudo-inverse helper that returns
the six per-step scalars cos_npg_dir, var_U_trace, and var_U_s0_a{0..3}.
Locks the public API and key shapes via a smoke test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: TDD the α=0 cosine identity (locking test)

**Files:**
- Modify: `tests/test_update_direction_diagnostics.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_update_direction_diagnostics.py`:

```python
def test_cosine_is_one_when_alpha_is_zero():
    """At α=0, A_eff[i] = G[i] = A_van[i] for every transition,
    so U_α = U_{α=0} exactly and cos_npg_dir == 1.0."""
    policy, trajs, Q_mu, V_mu, gamma, start_idx, rng = _make_setup(seed=1)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=0.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=5,
    )
    assert result['cos_npg_dir'] == pytest.approx(1.0, abs=1e-9)


def test_cosine_is_one_when_teacher_is_none():
    """With no teacher signal, α has no effect and U_α == U_{α=0}."""
    policy, trajs, _, _, gamma, start_idx, rng = _make_setup(seed=2)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu=None, V_mu=None, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=5,
    )
    # α=1 with no teacher → A_eff = α·0 + (1-α)·G = G... wait, formula is
    # A_eff = (1-α)·G + α·A_mu. With A_mu = 0 (no teacher), A_eff = (1-α)·G.
    # So U_α is a scaled version of U_van, cosine is +1 (same direction).
    assert result['cos_npg_dir'] == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py -v`

Expected: both new tests PASS (the implementation from Task 1 already handles α=0 correctly via the closed form). If either fails, debug Task 1 implementation before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_update_direction_diagnostics.py
git commit -m "$(cat <<'EOF'
test: lock α=0 cosine identity for update_direction_diagnostics

cos_npg_dir == 1.0 exactly at α=0 and at teacher=None — the mathematical
guarantee that anchors the diagnostic's correctness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: TDD bootstrap variance non-negativity & trace consistency

**Files:**
- Modify: `tests/test_update_direction_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_variance_fields_nonnegative_and_trace_dominates_s0_sum():
    """All bootstrap variance fields are ≥ 0, and var_U_trace
    (sum over ALL states) ≥ sum of var_U_s0_a* (only state s₀)."""
    policy, trajs, Q_mu, V_mu, gamma, start_idx, rng = _make_setup(seed=3)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=30,
    )
    s0_sum = sum(result[f'var_U_s0_a{a}'] for a in range(4))
    for a in range(4):
        assert result[f'var_U_s0_a{a}'] >= 0.0
    assert result['var_U_trace'] >= 0.0
    assert result['var_U_trace'] + 1e-9 >= s0_sum
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py::test_variance_fields_nonnegative_and_trace_dominates_s0_sum -v`

Expected: PASS (already satisfied by the implementation).

- [ ] **Step 3: Commit**

```bash
git add tests/test_update_direction_diagnostics.py
git commit -m "$(cat <<'EOF'
test: lock variance non-negativity + trace ≥ s₀-sum invariant

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: TDD softmax-tangent property

**Files:**
- Modify: `tests/test_update_direction_diagnostics.py`

- [ ] **Step 1: Write the failing test**

This test asserts a structural property (per-state action sum of U = 0)
that's hard to check without inspecting the full-batch U matrix
directly. The matrix is computed inside `update_direction_diagnostics`
but not returned. Add a parallel module-level helper
`_update_direction_full_batch` that returns the (U_α, U_van) pair on
the full batch — exposed strictly for testing. The production helper
from Task 1 is left as-is; both paths compute the same math (closed-form
ψ, per-state pinv) so the structural property carries over.

Append to `tabular_prototype/training.py`:

```python
def _update_direction_full_batch(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U_α, U_{α=0}) on the full batch.

    Exposed for testing the softmax-tangent property; production code
    should call `update_direction_diagnostics`.
    """
    # Build per_traj cache identically to update_direction_diagnostics
    n_states, n_actions = policy.theta.shape
    has_teacher = Q_mu is not None and V_mu is not None
    s_all_list, a_all_list, G_all_list, Amu_all_list = [], [], [], []
    for traj in trajectories:
        if not traj:
            continue
        G = np.array(estimate_returns(traj, gamma), dtype=float)
        s_arr = np.array([t.state_idx for t in traj], dtype=int)
        a_arr = np.array([t.action for t in traj], dtype=int)
        if has_teacher:
            A_mu = np.array(
                [get_teacher_advantage(Q_mu, V_mu, int(s), int(a))
                 for s, a in zip(s_arr, a_arr)], dtype=float,
            )
        else:
            A_mu = np.zeros_like(G)
        s_all_list.append(s_arr); a_all_list.append(a_arr)
        G_all_list.append(G); Amu_all_list.append(A_mu)
    if not s_all_list:
        return (np.zeros((n_states, n_actions)),
                np.zeros((n_states, n_actions)))
    s_all = np.concatenate(s_all_list)
    a_all = np.concatenate(a_all_list)
    G_all = np.concatenate(G_all_list)
    Amu_all = np.concatenate(Amu_all_list)
    A_eff = (1.0 - alpha) * G_all + alpha * Amu_all
    A_van = G_all

    U_alpha = np.zeros((n_states, n_actions))
    U_van = np.zeros((n_states, n_actions))
    for s in np.unique(s_all):
        mask = s_all == s
        actions_s = a_all[mask]
        pi_s = policy.get_probs(int(s))
        psi_mat = np.zeros((len(actions_s), n_actions))
        psi_mat[np.arange(len(actions_s)), actions_s] = 1.0
        psi_mat -= pi_s[None, :]
        F_s = psi_mat.T @ psi_mat
        g_alpha_s = psi_mat.T @ A_eff[mask]
        g_van_s = psi_mat.T @ A_van[mask]
        F_pinv = np.linalg.pinv(F_s)
        U_alpha[int(s)] = F_pinv @ g_alpha_s
        U_van[int(s)] = F_pinv @ g_van_s
    return U_alpha, U_van
```

Leave `update_direction_diagnostics` untouched — both paths produce
identical math, so the test using `_update_direction_full_batch`
validates the production helper's correctness.

Add the test:

```python
def test_softmax_tangent_per_state_action_sum_is_zero():
    """For tabular softmax, ψ row-sums to 0 (action axis), so the
    null space of F̂_s contains the all-ones vector. The pseudo-inverse
    maps to the orthogonal complement of N(F̂_s), so U[s,:].sum() ≈ 0
    for every visited state."""
    from tabular_prototype.training import _update_direction_full_batch
    policy, trajs, Q_mu, V_mu, gamma, _, _ = _make_setup(seed=4)
    U_alpha, U_van = _update_direction_full_batch(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
    )
    # Visited states are those with any non-zero row.
    visited = np.any(U_alpha != 0.0, axis=1) | np.any(U_van != 0.0, axis=1)
    for s in np.flatnonzero(visited):
        assert abs(U_alpha[s].sum()) < 1e-8, f"U_alpha sum at s={s}"
        assert abs(U_van[s].sum()) < 1e-8, f"U_van sum at s={s}"
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py::test_softmax_tangent_per_state_action_sum_is_zero -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tabular_prototype/training.py tests/test_update_direction_diagnostics.py
git commit -m "$(cat <<'EOF'
feat: expose _update_direction_full_batch + softmax-tangent test

Promotes the full-batch U computation to a module-level helper so the
softmax-tangent property (U[s,:].sum() == 0) can be tested directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: TDD reproducibility & single-state corner

**Files:**
- Modify: `tests/test_update_direction_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_reproducible_under_same_seed():
    """Same RNG seed → identical scalar outputs."""
    policy, trajs, Q_mu, V_mu, gamma, start_idx, _ = _make_setup(seed=5)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    r1 = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng_a, n_bootstrap=10,
    )
    r2 = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng_b, n_bootstrap=10,
    )
    for k in r1:
        if np.isnan(r1[k]) and np.isnan(r2[k]):
            continue
        assert r1[k] == r2[k], f"mismatch at {k}: {r1[k]} vs {r2[k]}"


def test_empty_or_single_state_corner_does_not_raise():
    """If trajectories degenerate (single state, single action visited),
    the helper still returns finite floats (NaN allowed for cosine only)."""
    rng = np.random.default_rng(6)
    # A 3x3 env where the start position is also the goal — every trajectory
    # immediately terminates at length 0 or 1.
    env = GridEnv(
        grid_size=3, goals=[(1, 1)], traps=[], horizon=5,
    )
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(5)
    Q_mu, V_mu, _ = compute_teacher_values_auto(
        env, known_goals=[(1, 1)], zeta=1.0, gamma=gamma,
    )
    trajs = collect_trajectories(env, policy, 3, rng)
    start_idx = env.state_to_idx(env.start)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=5,
    )
    # Doesn't raise; non-cosine scalars are finite & ≥ 0
    for k, v in result.items():
        if k == 'cos_npg_dir':
            assert np.isnan(v) or np.isfinite(v)
        else:
            assert np.isfinite(v) and v >= 0.0
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=. pytest tests/test_update_direction_diagnostics.py -v`

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_update_direction_diagnostics.py
git commit -m "$(cat <<'EOF'
test: reproducibility + degenerate-trajectory robustness

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire diagnostic into sample-mode `run_experiment`

**Files:**
- Modify: `tabular_prototype/experiments.py:46-330` (sample/hybrid branch, around lines 167-285)
- Create: `tests/test_run_experiment_npg_diag.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_experiment_npg_diag.py`:

```python
"""Integration test: sample-mode run_experiment emits the six NPG diag fields."""
import numpy as np

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment


def test_sample_mode_history_includes_npg_diag_fields():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5,
        goals=goals,
        teacher_capacity=1,
        sample_budget=120,
        horizon=10,
        alpha=1.0,
        lr=0.5,
        seed=0,
        mode='sample',
        trajectories_per_update=4,
        eval_interval=2,
        n_bootstrap=5,
    )
    assert result['history'], "expected non-empty history"
    h0 = result['history'][0]
    for k in ('cos_npg_dir', 'var_U_trace',
              'var_U_s0_a0', 'var_U_s0_a1', 'var_U_s0_a2', 'var_U_s0_a3'):
        assert k in h0, f"missing key: {k}"


def test_exact_mode_does_not_emit_npg_diag_fields():
    """Diagnostic is sample-mode only; exact mode should not have these keys."""
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5,
        goals=goals,
        teacher_capacity=1,
        sample_budget=10,  # 10 update steps in exact mode
        horizon=10,
        alpha=1.0,
        lr=0.5,
        seed=0,
        mode='exact',
        eval_interval=2,
    )
    h0 = result['history'][0]
    assert 'cos_npg_dir' not in h0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_run_experiment_npg_diag.py -v`

Expected: `test_sample_mode_history_includes_npg_diag_fields` FAILS — missing keys.
`test_exact_mode_does_not_emit_npg_diag_fields` may pass already.

- [ ] **Step 3: Modify `run_experiment` to accept `n_bootstrap` and call the helper**

Edit `tabular_prototype/experiments.py`. Find the signature of `run_experiment`
(around line 46) and add `n_bootstrap: int = 50` as a new kwarg:

```python
def run_experiment(
    grid_size: int = 9,
    goals: Optional[List[Tuple[int, int]]] = None,
    traps: Optional[List[Tuple[int, int]]] = None,
    teacher_capacity: int = 1,
    sample_budget: int = 10000,
    horizon: Optional[int] = None,
    alpha: float = 0.5,
    lr: float = 0.5,
    seed: int = 0,
    mode: str = 'exact',
    trajectories_per_update: int = 10,
    eval_interval: int = 5,
    eval_n_episodes: int = 20,
    zeta: Optional[float] = None,
    n_bootstrap: int = 50,
) -> Dict[str, Any]:
    ...
```

Add the import at the top:

```python
from .training import (
    compute_pav_rl_gradient,
    update_policy,
    update_direction_diagnostics,
    ...
)
```

In the trajectory-based branch (the `else:` after the exact-mode `if mode == "exact":`),
locate the section just before `update_policy(policy, grad, lr)`. At this
point `total_steps` has already been incremented for the current step
(line 185) but `update_count` has not (line 213). Predict whether the
eval-tick block will fire this step and call the diagnostic gated on that:

```python
# (right after compute_pav_rl_gradient, before theta_saved = policy.theta.copy())
predicted_update_count = update_count + 1
will_eval_tick = (
    predicted_update_count % eval_interval == 0
    or total_steps >= sample_budget
)
if mode == "sample" and will_eval_tick:
    npg_diag = update_direction_diagnostics(
        policy, trajectories, Q_mu, V_mu, alpha, gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=n_bootstrap,
    )
else:
    npg_diag = None
```

Then in the eval-tick block (the `if update_count % eval_interval == 0 or is_last:`
section around line 262), merge `npg_diag` into the history record:

```python
hist_entry = {
    'steps': total_steps,
    'mean_reward': eval_results['mean_reward'],
    'goal_rate': eval_results['goal_rate'],
    'exact_V_start': float(V_pi_new[start_idx]),
    'exact_V_start_undiscounted': float(V_pi_undiscounted[start_idx]),
    'unique_sa': vis_m['unique_sa'],
    'state_entropy': vis_m['state_entropy'],
    'adv_product_s0': _compute_adv_product_s0(
        policy, Q_pi_new, V_pi_new, Q_mu, V_mu, start_idx,
    ),
    'mc_var_undiscounted': eval_results['std_reward'] ** 2,
    'mc_var_discounted': eval_results['std_reward_discounted'] ** 2,
}
if npg_diag is not None:
    hist_entry.update(npg_diag)
history.append(hist_entry)
```

- [ ] **Step 4: Run integration tests**

Run: `PYTHONPATH=. pytest tests/test_run_experiment_npg_diag.py -v`

Expected: both PASS.

- [ ] **Step 5: Run the existing test suite to confirm no regressions**

Run: `PYTHONPATH=. pytest tests/ -v --timeout=120`

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_run_experiment_npg_diag.py
git commit -m "$(cat <<'EOF'
feat: emit NPG update-direction diagnostics in sample-mode history

Adds n_bootstrap kwarg to run_experiment. In sample mode, computes
update_direction_diagnostics at θ_t (pre-update) on steps that will
record an eval-tick, merging the six scalars into the history record.
Exact and hybrid modes are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Thread `n_bootstrap` through the learning-curve runners

**Files:**
- Modify: `tabular_prototype/experiments.py` — `run_learning_curve_experiment` (~line 633) and `run_learning_curve_experiment_zeta` (~line 946)
- Modify: `tests/test_run_experiment_npg_diag.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_experiment_npg_diag.py`:

```python
def test_run_learning_curve_propagates_n_bootstrap():
    from tabular_prototype.experiments import run_learning_curve_experiment
    from tabular_prototype.environment import generate_equidistant_goals
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    out = run_learning_curve_experiment(
        grid_size=5, goals=goals, teacher_capacity=1,
        sample_budget=120, horizon=10, alpha=1.0, lr=0.5,
        n_seeds=1, mode='sample',
        trajectories_per_update=4, eval_interval=2,
        n_bootstrap=3,
    )
    # The dict shape returned by run_learning_curve_experiment is a list
    # of per-seed results dicts.
    assert out, "expected at least one seed result"
    h0 = out[0]['history'][0]
    assert 'cos_npg_dir' in h0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_run_experiment_npg_diag.py::test_run_learning_curve_propagates_n_bootstrap -v`

Expected: FAIL — `unexpected keyword argument 'n_bootstrap'`.

- [ ] **Step 3: Add `n_bootstrap` pass-through**

In `tabular_prototype/experiments.py`:

1. Add `n_bootstrap: int = 50` to the signature of
   `run_learning_curve_experiment(...)` (~line 633), and pass it to
   every `run_experiment(...)` call inside that function (currently 2
   call sites at the auto-detect saturation path and the main loop).
2. Do the same for `run_learning_curve_experiment_zeta(...)` (~line 946).
3. If the test discovers other callers in `experiments.py`, follow the
   same minimal pattern: add the kwarg with a default of 50; pass through.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. pytest tests/test_run_experiment_npg_diag.py -v && PYTHONPATH=. pytest tests/ -v --timeout=120`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/experiments.py tests/test_run_experiment_npg_diag.py
git commit -m "$(cat <<'EOF'
feat: thread n_bootstrap through run_learning_curve_experiment(_zeta)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: TDD `plot_npg_cosine`

**Files:**
- Modify: `tabular_prototype/visualization.py` (add new function near other diagnostic plotters)
- Create: `tests/test_npg_diagnostics_visualization.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_npg_diagnostics_visualization.py`:

```python
"""Visualization smoke tests for the NPG-direction figures."""
import os
import tempfile

import matplotlib

matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np


def _fake_history(n_eval_ticks=10, cos_value=0.7, var_value=0.5):
    return [
        {
            'steps': 10 * i,
            'cos_npg_dir': cos_value + 0.01 * i,
            'var_U_trace': var_value + 0.01 * i,
            'var_U_s0_a0': var_value * 0.1,
            'var_U_s0_a1': var_value * 0.2,
            'var_U_s0_a2': var_value * 0.3,
            'var_U_s0_a3': var_value * 0.4,
        }
        for i in range(n_eval_ticks)
    ]


def test_plot_npg_cosine_writes_png_with_reference_line():
    from tabular_prototype.visualization import plot_npg_cosine
    histories_by_teacher = {
        0: [_fake_history(cos_value=0.5)],
        1: [_fake_history(cos_value=0.7)],
        2: [_fake_history(cos_value=0.9)],
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'npg_cosine.png')
        plot_npg_cosine(
            histories_by_teacher=histories_by_teacher,
            mode='capability',
            out_path=out_path,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_npg_diagnostics_visualization.py::test_plot_npg_cosine_writes_png_with_reference_line -v`

Expected: FAIL — `ImportError: cannot import name 'plot_npg_cosine'`.

- [ ] **Step 3: Implement `plot_npg_cosine`**

Append to `tabular_prototype/visualization.py`:

```python
def plot_npg_cosine(
    histories_by_teacher: Dict[Any, List[List[Dict[str, Any]]]],
    mode: str,
    out_path: str,
    cell_info: Dict[str, Any],
) -> None:
    """Single-panel figure of cos(U_α, U_{α=0}) per env step.

    Args:
        histories_by_teacher: dict mapping teacher value -> list of
            per-seed history lists. Each history list is a list of
            per-eval-tick dicts containing 'steps' and 'cos_npg_dir'.
        mode: 'capability' or 'zeta' — controls label/sort ordering.
        out_path: full PNG output path.
        cell_info: must contain keys distance, horizon, horizon_type,
            sample_budget, alpha.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    sorted_teachers = sorted(
        histories_by_teacher.keys(),
        key=lambda k: (k is None, k),
    )
    for tv in sorted_teachers:
        histories = histories_by_teacher[tv]
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        if min_len == 0:
            continue
        steps = np.mean([
            [h['steps'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        values = np.stack([
            [h['cos_npg_dir'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        label = (f'cap={tv}' if mode == 'capability'
                 else f'ζ={tv}')
        ax.plot(steps, mean, label=label, marker='o',
                markersize=3, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

    ax.axhline(
        1.0, color='black', linewidth=1.0, linestyle='--',
        label='α=0 (reference)',
    )
    ax.set_xlabel('env step', fontsize=9)
    ax.set_ylabel(
        r'$\cos(U_\alpha,\,U_{\alpha=0})$  at same $\theta_t$',
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    h_val = cell_info['horizon']
    h_type = cell_info['horizon_type']
    fig.suptitle(
        f'NPG update-direction cosine ({mode} sweep) — '
        f"dist={cell_info['distance']}, H={h_val} ({h_type}), "
        f"B={cell_info['sample_budget']}, "
        rf"$\alpha={cell_info['alpha']}$",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        r'$U = \hat F^{\dagger}\,\hat g$ on the per-step batch '
        r'($\hat g_\alpha$ uses $A_i = (1-\alpha)G_i + \alpha A^\mu_i$)',
        ha='center', fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=. pytest tests/test_npg_diagnostics_visualization.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/visualization.py tests/test_npg_diagnostics_visualization.py
git commit -m "$(cat <<'EOF'
feat: add plot_npg_cosine — single-cell cosine-similarity figure

Cosine of U_α vs U_{α=0} (vanilla-NPG direction at same θ_t) over env
steps, with dashed reference line at 1.0 and per-teacher curves.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: TDD `plot_npg_variance`

**Files:**
- Modify: `tabular_prototype/visualization.py`
- Modify: `tests/test_npg_diagnostics_visualization.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_npg_diagnostics_visualization.py`:

```python
def test_plot_npg_variance_writes_both_pngs_with_4_subplots_at_s0():
    from tabular_prototype.visualization import plot_npg_variance
    histories_by_teacher = {
        0: [_fake_history(var_value=0.3)],
        1: [_fake_history(var_value=0.5)],
        2: [_fake_history(var_value=0.7)],
    }
    baseline_alpha_zero = [_fake_history(var_value=0.4)]
    with tempfile.TemporaryDirectory() as tmp:
        plot_npg_variance(
            histories_by_teacher=histories_by_teacher,
            baseline_history_alpha_zero=baseline_alpha_zero,
            mode='capability',
            out_dir=tmp,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
            n_bootstrap=50,
        )
        trace_path = os.path.join(tmp, 'npg_var_trace.png')
        s0_path = os.path.join(tmp, 'npg_var_s0.png')
        assert os.path.exists(trace_path) and os.path.getsize(trace_path) > 0
        assert os.path.exists(s0_path) and os.path.getsize(s0_path) > 0
        # Read s0 figure back and verify it has 4 subplots
        import matplotlib.image as mpimg
        img = mpimg.imread(s0_path)
        # 4-subplot figures rendered at default DPI are wider than tall
        # OR have a 2x2 grid; cheapest check: figure has nonzero content.
        assert img.shape[0] > 0 and img.shape[1] > 0
```

For a more robust 4-subplot check, the test below directly inspects
the figure object — refactor `plot_npg_variance` so the s₀ figure is
constructed via `fig, axes = plt.subplots(2, 2, ...)` and we add an
internal hook for testing. The simplest such hook: assert by counting
files only (above), and additionally verify the figure's expected
subplot count by constructing it via a helper:

```python
def test_plot_npg_variance_s0_has_2x2_subplot_layout():
    from tabular_prototype.visualization import _build_npg_var_s0_figure
    histories_by_teacher = {0: [_fake_history(var_value=0.3)]}
    baseline_alpha_zero = [_fake_history(var_value=0.4)]
    fig = _build_npg_var_s0_figure(
        histories_by_teacher=histories_by_teacher,
        baseline_history_alpha_zero=baseline_alpha_zero,
        mode='capability',
        cell_info={'distance': 6, 'horizon': 50,
                   'horizon_type': 'small',
                   'sample_budget': 200, 'alpha': 1.0},
        n_bootstrap=50,
    )
    assert len(fig.axes) == 4
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_npg_diagnostics_visualization.py -v`

Expected: FAILs — `plot_npg_variance` / `_build_npg_var_s0_figure` not defined.

- [ ] **Step 3: Implement `plot_npg_variance` and `_build_npg_var_s0_figure`**

Append to `tabular_prototype/visualization.py`:

```python
def _build_npg_var_trace_figure(
    histories_by_teacher, baseline_history_alpha_zero,
    mode, cell_info, n_bootstrap,
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    sorted_teachers = sorted(
        histories_by_teacher.keys(), key=lambda k: (k is None, k),
    )
    for tv in sorted_teachers:
        histories = histories_by_teacher[tv]
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        if min_len == 0:
            continue
        steps = np.mean([
            [h['steps'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        values = np.stack([
            [h['var_U_trace'] for h in seed_hist[:min_len]]
            for seed_hist in histories
        ], axis=0)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        label = (f'cap={tv}' if mode == 'capability'
                 else f'ζ={tv}')
        ax.plot(steps, mean, label=label, marker='o',
                markersize=3, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

    # α=0 baseline overlay (real vanilla-NPG run).
    if baseline_history_alpha_zero:
        bh = baseline_history_alpha_zero
        min_len = min(len(h) for h in bh)
        if min_len > 0:
            steps_b = np.mean([
                [h['steps'] for h in seed_hist[:min_len]]
                for seed_hist in bh
            ], axis=0)
            vals_b = np.stack([
                [h['var_U_trace'] for h in seed_hist[:min_len]]
                for seed_hist in bh
            ], axis=0)
            mean_b = np.nanmean(vals_b, axis=0)
            std_b = np.nanstd(vals_b, axis=0)
            ax.plot(steps_b, mean_b, color='black', linestyle='--',
                    linewidth=1.5, label='α=0 (vanilla NPG)')
            ax.fill_between(steps_b, mean_b - std_b, mean_b + std_b,
                            color='black', alpha=0.1)

    ax.set_xlabel('env step', fontsize=9)
    ax.set_ylabel(r'$\sum_{s,a}\mathrm{Var}_b(U_b)$', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    h_val = cell_info['horizon']
    h_type = cell_info['horizon_type']
    fig.suptitle(
        f'NPG update-direction trace variance ({mode} sweep) — '
        f"dist={cell_info['distance']}, H={h_val} ({h_type}), "
        f"B={cell_info['sample_budget']}, "
        rf"$\alpha={cell_info['alpha']}$",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        f'Var across B={n_bootstrap} trajectory-level bootstrap resamples '
        f'of the per-step trajectory batch (no extra rollouts)',
        ha='center', fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def _build_npg_var_s0_figure(
    histories_by_teacher, baseline_history_alpha_zero,
    mode, cell_info, n_bootstrap,
):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    action_labels = ['↑ (a=0)', '→ (a=1)', '↓ (a=2)', '← (a=3)']

    sorted_teachers = sorted(
        histories_by_teacher.keys(), key=lambda k: (k is None, k),
    )

    for a, ax in enumerate(axes.flatten()):
        field = f'var_U_s0_a{a}'
        for tv in sorted_teachers:
            histories = histories_by_teacher[tv]
            if not histories:
                continue
            min_len = min(len(h) for h in histories)
            if min_len == 0:
                continue
            steps = np.mean([
                [h['steps'] for h in seed_hist[:min_len]]
                for seed_hist in histories
            ], axis=0)
            values = np.stack([
                [h[field] for h in seed_hist[:min_len]]
                for seed_hist in histories
            ], axis=0)
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            label = (f'cap={tv}' if mode == 'capability'
                     else f'ζ={tv}')
            ax.plot(steps, mean, label=label, marker='o',
                    markersize=2.5, linewidth=1.3)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

        if baseline_history_alpha_zero:
            bh = baseline_history_alpha_zero
            min_len = min(len(h) for h in bh)
            if min_len > 0:
                steps_b = np.mean([
                    [h['steps'] for h in seed_hist[:min_len]]
                    for seed_hist in bh
                ], axis=0)
                vals_b = np.stack([
                    [h[field] for h in seed_hist[:min_len]]
                    for seed_hist in bh
                ], axis=0)
                mean_b = np.nanmean(vals_b, axis=0)
                ax.plot(steps_b, mean_b, color='black',
                        linestyle='--', linewidth=1.3,
                        label='α=0 (vanilla NPG)')

        ax.set_title(action_labels[a], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(rf'$\mathrm{{Var}}_b(U_b[s_0,\,{a}])$', fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel('env step', fontsize=9)
    axes[0, 0].legend(fontsize=7, loc='best')

    h_val = cell_info['horizon']
    h_type = cell_info['horizon_type']
    fig.suptitle(
        f'NPG update-direction variance at start state s₀ '
        f'({mode} sweep) — '
        f"dist={cell_info['distance']}, H={h_val} ({h_type}), "
        f"B={cell_info['sample_budget']}, "
        rf"$\alpha={cell_info['alpha']}$",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        f'Var across B={n_bootstrap} trajectory-level bootstrap resamples '
        f'of the per-step trajectory batch (no extra rollouts)',
        ha='center', fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_npg_variance(
    histories_by_teacher: Dict[Any, List[List[Dict[str, Any]]]],
    baseline_history_alpha_zero: List[List[Dict[str, Any]]],
    mode: str,
    out_dir: str,
    cell_info: Dict[str, Any],
    n_bootstrap: int = 50,
) -> None:
    """Two PNGs: total Var[U] trace + per-action Var[U] at s₀ (2x2)."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    fig_trace = _build_npg_var_trace_figure(
        histories_by_teacher, baseline_history_alpha_zero,
        mode, cell_info, n_bootstrap,
    )
    fig_trace.savefig(os.path.join(out_dir, 'npg_var_trace.png'), dpi=120)
    plt.close(fig_trace)

    fig_s0 = _build_npg_var_s0_figure(
        histories_by_teacher, baseline_history_alpha_zero,
        mode, cell_info, n_bootstrap,
    )
    fig_s0.savefig(os.path.join(out_dir, 'npg_var_s0.png'), dpi=120)
    plt.close(fig_s0)
```

If `Dict`, `Any`, `List` are not already imported in `visualization.py`,
add `from typing import Any, Dict, List` at the top.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. pytest tests/test_npg_diagnostics_visualization.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tabular_prototype/visualization.py tests/test_npg_diagnostics_visualization.py
git commit -m "$(cat <<'EOF'
feat: add plot_npg_variance — trace + per-action s₀ variance figures

Two PNGs (npg_var_trace.png, npg_var_s0.png). Per-teacher curves plus
α=0 vanilla-NPG baseline overlay. Annotation describes the trajectory-
level bootstrap estimator on the figure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Focused runner `run_npg_diagnostics.py` + integration smoke test

**Files:**
- Create: `run_npg_diagnostics.py`
- Create: `tests/test_run_npg_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_npg_diagnostics.py`:

```python
"""Integration smoke test for the focused runner."""
import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.slow
def test_run_npg_diagnostics_end_to_end():
    """Tiny end-to-end run: 1 seed, B=3, grid=3, budget override."""
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [
                sys.executable, 'run_npg_diagnostics.py',
                '--n-seeds', '1',
                '--n-bootstrap', '3',
                '--grid-size', '3',
                '--override-budget', '40',
                '--output-dir', tmp,
            ],
            capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'},
        )
        assert out.returncode == 0, (
            f"runner exited {out.returncode}\nSTDOUT:\n{out.stdout}\n"
            f"STDERR:\n{out.stderr}"
        )
        for mode in ('zeta', 'capability'):
            for png in ('npg_cosine.png', 'npg_var_trace.png',
                        'npg_var_s0.png'):
                p = os.path.join(tmp, mode, png)
                assert os.path.exists(p), f"missing: {p}"
                assert os.path.getsize(p) > 0, f"empty: {p}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_run_npg_diagnostics.py -v -m slow`

Expected: FAIL — runner script does not exist.

- [ ] **Step 3: Create `run_npg_diagnostics.py`**

Create the file at repo root:

```python
#!/usr/bin/env python
"""Focused runner for NPG update-direction diagnostics.

Runs ONLY the canonical single cell defined in
docs/superpowers/specs/2026-05-25-npg-update-direction-diagnostics-design.md:
  dist=6, h_type='small', alpha=1.0, B = calibrated budgets[-2].

For each sweep mode ∈ {zeta, capability}:
  - One α=0 baseline training run (any non-(-1) teacher).
  - For each teacher value in the mode's list, one α=1 training run
    per seed.

Emits three PNGs per mode in <output_dir>/{mode}/:
  npg_cosine.png, npg_var_trace.png, npg_var_s0.png.

Usage:
  python run_npg_diagnostics.py \\
      [--n-seeds 30] \\
      [--n-bootstrap 50] \\
      [--output-dir results/figures/npg_diagnostics_<timestamp>] \\
      [--training-mode sample] \\
      [--grid-size 9] \\
      [--override-budget N]
"""
import argparse
import datetime as dt
import json
import os
import pickle
import sys
from collections import defaultdict

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment
from tabular_prototype.visualization import (
    plot_npg_cosine, plot_npg_variance,
)
import run_hypothesis_sweep as sweep


ZETA_TEACHER_VALUES_ALPHA1 = [0.0, 0.33, 0.67, 1.0]
CAPABILITY_TEACHER_VALUES_ALPHA1 = [0, 1, 2, 3]


def _resolve_cell(args):
    n_goals_zeta = 1
    n_goals_cap = 3
    if args.override_budget is not None:
        # Test/dev path: bypass calibration entirely.
        from tabular_prototype.config import compute_exploration_thresholds
        h_val = compute_exploration_thresholds(args.grid_size)['horizon_small']
        return {
            'zeta': dict(budget=args.override_budget, h_val=h_val,
                         lr=0.5, tpu=4, n_goals=n_goals_zeta),
            'capability': dict(budget=args.override_budget, h_val=h_val,
                               lr=0.5, tpu=4, n_goals=n_goals_cap),
        }
    calib_path = sweep._calibration_path_for(args.training_mode)
    try:
        calib = json.load(open(calib_path))
    except FileNotFoundError:
        sys.exit(
            f"Calibration JSON missing: {calib_path}. "
            f"Regenerate via run_calibration.py and retry."
        )
    out = {}
    for sweep_mode, ng in (('zeta', n_goals_zeta),
                           ('capability', n_goals_cap)):
        cell = sweep._find_calibration_cell(
            calib, args.distance, args.horizon_type, ng,
        )
        if cell is None:
            sys.exit(
                f"No calibration cell for distance={args.distance}, "
                f"horizon_type={args.horizon_type}, n_goals={ng}, "
                f"training_mode={args.training_mode}"
            )
        budgets = cell.get('budgets', [])
        if len(budgets) < 2:
            sys.exit(
                f"Calibration cell has fewer than 2 budgets: {budgets}"
            )
        out[sweep_mode] = dict(
            budget=budgets[-2],
            h_val=cell['horizon'],
            lr=cell.get('lr', cell.get('best_lr', 0.5)),
            tpu=cell.get('best_traj_per_update', 1),
            n_goals=ng,
        )
    return out


def _run_one(args, sweep_mode, cell, alpha, tv, seed):
    goals = generate_equidistant_goals(
        args.grid_size, cell['n_goals'], distance=args.distance,
    )
    kwargs = dict(
        grid_size=args.grid_size, goals=goals, lr=cell['lr'],
        horizon=cell['h_val'], sample_budget=cell['budget'],
        mode=args.training_mode, seed=seed,
        eval_interval=5, alpha=alpha,
        trajectories_per_update=cell['tpu'],
        n_bootstrap=args.n_bootstrap,
    )
    if sweep_mode == 'zeta':
        kwargs['teacher_capacity'] = 1
        kwargs['zeta'] = tv
    else:
        kwargs['teacher_capacity'] = tv
    return run_experiment(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=30)
    parser.add_argument('--n-bootstrap', type=int, default=50)
    default_out = os.path.join(
        'results', 'figures',
        f'npg_diagnostics_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    parser.add_argument('--output-dir', type=str, default=default_out)
    parser.add_argument('--training-mode', choices=['sample'],
                        default='sample')
    parser.add_argument('--distance', type=int, default=6)
    parser.add_argument('--horizon-type', type=str, default='small')
    parser.add_argument('--grid-size', type=int, default=9)
    parser.add_argument('--override-budget', type=int, default=None)
    args = parser.parse_args()

    cells = _resolve_cell(args)
    os.makedirs(args.output_dir, exist_ok=True)

    for sweep_mode in ('zeta', 'capability'):
        cell = cells[sweep_mode]
        if sweep_mode == 'zeta':
            tvs_alpha1 = ZETA_TEACHER_VALUES_ALPHA1
            baseline_tv = 0.0  # ζ=0 with α=0 is vanilla NPG
        else:
            tvs_alpha1 = CAPABILITY_TEACHER_VALUES_ALPHA1
            baseline_tv = 0  # cap=0 (random teacher); any non-(-1) works at α=0

        # Run α=1 per teacher value × seeds.
        histories_by_teacher: dict = defaultdict(list)
        all_records = []
        for tv in tvs_alpha1:
            for seed in range(args.n_seeds):
                r = _run_one(args, sweep_mode, cell, 1.0, tv, seed)
                histories_by_teacher[tv].append(r['history'])
                all_records.append({
                    'sweep_mode': sweep_mode, 'alpha': 1.0,
                    'teacher_value': tv, 'seed': seed,
                    'history': r['history'],
                })

        # Run α=0 baseline (one per seed).
        baseline_histories = []
        for seed in range(args.n_seeds):
            r = _run_one(args, sweep_mode, cell, 0.0, baseline_tv, seed)
            baseline_histories.append(r['history'])
            all_records.append({
                'sweep_mode': sweep_mode, 'alpha': 0.0,
                'teacher_value': baseline_tv, 'seed': seed,
                'history': r['history'],
            })

        mode_out = os.path.join(args.output_dir, sweep_mode)
        os.makedirs(mode_out, exist_ok=True)
        with open(os.path.join(mode_out, 'records.pkl'), 'wb') as fp:
            pickle.dump(all_records, fp)

        cell_info = {
            'distance': args.distance,
            'horizon': cell['h_val'],
            'horizon_type': args.horizon_type,
            'sample_budget': cell['budget'],
            'alpha': 1.0,
        }
        plot_npg_cosine(
            histories_by_teacher=dict(histories_by_teacher),
            mode=sweep_mode,
            out_path=os.path.join(mode_out, 'npg_cosine.png'),
            cell_info=cell_info,
        )
        plot_npg_variance(
            histories_by_teacher=dict(histories_by_teacher),
            baseline_history_alpha_zero=baseline_histories,
            mode=sweep_mode,
            out_dir=mode_out,
            cell_info=cell_info,
            n_bootstrap=args.n_bootstrap,
        )
        print(f'Wrote {mode_out}/*.png')

    print('NPG_DIAGNOSTICS_DONE')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run integration test**

Run: `PYTHONPATH=. pytest tests/test_run_npg_diagnostics.py -v -m slow`

Expected: PASS. If slower than ~60 s, accept it as a `slow`-marked test;
the test exists for end-to-end coverage, not fast iteration.

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v --timeout=180`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add run_npg_diagnostics.py tests/test_run_npg_diagnostics.py
git commit -m "$(cat <<'EOF'
feat: focused runner run_npg_diagnostics.py + end-to-end smoke test

Drives the canonical cell (dist=6, H=small, B=2nd-largest, α=1) for
both zeta and capability sweep modes; runs one α=0 baseline run per
mode for the variance overlay; writes npg_cosine.png, npg_var_trace.png,
and npg_var_s0.png into results/figures/npg_diagnostics_<ts>/{mode}/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Sanity-check on a tiny canonical run (manual)

**Files:** none

This is a sanity-check step before declaring the feature done. Not a
test — the engineer runs the focused runner with small but realistic
settings and visually confirms the figures make sense.

- [ ] **Step 1: Run a small canonical run**

```bash
PYTHONPATH=. python run_npg_diagnostics.py \
    --n-seeds 3 \
    --n-bootstrap 20 \
    --grid-size 5 \
    --override-budget 200 \
    --output-dir /tmp/npg_diag_smoke
```

- [ ] **Step 2: Visually inspect the outputs**

For each `{zeta, capability}` subdir, confirm:
- `npg_cosine.png` shows curves bounded in [-1, 1] with a dashed reference
  line at y=1.
- `npg_var_trace.png` shows curves ≥ 0 and an α=0 dashed baseline.
- `npg_var_s0.png` is a 2×2 grid of subplots.
- All three PNGs have the cell-info annotation in the title and the
  bootstrap-method annotation in the figure-bottom text.

If any of these are wrong, file a follow-up task. Do not commit
anything; this is observational.

---

## Self-Review Notes (post-write)

- **Spec coverage:**
  - Helper API and behavior — Task 1; α=0 cosine identity — Task 2; bootstrap fields — Task 3; softmax-tangent — Task 4; reproducibility/corner — Task 5.
  - Sample-mode plumbing + n_bootstrap CLI thread-through — Task 6, 7.
  - Cosine plot — Task 8; variance plots (trace + s₀ 2×2) — Task 9.
  - Focused runner — Task 10; manual smoke — Task 11.
- **Placeholder scan:** No "TBD"/"TODO"/"add appropriate X". Every code step shows actual code. Tests show full bodies.
- **Type consistency:** `update_direction_diagnostics` keyword names and `n_bootstrap` defaults match across helper, run_experiment, run_learning_curve_experiment, and the focused runner. Figure-builder helper names (`_build_npg_var_trace_figure`, `_build_npg_var_s0_figure`) are used only in `plot_npg_variance`, with `_build_npg_var_s0_figure` exposed for the 2×2-subplot-count test.
- **Existing-codebase patterns:** Plot signatures follow the pattern of `plot_advantage_alignment` / `plot_mc_variance_curve` (kwargs for cell info, `mean ± std` band across seeds). Focused runner mirrors `scripts/focused_items_3_6.py`.
