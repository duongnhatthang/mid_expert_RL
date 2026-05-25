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
    """With no teacher and 0 < α < 1, A_eff = (1-α)·G is a positive
    scalar multiple of A_van = G, so U_α and U_van point the same way
    and cos = +1. (α=1 with no teacher zeroes A_eff → cos is NaN, see
    separate test.)"""
    policy, trajs, _, _, gamma, start_idx, rng = _make_setup(seed=2)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu=None, V_mu=None, alpha=0.5, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=5,
    )
    assert result['cos_npg_dir'] == pytest.approx(1.0, abs=1e-9)


def test_cosine_is_nan_when_update_direction_is_zero():
    """α=1 with no teacher zeros out A_eff, so U_α = 0 and cosine is
    undefined — the helper must report NaN rather than crash."""
    policy, trajs, _, _, gamma, start_idx, rng = _make_setup(seed=2)
    result = update_direction_diagnostics(
        policy, trajs, Q_mu=None, V_mu=None, alpha=1.0, gamma=gamma,
        start_idx=start_idx, rng=rng, n_bootstrap=5,
    )
    assert np.isnan(result['cos_npg_dir'])


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
