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
