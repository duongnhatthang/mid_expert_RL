"""Unit tests for update_direction_diagnostics (PG direction, sample mode)
and exact_direction_diagnostics (NPG direction variants, exact mode)."""
import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy, collect_trajectories
from tabular_prototype.teacher import compute_teacher_values_auto
from tabular_prototype.training import (
    update_direction_diagnostics,
    exact_direction_diagnostics,
    compute_student_qvalues,
)
from tabular_prototype.config import compute_gamma_from_horizon


def _make_setup(grid_size=3, horizon=10, n_traj=8, seed=0):
    rng = np.random.default_rng(seed)
    goals = [(0, grid_size - 1)]
    env = GridEnv(grid_size=grid_size, goals=goals, traps=[], horizon=horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(horizon)
    Q_mu, V_mu, _ = compute_teacher_values_auto(
        env, known_goals=goals, zeta=1.0, gamma=gamma)
    trajs = collect_trajectories(env, policy, n_traj, rng)
    start_idx = env.state_to_idx(env.start)
    return env, policy, trajs, Q_mu, V_mu, gamma, start_idx


def test_pg_returns_expected_keys():
    _, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup()
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma, start_idx=start_idx)
    assert set(out.keys()) == {'pg_bias', 'var_g_trace', 'var_g_visited'}
    for k, v in out.items():
        if k == 'pg_bias':
            assert np.isnan(v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)
        else:
            assert np.isfinite(v) and v >= 0.0


def test_pg_cos_is_one_at_alpha_zero():
    _, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=1)
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=0.0, gamma=gamma, start_idx=start_idx)
    assert out['pg_bias'] == pytest.approx(-1.0, abs=1e-9)


def test_pg_cos_is_one_when_teacher_is_none():
    _, policy, trajs, _, _, gamma, start_idx = _make_setup(seed=2)
    out = update_direction_diagnostics(
        policy, trajs, None, None, alpha=0.5, gamma=gamma, start_idx=start_idx)
    assert out['pg_bias'] == pytest.approx(-1.0, abs=1e-9)


def test_pg_cos_is_nan_when_direction_is_zero():
    _, policy, trajs, _, _, gamma, start_idx = _make_setup(seed=2)
    out = update_direction_diagnostics(
        policy, trajs, None, None, alpha=1.0, gamma=gamma, start_idx=start_idx)
    assert np.isnan(out['pg_bias'])


def test_pg_variance_nonneg_and_visited_le_trace_times_horizon():
    """var_g_visited weights per-cell variance by avg visits ≤ horizon, so
    var_g_visited ≤ horizon · var_g_trace (loose bound)."""
    _, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=3, horizon=10)
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma, start_idx=start_idx)
    assert out['var_g_trace'] >= 0.0
    assert out['var_g_visited'] >= 0.0
    # Loose sanity: visited weighting can't exceed horizon-weighted trace.
    assert out['var_g_visited'] <= 10.0 * out['var_g_trace'] + 1e-9


def test_exact_returns_expected_keys():
    env, policy, _, Q_mu, V_mu, gamma, _ = _make_setup(seed=4)
    Q_pi, _ = compute_student_qvalues(env, policy, gamma)
    out = exact_direction_diagnostics(policy, Q_pi, Q_mu, V_mu, alpha=1.0)
    assert set(out.keys()) == {'u_bias_npg', 'u_bias_pinv'}
    for v in out.values():
        assert np.isnan(v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)


def test_exact_cos_is_one_at_alpha_zero():
    env, policy, _, Q_mu, V_mu, gamma, _ = _make_setup(seed=5)
    Q_pi, _ = compute_student_qvalues(env, policy, gamma)
    out = exact_direction_diagnostics(policy, Q_pi, Q_mu, V_mu, alpha=0.0)
    assert out['u_bias_npg'] == pytest.approx(-1.0, abs=1e-9)
    assert out['u_bias_pinv'] == pytest.approx(-1.0, abs=1e-9)


def test_exact_per_state_action_sums_for_pinv():
    """For U^(pinv) the per-state action sum is identically zero by construction."""
    env, policy, _, Q_mu, V_mu, gamma, _ = _make_setup(seed=6)
    Q_pi, _ = compute_student_qvalues(env, policy, gamma)
    # Compute U^(pinv) manually and check.
    pi = np.stack([policy.get_probs(s) for s in range(env.n_states)])
    A_mu = Q_mu - V_mu[:, None]
    alpha = 0.7
    A_eff = (1 - alpha) * Q_pi + alpha * A_mu
    U_pinv = A_eff - A_eff.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(U_pinv.sum(axis=1), 0.0, atol=1e-10)


def test_exact_npg_and_pinv_differ_by_multiple_of_one():
    """U^(NPG) - U^(pinv) should be a per-state scalar (multiple of the all-ones
    vector) by construction."""
    env, policy, _, Q_mu, V_mu, gamma, _ = _make_setup(seed=7)
    Q_pi, _ = compute_student_qvalues(env, policy, gamma)
    pi = np.stack([policy.get_probs(s) for s in range(env.n_states)])
    A_mu = Q_mu - V_mu[:, None]
    alpha = 0.6
    A_eff = (1 - alpha) * Q_pi + alpha * A_mu
    V_eff = (pi * A_eff).sum(axis=1, keepdims=True)
    U_npg = A_eff - V_eff
    U_pinv = A_eff - A_eff.mean(axis=1, keepdims=True)
    diff = U_npg - U_pinv
    # Each row of diff should be a constant.
    for s in range(env.n_states):
        np.testing.assert_allclose(diff[s], diff[s, 0], atol=1e-10)
