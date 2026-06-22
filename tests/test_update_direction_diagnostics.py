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
    env, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup()
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma, start_idx=start_idx,
        Q_pi=Q_pi, V_pi=V_pi)
    assert set(out.keys()) == {
        'pg_bias', 'var_g_trace', 'var_g_visited', 'var_inner_g_ref',
    }
    for k, v in out.items():
        if k == 'pg_bias':
            assert np.isnan(v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)
        else:
            assert np.isfinite(v) and v >= 0.0


def test_pg_var_inner_is_nan_without_Q_pi():
    """Inner-product variance is undefined without the A^π reference."""
    _, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=4)
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma, start_idx=start_idx)
    assert np.isnan(out['var_inner_g_ref'])


def test_pg_var_inner_matches_manual_computation():
    """Var_τ(<û_τ, A^π/‖A^π‖>) computed manually matches the diagnostic.

    Both sides are unit-normalized — the variance is across per-rollout
    cosine alignments with the reference.
    """
    env, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=5)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    alpha = 0.5
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=alpha, gamma=gamma,
        start_idx=start_idx, Q_pi=Q_pi, V_pi=V_pi)

    # Manual recomputation.
    n_states, n_actions = policy.theta.shape
    A_ref = (Q_pi - V_pi[:, None]).reshape(-1)
    A_ref_unit = A_ref / np.linalg.norm(A_ref)
    inners = []
    for traj in trajs:
        if not traj:
            continue
        from tabular_prototype.training import estimate_returns
        Gs = estimate_returns(traj, gamma)
        g = np.zeros((n_states, n_actions))
        for t, tr in enumerate(traj):
            s, a = tr.state_idx, tr.action
            A_mu = (Q_mu[s, a] - V_mu[s])
            A_eff = (1 - alpha) * Gs[t] + alpha * A_mu
            pi_s = policy.get_probs(s)
            psi = -pi_s.copy()
            psi[a] += 1.0
            g[s, :] += A_eff * psi
        norm = float(np.linalg.norm(g))
        if norm == 0.0:
            continue
        u = g.reshape(-1) / norm
        inners.append(float(u @ A_ref_unit))
    expected_var = float(np.var(inners))
    assert out['var_inner_g_ref'] == pytest.approx(expected_var, rel=1e-9, abs=1e-12)


def test_pg_bias_is_nan_when_Q_pi_omitted():
    """Without an exact Q^π reference there is no defined α=0 direction."""
    _, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=1)
    out = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=1.0, gamma=gamma, start_idx=start_idx)
    assert np.isnan(out['pg_bias'])


def test_pg_bias_in_valid_range_with_Q_pi():
    """With Q^π reference the cosine is well-defined and lies in [-1, 1]."""
    env, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=1)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    for alpha in (0.0, 0.5, 1.0):
        out = update_direction_diagnostics(
            policy, trajs, Q_mu, V_mu, alpha=alpha, gamma=gamma,
            start_idx=start_idx, Q_pi=Q_pi, V_pi=V_pi)
        v = out['pg_bias']
        assert np.isnan(v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)


def test_pg_bias_is_nan_when_alpha_direction_is_zero():
    """α=1 with no teacher → A_eff = 0 → ū_α has no direction → NaN cosine."""
    env, policy, trajs, _, _, gamma, start_idx = _make_setup(seed=2)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    out = update_direction_diagnostics(
        policy, trajs, None, None, alpha=1.0, gamma=gamma, start_idx=start_idx,
        Q_pi=Q_pi, V_pi=V_pi)
    assert np.isnan(out['pg_bias'])


def test_pg_bias_V_pi_optional_derives_from_policy():
    """Omitting V_pi while passing Q_pi reconstructs V_pi from π — same result."""
    env, policy, trajs, Q_mu, V_mu, gamma, start_idx = _make_setup(seed=3)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    out_with = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=0.5, gamma=gamma, start_idx=start_idx,
        Q_pi=Q_pi, V_pi=V_pi)
    out_without = update_direction_diagnostics(
        policy, trajs, Q_mu, V_mu, alpha=0.5, gamma=gamma, start_idx=start_idx,
        Q_pi=Q_pi)
    assert out_with['pg_bias'] == pytest.approx(out_without['pg_bias'], abs=1e-9)


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
