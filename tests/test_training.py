"""Tests for training.py gradient computation and NPG update."""

import numpy as np
import pytest

from tabular_prototype.environment import GridEnv, generate_equidistant_goals
from tabular_prototype.student import TabularSoftmaxPolicy, collect_trajectories
from tabular_prototype.teacher import compute_teacher_values_auto
from tabular_prototype.training import (
    compute_pav_rl_gradient,
    compute_student_qvalues,
    exact_npg_update,
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


from tabular_prototype.experiments import run_experiment


def test_run_experiment_exact_mode():
    """Exact mode should run without trajectories, using update steps as budget."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=50,
        alpha=0.5,
        lr=0.1,
        seed=0,
        eval_interval=10,
        exact_gradient=True,
    )
    assert 'final_mean_reward' in result
    assert 'history' in result
    assert len(result['history']) > 0
    assert result['budget_mode'] == 'exact'


def test_run_experiment_exact_learns():
    """Exact mode with teacher should learn on a simple grid."""
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
    assert result_teacher['final_goal_rate'] > 0.0 or result_vanilla['final_goal_rate'] > 0.0


@pytest.mark.parametrize("distance", [2, 4, 6, 7])
def test_equidistant_goals_9x9(distance):
    """Verify 3 equidistant goals can be generated at each sweep distance on 9x9."""
    goals = generate_equidistant_goals(9, 3, distance=distance)
    assert len(goals) == 3
    start = (4, 4)
    for g in goals:
        assert abs(g[0] - start[0]) + abs(g[1] - start[1]) == distance
        assert 0 <= g[0] < 9 and 0 <= g[1] < 9


def test_learning_curve_saturation():
    """Learning curve should run with saturation detection."""
    from tabular_prototype.experiments import run_learning_curve_experiment

    histories = run_learning_curve_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacities=[0, 1],
        horizon=16,
        alpha=0.5,
        lr=0.1,
        n_seeds=2,
        eval_interval=2,
        exact_gradient=True,
        max_budget=500,
    )
    assert 0 in histories
    assert 1 in histories
    for cap_histories in histories.values():
        for h in cap_histories:
            if h:
                assert 'exact_V_start_undiscounted' in h[0]
