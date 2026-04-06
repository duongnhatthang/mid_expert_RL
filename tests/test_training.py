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
