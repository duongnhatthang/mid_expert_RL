"""Tests for unified teacher value functions."""

import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype.teacher import (
    build_mixture_policy,
    build_optimal_policy,
    build_uniform_policy,
    compute_teacher_values,
    compute_teacher_values_auto,
    evaluate_policy_values,
)
from tabular_prototype.config import compute_gamma_from_horizon


@pytest.fixture
def small_env():
    """Small 5x5 grid with 2 goals for fast tests."""
    goals = [(0, 4), (4, 0)]
    return GridEnv(grid_size=5, goals=goals, horizon=50)


class TestUnifiedZeta1MatchesOptimal:
    """zeta=1.0 with all goals should match old compute_teacher_values."""

    def test_unified_zeta1_matches_optimal(self, small_env):
        env = small_env
        gamma = compute_gamma_from_horizon(env.horizon)

        # Old path: compute_teacher_values (optimal policy, all goals)
        Q_old, V_old = compute_teacher_values(env, env.goals, gamma)

        # New unified path: zeta=1.0, known_goals=env.goals
        Q_new, V_new, gamma_new = compute_teacher_values_auto(
            env, env.goals, zeta=1.0, gamma=gamma
        )

        np.testing.assert_allclose(Q_new, Q_old, atol=1e-10)
        np.testing.assert_allclose(V_new, V_old, atol=1e-10)
        assert gamma_new == gamma


class TestUnifiedZeta0MatchesUniform:
    """zeta=0.0 should match uniform random teacher."""

    def test_unified_zeta0_matches_uniform(self, small_env):
        env = small_env
        gamma = compute_gamma_from_horizon(env.horizon)

        # Direct uniform policy evaluation
        uniform_policy = build_uniform_policy(env.n_states, env.n_actions)
        Q_uniform, V_uniform = evaluate_policy_values(env, uniform_policy, gamma)

        # Unified path with zeta=0.0
        Q_new, V_new, gamma_new = compute_teacher_values_auto(
            env, env.goals, zeta=0.0, gamma=gamma
        )

        np.testing.assert_allclose(Q_new, Q_uniform, atol=1e-10)
        np.testing.assert_allclose(V_new, V_uniform, atol=1e-10)


class TestUnifiedMidZetaSubsetGoals:
    """zeta=0.5 with subset goals should mix that subset's optimal with uniform."""

    def test_unified_mid_zeta_subset_goals(self, small_env):
        env = small_env
        gamma = compute_gamma_from_horizon(env.horizon)
        subset_goals = [env.goals[0]]  # only first goal

        # Manual: build mixture policy with subset goals, evaluate
        optimal_policy = build_optimal_policy(env, subset_goals, gamma)
        uniform_policy = build_uniform_policy(env.n_states, env.n_actions)
        mixed_policy = 0.5 * optimal_policy + 0.5 * uniform_policy
        Q_manual, V_manual = evaluate_policy_values(env, mixed_policy, gamma)

        # Unified path
        Q_new, V_new, gamma_new = compute_teacher_values_auto(
            env, subset_goals, zeta=0.5, gamma=gamma
        )

        np.testing.assert_allclose(Q_new, Q_manual, atol=1e-10)
        np.testing.assert_allclose(V_new, V_manual, atol=1e-10)


class TestBuildMixturePolicyWithKnownGoals:
    """build_mixture_policy with known_goals uses subset's optimal."""

    def test_build_mixture_policy_with_known_goals(self, small_env):
        env = small_env
        gamma = compute_gamma_from_horizon(env.horizon)
        subset_goals = [env.goals[0]]

        # build_mixture_policy with known_goals
        policy = build_mixture_policy(env, zeta=0.7, gamma=gamma, known_goals=subset_goals)

        # Manual construction
        optimal_policy = build_optimal_policy(env, subset_goals, gamma)
        uniform_policy = build_uniform_policy(env.n_states, env.n_actions)
        expected = 0.7 * optimal_policy + 0.3 * uniform_policy

        np.testing.assert_allclose(policy, expected, atol=1e-10)

        # Should differ from using all goals
        policy_all = build_mixture_policy(env, zeta=0.7, gamma=gamma, known_goals=env.goals)
        assert not np.allclose(policy, policy_all), (
            "Subset policy should differ from all-goals policy"
        )
