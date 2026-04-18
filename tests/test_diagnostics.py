"""Tests for per-step diagnostic collection."""

import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy
from tabular_prototype.teacher import compute_teacher_values_auto
from tabular_prototype.training import exact_npg_update, compute_student_qvalues
from tabular_prototype.config import compute_gamma_from_horizon


@pytest.fixture
def setup():
    goals = [(3, 3)]
    env = GridEnv(grid_size=4, goals=goals, horizon=16)
    gamma = compute_gamma_from_horizon(env.horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    Q_mu, V_mu, _ = compute_teacher_values_auto(env, goals)
    return env, policy, Q_pi, Q_mu, V_mu, gamma


def test_diagnostics_returned(setup):
    """exact_npg_update returns a diagnostics dict."""
    env, policy, Q_pi, Q_mu, V_mu, gamma = setup
    diag = exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=0.5, lr=0.1)
    assert isinstance(diag, dict)


def test_diagnostics_keys(setup):
    """All expected diagnostic keys are present."""
    env, policy, Q_pi, Q_mu, V_mu, gamma = setup
    diag = exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=0.5, lr=0.1)
    expected_keys = {
        'q_pi_l2', 'q_pi_max', 'a_mu_l2', 'a_mu_max',
        'a_mu_mean', 'a_mu_var', 'a_mu_kurtosis',
        'a_mu_min_val', 'a_mu_max_val',
    }
    assert expected_keys.issubset(diag.keys())


def test_diagnostics_norms_correct(setup):
    """Verify norm calculations match manual computation."""
    env, policy, Q_pi, Q_mu, V_mu, gamma = setup
    alpha = 0.5
    lr = 0.1
    A_mu = Q_mu - V_mu[:, None]
    q_component = (1.0 - alpha) * Q_pi
    a_component = alpha * A_mu
    diag = exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=alpha, lr=lr)
    np.testing.assert_allclose(diag['q_pi_l2'], np.linalg.norm(q_component), rtol=1e-10)
    np.testing.assert_allclose(diag['q_pi_max'], np.abs(q_component).max(), rtol=1e-10)
    np.testing.assert_allclose(diag['a_mu_l2'], np.linalg.norm(a_component), rtol=1e-10)
    np.testing.assert_allclose(diag['a_mu_max'], np.abs(a_component).max(), rtol=1e-10)


def test_diagnostics_distribution_stats(setup):
    """Verify A^mu distribution statistics."""
    env, policy, Q_pi, Q_mu, V_mu, gamma = setup
    diag = exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=0.5, lr=0.1)
    A_mu = Q_mu - V_mu[:, None]
    np.testing.assert_allclose(diag['a_mu_mean'], A_mu.mean(), rtol=1e-10)
    np.testing.assert_allclose(diag['a_mu_var'], A_mu.var(), rtol=1e-10)
    np.testing.assert_allclose(diag['a_mu_min_val'], A_mu.min(), rtol=1e-10)
    np.testing.assert_allclose(diag['a_mu_max_val'], A_mu.max(), rtol=1e-10)


def test_diagnostics_no_teacher(setup):
    """Without teacher, A^mu diagnostics should be zero."""
    env, policy, Q_pi, _, _, gamma = setup
    diag = exact_npg_update(policy, Q_pi, None, None, alpha=0.5, lr=0.1)
    assert diag['a_mu_l2'] == 0.0
    assert diag['a_mu_max'] == 0.0
    assert diag['q_pi_l2'] > 0.0


def test_diagnostics_alpha_zero(setup):
    """At alpha=0, A^mu component norms should be zero."""
    env, policy, Q_pi, Q_mu, V_mu, gamma = setup
    diag = exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha=0.0, lr=0.1)
    assert diag['a_mu_l2'] == 0.0
    assert diag['a_mu_max'] == 0.0
    assert diag['q_pi_l2'] > 0.0


# =========================================================================
# Integration tests: run_experiment diagnostics collection
# =========================================================================

from tabular_prototype.experiments import run_experiment


def test_run_experiment_collects_diagnostics():
    """run_experiment in exact mode should collect per-step diagnostics."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=20,
        alpha=0.5,
        lr=0.1,
        seed=0,
        eval_interval=5,
        exact_gradient=True,
    )
    assert 'diagnostics' in result
    assert len(result['diagnostics']) == 20  # one per step

    d = result['diagnostics'][0]
    assert 'q_pi_l2' in d
    assert 'a_mu_l2' in d
    assert 'delta_v_total' in d
    assert 'delta_v_qpi' in d
    assert 'delta_v_amu' in d
    assert 'policy_entropy_start' in d


def test_run_experiment_cap_zeta():
    """Combined capacity + zeta mode should work."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3), (0, 3)],
        teacher_capacity=1,
        zeta=0.5,
        horizon=16,
        sample_budget=10,
        alpha=0.5,
        lr=0.1,
        seed=0,
        eval_interval=5,
        exact_gradient=True,
    )
    assert 'final_mean_reward' in result
    assert result['diagnostics'] is not None
    assert len(result['diagnostics']) == 10


def test_run_experiment_diagnostics_alpha_zero():
    """At alpha=0, A^mu contributions should be zero in diagnostics."""
    result = run_experiment(
        grid_size=4,
        goals=[(3, 3)],
        teacher_capacity=1,
        horizon=16,
        sample_budget=10,
        alpha=0.0,
        lr=0.1,
        seed=0,
        eval_interval=5,
        exact_gradient=True,
    )
    for d in result['diagnostics']:
        assert d['a_mu_l2'] == 0.0
        assert d['delta_v_amu'] == pytest.approx(0.0, abs=1e-8)
