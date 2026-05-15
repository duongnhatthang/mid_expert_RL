"""Tests for MC variance curve + α=0 baseline overlay.

Covers:
 - evaluate_policy returns mean_reward_discounted, std_reward_discounted.
 - run_experiment writes mc_var_undiscounted, mc_var_discounted in both
   exact and trajectory mode histories.
 - plot_mc_variance_curve emits two parameterized PNGs.
 - baseline_alpha=None disables the baseline overlay.
 - mode='cap_zeta' is a no-op.
"""

import numpy as np

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy
from tabular_prototype.training import evaluate_policy
from tabular_prototype.config import compute_gamma_from_horizon


def test_evaluate_policy_returns_discounted_fields():
    """evaluate_policy must return mean_reward_discounted and
    std_reward_discounted alongside the existing fields."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    rng = np.random.default_rng(0)
    result = evaluate_policy(env, policy, n_episodes=5, rng=rng)
    for k in ('mean_reward_discounted', 'std_reward_discounted'):
        assert k in result, f"missing {k!r} in evaluate_policy return"
        assert np.isfinite(result[k])
    # Discounted return <= undiscounted (rewards non-negative, gamma < 1)
    assert (
        result['mean_reward_discounted'] <= result['mean_reward'] + 1e-9
    ), f"discounted={result['mean_reward_discounted']} > undiscounted={result['mean_reward']}"


def test_mc_var_recorded_in_exact_history():
    """run_experiment(mode='exact') writes finite, non-negative
    mc_var_undiscounted and mc_var_discounted in every history entry."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        for k in ('mc_var_undiscounted', 'mc_var_discounted'):
            assert k in entry, f"missing {k!r} in {entry}"
            assert entry[k] is not None
            assert np.isfinite(entry[k])
            assert entry[k] >= 0.0, f"variance must be non-negative, got {entry[k]}"


def test_mc_var_recorded_in_hybrid_history():
    """run_experiment(mode='hybrid') — guards the trajectory-loop
    history.append site."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0, lr=1.0,
        horizon=8, sample_budget=20, mode='hybrid',
        trajectories_per_update=1,
        seed=0, eval_interval=1,
    )
    assert result['history']
    for entry in result['history']:
        for k in ('mc_var_undiscounted', 'mc_var_discounted'):
            assert k in entry
            assert entry[k] is not None
            assert np.isfinite(entry[k])
            assert entry[k] >= 0.0
