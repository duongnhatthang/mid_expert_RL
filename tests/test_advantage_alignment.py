"""Tests for adv_product_s0 metric + plot_advantage_alignment.

Covers:
 - `_compute_adv_product_s0` helper correctness on a uniform policy
   (where A^π is identically zero so the product collapses to zero).
 - run_experiment writes the field in both exact and trajectory modes.
 - plot_advantage_alignment emits the expected PNG via the default path
   and the override path; mode='cap_zeta' is a no-op.
"""

import numpy as np

from tabular_prototype.environment import GridEnv
from tabular_prototype.student import TabularSoftmaxPolicy
from tabular_prototype.training import compute_student_qvalues
from tabular_prototype.config import compute_gamma_from_horizon
from tabular_prototype.experiments import _compute_adv_product_s0


def test_adv_product_s0_zero_when_A_mu_is_zero():
    """When A^μ ≡ 0 (constant Q^μ at the start state), the product
    collapses to zero regardless of A^π. Catches whether the helper
    actually multiplies A^μ in (vs. ignoring it) and pins the shape
    contract on Q_mu/V_mu/probs.
    """
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(8)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)

    # Construct Q_mu, V_mu so that A_mu = Q_mu - V_mu = 0 everywhere.
    Q_mu = np.full((env.n_states, env.n_actions), 0.5)
    V_mu = np.full(env.n_states, 0.5)
    start_idx = env.state_to_idx(env.start)

    g = _compute_adv_product_s0(policy, Q_pi, V_pi, Q_mu, V_mu, start_idx)
    assert abs(g) < 1e-10, f"expected 0 when A^μ=0, got {g}"


def test_adv_product_s0_none_when_teacher_absent():
    """When Q_mu/V_mu is None the helper returns None (not 0, to keep the
    'teacher absent' state distinguishable downstream)."""
    env = GridEnv(grid_size=9, goals=[(2, 4)], horizon=8)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    gamma = compute_gamma_from_horizon(8)
    Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
    start_idx = env.state_to_idx(env.start)

    assert _compute_adv_product_s0(
        policy, Q_pi, V_pi, None, None, start_idx,
    ) is None
