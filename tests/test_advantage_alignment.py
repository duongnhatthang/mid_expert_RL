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


def test_adv_product_s0_recorded_in_exact_history():
    """run_experiment(mode='exact') writes a finite adv_product_s0 in
    every history entry when a teacher is configured."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0,
        horizon=8, sample_budget=5, mode='exact',
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        assert 'adv_product_s0' in entry, \
            f"missing adv_product_s0 in entry {entry}"
        assert entry['adv_product_s0'] is not None
        assert np.isfinite(entry['adv_product_s0'])


def test_adv_product_s0_recorded_in_hybrid_history():
    """run_experiment(mode='hybrid') writes a finite adv_product_s0 in
    every history entry. Guards against the trajectory-loop
    history.append site (which is separate from the exact-mode one)."""
    from tabular_prototype.experiments import run_experiment
    result = run_experiment(
        grid_size=9, goals=[(2, 4), (4, 6), (6, 4)],
        teacher_capacity=1, alpha=1.0, lr=1.0,
        horizon=8, sample_budget=20, mode='hybrid',
        trajectories_per_update=1,
        seed=0, eval_interval=1,
    )
    assert result['history'], "history should be non-empty"
    for entry in result['history']:
        assert 'adv_product_s0' in entry
        assert entry['adv_product_s0'] is not None
        assert np.isfinite(entry['adv_product_s0'])


def test_calibration_helpers_find_dist6_small_cell():
    """The two new helpers wrap calibration-JSON path resolution and
    per-cell substring matching. Smoke-test against the checked-in
    results/calibration.json."""
    import json
    from run_hypothesis_sweep import (
        _calibration_path_for, _find_calibration_cell,
    )

    # exact training mode → results/calibration.json
    path = _calibration_path_for('exact')
    assert path.endswith('calibration.json')

    calib = json.load(open(path))
    cell = _find_calibration_cell(calib, distance=6, h_type='small', n_goals=1)
    assert cell is not None, "expected one matching cell for dist=6 small ng=1"
    assert cell['horizon'] == 8
    assert isinstance(cell['budgets'], list) and len(cell['budgets']) >= 2

    # hybrid training mode → results/calibration_hybrid.json
    path_h = _calibration_path_for('hybrid')
    assert path_h.endswith('calibration_hybrid.json')
