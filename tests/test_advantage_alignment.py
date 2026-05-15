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


def _adv_history(n_points: int, base: float):
    """Synthetic history list. Mirrors the schema produced by
    run_experiment, including the new adv_product_s0 field."""
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': 0.0,
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
            'adv_product_s0': base + 0.05 * i,
        }
        for i in range(n_points)
    ]


def _zeta_results_for_advantage_alignment():
    """Synthetic all_results filling the default cell
    (distance=6, horizon_type='small', alpha=1.0, B=budgets[-2]).
    4 zetas × 3 seeds = 12 entries."""
    import json
    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    h_val = cell['horizon']

    out = []
    for zeta in [0.0, 0.33, 0.67, 1.0]:
        for seed in [0, 1, 2]:
            out.append({
                'distance': 6,
                'alpha': 1.0,
                'horizon_type': 'small',
                'horizon': h_val,
                'sample_budget': budget,
                'zeta': zeta,
                'seed': seed,
                'mode': 'exact',
                'history': _adv_history(4, base=0.05 * zeta + 0.01 * seed),
            })
    return out


def test_plot_advantage_alignment_default_path(tmp_path, monkeypatch):
    """Default invocation must emit exactly one PNG matching the
    parameterized filename for the default cell."""
    import json
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import run_hypothesis_sweep as sweep

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_advantage_alignment(
        _zeta_results_for_advantage_alignment(),
        mode='zeta',
        figures_dir=str(tmp_path),
    )

    assert captured_figures, "expected at least one figure saved"

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    expected_name = (
        f'advantage_alignment_dist6_small_B{budget}_alpha1.00.png'
    )
    pngs = list(tmp_path.glob('*.png'))
    assert len(pngs) == 1, f"expected 1 PNG, got {[p.name for p in pngs]}"
    assert pngs[0].name == expected_name, \
        f"unexpected filename {pngs[0].name}, expected {expected_name}"
    assert pngs[0].stat().st_size > 1000

    plt.close('all')


def test_plot_advantage_alignment_override_budget_rank(tmp_path):
    """Passing budget_rank=-1 selects the largest budget; filename
    parameterizes on the resolved budget."""
    import json
    import matplotlib.pyplot as plt
    import run_hypothesis_sweep as sweep

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    largest_budget = cell['budgets'][-1]
    h_val = cell['horizon']

    # Synthetic data at the LARGEST budget so the override matches it.
    results = []
    for zeta in [0.0, 0.33, 0.67, 1.0]:
        for seed in [0, 1, 2]:
            results.append({
                'distance': 6,
                'alpha': 1.0,
                'horizon_type': 'small',
                'horizon': h_val,
                'sample_budget': largest_budget,
                'zeta': zeta,
                'seed': seed,
                'mode': 'exact',
                'history': _adv_history(4, base=0.05 * zeta + 0.01 * seed),
            })

    sweep.plot_advantage_alignment(
        results, mode='zeta', figures_dir=str(tmp_path),
        budget_rank=-1,
    )

    expected = (
        f'advantage_alignment_dist6_small_B{largest_budget}_alpha1.00.png'
    )
    pngs = list(tmp_path.glob('*.png'))
    assert len(pngs) == 1, f"expected 1 PNG, got {[p.name for p in pngs]}"
    assert pngs[0].name == expected

    plt.close('all')


def test_plot_advantage_alignment_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function returns
    immediately and writes nothing."""
    import run_hypothesis_sweep as sweep
    sweep.plot_advantage_alignment([], mode='cap_zeta',
                                    figures_dir=str(tmp_path))
    assert not list(tmp_path.glob('*.png'))
