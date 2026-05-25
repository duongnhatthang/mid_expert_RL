"""Integration test: sample-mode run_experiment emits the six NPG diag fields."""
import numpy as np

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment


def test_sample_mode_history_includes_npg_diag_fields():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5,
        goals=goals,
        teacher_capacity=1,
        sample_budget=120,
        horizon=10,
        alpha=1.0,
        lr=0.5,
        seed=0,
        mode='sample',
        trajectories_per_update=4,
        eval_interval=2,
        n_bootstrap=5,
    )
    assert result['history'], "expected non-empty history"
    h0 = result['history'][0]
    for k in ('cos_npg_dir', 'var_U_trace',
              'var_U_s0_a0', 'var_U_s0_a1', 'var_U_s0_a2', 'var_U_s0_a3'):
        assert k in h0, f"missing key: {k}"


def test_exact_mode_does_not_emit_npg_diag_fields():
    """Diagnostic is sample-mode only; exact mode should not have these keys."""
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5,
        goals=goals,
        teacher_capacity=1,
        sample_budget=10,  # 10 update steps in exact mode
        horizon=10,
        alpha=1.0,
        lr=0.5,
        seed=0,
        mode='exact',
        eval_interval=2,
    )
    h0 = result['history'][0]
    assert 'cos_npg_dir' not in h0
