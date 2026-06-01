"""Integration tests: run_experiment emits the right diagnostic fields."""
import numpy as np

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment, run_learning_curve_experiment


def test_sample_mode_history_includes_pg_diag_fields():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5, goals=goals, teacher_capacity=1, sample_budget=120,
        horizon=10, alpha=1.0, lr=0.5, seed=0, mode='sample',
        trajectories_per_update=4, eval_interval=2,
        pg_diag_enabled=True,
    )
    h0 = result['history'][0]
    for k in ('cos_pg_dir', 'var_g_trace', 'var_g_visited'):
        assert k in h0


def test_exact_mode_history_includes_u_cos_fields_when_enabled():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    result = run_experiment(
        grid_size=5, goals=goals, teacher_capacity=1, sample_budget=10,
        horizon=10, alpha=1.0, lr=0.5, seed=0, mode='exact',
        eval_interval=2,
        pg_diag_enabled=True,
    )
    h0 = result['history'][0]
    for k in ('cos_u_npg', 'cos_u_pinv'):
        assert k in h0


def test_disabled_modes_omit_diag_fields():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    for mode in ('sample', 'exact'):
        sb = 120 if mode == 'sample' else 10
        result = run_experiment(
            grid_size=5, goals=goals, teacher_capacity=1, sample_budget=sb,
            horizon=10, alpha=1.0, lr=0.5, seed=0, mode=mode,
            trajectories_per_update=4, eval_interval=2,
            # pg_diag_enabled omitted → default False
        )
        h0 = result['history'][0]
        assert 'cos_pg_dir' not in h0
        assert 'cos_u_npg' not in h0
        assert 'cos_u_pinv' not in h0


def test_run_learning_curve_propagates_pg_diag_enabled():
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    out = run_learning_curve_experiment(
        grid_size=5, goals=goals, teacher_capacities=[1],
        sample_budget=120, horizon=10, alpha=1.0, lr=0.5,
        n_seeds=1, mode='sample',
        trajectories_per_update=4, eval_interval=2,
        pg_diag_enabled=True,
    )
    h0 = out[1][0][0]
    assert 'cos_pg_dir' in h0
