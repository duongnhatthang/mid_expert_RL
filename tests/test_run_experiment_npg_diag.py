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


def test_sample_mode_skips_npg_diag_when_n_bootstrap_is_zero():
    """When n_bootstrap=0 (the default), sample mode preserves the legacy
    history schema — no NPG-direction keys appear."""
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
        # n_bootstrap omitted → default 0
    )
    h0 = result['history'][0]
    assert 'cos_npg_dir' not in h0


def test_run_learning_curve_propagates_n_bootstrap():
    from tabular_prototype.experiments import run_learning_curve_experiment
    from tabular_prototype.environment import generate_equidistant_goals
    goals = generate_equidistant_goals(grid_size=5, n_goals=1, distance=2)
    out = run_learning_curve_experiment(
        grid_size=5, goals=goals, teacher_capacities=[1],
        sample_budget=120, horizon=10, alpha=1.0, lr=0.5,
        n_seeds=1, mode='sample',
        trajectories_per_update=4, eval_interval=2,
        n_bootstrap=3,
    )
    # run_learning_curve_experiment returns Dict[int, list] mapping
    # teacher_capacity -> list of per-seed histories (each history is a
    # list of per-eval step dicts).
    assert out, "expected at least one capacity result"
    assert 1 in out, "expected capacity=1 in results"
    assert out[1], "expected at least one seed history"
    h0 = out[1][0][0]
    assert 'cos_npg_dir' in h0
