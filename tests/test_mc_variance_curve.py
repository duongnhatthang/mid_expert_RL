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


def _var_history(n_points: int, base: float):
    """Synthetic history list with mc_var_* fields populated."""
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': 0.0,
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
            'adv_product_s0': 0.0,
            'mc_var_undiscounted': base + 0.01 * i,
            'mc_var_discounted': 0.5 * (base + 0.01 * i),
        }
        for i in range(n_points)
    ]


def _zeta_results_for_mc_variance():
    """Synthetic all_results for the default variance cell:
    distance=6, horizon_type='small', alpha=1.0, B=budgets[-2] plus
    α=0 baseline. 4 zetas × 3 seeds + 1 zeta × 3 seeds = 15 entries."""
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
                'history': _var_history(4, base=0.05 * zeta + 0.01 * seed),
            })
    for seed in [0, 1, 2]:
        out.append({
            'distance': 6,
            'alpha': 0.0,
            'horizon_type': 'small',
            'horizon': h_val,
            'sample_budget': budget,
            'zeta': 0.0,
            'seed': seed,
            'mode': 'exact',
            'history': _var_history(4, base=0.02 + 0.005 * seed),
        })
    return out


def test_plot_mc_variance_curve_default_emits_two_pngs(tmp_path, monkeypatch):
    """Default invocation must emit exactly two PNGs (undiscounted +
    discounted) with the expected parameterized filenames."""
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

    sweep.plot_mc_variance_curve(
        _zeta_results_for_mc_variance(),
        mode='zeta',
        figures_dir=str(tmp_path),
    )

    calib = json.load(open('results/calibration.json'))
    cell = next(v for k, v in calib.items()
                if k.startswith('dist=6_small_ng=1_'))
    budget = cell['budgets'][-2]
    expected = {
        f'mc_variance_undiscounted_dist6_small_B{budget}_alpha1.00.png',
        f'mc_variance_discounted_dist6_small_B{budget}_alpha1.00.png',
    }
    pngs = {p.name for p in tmp_path.glob('*.png')}
    assert pngs == expected, f"expected {expected}, got {pngs}"
    for p in tmp_path.glob('*.png'):
        assert p.stat().st_size > 1000

    # Every captured figure must have a dashed black α=0 baseline line.
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert dashed_black, (
                "expected α=0 baseline overlay (dashed black) on "
                "every captured figure"
            )

    plt.close('all')


def test_plot_mc_variance_curve_baseline_alpha_none(tmp_path, monkeypatch):
    """Passing baseline_alpha=None must suppress the α=0 baseline
    overlay in every emitted figure."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import run_hypothesis_sweep as sweep

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_mc_variance_curve(
        _zeta_results_for_mc_variance(),
        mode='zeta',
        figures_dir=str(tmp_path),
        baseline_alpha=None,
    )

    assert captured_figures, "expected at least one figure saved"
    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert not dashed_black, (
                "expected NO dashed black baseline line when "
                f"baseline_alpha=None, got {len(dashed_black)}"
            )

    plt.close('all')


def test_plot_mc_variance_curve_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function returns
    immediately and writes nothing."""
    import run_hypothesis_sweep as sweep
    sweep.plot_mc_variance_curve([], mode='cap_zeta',
                                  figures_dir=str(tmp_path))
    assert not list(tmp_path.glob('*.png'))
