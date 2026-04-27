"""Tests for plot_learning_curves in run_hypothesis_sweep.

Builds a small synthetic all_results list (the same shape produced by the
sweep pipeline and pickled to <mode>_sweep_results.pkl) and asserts the
expected PNGs are written to <figures_dir>/learning_curves/.

No image-diff testing — only file existence and non-trivial size.
"""

import pytest

import run_hypothesis_sweep as sweep


def _make_history(n_points: int, base_v: float):
    """Synthetic history list of length n_points, exact_V_start increasing
    linearly from base_v toward base_v + 0.5.
    """
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': base_v + 0.5 * (i / max(1, n_points - 1)),
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
        }
        for i in range(n_points)
    ]


def _zeta_results():
    """2 distances × 2 alphas × 2 horizons × 2 budgets × 2 zetas × 2 seeds = 64 entries."""
    out = []
    for dist in [4, 6]:
        for alpha in [0.0, 0.5]:
            for h_type, h_val in [('small', 8), ('large', 16)]:
                for budget in [10, 30]:
                    for zeta in [0.0, 1.0]:
                        for seed in [0, 1]:
                            out.append({
                                'distance': dist,
                                'alpha': alpha,
                                'horizon_type': h_type,
                                'horizon': h_val,
                                'sample_budget': budget,
                                'zeta': zeta,
                                'seed': seed,
                                'history': _make_history(4, base_v=0.1 * zeta + 0.01 * seed),
                            })
    return out


def test_plot_learning_curves_zeta(tmp_path):
    figures_dir = tmp_path
    sweep.plot_learning_curves(_zeta_results(), mode='zeta',
                               figures_dir=str(figures_dir))

    out_dir = figures_dir / 'learning_curves'
    assert out_dir.is_dir(), "learning_curves subdir not created"

    # 2 distances × 2 alphas = 4 PNGs
    expected = {
        f'learning_curve_dist{d}_alpha{a:.2f}.png'
        for d in [4, 6] for a in [0.0, 0.5]
    }
    actual = {p.name for p in out_dir.glob('*.png')}
    assert actual == expected, f"PNG set mismatch: {actual} vs {expected}"

    for p in out_dir.glob('*.png'):
        assert p.stat().st_size > 1000, f"{p.name} suspiciously small ({p.stat().st_size} B)"


def _capability_results():
    """Capability sweep: 2 distances × 2 alphas × 2 horizons × 2 budgets ×
    capability values × 2 seeds.

    At alpha=0, includes c=-1 (no teacher) and c=0 (uniform). At alpha>0,
    only c in {1, 2, 3} — matches sweep behavior where redundant teacher
    variations are skipped at alpha=0.
    """
    out = []
    for dist in [4, 6]:
        for alpha in [0.0, 0.5]:
            caps = [-1, 0, 1, 2, 3] if alpha == 0.0 else [1, 2, 3]
            for h_type, h_val in [('small', 8), ('large', 16)]:
                for budget in [10, 30]:
                    for cap in caps:
                        for seed in [0, 1]:
                            out.append({
                                'distance': dist,
                                'alpha': alpha,
                                'horizon_type': h_type,
                                'horizon': h_val,
                                'sample_budget': budget,
                                'teacher_capacity': cap,
                                'seed': seed,
                                'history': _make_history(
                                    4, base_v=0.05 * (cap + 1) + 0.01 * seed
                                ),
                            })
    return out


def test_plot_learning_curves_capability(tmp_path):
    figures_dir = tmp_path
    sweep.plot_learning_curves(_capability_results(), mode='capability',
                               figures_dir=str(figures_dir))

    out_dir = figures_dir / 'learning_curves'
    expected = {
        f'learning_curve_dist{d}_alpha{a:.2f}.png'
        for d in [4, 6] for a in [0.0, 0.5]
    }
    actual = {p.name for p in out_dir.glob('*.png')}
    assert actual == expected, f"PNG set mismatch: {actual} vs {expected}"

    for p in out_dir.glob('*.png'):
        assert p.stat().st_size > 1000


def test_plot_learning_curves_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function early-returns
    and writes nothing.
    """
    figures_dir = tmp_path
    # Pass an arbitrary non-empty list; should not be inspected.
    sweep.plot_learning_curves(
        [{'distance': 4, 'alpha': 0.0, 'horizon_type': 'small',
          'sample_budget': 10, 'cap_zeta': 'cap=1_z=0.5', 'seed': 0,
          'history': []}],
        mode='cap_zeta',
        figures_dir=str(figures_dir),
    )
    out_dir = figures_dir / 'learning_curves'
    # Either dir doesn't exist or it exists but is empty.
    if out_dir.exists():
        assert list(out_dir.glob('*.png')) == [], \
            "cap_zeta mode must not write learning-curve PNGs"
