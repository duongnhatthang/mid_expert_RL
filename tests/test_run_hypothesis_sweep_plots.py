"""Tests for plot_learning_curves in run_hypothesis_sweep.

Builds a small synthetic all_results list (the same shape produced by the
sweep pipeline and pickled to <mode>_sweep_results.pkl) and asserts the
expected PNGs are written to <figures_dir>/learning_curves/.

No image-diff testing — only file existence and non-trivial size.
"""

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


def test_plot_learning_curves_ragged_histories(tmp_path):
    """In hybrid/sample mode, update count per seed varies because the
    budget bounds env steps, not updates. plot_learning_curves must
    truncate to the shortest seed's prefix and proceed silently rather
    than raising.
    """
    out = []
    # Single (dist, alpha, h_type, budget, zeta) cell, 3 seeds with
    # history lengths {3, 4, 5}.
    for seed, n_pts in zip([0, 1, 2], [3, 4, 5]):
        out.append({
            'distance': 4,
            'alpha': 0.0,
            'horizon_type': 'small',
            'horizon': 8,
            'sample_budget': 10,
            'zeta': 0.0,
            'seed': seed,
            'history': _make_history(n_pts, base_v=0.0),
        })

    figures_dir = tmp_path
    sweep.plot_learning_curves(out, mode='zeta', figures_dir=str(figures_dir))

    out_dir = figures_dir / 'learning_curves'
    pngs = list(out_dir.glob('*.png'))
    assert len(pngs) == 1, f"Expected 1 PNG, got {len(pngs)}"
    assert pngs[0].stat().st_size > 1000


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


def test_plot_learning_curves_draws_v_star_skyline(tmp_path, monkeypatch):
    """Every visible subplot must carry a black dashed V*(s_0) horizontal line."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_learning_curves(_zeta_results(), mode='zeta',
                               figures_dir=str(tmp_path))

    assert captured_figures, "expected at least one figure saved"

    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert dashed_black, (
                f"axes titled {ax.get_title()!r} missing a black dashed "
                f"V*(s_0) line"
            )

    plt.close('all')


def test_v_star_at_s0_is_finite_and_in_range():
    """Sanity-check the V*(s_0) computation we wire into learning curves.

    Build a GridEnv with the same parameters used by the sweep (grid_size=9,
    1 goal at distance 4, horizon=8 i.e. the 'small' bucket for 9x9), compute
    pi* and V*, and assert 0 < V*(s_0) <= 1.

    Lower bound: the goal is reachable in 4 steps (Manhattan distance), well
    within H=8, so the optimal discounted return is strictly positive.
    Upper bound: maximum undiscounted return is 1 (single absorbing goal
    with reward=1); discounting shrinks it further. Values above 1 would
    indicate a sign / state-index bug.
    """
    from tabular_prototype.environment import GridEnv, generate_equidistant_goals
    from tabular_prototype.teacher import build_optimal_policy, evaluate_policy_values
    from tabular_prototype.config import compute_gamma_from_horizon

    grid_size = 9
    horizon = 8  # 'small' bucket for 9x9 (corner_dist = 8)
    goals = generate_equidistant_goals(grid_size, n_goals=1, distance=4)
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)
    gamma = compute_gamma_from_horizon(horizon)

    pi_star = build_optimal_policy(env, env.goals, gamma)
    _, V_star = evaluate_policy_values(env, pi_star, gamma)
    v_star_s0 = float(V_star[env.state_to_idx(env.start)])

    assert 0.0 < v_star_s0 <= 1.0, (
        f"V*(s_0) out of expected (0, 1] range: {v_star_s0}"
    )
