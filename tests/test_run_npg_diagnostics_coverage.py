# tests/test_run_npg_diagnostics_coverage.py
import os
import pickle
import numpy as np
import run_npg_diagnostics as rnd

CELL = {'distance': 6, 'horizon': 8, 'horizon_type': 'small',
        'sample_budget': 12, 'alpha': 1.0}
METRICS = ['max_mu_over_pi', 'max_pi_over_mu', 'chi2_pi_mu',
           'kl_pi_mu', 'kl_mu_pi', 'tv', 'renyi_inf']


def _hist():
    out = []
    for tick in range(2):
        e = {'steps': (tick + 1) * 2}
        for ref in ('analytic', 'learned'):
            for m in METRICS:
                e[f'cov_{ref}_{m}'] = 1.0 + 0.1 * tick
        out.append(e)
    return out


def _grids():
    n_s, n_a = 25, 4
    g = lambda s: np.abs(np.random.default_rng(s).random((n_s, n_a)))
    return {ref: {'d_mu': g(0), 'd_pi': g(1),
                  'ratio_mu_over_pi': g(2), 'ratio_pi_over_mu': g(3)}
            for ref in ('analytic', 'learned')}


def test_emit_coverage_plots_capability_one_heatmap_per_teacher(tmp_path):
    hbt = {0: [_hist()], 3: [_hist()]}
    grids_by_teacher = {0: [_grids()], 3: [_grids()]}
    rnd._emit_coverage_plots(
        out_dir=str(tmp_path), mode='capability', cell_info=CELL,
        histories_by_teacher=hbt, baseline_histories=[_hist()],
        grids_by_teacher=grids_by_teacher, baseline_grids=[_grids()],
        grid_size=5,
    )
    for ref in ('analytic', 'learned'):
        for m in METRICS:
            assert os.path.exists(os.path.join(tmp_path, f'cov_{ref}_{m}.png'))
        # One heatmap per α=1 teacher value, plus the α=0 baseline.
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_cap0.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_cap3.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_baseline.png'))
        assert not os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_top.png'))


def test_emit_coverage_plots_zeta_slugs(tmp_path):
    hbt = {0.0: [_hist()], 1.0: [_hist()]}
    grids_by_teacher = {0.0: [_grids()], 1.0: [_grids()]}
    rnd._emit_coverage_plots(
        out_dir=str(tmp_path), mode='zeta', cell_info=CELL,
        histories_by_teacher=hbt, baseline_histories=[_hist()],
        grids_by_teacher=grids_by_teacher, baseline_grids=[_grids()],
        grid_size=5,
    )
    for ref in ('analytic', 'learned'):
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_zeta0.0.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_zeta1.0.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_baseline.png'))


def test_replot_coverage_from_cache_redraws_without_running(tmp_path):
    # Lay out a cached coverage_data.pkl as a real run would, then redraw.
    cell_dir = tmp_path / 'exact' / 'capability'
    cell_dir.mkdir(parents=True)
    with open(cell_dir / 'coverage_data.pkl', 'wb') as fp:
        pickle.dump(dict(
            sweep_mode='capability', cell_info=CELL, grid_size=5,
            histories_by_teacher={0: [_hist()], 3: [_hist()]},
            baseline_histories=[_hist()],
            grids_by_teacher={0: [_grids()], 3: [_grids()]},
            baseline_grids=[_grids()],
        ), fp)
    rnd._replot_coverage_from(str(tmp_path))
    for ref in ('analytic', 'learned'):
        assert os.path.exists(cell_dir / f'cov_{ref}_heatmap_cap0.png')
        assert os.path.exists(cell_dir / f'cov_{ref}_heatmap_cap3.png')
        assert os.path.exists(cell_dir / f'cov_{ref}_tv.png')
