import os
import numpy as np
from tabular_prototype import visualization as viz

CELL = {'distance': 6, 'horizon': 8, 'horizon_type': 'small',
        'sample_budget': 12, 'alpha': 1.0}

METRICS = ['max_mu_over_pi', 'max_pi_over_mu', 'chi2_pi_mu',
           'kl_pi_mu', 'kl_mu_pi', 'tv', 'renyi_inf']


def _hist(seed_offset):
    out = []
    for tick in range(3):
        entry = {'steps': (tick + 1) * 2}
        for m in METRICS:
            entry[f'cov_analytic_{m}'] = 1.0 + 0.1 * tick + 0.01 * seed_offset
        out.append(entry)
    return out


def test_plot_coverage_curves_writes_all_metrics(tmp_path):
    hbt = {0: [_hist(0), _hist(1)], 3: [_hist(2), _hist(3)]}
    baseline = [_hist(4), _hist(5)]
    paths = viz.plot_coverage_curves(
        histories_by_teacher=hbt,
        baseline_history_alpha_zero=baseline,
        mode='capability', out_dir=str(tmp_path), cell_info=CELL,
        ref='analytic',
    )
    assert len(paths) == len(METRICS)
    for p in paths:
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_plot_coverage_heatmaps_writes_file(tmp_path):
    n_s, n_a, gs = 25, 4, 5
    grids = {
        'd_mu': np.abs(np.random.default_rng(0).random((n_s, n_a))),
        'd_pi': np.abs(np.random.default_rng(1).random((n_s, n_a))),
        'ratio_mu_over_pi': np.abs(np.random.default_rng(2).random((n_s, n_a))),
        'ratio_pi_over_mu': np.abs(np.random.default_rng(3).random((n_s, n_a))),
    }
    out = str(tmp_path / 'cov_analytic_heatmap.png')
    viz.plot_coverage_heatmaps('analytic', grids, gs, out, CELL,
                               student_label='cap=3 (α=1)')
    assert os.path.exists(out) and os.path.getsize(out) > 0
