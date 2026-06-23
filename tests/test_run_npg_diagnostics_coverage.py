# tests/test_run_npg_diagnostics_coverage.py
import os
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


def test_emit_coverage_plots_writes_both_refs(tmp_path):
    hbt = {0: [_hist()], 3: [_hist()]}
    grids_by_teacher = {0: [_grids()], 3: [_grids()]}
    rnd._emit_coverage_plots(
        out_dir=str(tmp_path), mode='capability', cell_info=CELL,
        histories_by_teacher=hbt, baseline_histories=[_hist()],
        grids_by_teacher=grids_by_teacher, baseline_grids=[_grids()],
        top_tv=3, grid_size=5,
    )
    for ref in ('analytic', 'learned'):
        for m in METRICS:
            assert os.path.exists(os.path.join(tmp_path, f'cov_{ref}_{m}.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_top.png'))
        assert os.path.exists(
            os.path.join(tmp_path, f'cov_{ref}_heatmap_baseline.png'))
