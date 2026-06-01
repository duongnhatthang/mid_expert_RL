"""Visualization smoke tests for the PG/U direction figures."""
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _fake_pg_history(n_eval_ticks=10, cos_value=0.7):
    return [{
        'steps': 10 * i,
        'cos_pg_dir': cos_value + 0.01 * i,
        'var_g_trace': 0.5 + 0.01 * i,
        'var_g_visited': 0.3 + 0.01 * i,
    } for i in range(n_eval_ticks)]


def _fake_u_history(n_eval_ticks=10, npg_value=0.5, pinv_value=0.6):
    return [{
        'steps': 10 * i,
        'cos_u_npg': npg_value + 0.01 * i,
        'cos_u_pinv': pinv_value + 0.01 * i,
    } for i in range(n_eval_ticks)]


def test_plot_pg_cosine_writes_png():
    from tabular_prototype.visualization import plot_pg_cosine
    histories = {0: [_fake_pg_history()], 1: [_fake_pg_history(cos_value=0.8)]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pg_cosine.png')
        plot_pg_cosine(
            histories_by_teacher=histories,
            mode='capability',
            out_path=path,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
        )
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_pg_variance_writes_two_pngs():
    from tabular_prototype.visualization import plot_pg_variance
    histories = {0: [_fake_pg_history()], 1: [_fake_pg_history()]}
    baseline = [_fake_pg_history()]
    with tempfile.TemporaryDirectory() as tmp:
        plot_pg_variance(
            histories_by_teacher=histories,
            baseline_history_alpha_zero=baseline,
            mode='capability',
            out_dir=tmp,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
        )
        assert os.path.exists(os.path.join(tmp, 'pg_var_trace.png'))
        assert os.path.exists(os.path.join(tmp, 'pg_var_visited.png'))


def test_plot_u_cosine_writes_both_centerings():
    from tabular_prototype.visualization import plot_u_cosine
    histories = {0: [_fake_u_history()], 1: [_fake_u_history()]}
    with tempfile.TemporaryDirectory() as tmp:
        for centering in ('npg', 'pinv'):
            path = os.path.join(tmp, f'u_cosine_{centering}.png')
            plot_u_cosine(
                histories_by_teacher=histories,
                mode='capability',
                out_path=path,
                cell_info={'distance': 6, 'horizon': 50,
                           'horizon_type': 'small',
                           'sample_budget': 200, 'alpha': 1.0},
                centering=centering,
            )
            assert os.path.exists(path) and os.path.getsize(path) > 0
