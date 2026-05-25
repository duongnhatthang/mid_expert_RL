"""Visualization smoke tests for the NPG-direction figures."""
import os
import tempfile

import matplotlib

matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np


def _fake_history(n_eval_ticks=10, cos_value=0.7, var_value=0.5):
    return [
        {
            'steps': 10 * i,
            'cos_npg_dir': cos_value + 0.01 * i,
            'var_U_trace': var_value + 0.01 * i,
            'var_U_s0_a0': var_value * 0.1,
            'var_U_s0_a1': var_value * 0.2,
            'var_U_s0_a2': var_value * 0.3,
            'var_U_s0_a3': var_value * 0.4,
        }
        for i in range(n_eval_ticks)
    ]


def test_plot_npg_cosine_writes_png_with_reference_line():
    from tabular_prototype.visualization import plot_npg_cosine
    histories_by_teacher = {
        0: [_fake_history(cos_value=0.5)],
        1: [_fake_history(cos_value=0.7)],
        2: [_fake_history(cos_value=0.9)],
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'npg_cosine.png')
        plot_npg_cosine(
            histories_by_teacher=histories_by_teacher,
            mode='capability',
            out_path=out_path,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


def test_plot_npg_variance_writes_both_pngs_with_4_subplots_at_s0():
    from tabular_prototype.visualization import plot_npg_variance
    histories_by_teacher = {
        0: [_fake_history(var_value=0.3)],
        1: [_fake_history(var_value=0.5)],
        2: [_fake_history(var_value=0.7)],
    }
    baseline_alpha_zero = [_fake_history(var_value=0.4)]
    with tempfile.TemporaryDirectory() as tmp:
        plot_npg_variance(
            histories_by_teacher=histories_by_teacher,
            baseline_history_alpha_zero=baseline_alpha_zero,
            mode='capability',
            out_dir=tmp,
            cell_info={'distance': 6, 'horizon': 50,
                       'horizon_type': 'small',
                       'sample_budget': 200, 'alpha': 1.0},
            n_bootstrap=50,
        )
        trace_path = os.path.join(tmp, 'npg_var_trace.png')
        s0_path = os.path.join(tmp, 'npg_var_s0.png')
        assert os.path.exists(trace_path) and os.path.getsize(trace_path) > 0
        assert os.path.exists(s0_path) and os.path.getsize(s0_path) > 0
        import matplotlib.image as mpimg
        img = mpimg.imread(s0_path)
        assert img.shape[0] > 0 and img.shape[1] > 0


def test_plot_npg_variance_s0_has_2x2_subplot_layout():
    from tabular_prototype.visualization import _build_npg_var_s0_figure
    histories_by_teacher = {0: [_fake_history(var_value=0.3)]}
    baseline_alpha_zero = [_fake_history(var_value=0.4)]
    fig = _build_npg_var_s0_figure(
        histories_by_teacher=histories_by_teacher,
        baseline_history_alpha_zero=baseline_alpha_zero,
        mode='capability',
        cell_info={'distance': 6, 'horizon': 50,
                   'horizon_type': 'small',
                   'sample_budget': 200, 'alpha': 1.0},
        n_bootstrap=50,
    )
    assert len(fig.axes) == 4
    suptitle = fig._suptitle.get_text() if fig._suptitle else ''
    assert 'dist=6' in suptitle
    assert 'B=200' in suptitle
    assert r'$\alpha=1.0$' in suptitle or 'alpha=1.0' in suptitle.lower()
    bootstrap_anno = '\n'.join(t.get_text() for t in fig.texts)
    assert 'B=50' in bootstrap_anno
    assert 'bootstrap' in bootstrap_anno.lower()
    plt.close(fig)


def test_plot_npg_cosine_figure_has_cell_info_annotation():
    from tabular_prototype.visualization import _build_npg_cosine_figure
    histories_by_teacher = {0: [_fake_history(cos_value=0.7)]}
    fig = _build_npg_cosine_figure(
        histories_by_teacher=histories_by_teacher,
        mode='capability',
        cell_info={'distance': 6, 'horizon': 50,
                   'horizon_type': 'small',
                   'sample_budget': 200, 'alpha': 1.0},
    )
    suptitle = fig._suptitle.get_text() if fig._suptitle else ''
    assert 'dist=6' in suptitle
    assert 'B=200' in suptitle
    formula = '\n'.join(t.get_text() for t in fig.texts)
    assert 'F' in formula and ('g' in formula or 'hat g' in formula)
    plt.close(fig)
