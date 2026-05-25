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
