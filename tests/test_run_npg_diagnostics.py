"""Integration smoke test for the focused runner."""
import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.slow
def test_run_npg_diagnostics_end_to_end():
    """Tiny run: both training modes, both sweep modes, smallest configs."""
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, 'run_npg_diagnostics.py',
             '--n-seeds', '1',
             '--grid-size', '3',
             '--distance', '1',
             '--override-budget', '40',
             '--training-modes', 'sample,exact',
             '--output-dir', tmp],
            capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': '.'},
        )
        assert out.returncode == 0, (
            f"runner exited {out.returncode}\n"
            f"STDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")
        # Sample mode: 3 PNGs per sweep mode.
        for sweep_mode in ('zeta', 'capability'):
            for png in ('pg_cosine.png', 'pg_var_trace.png', 'pg_var_visited.png'):
                p = os.path.join(tmp, 'sample', sweep_mode, png)
                assert os.path.exists(p), f"missing: {p}"
                assert os.path.getsize(p) > 0
            for png in ('u_cosine_npg.png', 'u_cosine_pinv.png'):
                p = os.path.join(tmp, 'exact', sweep_mode, png)
                assert os.path.exists(p), f"missing: {p}"
                assert os.path.getsize(p) > 0
