"""Integration smoke test for the focused runner."""
import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.slow
def test_run_npg_diagnostics_end_to_end():
    """Tiny end-to-end run: 1 seed, B=3, grid=3, budget override."""
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [
                sys.executable, 'run_npg_diagnostics.py',
                '--n-seeds', '1',
                '--n-bootstrap', '3',
                '--grid-size', '3',
                '--distance', '1',  # 3x3 grid only supports distance ∈ {1, 2}
                '--override-budget', '40',
                '--output-dir', tmp,
            ],
            capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': '.'},
        )
        assert out.returncode == 0, (
            f"runner exited {out.returncode}\nSTDOUT:\n{out.stdout}\n"
            f"STDERR:\n{out.stderr}"
        )
        for mode in ('zeta', 'capability'):
            for png in ('npg_cosine.png', 'npg_var_trace.png',
                        'npg_var_s0.png'):
                p = os.path.join(tmp, mode, png)
                assert os.path.exists(p), f"missing: {p}"
                assert os.path.getsize(p) > 0, f"empty: {p}"
