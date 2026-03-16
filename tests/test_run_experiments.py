"""
Unit tests for run_experiments, including visualize_2x2_policies.
"""

import os

import pytest

from run_experiments import visualize_2x2_policies


def test_visualize_2x2_policies(tmp_path):
    """Visualize policies: 2 figures (cap 1 and n_goals)."""
    saved = visualize_2x2_policies(
        grid_size=4,
        n_goals=2,
        figures_dir=str(tmp_path),
    )
    assert len(saved) == 2, "Expected 2 figures (teacher_cap 1 and 2)"
    for path in saved:
        assert os.path.isfile(path), f"Figure not created: {path}"
