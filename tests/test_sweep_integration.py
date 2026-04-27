"""End-to-end integration test for run_hypothesis_sweep.run_sweep.

Exercises the full wiring (calibration dispatch, training-mode dispatch,
CSV + pickle output) for every (sweep_mode, training_mode) pair the
default suite cares about. Uses fake calibration files in a tempdir and
a reduced distance set so a full sweep finishes in seconds rather than
hours.

Marked @pytest.mark.slow — skipped by default. Run with:

    PYTHONPATH=. pytest tests/test_sweep_integration.py --run-slow -v
"""

import json
import os

import pytest

import run_hypothesis_sweep as sweep


FAKE_T_SAT = 20
FAKE_BUDGETS = [3, 6, 20, 40]
FAKE_LR_EXACT = 0.5
FAKE_LR_TRAJ = 1.0
FAKE_TPU = 2
FAKE_DISTANCES = [4]


def _fake_exact_entry(dist, h_type, n_goals):
    return {
        'T_sat': FAKE_T_SAT,
        'T_sat_mean': FAKE_T_SAT,
        'T_sat_max': FAKE_T_SAT,
        'per_seed': [FAKE_T_SAT],
        'budgets': list(FAKE_BUDGETS),
        'saturated': True,
        'best_final_reward': 1.0,
        'horizon': 90,
        'n_goals': n_goals,
        'distance': dist,
        'horizon_type': h_type,
        'lr': FAKE_LR_EXACT,
        'grid_size': sweep.GRID_SIZE,
        'threshold': 0.95,
        'max_budget': 2000,
        'n_seeds': 1,
    }


def _fake_traj_entry(dist, h_type, n_goals, training_mode):
    return {
        'T_sat': FAKE_T_SAT,
        'saturated': True,
        'budgets': list(FAKE_BUDGETS),
        'best_lr': FAKE_LR_TRAJ,
        'best_traj_per_update': FAKE_TPU,
        'best_final_reward': 1.0,
        'all_combos': {},
        'horizon': 90,
        'n_goals': n_goals,
        'distance': dist,
        'horizon_type': h_type,
        'grid_size': sweep.GRID_SIZE,
        'threshold': 0.95,
        'max_budget': 2000,
        'n_seeds': 1,
        'training_mode': training_mode,
    }


@pytest.fixture
def fake_calibration_paths(tmp_path):
    """Writes fake calibration JSONs into tmp_path for all three modes."""
    exact_cal, hybrid_cal, sample_cal = {}, {}, {}
    for dist in FAKE_DISTANCES:
        for h_type in sweep.HORIZON_TYPES:
            for n_goals in (1, 3):
                exact_key = (f"dist={dist}_{h_type}_ng={n_goals}_"
                             f"lr={FAKE_LR_EXACT}_grid={sweep.GRID_SIZE}")
                traj_key = (f"dist={dist}_{h_type}_ng={n_goals}_"
                            f"grid={sweep.GRID_SIZE}")
                exact_cal[exact_key] = _fake_exact_entry(
                    dist, h_type, n_goals)
                hybrid_cal[traj_key] = _fake_traj_entry(
                    dist, h_type, n_goals, 'hybrid')
                sample_cal[traj_key] = _fake_traj_entry(
                    dist, h_type, n_goals, 'sample')

    paths = {
        'exact': str(tmp_path / 'calibration.json'),
        'hybrid': str(tmp_path / 'calibration_hybrid.json'),
        'sample': str(tmp_path / 'calibration_sample.json'),
    }
    for name, cal in [('exact', exact_cal),
                      ('hybrid', hybrid_cal),
                      ('sample', sample_cal)]:
        with open(paths[name], 'w') as f:
            json.dump(cal, f)
    return paths


@pytest.mark.slow
@pytest.mark.parametrize("sweep_mode", ["capability", "zeta"])
@pytest.mark.parametrize("training_mode", ["exact", "hybrid", "sample"])
def test_run_sweep_end_to_end(tmp_path, fake_calibration_paths,
                                sweep_mode, training_mode):
    """Verify run_sweep produces CSV + pickle without error for every
    combo of {sweep_mode, training_mode} in the default suite."""
    out_dir = tmp_path / f"{sweep_mode}_{training_mode}"
    out_dir.mkdir()

    results = sweep.run_sweep(
        mode=sweep_mode,
        output_dir=str(out_dir),
        n_seeds=1,
        n_workers=1,
        training_mode=training_mode,
        calibration_paths=fake_calibration_paths,
        distances=FAKE_DISTANCES,
    )

    assert len(results) > 0, "run_sweep returned no results"

    csv_path = out_dir / f"{sweep_mode}_sweep_results.csv"
    pkl_path = out_dir / f"{sweep_mode}_sweep_results.pkl"
    assert csv_path.exists(), f"expected CSV at {csv_path}"
    assert pkl_path.exists(), f"expected pickle at {pkl_path}"

    with csv_path.open() as f:
        n_lines = sum(1 for _ in f)
    # header + at least a couple of result rows
    assert n_lines >= 3, f"CSV too short ({n_lines} lines)"


@pytest.mark.slow
def test_fake_configs_do_not_touch_real_files(fake_calibration_paths):
    """The fixture writes to tmp_path; assert no real results/ paths were
    mutated."""
    for path in fake_calibration_paths.values():
        # Each fake path should live under a pytest tmp dir, never under
        # the repo's results/ directory.
        real_results = os.path.realpath('results')
        assert os.path.realpath(path).startswith(os.path.realpath(
            os.path.dirname(path))), (
            f"fake path leaked: {path}")
        assert not os.path.realpath(path).startswith(real_results), (
            f"fake path points at real results/: {path}")
