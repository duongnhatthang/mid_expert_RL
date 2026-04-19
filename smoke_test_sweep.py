"""Smoke test for the hypothesis sweep across all three training modes.

Generates tiny fake calibration files in a temp directory (T_sat=20,
tpu=2, lr=1.0 for all configs), then runs run_sweep for each of
{capability, zeta} × {exact, hybrid, sample} to verify the end-to-end
wiring (auto-cal dispatch, training-mode dispatch, CSV output) works.

This NEVER touches results/calibration*.json. Real configs are left alone.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

import run_hypothesis_sweep as sweep


# ---------------------------------------------------------------------------
# Fake calibration payloads. Keep T_sat small so each training run is fast.
# ---------------------------------------------------------------------------

SMOKE_DISTANCES = [4]            # Override DISTANCES for speed
SMOKE_T_SAT = 20
SMOKE_BUDGETS = [3, 6, 20, 40]
SMOKE_LR_EXACT = 0.5
SMOKE_LR_TRAJ = 1.0
SMOKE_TPU = 2


def _fake_exact_entry(dist, h_type, n_goals):
    return {
        'T_sat': SMOKE_T_SAT,
        'T_sat_mean': SMOKE_T_SAT,
        'T_sat_max': SMOKE_T_SAT,
        'per_seed': [SMOKE_T_SAT],
        'budgets': list(SMOKE_BUDGETS),
        'saturated': True,
        'best_final_reward': 1.0,
        'horizon': 90,  # overwritten by run_experiment from thresholds
        'n_goals': n_goals,
        'distance': dist,
        'horizon_type': h_type,
        'lr': SMOKE_LR_EXACT,
        'grid_size': sweep.GRID_SIZE,
        'threshold': 0.95,
        'max_budget': 2000,
        'n_seeds': 1,
    }


def _fake_traj_entry(dist, h_type, n_goals, training_mode):
    return {
        'T_sat': SMOKE_T_SAT,
        'saturated': True,
        'budgets': list(SMOKE_BUDGETS),
        'best_lr': SMOKE_LR_TRAJ,
        'best_traj_per_update': SMOKE_TPU,
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


def _write_fake_configs(cfg_dir):
    """Populate fake calibration files covering every key the sweep asks
    for: distances × horizon_types × {n_goals=1 (zeta), n_goals=3
    (capability)}."""
    exact_cal = {}
    hybrid_cal = {}
    sample_cal = {}
    for dist in SMOKE_DISTANCES:
        for h_type in sweep.HORIZON_TYPES:
            for n_goals in (1, 3):
                exact_key = (f"dist={dist}_{h_type}_ng={n_goals}_"
                             f"lr={SMOKE_LR_EXACT}_grid={sweep.GRID_SIZE}")
                traj_key = (f"dist={dist}_{h_type}_ng={n_goals}_"
                            f"grid={sweep.GRID_SIZE}")
                exact_cal[exact_key] = _fake_exact_entry(dist, h_type, n_goals)
                hybrid_cal[traj_key] = _fake_traj_entry(
                    dist, h_type, n_goals, 'hybrid')
                sample_cal[traj_key] = _fake_traj_entry(
                    dist, h_type, n_goals, 'sample')

    paths = {
        'exact': os.path.join(cfg_dir, 'calibration.json'),
        'hybrid': os.path.join(cfg_dir, 'calibration_hybrid.json'),
        'sample': os.path.join(cfg_dir, 'calibration_sample.json'),
    }
    for name, cal in [('exact', exact_cal),
                      ('hybrid', hybrid_cal),
                      ('sample', sample_cal)]:
        with open(paths[name], 'w') as f:
            json.dump(cal, f, indent=2)
    return paths


def _run_one(sweep_mode, training_mode, output_dir):
    """Run a single tiny sweep and verify a CSV was produced."""
    print(f"\n{'=' * 60}")
    print(f"  sweep_mode={sweep_mode}, training_mode={training_mode}")
    print(f"{'=' * 60}", flush=True)

    sub_out = os.path.join(output_dir, f"{sweep_mode}_{training_mode}")
    os.makedirs(sub_out, exist_ok=True)

    results = sweep.run_sweep(
        mode=sweep_mode,
        output_dir=sub_out,
        n_seeds=1,
        n_workers=1,
        training_mode=training_mode,
    )

    csv_path = os.path.join(sub_out, f"{sweep_mode}_sweep_results.csv")
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    # Sanity: at least a few rows written
    with open(csv_path) as f:
        n_lines = sum(1 for _ in f)
    assert n_lines >= 2, f"CSV too short ({n_lines} lines): {csv_path}"
    print(f"  OK: {len(results)} results, CSV has {n_lines} lines")


def main():
    tmp_root = tempfile.mkdtemp(prefix="smoke_sweep_")
    print(f"[smoke] workspace: {tmp_root}")

    orig = {
        'CALIBRATION_PATH': sweep.CALIBRATION_PATH,
        'HYBRID_CALIBRATION_PATH': sweep.HYBRID_CALIBRATION_PATH,
        'SAMPLE_CALIBRATION_PATH': sweep.SAMPLE_CALIBRATION_PATH,
        'DISTANCES': list(sweep.DISTANCES),
    }

    failures = []
    try:
        cfg_dir = os.path.join(tmp_root, 'configs')
        os.makedirs(cfg_dir, exist_ok=True)
        paths = _write_fake_configs(cfg_dir)

        # Point the sweep module at the fake configs + shrink distances
        sweep.CALIBRATION_PATH = paths['exact']
        sweep.HYBRID_CALIBRATION_PATH = paths['hybrid']
        sweep.SAMPLE_CALIBRATION_PATH = paths['sample']
        sweep.DISTANCES = list(SMOKE_DISTANCES)

        print(f"[smoke] fake calibration paths:")
        for name, p in paths.items():
            print(f"  {name}: {p}")
        print(f"[smoke] DISTANCES overridden to {sweep.DISTANCES}")

        outputs_dir = os.path.join(tmp_root, 'runs')
        os.makedirs(outputs_dir, exist_ok=True)

        for training_mode in ('exact', 'hybrid', 'sample'):
            for sweep_mode in ('capability', 'zeta'):
                try:
                    _run_one(sweep_mode, training_mode, outputs_dir)
                except Exception:
                    failures.append((sweep_mode, training_mode))
                    traceback.print_exc()

    finally:
        # Restore module state so we don't corrupt other imports in the
        # same interpreter (e.g. pytest collection).
        for k, v in orig.items():
            setattr(sweep, k, v)
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL: {len(failures)} combos failed: {failures}")
        sys.exit(1)
    else:
        print("PASS: all 6 combos ran successfully with fake configs.")


if __name__ == '__main__':
    main()
