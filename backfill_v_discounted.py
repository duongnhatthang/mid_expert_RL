"""Backfill final_V_discounted for rows with NaN from saved pickles.

Root cause: pre-patch, hybrid/sample mode only recorded history when
`update_count % eval_interval == 0`, so at small budgets where update_count
never reached eval_interval, history stayed empty and final_V_discounted
was left as NaN in the CSV.

Backfill: V_final(s0) = V_uniform(s0) + Σ delta_v_total
  - V_uniform(s0) computed exactly from the env under theta=0 (uniform
    policy), discounted with gamma = 1 - 1/horizon.
  - delta_v_total per update is recorded in diagnostics[] and is the exact
    change in V^π(s0) under the discounted γ across that update.

Therefore this backfill is mathematically exact, not an approximation.

Usage:
    python backfill_v_discounted.py \
        --sweep-dir results/full_suite_20260419_105552/zeta_hybrid \
        --mode zeta

Writes:
    <sweep-dir>/<mode>_sweep_results.csv  (overwrites with V_backfilled col,
    original preserved at <mode>_sweep_results.csv.orig if not already saved)
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd

from tabular_prototype.environment import GridEnv
from tabular_prototype.training import compute_student_qvalues
from tabular_prototype.student import TabularSoftmaxPolicy
import run_hypothesis_sweep as sweep


def _v_uniform_start(grid_size, goals, horizon):
    """Compute V^π(s0) under the uniform policy (theta=0), discounted by
    γ = 1 - 1/horizon."""
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)  # theta=0 → uniform
    gamma = 1.0 - 1.0 / horizon
    _, V_pi = compute_student_qvalues(env, policy, gamma)
    start_idx = env.state_to_idx(env.start)
    return float(V_pi[start_idx])


def backfill(sweep_dir, mode):
    pkl_path = os.path.join(sweep_dir, f'{mode}_sweep_results.pkl')
    csv_path = os.path.join(sweep_dir, f'{mode}_sweep_results.csv')
    orig_csv = csv_path + '.orig'

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    df = pd.read_csv(csv_path)

    # Build V_uniform cache keyed by (distance, horizon). Goals come from
    # sweep._goal_positions which maps sweep_mode → {dist: [goals]}.
    goal_pos = sweep._goal_positions(mode)
    v_uniform_cache = {}

    # Add V_backfilled column if absent
    if 'V_backfilled' not in df.columns:
        df['V_backfilled'] = False

    tcol = sweep._teacher_col(mode)
    n_backfilled = 0
    n_still_nan = 0

    for r in results:
        if not np.isnan(r.get('final_V_discounted', np.nan)):
            continue  # already have V, nothing to do

        dist = r['distance']
        horizon = r['horizon']
        key = (dist, horizon)
        if key not in v_uniform_cache:
            v_uniform_cache[key] = _v_uniform_start(
                sweep.GRID_SIZE, goal_pos[dist], horizon)
        v_init = v_uniform_cache[key]

        diags = r.get('diagnostics') or []
        if not diags:
            n_still_nan += 1
            continue

        delta_sum = float(sum(d.get('delta_v_total', 0.0) for d in diags))
        v_final = v_init + delta_sum

        # Locate matching CSV row
        if mode == 'cap_zeta':
            teacher_val = f"cap={r['teacher_capacity']}_z={r['zeta']}"
        else:
            teacher_val = r['zeta'] if mode == 'zeta' else r['teacher_capacity']

        mask = (
            (df[tcol] == teacher_val) &
            (df['alpha'] == r['alpha']) &
            (df['sample_budget'] == r['sample_budget']) &
            (df['horizon'] == r['horizon']) &
            (df['horizon_type'] == r['horizon_type']) &
            (df['distance'] == r['distance']) &
            (df['seed'] == r['seed'])
        )
        n_match = int(mask.sum())
        if n_match != 1:
            print(f"  WARN: expected 1 CSV row match, got {n_match} for "
                  f"seed={r['seed']} budget={r['sample_budget']} "
                  f"horizon={r['horizon']} dist={r['distance']} "
                  f"{tcol}={teacher_val}")
            continue

        df.loc[mask, 'final_V_discounted'] = v_final
        df.loc[mask, 'V_backfilled'] = True
        n_backfilled += 1

    if not os.path.exists(orig_csv):
        shutil.copy2(csv_path, orig_csv)
        print(f"  Saved original at {orig_csv}")

    df.to_csv(csv_path, index=False)
    print(f"  Backfilled {n_backfilled} rows, "
          f"{n_still_nan} rows remain NaN (no diagnostics), "
          f"wrote {csv_path}")
    return n_backfilled, n_still_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', required=True)
    ap.add_argument('--mode', required=True, choices=['zeta', 'capability', 'cap_zeta'])
    args = ap.parse_args()
    backfill(args.sweep_dir, args.mode)


if __name__ == '__main__':
    main()
