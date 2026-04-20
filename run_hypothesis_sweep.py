"""
Comprehensive parameter sweep to test the mid-teacher-wins hypothesis.

Three modes:
  --mode zeta        Continuous mixture teacher: mu(zeta) = zeta*pi* + (1-zeta)*uniform
                     Uses 1 goal per distance.
  --mode capability  Discrete knowledge teacher: capacity in {-1, 0, 1, 2, 3}
                     Uses 3 goals per distance (so cap=1,2 are "mid-capacity").
  --mode cap_zeta    Combined: sweep (capacity, zeta) pairs.
                     Uses 3 goals per distance, capacity in {0,1,2,3}, zeta in {0.25,0.5,0.75,1.0}.

Sweeps alpha, budget, horizon, and goal distance on a 9x9 grid.
Budgets are loaded from calibration.json (run run_calibration.py first).
"""

import argparse
import csv
import datetime
import itertools
import json
import multiprocessing as mp
import os
import pickle
import time
import numpy as np
import pandas as pd

from tabular_prototype import (
    run_experiment,
    GridEnv,
    compute_exploration_thresholds,
)
from tabular_prototype.environment import generate_equidistant_goals

# =========================================================================
# Constants
# =========================================================================

GRID_SIZE = 9
DISTANCES = [4, 6, 8]  # 3 values per sweep (standard)

# Zeta mode: 1 goal per distance
ZETA_GOAL_POSITIONS = {
    d: generate_equidistant_goals(9, 1, distance=d) for d in DISTANCES
}

# Capability mode: 3 goals per distance (so cap=1,2 are mid-capacity, cap=3 is best)
CAP_GOAL_POSITIONS = {
    d: generate_equidistant_goals(9, 3, distance=d) for d in DISTANCES
}

ZETA_VALUES = [0.0, 0.33, 0.67, 1.0]
CAP_VALUES = [-1, 0, 1, 2, 3]

# Cap-zeta mode: 3 goals per distance, sweep (capacity, zeta)
CAP_ZETA_GOAL_POSITIONS = CAP_GOAL_POSITIONS  # reuse capability mode goals
CAP_ZETA_CAPACITIES = [0, 1, 2, 3]
CAP_ZETA_ZETAS = [0.25, 0.5, 0.75, 1.0]

ALPHA_VALUES = [0.0, 0.33, 0.67, 1.0]

HORIZON_TYPES = ['small', 'large']

CALIBRATION_PATH = 'results/calibration.json'
HYBRID_CALIBRATION_PATH = 'results/calibration_hybrid.json'
SAMPLE_CALIBRATION_PATH = 'results/calibration_sample.json'

# Fallback budgets if calibration not available
FALLBACK_BUDGETS = [10, 30, 100, 200]

DEFAULT_CALIB_BUDGET = 2000
UNSATURATED_BUDGET_MULTIPLIER = 3


def _get_horizons():
    thresholds = compute_exploration_thresholds(GRID_SIZE)
    return {
        'small': thresholds['horizon_small'],
        'large': thresholds['horizon_large'],
    }


def _goal_positions(mode: str):
    if mode == 'zeta':
        return ZETA_GOAL_POSITIONS
    return CAP_GOAL_POSITIONS  # both 'capability' and 'cap_zeta' use 3-goal positions


def _load_calibrated_budgets(mode: str, n_seeds_calib: int = 3,
                               calibration_path: str = None,
                               distances=None):
    """Load per-config budgets from calibration.json for exact mode.

    Handles three cases per config:
    1. Missing from JSON → auto-calibrate with DEFAULT_CALIB_BUDGET, cache.
    2. Present but unsaturated (T_sat hit budget cap) and was run with
       DEFAULT_CALIB_BUDGET → re-run with 3× budget, cache.
    3. Present and saturated (or already extended) → use as-is; warn if still
       unsaturated after extended run.

    Legacy entries without a 'saturated' flag are treated as saturated (the
    original calibration file was populated before the flag existed).

    Args:
        calibration_path: path to the exact calibration JSON. Defaults to
            the module-level CALIBRATION_PATH.
        distances: iterable of distances to load. Defaults to module-level
            DISTANCES.
    """
    from run_calibration import calibrate_single

    if calibration_path is None:
        calibration_path = CALIBRATION_PATH
    if distances is None:
        distances = DISTANCES

    n_goals = 1 if mode == 'zeta' else 3
    lr = 0.5

    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            calibration = json.load(f)
    else:
        calibration = {}

    budgets_map = {}  # (dist, h_type) -> list of budgets
    updated = False
    for dist in distances:
        for h_type in HORIZON_TYPES:
            key = f"dist={dist}_{h_type}_ng={n_goals}_lr={lr}_grid={GRID_SIZE}"

            needs_run = False
            run_budget = DEFAULT_CALIB_BUDGET

            if key not in calibration:
                needs_run = True
                print(f"  Auto-calibrating missing exact config: {key} ...",
                      end=" ", flush=True)
            elif not calibration[key].get('saturated', True):
                prev_budget = calibration[key].get('max_budget',
                                                    DEFAULT_CALIB_BUDGET)
                extended_budget = (DEFAULT_CALIB_BUDGET
                                   * UNSATURATED_BUDGET_MULTIPLIER)
                if prev_budget < extended_budget:
                    needs_run = True
                    run_budget = extended_budget
                    print(f"  Re-calibrating unsaturated exact config: "
                          f"{key} (prev budget={prev_budget}, extending to "
                          f"{run_budget}) ...", end=" ", flush=True)

            if needs_run:
                result = calibrate_single(
                    dist, h_type, n_goals, lr, GRID_SIZE,
                    n_seeds_calib, run_budget, 0.95,
                )
                calibration[key] = result
                updated = True
                sat_str = ("saturated" if result.get('saturated')
                           else "NOT SATURATED")
                print(f"T_sat={result['T_sat']}, {sat_str}", flush=True)

            c = calibration[key]
            if not c.get('saturated', True):
                print(f"  WARNING: exact {key} did NOT saturate even at "
                      f"budget={c.get('max_budget')}! T_sat={c['T_sat']} is "
                      f"the budget cap, not true saturation. Best reward="
                      f"{c.get('best_final_reward', float('nan')):.3f} < "
                      f"0.95 threshold.", flush=True)

            budgets_map[(dist, h_type)] = c['budgets']

    if updated:
        os.makedirs(os.path.dirname(calibration_path) or '.', exist_ok=True)
        with open(calibration_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"  Saved updated calibration to {calibration_path}",
              flush=True)

    return budgets_map


def _trajectory_calibration_path(training_mode: str) -> str:
    return {
        'hybrid': HYBRID_CALIBRATION_PATH,
        'sample': SAMPLE_CALIBRATION_PATH,
    }[training_mode]


def _load_trajectory_calibration(sweep_mode: str, training_mode: str,
                                  n_seeds_calib: int = 3,
                                  calibration_path: str = None,
                                  distances=None):
    """Load per-config calibration (LR, traj_per_update, budgets) for a
    trajectory-based training mode ("hybrid" or "sample").

    Handles three cases per config:
    1. Missing from JSON → auto-calibrate with DEFAULT_CALIB_BUDGET, cache.
    2. Present but unsaturated (best combo didn't reach threshold) and was
       run with DEFAULT_CALIB_BUDGET → re-run with 3× budget, cache.
    3. Present and saturated (or already extended) → use as-is; warn loudly
       if still unsaturated after extended run.

    Args:
        calibration_path: path to the trajectory calibration JSON. Defaults
            to the module-level path for the given training_mode.
        distances: iterable of distances to load. Defaults to module-level
            DISTANCES.
    """
    from run_calibration import calibrate_trajectory_single

    if calibration_path is None:
        calibration_path = _trajectory_calibration_path(training_mode)
    if distances is None:
        distances = DISTANCES

    n_goals = 1 if sweep_mode == 'zeta' else 3
    cal_path = calibration_path

    if os.path.exists(cal_path):
        with open(cal_path) as f:
            calibration = json.load(f)
    else:
        calibration = {}

    cfg = {}  # (dist, h_type) -> {lr, tpu, budgets}
    updated = False
    for dist in distances:
        for h_type in HORIZON_TYPES:
            key = f"dist={dist}_{h_type}_ng={n_goals}_grid={GRID_SIZE}"

            needs_run = False
            run_budget = DEFAULT_CALIB_BUDGET

            if key not in calibration:
                needs_run = True
                print(f"  Auto-calibrating missing {training_mode} config: "
                      f"{key} ...", end=" ", flush=True)
            elif not calibration[key].get('saturated', True):
                prev_budget = calibration[key].get('max_budget',
                                                    DEFAULT_CALIB_BUDGET)
                extended_budget = (DEFAULT_CALIB_BUDGET
                                   * UNSATURATED_BUDGET_MULTIPLIER)
                if prev_budget < extended_budget:
                    needs_run = True
                    run_budget = extended_budget
                    print(f"  Re-calibrating unsaturated {training_mode} "
                          f"config: {key} (prev budget={prev_budget}, "
                          f"extending to {run_budget}) ...",
                          end=" ", flush=True)

            if needs_run:
                result = calibrate_trajectory_single(
                    dist, h_type, n_goals, GRID_SIZE,
                    n_seeds_calib, run_budget, 0.95,
                    training_mode=training_mode,
                )
                calibration[key] = result
                updated = True
                sat_str = ("saturated" if result.get('saturated')
                           else "NOT SATURATED")
                print(f"lr={result['best_lr']}, "
                      f"tpu={result['best_traj_per_update']}, "
                      f"T_sat={result['T_sat']}, {sat_str}", flush=True)

            c = calibration[key]
            if not c.get('saturated', True):
                print(f"  WARNING: {training_mode} {key} did NOT saturate "
                      f"even at budget={c.get('max_budget')}! "
                      f"T_sat={c['T_sat']} is the budget cap, not true "
                      f"saturation. Best reward="
                      f"{c['best_final_reward']:.3f} < 0.95 threshold.",
                      flush=True)

            cfg[(dist, h_type)] = {
                'lr': c['best_lr'],
                'trajectories_per_update': c['best_traj_per_update'],
                'budgets': c['budgets'],
            }

    if updated:
        os.makedirs(os.path.dirname(cal_path) or '.', exist_ok=True)
        with open(cal_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"  Saved updated calibration to {cal_path}", flush=True)

    return cfg


# =========================================================================
# Sweep runner
# =========================================================================

def _run_single_experiment(args_tuple):
    """Worker function for multiprocessing. Returns result dict."""
    (mode, teacher_val, alpha, budget, h_type, dist, seed,
     horizons, training_mode, lr, traj_per_update) = args_tuple
    goal_pos = _goal_positions(mode)
    goals = goal_pos[dist]
    horizon = horizons[h_type]
    kwargs = dict(
        grid_size=GRID_SIZE,
        goals=goals,
        horizon=horizon,
        sample_budget=budget,
        alpha=alpha,
        lr=lr,
        trajectories_per_update=traj_per_update,
        seed=seed,
        eval_interval=5,
        eval_n_episodes=50,
        mode=training_mode,
    )
    if mode == 'zeta':
        kwargs['zeta'] = teacher_val
        teacher_key = 'zeta'
    elif mode == 'cap_zeta':
        cap, zeta = teacher_val
        kwargs['teacher_capacity'] = cap
        kwargs['zeta'] = zeta
        teacher_key = 'cap_zeta'
    else:
        kwargs['teacher_capacity'] = teacher_val
        teacher_key = 'teacher_capacity'

    result = run_experiment(**kwargs)
    result['distance'] = dist
    result['horizon_type'] = h_type

    if mode == 'cap_zeta':
        result['teacher_capacity'] = cap
        result['zeta'] = zeta
        result['cap_zeta'] = f'cap={cap}_z={zeta}'
    else:
        result[teacher_key] = teacher_val

    # Extract final V^π(s_0) values from history
    if result.get('history'):
        last = result['history'][-1]
        result['final_V_discounted'] = last.get('exact_V_start', float('nan'))
    else:
        result['final_V_discounted'] = float('nan')

    return result


def _write_progress(output_dir, completed, total, elapsed):
    """Write atomic progress.json for monitoring from another tmux pane."""
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    progress = {
        "completed": completed,
        "total": total,
        "pct": round(100 * completed / total, 1) if total > 0 else 0,
        "elapsed_s": round(elapsed, 1),
        "eta_s": round(eta, 1),
        "rate_exp_per_s": round(rate, 3),
    }
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, "progress.json.tmp")
    final_path = os.path.join(output_dir, "progress.json")
    with open(tmp_path, "w") as f:
        json.dump(progress, f)
    os.replace(tmp_path, final_path)


def run_sweep(mode: str, output_dir: str, n_seeds: int, n_workers: int = 1,
              training_mode: str = "exact",
              calibration_paths: dict = None,
              distances=None) -> list:
    """Run all experiments for the given mode, save CSV, return result dicts.

    Args:
        training_mode: one of "exact", "hybrid", "sample" (see
            :func:`tabular_prototype.experiments.run_experiment`).
        calibration_paths: optional dict mapping training_mode → path of the
            calibration JSON to use. Keys: "exact", "hybrid", "sample".
            Missing keys fall back to module-level constants. Lets tests and
            batch jobs point at alternate calibration files without
            mutating module state.
        distances: optional iterable of distances to sweep. Defaults to the
            module-level DISTANCES. Tests use a reduced set to keep runs
            fast.
    """
    if distances is None:
        distances = DISTANCES
    resolved_paths = {
        'exact': CALIBRATION_PATH,
        'hybrid': HYBRID_CALIBRATION_PATH,
        'sample': SAMPLE_CALIBRATION_PATH,
    }
    if calibration_paths:
        resolved_paths.update(calibration_paths)

    horizons = _get_horizons()
    csv_path = os.path.join(output_dir, f'{mode}_sweep_results.csv')

    if mode == 'zeta':
        teacher_values = ZETA_VALUES
    elif mode == 'cap_zeta':
        teacher_values = [
            (cap, z) for cap in CAP_ZETA_CAPACITIES for z in CAP_ZETA_ZETAS
        ]
    else:
        teacher_values = CAP_VALUES

    if training_mode == "exact":
        budgets_map = _load_calibrated_budgets(
            mode,
            calibration_path=resolved_paths['exact'],
            distances=distances,
        )
        default_lr = 0.5
        default_tpu = 10
        traj_config = None
    else:
        traj_config = _load_trajectory_calibration(
            mode, training_mode,
            calibration_path=resolved_paths[training_mode],
            distances=distances,
        )
        budgets_map = (
            {k: v['budgets'] for k, v in traj_config.items()}
            if traj_config else None
        )
        default_lr = 0.1
        default_tpu = 10

    # Build configs with per-(dist, h_type) budgets
    configs = []
    all_budget_sets = set()
    for tv, alpha, h_type, dist, seed in itertools.product(
        teacher_values, ALPHA_VALUES, HORIZON_TYPES, distances, range(n_seeds)
    ):
        if alpha == 0.0 and tv != teacher_values[0]:
            continue

        # Cap=0 with any zeta is always uniform, skip duplicates
        if mode == 'cap_zeta':
            cap, zeta = tv
            if cap == 0 and zeta != CAP_ZETA_ZETAS[0]:
                continue

        budget_list = budgets_map[(dist, h_type)] if budgets_map else FALLBACK_BUDGETS
        for budget in budget_list:
            configs.append((tv, alpha, budget, h_type, dist, seed))
        all_budget_sets.add((dist, h_type, tuple(budget_list)))

    total = len(configs)

    print("=" * 70, flush=True)
    print(f"HYPOTHESIS SWEEP — mode={mode}", flush=True)
    print("=" * 70, flush=True)
    print(f"Teacher values: {teacher_values}", flush=True)
    print(f"Alpha values:   {ALPHA_VALUES}", flush=True)
    print(f"Horizons:       small={horizons['small']}, large={horizons['large']}", flush=True)
    print(f"Distances:      {DISTANCES}", flush=True)
    print(f"Seeds:          {n_seeds}", flush=True)
    print(f"Budgets per config:", flush=True)
    for dist, h_type, blist in sorted(all_budget_sets):
        print(f"  dist={dist}, {h_type}: {list(blist)}", flush=True)
    print(f"Total experiments: {total}", flush=True)
    print(f"Workers:        {n_workers}", flush=True)
    print(flush=True)

    worker_args = []
    for tv, alpha, budget, h_type, dist, seed in configs:
        if traj_config and (dist, h_type) in traj_config:
            sc = traj_config[(dist, h_type)]
            lr = sc['lr']
            tpu = sc['trajectories_per_update']
        else:
            lr = default_lr
            tpu = default_tpu
        worker_args.append(
            (mode, tv, alpha, budget, h_type, dist, seed,
             horizons, training_mode, lr, tpu)
        )

    t0 = time.time()

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            all_results = []
            for i, result in enumerate(pool.imap_unordered(_run_single_experiment, worker_args)):
                all_results.append(result)
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    print(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                          f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                          flush=True)
                    _write_progress(output_dir, i + 1, total, elapsed)
    else:
        all_results = []
        for i, args in enumerate(worker_args):
            if i % 50 == 0:
                elapsed = time.time() - t0
                rate = max(i, 1) / max(elapsed, 0.01)
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{total} ({100*i/total:.0f}%) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                      flush=True)
                _write_progress(output_dir, i, total, elapsed)
            all_results.append(_run_single_experiment(args))

    elapsed = time.time() - t0
    print(f"  Completed {total} experiments in {elapsed:.1f}s ({total/elapsed:.1f} exp/s)",
          flush=True)
    _write_progress(output_dir, total, total, elapsed)

    # Replicate alpha=0 results for all teacher values (teacher is irrelevant at alpha=0).
    # Keep visitation_counts reference so downstream plots show Vanilla NPG visitation for
    # every row. Shallow copy of the np.ndarray is cheap (pickle memoizes identical arrays).
    if mode == 'cap_zeta':
        tcol = 'cap_zeta'
    else:
        tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'

    alpha_zero_results = [r for r in all_results if r.get('alpha') == 0.0]
    for r in list(alpha_zero_results):
        for tv in teacher_values:
            if mode == 'cap_zeta':
                tv_key = f'cap={tv[0]}_z={tv[1]}'
                if tv_key != r.get(tcol):
                    r_copy = r.copy()
                    r_copy[tcol] = tv_key
                    r_copy['teacher_capacity'] = tv[0]
                    r_copy['zeta'] = tv[1]
                    all_results.append(r_copy)
            else:
                if tv != r[tcol]:
                    r_copy = r.copy()
                    r_copy[tcol] = tv
                    all_results.append(r_copy)

    # Save CSV
    if mode == 'cap_zeta':
        fieldnames = ['teacher_capacity', 'zeta', 'cap_zeta', 'alpha', 'sample_budget', 'horizon',
                       'horizon_type', 'distance', 'seed', 'final_mean_reward',
                       'final_V_discounted', 'final_goal_rate',
                       'final_unique_sa', 'final_state_entropy']
    elif mode == 'zeta':
        fieldnames = ['zeta', 'alpha', 'sample_budget', 'horizon', 'horizon_type',
                       'distance', 'seed', 'final_mean_reward', 'final_V_discounted',
                       'final_goal_rate', 'final_unique_sa', 'final_state_entropy']
    else:
        fieldnames = ['teacher_capacity', 'alpha', 'sample_budget', 'horizon', 'horizon_type',
                       'distance', 'seed', 'final_mean_reward', 'final_V_discounted',
                       'final_goal_rate', 'final_unique_sa', 'final_state_entropy']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\nCSV saved to {csv_path}", flush=True)

    # Cache full results (including visitation_counts) for replotting
    cache_path = os.path.join(output_dir, f'{mode}_sweep_results.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Cached full results to {cache_path}", flush=True)

    return all_results


# =========================================================================
# Plotting helpers
# =========================================================================

METRICS = [
    ('final_mean_reward', 'Mean Reward'),
    ('final_V_discounted', 'V\u03c0(s\u2080) discounted'),
]


def _teacher_col(mode: str) -> str:
    if mode == 'zeta':
        return 'zeta'
    if mode == 'cap_zeta':
        return 'cap_zeta'
    return 'teacher_capacity'


def _teacher_label(mode: str, val) -> str:
    """Render a matplotlib-friendly teacher label using LaTeX math where
    applicable. Return values are intended to be consumed as `label=...` on
    plot elements — strings may contain `$...$` math segments."""
    if mode == 'zeta':
        return rf'$\zeta={float(val):.2f}$'
    if mode == 'cap_zeta':
        if isinstance(val, str):
            cap, zeta = _parse_cap_zeta(val)
            return rf'$c={cap},\;\zeta={zeta:.2f}$'
        return str(val)
    iv = int(val)
    if iv == -1:
        return 'no teacher'
    if iv == 0:
        return 'uniform'
    return rf'$c={iv}$'


def _sort_teacher_vals(teacher_series, mode: str):
    """Sort unique teacher values from a DataFrame column.

    Returns list of sorted values. For cap_zeta, strings are parsed as (cap, zeta)
    tuples for ordering but returned as the original string form.
    """
    vals = list(teacher_series.unique())
    if mode == 'cap_zeta':
        def _key(s):
            # 'cap=C_z=Z' -> (C, Z)
            parts = str(s).replace('cap=', '').replace('z=', '').split('_')
            return (int(parts[0]), float(parts[1]))
        return sorted(vals, key=_key)
    return sorted(vals, key=float)


def _teacher_x_positions(teacher_vals, mode: str):
    """Return numeric x-axis positions for teacher values.

    For cap_zeta, uses indices (0, 1, 2, ...). For zeta/capability, uses the
    float values themselves.
    """
    if mode == 'cap_zeta':
        return list(range(len(teacher_vals)))
    return [float(v) for v in teacher_vals]


def _mode_subtitle(mode: str) -> str:
    """Explanation subtitle for α and teacher param."""
    if mode == 'zeta':
        return ('\u03b1: learner reliance on teacher (0=self-only, 1=teacher-only)    '
                '\u03b6: teacher expertise (0=uniform random, 1=optimal)')
    return ('\u03b1: learner reliance on teacher (0=self-only, 1=teacher-only)    '
            'cap: # goals teacher knows (-1=no signal, 0=uniform random policy, 1..3=partial..full)')


# =========================================================================
# Plot functions — each produces side-by-side (mean_reward | V_discounted)
# =========================================================================

def _get_bh_pairs(dist_df):
    """Get sorted (budget, horizon_type, horizon_value) triples that exist in the data."""
    pairs = dist_df.groupby(['sample_budget', 'horizon_type', 'horizon']).size().reset_index()
    return sorted(
        pairs[['sample_budget', 'horizon_type', 'horizon']].values.tolist(),
        key=lambda x: (x[1], x[0]),
    )


def _bh_title(budget, h_type, horizon):
    """Readable subplot title for a (budget, horizon_type, horizon) combo."""
    return f'budget={budget}, H={int(horizon)} ({h_type})'


def plot_heatmaps(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per distance: one row per (budget, horizon) that has data, side-by-side metrics."""
    import matplotlib.pyplot as plt

    tcol = _teacher_col(mode)
    goal_pos = _goal_positions(mode)
    teacher_vals = _sort_teacher_vals(df[tcol], mode)

    for dist in DISTANCES:
        dist_df = df[df['distance'] == dist]
        bh_pairs = _get_bh_pairs(dist_df)
        n_rows = len(bh_pairs)
        if n_rows == 0:
            continue

        n_metrics = len(METRICS)
        fig, axes = plt.subplots(n_rows, n_metrics,
                                 figsize=(5 * n_metrics, 3 * n_rows + 1.5),
                                 squeeze=False)

        for mi, (metric, mlabel) in enumerate(METRICS):
            vmax = max(0.01, dist_df[metric].max())
            for row, (budget, h_type, horizon) in enumerate(bh_pairs):
                ax = axes[row, mi]
                sub = dist_df[(dist_df['sample_budget'] == budget) &
                              (dist_df['horizon_type'] == h_type)]

                pivot = sub.pivot_table(
                    values=metric, index=tcol, columns='alpha', aggfunc='mean'
                )
                pivot = pivot.reindex(index=teacher_vals, columns=ALPHA_VALUES)

                ax.imshow(pivot.values, aspect='auto', cmap='viridis',
                          vmin=0, vmax=vmax)
                for yi in range(pivot.shape[0]):
                    for xi in range(pivot.shape[1]):
                        val = pivot.values[yi, xi]
                        if not np.isnan(val):
                            ax.text(xi, yi, f'{val:.2f}', ha='center',
                                    va='center', fontsize=6,
                                    color='white' if val < vmax * 0.7 else 'black')

                ax.set_xticks(range(len(ALPHA_VALUES)))
                ax.set_xticklabels(
                    ['NPG' if a == 0.0 else str(a) for a in ALPHA_VALUES], fontsize=7)
                ax.set_yticks(range(len(teacher_vals)))
                ax.set_yticklabels(
                    [_teacher_label(mode, v) for v in teacher_vals], fontsize=7)

                ax.set_title(f'{mlabel}\n{_bh_title(budget, h_type, horizon)}', fontsize=8)
                if mi == 0:
                    ax.set_ylabel('Teacher')
                if row == n_rows - 1:
                    ax.set_xlabel('Alpha')

        goals = goal_pos[dist]
        fig.suptitle(f'Heatmap — distance={dist}, goals={goals}\n'
                     f'{_mode_subtitle(mode)}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = os.path.join(figures_dir, f'heatmap_dist{dist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_reward_vs_teacher_cap_zeta(df: pd.DataFrame, figures_dir: str):
    """Cap×Zeta mode: one heatmap per (distance, budget, horizon, alpha) showing
    mean_reward as a function of (capacity, zeta).

    Laid out as a grid per distance: rows=(budget, horizon), cols=alpha,
    with (cap, zeta) heatmaps. cap=0 rows are filled with the single collected
    value across all zeta columns since zeta is a no-op when cap=0.
    """
    import matplotlib.pyplot as plt

    goal_pos = _goal_positions('cap_zeta')
    caps = CAP_ZETA_CAPACITIES
    zetas = CAP_ZETA_ZETAS
    # Parse cap_zeta strings in the df into two integer/float columns for pivoting
    parsed = df['cap_zeta'].astype(str).apply(_parse_cap_zeta)
    df = df.copy()
    df['_cap'] = [p[0] for p in parsed]
    df['_zeta'] = [p[1] for p in parsed]

    metric = 'final_mean_reward'
    mlabel = 'Mean Reward'

    for dist in DISTANCES:
        dist_df = df[df['distance'] == dist]
        bh_pairs = _get_bh_pairs(dist_df)
        if not bh_pairs:
            continue

        n_rows = len(bh_pairs)
        n_cols = len(ALPHA_VALUES)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(2.4 * n_cols + 1.0, 2.2 * n_rows + 1.2),
                                  squeeze=False)

        vmax = max(0.01, dist_df[metric].max())

        def _cell_stats(cell_vals):
            """Return (mean, sem) from a Series of per-seed rewards."""
            n = len(cell_vals)
            if n == 0:
                return np.nan, np.nan
            mean = float(cell_vals.mean())
            sem = float(cell_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            return mean, sem

        for ri, (budget, h_type, horizon) in enumerate(bh_pairs):
            sub = dist_df[(dist_df['sample_budget'] == budget) &
                          (dist_df['horizon_type'] == h_type)]

            # Vanilla NPG baseline (α=0) for this (budget, horizon). Teacher is
            # irrelevant at α=0 and the sweep replicates a single α=0 run
            # across all (cap, ζ) slots, so we MUST deduplicate by seed before
            # averaging — summing 160 replicated floats accumulates enough
            # rounding error that the mean differs from any individual cell's
            # displayed value at 2-decimal precision (e.g. cells show "0.11"
            # but the summed mean rounds to "0.10"). Taking the per-seed mean
            # on deduplicated rows yields exactly the same number that the
            # α=0 column cells display.
            vanilla_rows_all = sub[sub['alpha'] == 0.0]
            if len(vanilla_rows_all):
                vanilla_by_seed = vanilla_rows_all.drop_duplicates(subset='seed')
                vanilla_mean = float(vanilla_by_seed[metric].mean())
            else:
                vanilla_mean = np.nan

            for ci, alpha in enumerate(ALPHA_VALUES):
                ax = axes[ri, ci]
                asub = sub[sub['alpha'] == alpha]
                # Build (cap x zeta) grid of (mean, sem) cells
                grid_mean = np.full((len(caps), len(zetas)), np.nan)
                grid_sem = np.full((len(caps), len(zetas)), np.nan)
                for i, cap in enumerate(caps):
                    cap_rows = asub[asub['_cap'] == cap]
                    if cap == 0:
                        # cap=0 is uniform regardless of zeta: collect all seeds
                        # and replicate across the zeta axis.
                        mean, sem = _cell_stats(cap_rows[metric])
                        grid_mean[i, :] = mean
                        grid_sem[i, :] = sem
                        continue
                    for j, z in enumerate(zetas):
                        cell_vals = cap_rows[np.isclose(cap_rows['_zeta'], z)][metric]
                        mean, sem = _cell_stats(cell_vals)
                        grid_mean[i, j] = mean
                        grid_sem[i, j] = sem

                im = ax.imshow(grid_mean, aspect='auto', cmap='viridis',
                               vmin=0, vmax=vmax, origin='lower')

                # Annotate cells: mean on top, ±sem below, and highlight cells
                # that match or beat the vanilla NPG baseline.
                for yi in range(grid_mean.shape[0]):
                    for xi in range(grid_mean.shape[1]):
                        val = grid_mean[yi, xi]
                        sem = grid_sem[yi, xi]
                        if np.isnan(val):
                            continue
                        text_color = 'white' if val < vmax * 0.6 else 'black'
                        ax.text(xi, yi - 0.15, f'{val:.2f}',
                                ha='center', va='center',
                                fontsize=6, color=text_color)
                        if not np.isnan(sem):
                            ax.text(xi, yi + 0.22,
                                    rf'$\pm${sem:.2f}',
                                    ha='center', va='center',
                                    fontsize=5, color=text_color)
                        # Highlight cells relative to the vanilla NPG baseline.
                        # Skip the α=0 column (trivially equal) and undefined
                        # baselines. Comparison MUST use the same rounding rule
                        # as the cell text (f-string {:.2f}), not Python's
                        # round(), because the two disagree on floating-point
                        # edge cases (e.g. round(0.885, 2) == 0.88 but
                        # f'{0.885:.2f}' == '0.89'). We want the highlight to
                        # match what the reader sees in the cell.
                        if alpha != 0.0 and not np.isnan(vanilla_mean):
                            val_disp = float(f'{val:.2f}')
                            van_disp = float(f'{vanilla_mean:.2f}')
                            if val_disp > van_disp:
                                edge = '#ff0000'  # pure red: strictly better
                                lw = 2.0
                            elif val_disp == van_disp:
                                edge = '#ff7eb6'  # warm pink: equal at display precision
                                lw = 1.6
                            else:
                                edge = None
                            if edge is not None:
                                # Inset rectangle slightly so neighbouring
                                # outlined cells don't visually merge their
                                # borders.
                                inset = 0.08
                                ax.add_patch(plt.Rectangle(
                                    (xi - 0.5 + inset, yi - 0.5 + inset),
                                    1 - 2 * inset, 1 - 2 * inset,
                                    fill=False, edgecolor=edge,
                                    linewidth=lw, zorder=5))

                ax.set_xticks(range(len(zetas)))
                ax.set_yticks(range(len(caps)))
                if ri == n_rows - 1:
                    ax.set_xticklabels([f'{z}' for z in zetas], fontsize=7)
                    ax.set_xlabel(r'$\zeta$', fontsize=9)
                else:
                    ax.set_xticklabels([])
                if ci == 0:
                    ax.set_yticklabels([f'{c}' for c in caps], fontsize=7)
                    ax.set_ylabel(f'{_bh_title(budget, h_type, horizon)}\ncapacity $c$',
                                  fontsize=7)
                else:
                    ax.set_yticklabels([])
                if ri == 0:
                    ax.set_title(
                        'Vanilla NPG' if alpha == 0.0 else rf'$\alpha={alpha}$',
                        fontsize=9)

        fig.subplots_adjust(right=0.90)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label='Mean reward')

        goals = goal_pos[dist]
        fig.suptitle(
            rf'Cap$\times\zeta$ — distance={dist}, goals={goals}'
            '\n'
            r'Rows: (budget, horizon). Columns: $\alpha$. '
            r'Cells: mean reward $\pm$ SEM (10 seeds) as a function of '
            r'$(c,\;\zeta)$.'
            '\n'
            r'Row $c=0$ is uniform-random teacher ($\zeta$ is a no-op), '
            r'replicated across $\zeta$ for display.'
            '\n'
            r'Outlines (vs. vanilla NPG baseline at $\alpha=0$, same (budget, horizon), '
            r'compared at displayed 2-decimal precision): '
            r'RED = strictly better,  PINK = equal.',
            fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.90, 0.92])
        save_path = os.path.join(figures_dir, f'cap_zeta_reward_dist{dist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_reward_vs_teacher(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per distance: one row per (budget, horizon) that has data, side-by-side metrics.

    For cap_zeta mode, delegates to plot_reward_vs_teacher_cap_zeta which renders
    a 2D (capacity × zeta) heatmap/surface per (budget, horizon, alpha) slice.
    """
    if mode == 'cap_zeta':
        return plot_reward_vs_teacher_cap_zeta(df, figures_dir)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    tcol = _teacher_col(mode)
    goal_pos = _goal_positions(mode)
    teacher_vals = _sort_teacher_vals(df[tcol], mode)
    x_positions = _teacher_x_positions(teacher_vals, mode)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(ALPHA_VALUES)))

    for dist in DISTANCES:
        dist_df = df[df['distance'] == dist]
        bh_pairs = _get_bh_pairs(dist_df)
        n_rows = len(bh_pairs)
        if n_rows == 0:
            continue

        n_metrics = len(METRICS)
        dist_has_bf = ('V_backfilled' in dist_df.columns
                       and bool(dist_df['V_backfilled'].any()))
        # Reserve vertical space for the suptitle (more if we need the
        # backfill explanation below it).
        title_in = 1.35 if dist_has_bf else 0.55
        fig_h = 3.5 * n_rows + title_in + 0.2
        fig, axes = plt.subplots(n_rows, n_metrics,
                                 figsize=(5.5 * n_metrics, fig_h),
                                 squeeze=False)

        any_backfilled = False
        for mi, (metric, mlabel) in enumerate(METRICS):
            for row, (budget, h_type, horizon) in enumerate(bh_pairs):
                ax = axes[row, mi]
                sub = dist_df[(dist_df['sample_budget'] == budget) &
                              (dist_df['horizon_type'] == h_type)]

                for ai, alpha in enumerate(ALPHA_VALUES):
                    alpha_sub = sub[sub['alpha'] == alpha]
                    means, sems, bf_flags = [], [], []
                    for tv in teacher_vals:
                        tv_rows = alpha_sub[alpha_sub[tcol] == tv]
                        tv_data = tv_rows[metric]
                        means.append(tv_data.mean())
                        sems.append(tv_data.std() / np.sqrt(max(1, len(tv_data))))
                        if ('V_backfilled' in tv_rows.columns
                                and metric == 'final_V_discounted'):
                            bf_flags.append(bool(tv_rows['V_backfilled'].any()))
                        else:
                            bf_flags.append(False)

                    means_arr = np.array(means)
                    sems_arr = np.array(sems)
                    c = colors[ai]
                    ax.plot(x_positions, means_arr, marker='o', markersize=3,
                            label=f'\u03b1={alpha}', color=c, linewidth=1.5)
                    ax.fill_between(x_positions, means_arr - sems_arr,
                                    means_arr + sems_arr, alpha=0.15, color=c)
                    # Overlay open rings on cells containing backfilled seeds
                    bf_x = [x for x, f in zip(x_positions, bf_flags) if f]
                    bf_y = [m for m, f in zip(means_arr, bf_flags) if f]
                    if bf_x:
                        any_backfilled = True
                        ax.scatter(bf_x, bf_y, s=60, facecolors='none',
                                   edgecolors=c, linewidths=1.2, zorder=5)

                ax.set_title(f'{mlabel}\n{_bh_title(budget, h_type, horizon)}',
                             fontsize=8)
                ax.set_ylabel(mlabel, fontsize=7)
                ax.grid(True, alpha=0.3)

                # Mode-specific x-axis
                if mode == 'zeta':
                    ax.set_xlabel('\u03b6 (teacher expertise)', fontsize=7)
                elif mode == 'cap_zeta':
                    ax.set_xlabel('(capacity, \u03b6)', fontsize=7)
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(
                        [_teacher_label(mode, v) for v in teacher_vals],
                        fontsize=5, rotation=45, ha='right')
                else:
                    ax.set_xlabel('Teacher Capacity', fontsize=7)
                    ax.set_xticks([int(v) for v in teacher_vals])

                if row == 0 and mi == n_metrics - 1:
                    ax.legend(fontsize=6, loc='upper right')

        goals = goal_pos[dist]
        title = (f'Performance vs Teacher — distance={dist}, goals={goals}\n'
                 f'{_mode_subtitle(mode)}')
        if any_backfilled:
            title += (
                '\nOpen rings around markers \u2192 V\u03c0(s\u2080) '
                'reconstructed from training-time \u0394V diagnostics, not '
                'measured directly at the end.\n'
                'V is logged only every few policy updates; at small sample '
                'budgets the run ends before enough updates accumulate to '
                'trigger a logged snapshot.\n'
                'Mean reward is evaluated separately and is unaffected.'
            )
        fig.suptitle(title, fontsize=13, fontweight='bold')
        rect_top = 1.0 - title_in / fig_h
        plt.tight_layout(rect=[0, 0, 1, rect_top])
        save_path = os.path.join(figures_dir, f'reward_vs_teacher_dist{dist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_distance_effect(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per alpha: rows = (horizon_type x budget_rank), cols = metrics.
    Each subplot has one line per teacher value with shaded SEM — clean and readable."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    tcol = _teacher_col(mode)
    teacher_vals = _sort_teacher_vals(df[tcol], mode)
    colors = cm.tab10(np.linspace(0, 1, max(10, len(teacher_vals))))[:len(teacher_vals)]

    budget_rank_labels = ['T_sat/5', 'T_sat/3', 'T_sat', '2\u00d7T_sat']

    for alpha in [0.0, 0.5, 1.0]:
        alpha_df = df[df['alpha'] == alpha]
        if len(alpha_df) == 0:
            continue

        # Determine budget ranks per (horizon_type, distance)
        rank_data = {}  # (h_type, rank) -> exists
        budget_ranks = {}  # (h_type, distance) -> sorted budget list
        for h_type in HORIZON_TYPES:
            h_df = alpha_df[alpha_df['horizon_type'] == h_type]
            for d in DISTANCES:
                d_budgets = sorted(h_df[h_df['distance'] == d]['sample_budget'].unique())
                budget_ranks[(h_type, d)] = d_budgets
                for r in range(len(d_budgets)):
                    rank_data[(h_type, r)] = True

        # Build row list: (h_type, rank)
        row_configs = sorted(rank_data.keys(), key=lambda x: (x[0], x[1]))
        n_rows = len(row_configs)
        n_metrics = len(METRICS)

        fig, axes = plt.subplots(n_rows, n_metrics,
                                 figsize=(5.5 * n_metrics, 3 * n_rows + 1.5),
                                 squeeze=False)

        for mi, (metric, mlabel) in enumerate(METRICS):
            for row_idx, (h_type, rank) in enumerate(row_configs):
                ax = axes[row_idx, mi]
                h_df = alpha_df[alpha_df['horizon_type'] == h_type]
                horizon_val = h_df['horizon'].iloc[0] if len(h_df) > 0 else '?'

                for ti, tv in enumerate(teacher_vals):
                    means, sems, dists_used = [], [], []
                    budgets_used = []
                    for d in DISTANCES:
                        blist = budget_ranks.get((h_type, d), [])
                        if rank >= len(blist):
                            continue
                        budget = blist[rank]
                        d_data = h_df[(h_df['distance'] == d) &
                                      (h_df['sample_budget'] == budget) &
                                      (h_df[tcol] == tv)][metric]
                        if len(d_data) > 0:
                            means.append(d_data.mean())
                            sems.append(d_data.std() / np.sqrt(len(d_data)))
                            dists_used.append(d)
                            budgets_used.append(budget)

                    if not means:
                        continue
                    means_arr = np.array(means)
                    sems_arr = np.array(sems)
                    c = colors[ti]
                    ax.plot(dists_used, means_arr, marker='o', markersize=4,
                            color=c, linewidth=1.5,
                            label=_teacher_label(mode, tv))
                    ax.fill_between(dists_used, means_arr - sems_arr,
                                    means_arr + sems_arr, alpha=0.15, color=c)

                rank_label = budget_rank_labels[rank] if rank < len(budget_rank_labels) else f'rank{rank}'
                ax.set_title(f'{mlabel}\nH={int(horizon_val)} ({h_type}), budget~{rank_label}',
                             fontsize=8)
                ax.set_ylabel(mlabel, fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(DISTANCES)

                if row_idx == 0 and mi == n_metrics - 1:
                    ax.legend(fontsize=6, loc='upper right')
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Goal Distance (Manhattan)', fontsize=8)

        fig.suptitle(f'Distance Effect — \u03b1={alpha}\n'
                     f'{_mode_subtitle(mode)}',
                     fontsize=10, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = os.path.join(figures_dir, f'distance_effect_alpha{alpha}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def _parse_cap_zeta(tv_str: str):
    """Parse 'cap=C_z=Z' string -> (int, float)."""
    parts = str(tv_str).replace('cap=', '').replace('z=', '').split('_')
    return int(parts[0]), float(parts[1])


def _compute_teacher_advantages(mode, teacher_vals, env, goals, gamma):
    """Compute teacher advantage A^mu(s,a) for each teacher value.
    Returns dict: teacher_val -> (n_states, n_actions) array, or None if no teacher."""
    from tabular_prototype.teacher import compute_teacher_values_auto

    advantages = {}
    for tv in teacher_vals:
        if mode == 'zeta':
            Q_mu, V_mu, _ = compute_teacher_values_auto(
                env, env.goals, zeta=float(tv), gamma=gamma)
            advantages[tv] = Q_mu - V_mu[:, None]
        elif mode == 'cap_zeta':
            cap, zeta = _parse_cap_zeta(tv)
            if cap == 0:
                # cap=0 is uniform regardless of zeta
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, env.goals, zeta=0.0, gamma=gamma)
            else:
                known_goals = goals[:cap]
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, known_goals, zeta=zeta, gamma=gamma)
            advantages[tv] = Q_mu - V_mu[:, None]
        else:
            tv_int = int(tv)
            if tv_int == -1:
                advantages[tv] = None
            elif tv_int == 0:
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, env.goals, zeta=0.0, gamma=gamma)
                advantages[tv] = Q_mu - V_mu[:, None]
            else:
                known_goals = goals[:tv_int]
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, known_goals, zeta=1.0, gamma=gamma)
                advantages[tv] = Q_mu - V_mu[:, None]
    return advantages


def plot_visitation_grids_cap_zeta(all_results: list, figures_dir: str):
    """Cap×ζ visitation grids: one figure per (distance, horizon_type, alpha).

    Each figure stacks all budgets vertically. For each budget, rows=cap,
    cols=ζ visitation columns + 1 advantage column. cap=0 row is replicated
    across ζ (no-op); α=0 visitation is shared across all (cap, ζ).
    """
    from collections import defaultdict
    from tabular_prototype.config import compute_gamma_from_horizon
    import matplotlib.pyplot as plt

    caps = CAP_ZETA_CAPACITIES
    zetas = CAP_ZETA_ZETAS
    goal_pos = _goal_positions('cap_zeta')
    horizons = _get_horizons()
    budget_vals_all = sorted({r['sample_budget'] for r in all_results})

    for dist in DISTANCES:
        goals = goal_pos[dist]
        env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizons['large'])

        for h_type in HORIZON_TYPES:
            horizon_val = horizons[h_type]
            gamma = compute_gamma_from_horizon(horizon_val)

            # Pre-compute teacher advantages once per (dist, h_type) — they don't
            # depend on budget or alpha.
            tv_strs = [f'cap={c}_z={z}' for c in caps for z in zetas]
            teacher_advantages = _compute_teacher_advantages(
                'cap_zeta', tv_strs, env, goals, gamma)
            # Store raw A matrices for per-action compass-rose rendering, plus
            # per-state action-variance grids. Both share color scales over the
            # whole figure (all (c, ζ) pairs).
            A_full = {}
            var_full = {}
            vabs_adv = 0.0
            vmax_var = 0.0
            for c in caps:
                for z in zetas:
                    A = teacher_advantages.get(f'cap={c}_z={z}')
                    if A is None:
                        continue
                    A_full[(c, z)] = A
                    vabs_adv = max(vabs_adv, float(np.abs(A).max()))
                    vg = _variance_grid(A, env)
                    var_full[(c, z)] = vg
                    vmax_var = max(vmax_var, float(vg.max()))
            if vabs_adv == 0:
                vabs_adv = 1.0
            if vmax_var == 0:
                vmax_var = 1.0

            # Collect groups: (budget, cap, zeta, alpha) -> list of visitations
            raw_groups = defaultdict(list)
            for r in all_results:
                if (r.get('distance') != dist or
                    r.get('horizon_type') != h_type or
                    'visitation_counts' not in r):
                    continue
                cap = r.get('teacher_capacity')
                zeta = r.get('zeta')
                if cap is None or zeta is None:
                    continue
                raw_groups[(
                    r['sample_budget'], int(cap), float(zeta), float(r.get('alpha', 0.0))
                )].append(r['visitation_counts'])

            if not raw_groups:
                continue

            budget_vals = sorted({k[0] for k in raw_groups})

            # Fill cap=0 across ζ and α=0 across all (cap, ζ) — backward-compat
            # for old pickles that popped visitation_counts on replication.
            for budget in budget_vals:
                cap0_per_alpha = defaultdict(list)
                for (b, cap, z, a), vs in raw_groups.items():
                    if b == budget and cap == 0:
                        cap0_per_alpha[a].extend(vs)
                for a, vs in cap0_per_alpha.items():
                    if not vs:
                        continue
                    for z in zetas:
                        raw_groups.setdefault((budget, 0, z, a), list(vs))

                any_alpha0 = []
                for (b, cap, z, a), vs in list(raw_groups.items()):
                    if b == budget and a == 0.0:
                        any_alpha0.extend(vs)
                if any_alpha0:
                    for c in caps:
                        for z in zetas:
                            raw_groups.setdefault((budget, c, z, 0.0), list(any_alpha0))

            visitation_data = {k: np.mean(v, axis=0) for k, v in raw_groups.items()}

            # One plot per alpha: (budget × cap) rows, (zeta + 1 adv) cols
            for alpha in ALPHA_VALUES:
                alpha_vis = {(b, c, z): v for (b, c, z, a), v in visitation_data.items()
                             if a == alpha}
                if not alpha_vis:
                    continue

                # Per-budget vmax (budgets have very different visit scales)
                vmax_vis_per_budget = {}
                vis_grids = {}
                for (b, c, z), vis in alpha_vis.items():
                    grid = vis.sum(axis=1).reshape(env.grid_size, env.grid_size)
                    vis_grids[(b, c, z)] = grid
                    vmax_vis_per_budget[b] = max(
                        vmax_vis_per_budget.get(b, 0), grid.max())

                n_cap = len(caps)
                n_zeta = len(zetas)
                n_budget = len(budget_vals)
                # Triple per (cap, ζ) cell: visitation | per-action advantage | variance
                n_cols = 3 * n_zeta
                n_rows = n_budget * n_cap

                cell_size = 1.0
                fig_w = cell_size * n_cols + 4.0
                fig_h = cell_size * n_rows + 1.6
                fig, axes = plt.subplots(n_rows, n_cols,
                                          figsize=(fig_w, fig_h),
                                          squeeze=False)

                vis_mappables = {}  # budget -> im
                adv_mappable = None
                var_mappable = None

                for bi, budget in enumerate(budget_vals):
                    vmax_vis = vmax_vis_per_budget.get(budget, 0) or 1
                    for ci, c in enumerate(caps):
                        global_row = bi * n_cap + ci
                        for zi, z in enumerate(zetas):
                            vis_col = 3 * zi
                            adv_col = 3 * zi + 1
                            var_col = 3 * zi + 2

                            # Visitation
                            ax_v = axes[global_row, vis_col]
                            grid = vis_grids.get((budget, c, z))
                            if grid is not None:
                                im = ax_v.imshow(grid, cmap='hot', origin='upper',
                                                 vmin=0, vmax=vmax_vis)
                                vis_mappables[budget] = im
                                _annotate_grid(ax_v, env, goals, compact=True)
                            else:
                                ax_v.axis('off')
                            ax_v.set_xticks([])
                            ax_v.set_yticks([])

                            # Per-action advantage (compass rose) for THIS (cap, ζ)
                            ax_a = axes[global_row, adv_col]
                            A = A_full.get((c, z))
                            if A is not None:
                                sm = _draw_per_action_advantage(
                                    ax_a, env, A, vmin=-vabs_adv, vmax=vabs_adv)
                                adv_mappable = sm
                                _annotate_grid(ax_a, env, goals, compact=True)
                            else:
                                ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                          transform=ax_a.transAxes, fontsize=8, color='gray')
                                ax_a.set_xlim(0, 1)
                                ax_a.set_ylim(0, 1)
                            ax_a.set_xticks([])
                            ax_a.set_yticks([])

                            # Variance of advantage across actions
                            ax_var = axes[global_row, var_col]
                            vg = var_full.get((c, z))
                            if vg is not None:
                                im_var = ax_var.imshow(vg, cmap='viridis',
                                                       origin='upper',
                                                       vmin=0, vmax=vmax_var)
                                var_mappable = im_var
                                _annotate_grid(ax_var, env, goals, compact=True)
                            else:
                                ax_var.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                            transform=ax_var.transAxes, fontsize=8, color='gray')
                                ax_var.set_xlim(0, 1)
                                ax_var.set_ylim(0, 1)
                            ax_var.set_xticks([])
                            ax_var.set_yticks([])

                            if global_row == 0:
                                ax_v.set_title('visit', fontsize=7)
                                ax_a.set_title('adv', fontsize=7)
                                ax_var.set_title('var', fontsize=7)

                fig.subplots_adjust(
                    left=0.13, right=0.80, bottom=0.04, top=0.82,
                    wspace=0.06, hspace=0.30)

                # Single ζ header centered over each (vis, adv, var) triple in the top row
                for zi, z in enumerate(zetas):
                    vis_col = 3 * zi
                    var_col = 3 * zi + 2
                    ax_v = axes[0, vis_col]
                    ax_end = axes[0, var_col]
                    bbox_v = ax_v.get_position()
                    bbox_end = ax_end.get_position()
                    x_center = (bbox_v.x0 + bbox_end.x1) / 2.0
                    y_top = bbox_v.y1 + 0.025
                    fig.text(x_center, y_top, rf'$\zeta={z}$',
                             ha='center', va='bottom', fontsize=11, fontweight='bold')

                # Row labels (c=0, c=1, ...) via fig.text just left of the first column
                for bi, budget in enumerate(budget_vals):
                    for ci, c in enumerate(caps):
                        global_row = bi * n_cap + ci
                        ax = axes[global_row, 0]
                        bbox = ax.get_position()
                        y_center = (bbox.y0 + bbox.y1) / 2.0
                        fig.text(bbox.x0 - 0.005, y_center, rf'$c={c}$',
                                 ha='right', va='center', fontsize=10)

                # "Budget=X" band label on the far left, centered per band
                for bi, budget in enumerate(budget_vals):
                    first = axes[bi * n_cap, 0]
                    last = axes[bi * n_cap + n_cap - 1, 0]
                    bbox_first = first.get_position()
                    bbox_last = last.get_position()
                    y_center = (bbox_first.y1 + bbox_last.y0) / 2.0
                    fig.text(0.015, y_center, rf'$T={budget}$',
                             ha='left', va='center',
                             fontsize=11, fontweight='bold')

                # Per-budget visit colorbars aligned with each budget band.
                # Three colorbar columns spaced to avoid tick-label collisions:
                #   visit x=0.83, adv x=0.90, var x=0.96
                for bi, budget in enumerate(budget_vals):
                    im = vis_mappables.get(budget)
                    if im is None:
                        continue
                    first = axes[bi * n_cap, 0]
                    last = axes[bi * n_cap + n_cap - 1, 0]
                    bbox_first = first.get_position()
                    bbox_last = last.get_position()
                    y0 = bbox_last.y0
                    height = bbox_first.y1 - bbox_last.y0
                    cax = fig.add_axes([0.83, y0, 0.010, height])
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_label('Visits', fontsize=8)
                    cb.ax.tick_params(labelsize=6)

                if adv_mappable is not None:
                    top_bbox = axes[0, 0].get_position()
                    bot_bbox = axes[-1, 0].get_position()
                    y0 = bot_bbox.y0
                    height = top_bbox.y1 - bot_bbox.y0
                    cax_adv = fig.add_axes([0.90, y0, 0.010, height])
                    cb_adv = fig.colorbar(adv_mappable, cax=cax_adv)
                    cb_adv.set_label(r'$A^{\mu}(s,a)$', fontsize=9)
                    cb_adv.ax.tick_params(labelsize=7)

                if var_mappable is not None:
                    top_bbox = axes[0, 0].get_position()
                    bot_bbox = axes[-1, 0].get_position()
                    y0 = bot_bbox.y0
                    height = top_bbox.y1 - bot_bbox.y0
                    cax_var = fig.add_axes([0.96, y0, 0.010, height])
                    cb_var = fig.colorbar(var_mappable, cax=cax_var)
                    cb_var.set_label(r'$\mathrm{Var}_a\,A^{\mu}$', fontsize=9)
                    cb_var.ax.tick_params(labelsize=7)

                # Section separators in figure coordinates. Thick horizontal
                # lines split budget bands; thin vertical lines split each
                # (vis|adv|var) ζ triple from the next so dense grids stay parseable.
                from matplotlib.lines import Line2D
                grid_left = axes[0, 0].get_position().x0
                grid_right = axes[0, n_cols - 1].get_position().x1
                for bi in range(n_budget - 1):
                    bottom_row = bi * n_cap + n_cap - 1
                    top_row = (bi + 1) * n_cap
                    bot_bbox = axes[bottom_row, 0].get_position()
                    top_bbox = axes[top_row, 0].get_position()
                    y = (bot_bbox.y0 + top_bbox.y1) / 2.0
                    fig.add_artist(Line2D(
                        [grid_left, grid_right], [y, y],
                        color='black', linewidth=1.3, zorder=10))
                # Vertical separators between each ζ triple (zi=0..n_zeta-2)
                grid_top = axes[0, 0].get_position().y1
                grid_bot = axes[-1, 0].get_position().y0
                for zi in range(n_zeta - 1):
                    right_of_triple = axes[0, 3 * zi + 2].get_position()
                    left_of_next = axes[0, 3 * zi + 3].get_position()
                    x = (right_of_triple.x1 + left_of_next.x0) / 2.0
                    fig.add_artist(Line2D(
                        [x, x], [grid_bot, grid_top],
                        color='black', linewidth=1.0, zorder=10))

                alpha_label = 'Vanilla NPG' if alpha == 0.0 else rf'$\alpha={alpha}$'
                fig.suptitle(
                    rf'Cap$\times\zeta$ visitation + teacher advantage — '
                    rf'distance={dist}, horizon={horizon_val} ({h_type}), {alpha_label}'
                    '\n'
                    r'Each $(c,\;\zeta)$ cell: visitation (left), per-action '
                    r'$A^{\mu}(s,a)$ compass rose (middle), '
                    r'$\mathrm{Var}_a\,A^{\mu}(s,\cdot)$ (right). '
                    r'Rows grouped by budget $T$ (per-budget visit colour scale).',
                    fontsize=11, fontweight='bold', y=0.97)

                save_path = os.path.join(
                    figures_dir,
                    f'cap_zeta_visit_dist{dist}_{h_type}_alpha{alpha}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)


def plot_visitation_grids(all_results: list, mode: str, figures_dir: str):
    """Consolidated visitation grids + teacher advantage column.

    For cap_zeta mode, delegates to plot_visitation_grids_cap_zeta which produces
    (capacity × zeta) grid figures per (distance, budget, horizon, alpha).

    For zeta/capability modes, produces one figure per (distance, horizon_type).
    The figure stacks all budgets vertically using thick horizontal separators.
    Each budget band is a (teacher × alpha) grid of visitation minicells with a
    shared teacher-advantage column on the right. This reduces the figure count
    from ~40 (4 dist × 4 budgets × 2 horizons) down to 8 (4 dist × 2 horizons).
    """
    if mode == 'cap_zeta':
        plot_visitation_grids_cap_zeta(all_results, figures_dir)
        return

    from collections import defaultdict
    from tabular_prototype.config import compute_gamma_from_horizon
    import matplotlib.pyplot as plt

    tcol = _teacher_col(mode)
    goal_pos = _goal_positions(mode)
    horizons = _get_horizons()

    for dist in DISTANCES:
        goals = goal_pos[dist]
        env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizons['large'])

        for h_type in HORIZON_TYPES:
            # Collect (budget, teacher_val, alpha) -> list of visitation arrays
            groups = defaultdict(list)
            for r in all_results:
                if (r.get('distance') == dist and
                    r.get('horizon_type') == h_type and
                    'visitation_counts' in r):
                    key = (r['sample_budget'], r[tcol], r['alpha'])
                    groups[key].append(r['visitation_counts'])

            if not groups:
                continue

            visitation_data = {k: np.mean(v, axis=0) for k, v in groups.items()}

            budget_vals = sorted({k[0] for k in visitation_data})
            if mode == 'zeta':
                row_keys = sorted({k[1] for k in visitation_data})
                row_label_fn = lambda k: rf'$\zeta={float(k):.2f}$'
            else:
                row_keys = sorted({k[1] for k in visitation_data})
                cap_labels = {-1: 'no teacher', 0: 'uniform'}
                row_label_fn = lambda k: cap_labels.get(int(k), rf'$c={int(k)}$')

            col_keys = sorted({k[2] for k in visitation_data})
            col_label_fn = lambda a: 'Vanilla NPG' if a == 0.0 else rf'$\alpha={a}$'

            horizon_val = horizons[h_type]
            gamma = compute_gamma_from_horizon(horizon_val)
            teacher_advantages = _compute_teacher_advantages(
                mode, row_keys, env, goals, gamma)

            n_tv = len(row_keys)
            n_alpha = len(col_keys)
            n_budget = len(budget_vals)
            # Two extra columns: per-action advantage (compass rose) + variance
            n_cols = n_alpha + 2
            ADV_COL = n_alpha
            VAR_COL = n_alpha + 1
            n_rows = n_budget * n_tv

            cell_size = 1.4
            fig_w = cell_size * n_cols + 4.0  # extra room for labels + colorbars
            fig_h = cell_size * n_rows + 1.6
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(fig_w, fig_h),
                                      squeeze=False)

            # Compute shared vmax PER BUDGET (different budgets have very different
            # visitation magnitudes, so sharing globally would wash out small budgets)
            vmax_vis_per_budget = {}
            vis_grids = {}
            for (budget, rk, ck), vis in visitation_data.items():
                grid = vis.sum(axis=1).reshape(env.grid_size, env.grid_size)
                vis_grids[(budget, rk, ck)] = grid
                vmax_vis_per_budget[budget] = max(
                    vmax_vis_per_budget.get(budget, 0), grid.max())

            # Per-action advantage + variance share colormaps across all rows
            # (neither depends on budget). Advantage uses symmetric diverging
            # scale; variance uses viridis (>=0).
            A_matrices = {}
            var_grids = {}
            vabs_adv = 0.0
            vmax_var = 0.0
            for rk in row_keys:
                A = teacher_advantages.get(rk)
                if A is not None:
                    A_matrices[rk] = A
                    vabs_adv = max(vabs_adv, float(np.abs(A).max()))
                    vg = _variance_grid(A, env)
                    var_grids[rk] = vg
                    vmax_var = max(vmax_var, float(vg.max()))
            if vabs_adv == 0:
                vabs_adv = 1.0
            if vmax_var == 0:
                vmax_var = 1.0

            vis_mappables = {}  # budget -> matplotlib image (for per-budget colorbars)
            adv_mappable = None
            var_mappable = None

            for bi, budget in enumerate(budget_vals):
                vmax_vis = vmax_vis_per_budget.get(budget, 0) or 1
                for ri, rk in enumerate(row_keys):
                    global_row = bi * n_tv + ri
                    for ci, ck in enumerate(col_keys):
                        ax = axes[global_row, ci]
                        grid = vis_grids.get((budget, rk, ck))
                        if grid is not None:
                            im = ax.imshow(grid, cmap='hot', origin='upper',
                                           vmin=0, vmax=vmax_vis)
                            vis_mappables[budget] = im
                            _annotate_grid(ax, env, goals, compact=True)
                        else:
                            ax.axis('off')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if global_row == 0:
                            ax.set_title(col_label_fn(ck), fontsize=9)

                    # Per-action advantage column (compass rose)
                    ax_adv = axes[global_row, ADV_COL]
                    A = A_matrices.get(rk)
                    if A is not None:
                        sm = _draw_per_action_advantage(
                            ax_adv, env, A, vmin=-vabs_adv, vmax=vabs_adv)
                        adv_mappable = sm
                        _annotate_grid(ax_adv, env, goals, compact=True)
                    else:
                        ax_adv.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=ax_adv.transAxes, fontsize=9, color='gray')
                        ax_adv.set_xlim(0, 1)
                        ax_adv.set_ylim(0, 1)
                    ax_adv.set_xticks([])
                    ax_adv.set_yticks([])
                    if global_row == 0:
                        ax_adv.set_title(r'$A^{\mu}(s,a)$ per action', fontsize=9)

                    # Variance column: Var_a A^mu(s, .)
                    ax_var = axes[global_row, VAR_COL]
                    vg = var_grids.get(rk)
                    if vg is not None:
                        im_var = ax_var.imshow(vg, cmap='viridis', origin='upper',
                                               vmin=0, vmax=vmax_var)
                        var_mappable = im_var
                        _annotate_grid(ax_var, env, goals, compact=True)
                    else:
                        ax_var.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=ax_var.transAxes, fontsize=9, color='gray')
                        ax_var.set_xlim(0, 1)
                        ax_var.set_ylim(0, 1)
                    ax_var.set_xticks([])
                    ax_var.set_yticks([])
                    if global_row == 0:
                        ax_var.set_title(r'$\mathrm{Var}_a\,A^{\mu}(s,\cdot)$', fontsize=9)

            # Layout: leave room on left for band label + row labels, on right
            # for per-budget visit colorbars + advantage + variance colorbars,
            # on top for the 2-line suptitle.
            fig.subplots_adjust(
                left=0.14, right=0.80, bottom=0.04, top=0.88,
                wspace=0.10, hspace=0.25)

            # Use fig.text (not set_ylabel) for row labels so they render reliably
            # in dense grids. Position each label just left of the first column.
            for bi, budget in enumerate(budget_vals):
                for ri, rk in enumerate(row_keys):
                    global_row = bi * n_tv + ri
                    ax = axes[global_row, 0]
                    bbox = ax.get_position()
                    y_center = (bbox.y0 + bbox.y1) / 2.0
                    fig.text(bbox.x0 - 0.005, y_center, row_label_fn(rk),
                             ha='right', va='center', fontsize=9)

            # "Budget=X" band label further left, centered on each budget band
            for bi, budget in enumerate(budget_vals):
                first = axes[bi * n_tv, 0]
                last = axes[bi * n_tv + n_tv - 1, 0]
                bbox_first = first.get_position()
                bbox_last = last.get_position()
                y_center = (bbox_first.y1 + bbox_last.y0) / 2.0
                fig.text(0.015, y_center, rf'$T={budget}$',
                         ha='left', va='center',
                         fontsize=11, fontweight='bold')

            # Pull axes up tight against the suptitle BEFORE reading their
            # positions for colorbars / separators — otherwise subplots_adjust
            # afterwards moves the axes but leaves the figure-coord artists
            # misaligned.
            fig.subplots_adjust(top=1.0 - 1.0 / fig_h)

            # Per-budget visit colorbars precisely aligned with each budget band.
            # Three colorbar columns are spaced so tick labels don't collide:
            #   visit: x=0.83  (per band)
            #   adv:   x=0.90  (shared)
            #   var:   x=0.96  (shared)
            for bi, budget in enumerate(budget_vals):
                im = vis_mappables.get(budget)
                if im is None:
                    continue
                first = axes[bi * n_tv, 0]
                last = axes[bi * n_tv + n_tv - 1, 0]
                bbox_first = first.get_position()
                bbox_last = last.get_position()
                y0 = bbox_last.y0
                height = bbox_first.y1 - bbox_last.y0
                cax = fig.add_axes([0.83, y0, 0.010, height])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label('Visits', fontsize=8)
                cb.ax.tick_params(labelsize=6)

            if adv_mappable is not None:
                top_bbox = axes[0, 0].get_position()
                bot_bbox = axes[-1, 0].get_position()
                y0 = bot_bbox.y0
                height = top_bbox.y1 - bot_bbox.y0
                cax_adv = fig.add_axes([0.90, y0, 0.010, height])
                cb_adv = fig.colorbar(adv_mappable, cax=cax_adv)
                cb_adv.set_label(r'$A^{\mu}(s,a)$', fontsize=9)
                cb_adv.ax.tick_params(labelsize=7)

            if var_mappable is not None:
                top_bbox = axes[0, 0].get_position()
                bot_bbox = axes[-1, 0].get_position()
                y0 = bot_bbox.y0
                height = top_bbox.y1 - bot_bbox.y0
                cax_var = fig.add_axes([0.96, y0, 0.010, height])
                cb_var = fig.colorbar(var_mappable, cax=cax_var)
                cb_var.set_label(r'$\mathrm{Var}_a\,A^{\mu}$', fontsize=9)
                cb_var.ax.tick_params(labelsize=7)

            # Section separators drawn in figure coordinates so they can cross
            # subplot boundaries. Horizontal lines split the stacked budget
            # bands; vertical lines separate α columns from the adv/var panels.
            from matplotlib.lines import Line2D
            grid_left = axes[0, 0].get_position().x0
            grid_right = axes[0, n_cols - 1].get_position().x1
            for bi in range(n_budget - 1):
                bottom_row = bi * n_tv + n_tv - 1
                top_row = (bi + 1) * n_tv
                bot_bbox = axes[bottom_row, 0].get_position()
                top_bbox = axes[top_row, 0].get_position()
                y = (bot_bbox.y0 + top_bbox.y1) / 2.0
                fig.add_artist(Line2D(
                    [grid_left, grid_right], [y, y],
                    color='black', linewidth=1.3, zorder=10))
            # Vertical separator between α-block and adv column
            right_alpha = axes[0, n_alpha - 1].get_position()
            adv_col_bbox = axes[0, ADV_COL].get_position()
            x_sep = (right_alpha.x1 + adv_col_bbox.x0) / 2.0
            grid_top = axes[0, 0].get_position().y1
            grid_bot = axes[-1, 0].get_position().y0
            fig.add_artist(Line2D(
                [x_sep, x_sep], [grid_bot, grid_top],
                color='black', linewidth=1.3, zorder=10))
            # Vertical separator between adv and var columns
            adv_bbox = axes[0, ADV_COL].get_position()
            var_bbox = axes[0, VAR_COL].get_position()
            x_sep2 = (adv_bbox.x1 + var_bbox.x0) / 2.0
            fig.add_artist(Line2D(
                [x_sep2, x_sep2], [grid_bot, grid_top],
                color='black', linewidth=0.8, linestyle='--', zorder=10))

            # Title placement (axes already pulled up via subplots_adjust
            # above, so here we just position the suptitle near the top).
            title_top = 1.0 - 0.35 / fig_h
            fig.suptitle(
                rf'State visitation + teacher advantage — distance={dist}, '
                rf'horizon={horizon_val} ({h_type})'
                '\n'
                rf'Rows grouped by budget $T\in\{{{",".join(str(b) for b in budget_vals)}\}}$. '
                r'Each budget band has its own visit colour scale.',
                fontsize=11, fontweight='bold', y=title_top)

            save_path = os.path.join(
                figures_dir,
                f'visitation_grid_dist{dist}_{h_type}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {save_path}")


def _draw_per_action_advantage(ax, env, A, vmin, vmax, cmap_name='RdBu_r'):
    """Draw per-action advantage as a compass rose: 4 triangles per cell.

    Cell (i, j) is split into 4 triangles meeting at the center:
      - Top triangle    → A(s, 0)  (UP)
      - Bottom triangle → A(s, 1)  (DOWN)
      - Left triangle   → A(s, 2)  (LEFT)
      - Right triangle  → A(s, 3)  (RIGHT)

    Args:
        A: shape (n_states, n_actions) — advantage values.
        vmin, vmax: shared color range (use symmetric around 0 for diverging).
    Returns:
        A ScalarMappable for colorbar use.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    G = env.grid_size

    # Vectorize: 4 triangles per cell, one PolyCollection for the whole grid.
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    cx = jj.ravel().astype(float)
    cy = ii.ravel().astype(float)
    tl = np.stack([cx - 0.5, cy - 0.5], axis=1)
    tr = np.stack([cx + 0.5, cy - 0.5], axis=1)
    bl = np.stack([cx - 0.5, cy + 0.5], axis=1)
    br = np.stack([cx + 0.5, cy + 0.5], axis=1)
    cc = np.stack([cx, cy], axis=1)

    # Triangle stacks — shape (G*G, 3, 2) per action
    tri_up = np.stack([tl, tr, cc], axis=1)
    tri_dn = np.stack([bl, br, cc], axis=1)
    tri_lf = np.stack([tl, bl, cc], axis=1)
    tri_rt = np.stack([tr, br, cc], axis=1)

    state_idx = np.array([env.state_to_idx((int(i), int(j)))
                          for i, j in zip(ii.ravel(), jj.ravel())])
    a_up, a_dn, a_lf, a_rt = A[state_idx, 0], A[state_idx, 1], A[state_idx, 2], A[state_idx, 3]

    verts = np.concatenate([tri_up, tri_dn, tri_lf, tri_rt], axis=0)
    vals = np.concatenate([a_up, a_dn, a_lf, a_rt], axis=0)

    pc = PolyCollection(verts, array=vals, cmap=cmap, norm=norm,
                        edgecolors='gray', linewidths=0.15)
    ax.add_collection(pc)
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_aspect('equal')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def _variance_grid(A, env):
    """Return per-state variance of advantage across actions, shaped to grid."""
    return A.var(axis=1).reshape(env.grid_size, env.grid_size)


def _annotate_grid(ax, env, goals, compact: bool = False):
    """Add start and goal markers to a grid subplot.

    Uses patches only (no text overlays) — text inside small cells renders
    unreliably and often clips outside the imshow area. Start is a cyan dot,
    goals are green outline squares.

    Args:
        compact: If True, use smaller marker sizes suitable for dense grids.
    """
    import matplotlib.patches as patches
    start = env.start
    # Clip annotations strictly to the imshow extent so markers never stray
    # outside the cell (fixes "random" G/S glyphs that appeared in empty axes).
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(env.grid_size - 0.5, -0.5)  # matches origin='upper'

    start_r = 0.18 if compact else 0.25
    goal_lw = 1.0 if compact else 1.5
    goal_half = 0.30 if compact else 0.35

    ax.add_patch(patches.Circle(
        (start[1], start[0]), start_r,
        fill=True, facecolor='cyan', edgecolor='black',
        linewidth=0.4, alpha=0.9, zorder=5))
    if goals:
        for goal in goals:
            ax.add_patch(patches.Rectangle(
                (goal[1] - goal_half, goal[0] - goal_half),
                2 * goal_half, 2 * goal_half,
                fill=False, edgecolor='lime', linewidth=goal_lw, zorder=5))


DELTA_V_SUBTITLE = (
    r'Each line shows the TOTAL per-step change in student value at the start state: '
    r'$\Delta V^{\pi}(s_0)\;=\;V^{\pi_{\mathrm{new}}}(s_0)\;-\;V^{\pi_{\mathrm{old}}}(s_0)$'
    r' (one curve per teacher value).'
    '\n'
    r'Full NPG update: '
    r'$\theta\leftarrow\theta+\eta\,[(1-\alpha)\,Q^{\pi}+\alpha\,A^{\mu}]$, '
    r'so $\Delta V^{\pi}(s_0)\approx\Delta_{Q^{\pi}}+\Delta_{A^{\mu}}$.'
    '\n'
    r'Decomposition: $\Delta_{Q^{\pi}}$ is the hypothetical value change from a '
    r'Qπ-only update $\theta\leftarrow\theta+\eta\,Q^{\pi}$ evaluated from the same '
    r"$\theta_{\mathrm{old}}$; $\Delta_{A^{\mu}}\equiv\Delta V^{\pi}(s_0)-\Delta_{Q^{\pi}}$."
    '\n'
    r'(Equality is approximate since softmax couples the two directions.)'
    '\n'
    r'AUC box (upper-right of each subplot, top-5 only): '
    r'$\int\Delta V^{\pi}(s_0)\,dt$ '
    r'(trapezoidal rule) $\approx V^{\pi_{\mathrm{final}}}(s_0)-V^{\pi_{\mathrm{init}}}(s_0)$, '
    r'ranked best$\to$worst — larger $=$ more total improvement.'
)


def _avg_diagnostics_by_tv(group_diags: dict) -> dict:
    """Average per-step diagnostic dicts across seeds for each teacher value.

    Args:
        group_diags: {teacher_val: [diag_list_seed1, diag_list_seed2, ...]}

    Returns:
        {teacher_val: [avg_diag_per_step, ...]}
    """
    averaged = {}
    for tv, seed_diag_lists in group_diags.items():
        n_steps = min(len(d) for d in seed_diag_lists)
        avg_diags = []
        for step_i in range(n_steps):
            step_dicts = [d[step_i] for d in seed_diag_lists]
            avg = {k: float(np.mean([s[k] for s in step_dicts]))
                   for k in step_dicts[0]
                   if isinstance(step_dicts[0][k], (int, float))}
            avg['step'] = step_i
            avg_diags.append(avg)
        averaged[tv] = avg_diags
    return averaged


def _plot_consolidated_diagnostics(
    groups: dict, mode: str, metric_keys: list, plot_kind: str,
    title: str, ylabel: str, save_path: str,
):
    """Grid plot: rows=budgets, cols=alphas, one line per teacher value.

    metric_keys: list of (key, legend_label) to plot. For Δ-V plot_kind='delta_v',
    uses stacked bars for qpi+amu with total as a line overlay.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    budgets = sorted({k[2] for k in groups})
    alphas = sorted({k[3] for k in groups if k[3] != 0.0})
    if not budgets or not alphas:
        return

    # Collect all teacher vals appearing in this group
    tv_set = set()
    for diags_by_tv in groups.values():
        tv_set.update(diags_by_tv.keys())
    teacher_vals = _sort_teacher_vals(pd.Series(list(tv_set)), mode)
    if not teacher_vals:
        return
    colors = cm.tab10(np.linspace(0, 1, max(10, len(teacher_vals))))[:len(teacher_vals)]
    tv_color = dict(zip(teacher_vals, colors))

    n_rows = len(budgets)
    n_cols = len(alphas)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.0 * n_cols, 2.2 * n_rows + 1.2),
                              squeeze=False, sharex='col')

    for ri, budget in enumerate(budgets):
        for ci, alpha in enumerate(alphas):
            ax = axes[ri, ci]
            key = next((k for k in groups if k[2] == budget and k[3] == alpha), None)
            if key is None:
                ax.set_visible(False)
                continue
            diags_by_tv = groups[key]

            if plot_kind == 'delta_v':
                # One thin line per teacher value with total Δ-V only (stacked bars
                # would be unreadable at this density). We also compute the AUC
                # (cumulative ΔV across all update steps) per teacher and show
                # it as a small annotation so baselines can be compared at a
                # glance — larger AUC = larger total improvement in V^π(s₀).
                auc_by_tv = {}
                for tv in teacher_vals:
                    if tv not in diags_by_tv:
                        continue
                    diags = diags_by_tv[tv]
                    steps = [d['step'] for d in diags]
                    dv_total = [d['delta_v_total'] for d in diags]
                    ax.plot(steps, dv_total, color=tv_color[tv],
                            linewidth=1.2, label=_teacher_label(mode, tv))
                    # AUC via trapezoidal rule over the step grid. This
                    # approximates ∫ΔV dt and, since ΔV is per-step change in
                    # V^π(s₀), telescopes to V^π_final(s₀) − V^π_init(s₀).
                    if len(steps) >= 2:
                        # np.trapezoid replaces np.trapz in numpy 2.x; fall
                        # back to trapz for older numpy.
                        trap = getattr(np, 'trapezoid', None) or np.trapz
                        auc_by_tv[tv] = float(trap(dv_total, steps))
                    elif len(steps) == 1:
                        auc_by_tv[tv] = float(dv_total[0])
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                # Rank teachers by AUC (largest first) and show only the
                # TOP 5 in the upper-right corner — listing all teachers
                # (16 for cap_zeta) blocks too much of the curves. The
                # suptitle notes that we truncate to top-5.
                if auc_by_tv:
                    ranked = sorted(auc_by_tv.items(), key=lambda x: -x[1])
                    top_n = 5
                    lines = [
                        rf'{_teacher_label(mode, tv)}: {auc:+.2f}'
                        for tv, auc in ranked[:top_n]
                    ]
                    suffix = ''
                    if len(ranked) > top_n:
                        suffix = f'\n(+{len(ranked) - top_n} more)'
                    auc_text = f'Top-{min(top_n, len(ranked))} AUC\n' + '\n'.join(lines) + suffix
                    ax.text(0.98, 0.98, auc_text,
                            transform=ax.transAxes,
                            ha='right', va='top', fontsize=5,
                            family='monospace',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white',
                                      edgecolor='lightgray',
                                      alpha=0.85))
            else:
                for tv in teacher_vals:
                    if tv not in diags_by_tv:
                        continue
                    diags = diags_by_tv[tv]
                    steps = [d['step'] for d in diags]
                    for mk_i, (metric, style) in enumerate(metric_keys):
                        vals = [d.get(metric, 0.0) for d in diags]
                        # Only label the first metric line per teacher — the
                        # suptitle explains what each line style represents.
                        line_label = (_teacher_label(mode, tv)
                                      if mk_i == 0 else None)
                        ax.plot(steps, vals, color=tv_color[tv],
                                linewidth=1.2, linestyle=style,
                                label=line_label)
                if plot_kind == 'cosine':
                    # Cosine is in [-1, 1]; pin the axis and draw zero line so
                    # orthogonal directions are visually obvious.
                    ax.set_ylim(-1.05, 1.05)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(r'$\alpha=' + f'{alpha}$', fontsize=9)
            if ci == 0:
                ax.set_ylabel(rf'$T={budget}$' + '\n' + ylabel, fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel('NPG update step', fontsize=8)

    # Single legend for the whole figure
    handles_labels = {}
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in handles_labels:
                handles_labels[l] = h
    if handles_labels:
        fig.legend(handles_labels.values(), handles_labels.keys(),
                   loc='lower center', ncol=min(len(handles_labels), 6),
                   fontsize=7, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_sweep_diagnostics(all_results: list, mode: str, figures_dir: str):
    """Generate consolidated diagnostic plots from sweep results.

    Produces one plot per (distance, horizon_type) × metric:
      - magnitude_dist{D}_{h_type}.png   — ||(1-α)Q^π|| and ||α A^μ|| L2 norms
      - delta_v_dist{D}_{h_type}.png     — per-step ΔV^π(s₀)
      - entropy_dist{D}_{h_type}.png     — policy entropy at s₀

    Each plot is a grid of (budget × alpha) subplots with one line per teacher value.
    A^μ distribution stats are not plotted (they are constant — A^μ is fixed).
    """
    from collections import defaultdict

    tcol = _teacher_col(mode)
    diag_dir = os.path.join(figures_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    # Group: (dist, h_type, budget, alpha) -> {teacher_val: [seed_diag_lists]}
    raw_groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if not r.get('diagnostics'):
            continue
        key = (r.get('distance'), r.get('horizon_type'),
               r.get('sample_budget'), r.get('alpha', 0.0))
        raw_groups[key][r[tcol]].append(r['diagnostics'])

    # Average diagnostics per (dist, h_type, budget, alpha) -> tv -> [avg_step_diags]
    avg_groups = {k: _avg_diagnostics_by_tv(v) for k, v in raw_groups.items()}

    # Further organize by (dist, h_type)
    by_dist_h = defaultdict(dict)
    for (dist, h_type, budget, alpha), tv_diags in avg_groups.items():
        by_dist_h[(dist, h_type)][(dist, h_type, budget, alpha)] = tv_diags

    for (dist, h_type), sub_groups in by_dist_h.items():
        suffix = f'dist{dist}_{h_type}'
        base_title = f'distance={dist}, horizon={h_type}'

        _plot_consolidated_diagnostics(
            sub_groups, mode,
            metric_keys=[('q_pi_l2', '-'), ('a_mu_l2', '--')],
            plot_kind='magnitude',
            title=(
                r'Signal magnitude ($L_2$) — ' + base_title + '\n'
                r'Solid: $\|(1-\alpha)\,Q^{\pi}\|_2$,'
                r'   Dashed: $\|\alpha\,A^{\mu}\|_2$.'
                r'  One colour per teacher value (see legend).'
            ),
            ylabel=r'$L_2$ norm',
            save_path=os.path.join(diag_dir, f'magnitude_{suffix}.png'),
        )

        _plot_consolidated_diagnostics(
            sub_groups, mode,
            metric_keys=[],
            plot_kind='delta_v',
            title=(
                r'Per-step $\Delta V^{\pi}(s_0)$ — ' + base_title + '\n'
                + DELTA_V_SUBTITLE
            ),
            ylabel=r'$\Delta V^{\pi}(s_0)$',
            save_path=os.path.join(diag_dir, f'delta_v_{suffix}.png'),
        )

        _plot_consolidated_diagnostics(
            sub_groups, mode,
            metric_keys=[('policy_entropy_start', '-')],
            plot_kind='entropy',
            title=(
                r'Policy entropy at $s_0$ — ' + base_title + '\n'
                r'$H(\pi(\cdot\mid s_0))$ over NPG update steps, '
                r'one curve per teacher value.'
            ),
            ylabel=r'$H(\pi(\cdot\mid s_0))$  (nats)',
            save_path=os.path.join(diag_dir, f'entropy_{suffix}.png'),
        )

        _plot_consolidated_diagnostics(
            sub_groups, mode,
            metric_keys=[('cos_q_a', '-')],
            plot_kind='cosine',
            title=(
                r'Update-direction cosine similarity — ' + base_title + '\n'
                r'$\cos\theta$ between $(1-\alpha)\,Q^{\pi}$ and '
                r'$\alpha\,A^{\mu}$, flattened over all $(s,a)$, per NPG step.'
                '\n'
                r'$\cos\to 1$: teacher AMPLIFIES $Q^{\pi}$ direction. '
                r'$\cos\to 0$: orthogonal. '
                r'$\cos\to -1$: teacher REDIRECTS against $Q^{\pi}$.'
            ),
            ylabel=r'$\cos\theta$',
            save_path=os.path.join(diag_dir, f'cosine_{suffix}.png'),
        )

    print(f"Diagnostic plots saved to {diag_dir}/")


def plot_exact_vs_sample_comparison(
    exact_csv: str, sample_csv: str, mode: str, figures_dir: str
):
    """Side-by-side and difference heatmaps comparing exact vs sample."""
    import matplotlib.pyplot as plt

    df_exact = pd.read_csv(exact_csv)
    df_sample = pd.read_csv(sample_csv)

    tcol = _teacher_col(mode)
    teacher_vals = _sort_teacher_vals(df_exact[tcol], mode)
    os.makedirs(figures_dir, exist_ok=True)

    for dist in DISTANCES:
        de = df_exact[df_exact['distance'] == dist]
        ds = df_sample[df_sample['distance'] == dist]

        for h_type in HORIZON_TYPES:
            he = de[de['horizon_type'] == h_type]
            hs = ds[ds['horizon_type'] == h_type]

            budgets_e = sorted(he['sample_budget'].unique())
            budgets_s = sorted(hs['sample_budget'].unique())
            n_budgets = min(len(budgets_e), len(budgets_s))
            if n_budgets == 0:
                continue

            fig, axes = plt.subplots(n_budgets, 3, figsize=(15, 4 * n_budgets),
                                      squeeze=False)

            for bi in range(n_budgets):
                for mi, (df_cur, budget, label) in enumerate([
                    (he, budgets_e[bi], f'Exact (budget={budgets_e[bi]})'),
                    (hs, budgets_s[bi], f'Sample (budget={budgets_s[bi]})'),
                ]):
                    ax = axes[bi, mi]
                    sub = df_cur[df_cur['sample_budget'] == budget]
                    pivot = sub.pivot_table(
                        values='final_mean_reward', index=tcol,
                        columns='alpha', aggfunc='mean'
                    )
                    pivot = pivot.reindex(index=teacher_vals, columns=ALPHA_VALUES)
                    vmax = max(0.01, pivot.values[~np.isnan(pivot.values)].max()) \
                        if not np.all(np.isnan(pivot.values)) else 1.0

                    ax.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
                    for yi in range(pivot.shape[0]):
                        for xi in range(pivot.shape[1]):
                            val = pivot.values[yi, xi]
                            if not np.isnan(val):
                                ax.text(xi, yi, f'{val:.2f}', ha='center', va='center',
                                        fontsize=6, color='white' if val < vmax * 0.7 else 'black')
                    ax.set_title(label, fontsize=8)
                    ax.set_xticks(range(len(ALPHA_VALUES)))
                    ax.set_xticklabels([str(a) for a in ALPHA_VALUES], fontsize=7)
                    ax.set_yticks(range(len(teacher_vals)))
                    ax.set_yticklabels([_teacher_label(mode, v) for v in teacher_vals], fontsize=7)

                # Difference heatmap
                ax_diff = axes[bi, 2]
                sub_e = he[he['sample_budget'] == budgets_e[bi]]
                sub_s = hs[hs['sample_budget'] == budgets_s[bi]]
                pivot_e = sub_e.pivot_table(values='final_mean_reward', index=tcol,
                                            columns='alpha', aggfunc='mean')
                pivot_s = sub_s.pivot_table(values='final_mean_reward', index=tcol,
                                            columns='alpha', aggfunc='mean')
                pivot_e = pivot_e.reindex(index=teacher_vals, columns=ALPHA_VALUES)
                pivot_s = pivot_s.reindex(index=teacher_vals, columns=ALPHA_VALUES)
                diff = pivot_s.values - pivot_e.values
                vabs = max(0.01, np.nanmax(np.abs(diff)))

                ax_diff.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
                for yi in range(diff.shape[0]):
                    for xi in range(diff.shape[1]):
                        val = diff[yi, xi]
                        if not np.isnan(val):
                            ax_diff.text(xi, yi, f'{val:+.2f}', ha='center', va='center',
                                         fontsize=6)
                ax_diff.set_title(f'Diff (Sample - Exact)', fontsize=8)
                ax_diff.set_xticks(range(len(ALPHA_VALUES)))
                ax_diff.set_xticklabels([str(a) for a in ALPHA_VALUES], fontsize=7)
                ax_diff.set_yticks(range(len(teacher_vals)))
                ax_diff.set_yticklabels([_teacher_label(mode, v) for v in teacher_vals], fontsize=7)

            fig.suptitle(f'Exact vs Sample: dist={dist}, {h_type} horizon',
                         fontsize=11, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(figures_dir, f'comparison_dist{dist}_{h_type}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {save_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive hypothesis sweep (zeta or capability)')
    parser.add_argument('--mode', choices=['zeta', 'capability', 'cap_zeta'], required=True,
                        help='Teacher parameterization to sweep')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/{mode}_sweep/<timestamp>)')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds per config')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, use CPU count for max)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip experiments, only generate plots from existing CSV')
    parser.add_argument('--all-plots', action='store_true',
                        help='Also generate heatmaps and distance_effect plots (off by default)')
    parser.add_argument('--training-mode', type=str, default='exact',
                        choices=['exact', 'hybrid', 'sample'],
                        help='Training mode: exact NPG (default), hybrid '
                             '(trajectory gradient + exact Q^π), or sample '
                             '(trajectory gradient + MC return estimates).')
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/{args.mode}_sweep/{timestamp}'

    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f'{args.mode}_sweep_results.csv')

    pkl_path = os.path.join(args.output_dir, f'{args.mode}_sweep_results.pkl')

    if args.skip_run:
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV not found at {csv_path}. Run without --skip-run first.")
            return
        # Try to load cached full results (with visitation data)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                all_results = pickle.load(f)
            print(f"Loaded cached results from {pkl_path} ({len(all_results)} entries)")
        else:
            all_results = None
            print(f"No pickle cache found at {pkl_path} — visitation grids will be skipped")
    else:
        all_results = run_sweep(args.mode, args.output_dir, args.n_seeds,
                                args.n_workers,
                                training_mode=args.training_mode)

    # Load CSV for plotting
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}")

    print("\nGenerating plots...")
    plot_reward_vs_teacher(df, args.mode, figures_dir)

    if all_results is not None:
        print("\nGenerating visitation grids...")
        plot_visitation_grids(all_results, args.mode, figures_dir)

        print("\nGenerating diagnostic plots...")
        _plot_sweep_diagnostics(all_results, args.mode, figures_dir)
    else:
        print("\nSkipping visitation grids (no pickle cache with visitation data)")

    if args.all_plots:
        print("\nGenerating additional plots (heatmaps, distance_effect)...")
        plot_heatmaps(df, args.mode, figures_dir)
        plot_distance_effect(df, args.mode, figures_dir)

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
