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
DISTANCES = [4, 6, 7, 8]

# Zeta mode: 1 goal per distance
ZETA_GOAL_POSITIONS = {
    d: generate_equidistant_goals(9, 1, distance=d) for d in DISTANCES
}

# Capability mode: 3 goals per distance (so cap=1,2 are mid-capacity, cap=3 is best)
CAP_GOAL_POSITIONS = {
    d: generate_equidistant_goals(9, 3, distance=d) for d in DISTANCES
}

ZETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
CAP_VALUES = [-1, 0, 1, 2, 3]

# Cap-zeta mode: 3 goals per distance, sweep (capacity, zeta)
CAP_ZETA_GOAL_POSITIONS = CAP_GOAL_POSITIONS  # reuse capability mode goals
CAP_ZETA_CAPACITIES = [0, 1, 2, 3]
CAP_ZETA_ZETAS = [0.25, 0.5, 0.75, 1.0]

ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

HORIZON_TYPES = ['small', 'large']

CALIBRATION_PATH = 'results/calibration.json'

# Fallback budgets if calibration not available
FALLBACK_BUDGETS = [10, 30, 100, 200]


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


def _load_calibrated_budgets(mode: str):
    """Load per-config budgets from calibration.json."""
    n_goals = 1 if mode == 'zeta' else 3
    lr = 0.5

    if not os.path.exists(CALIBRATION_PATH):
        print("=" * 70, flush=True)
        print("WARNING: Calibration file not found!", flush=True)
        print(f"  Expected: {CALIBRATION_PATH}", flush=True)
        print(f"  Using fallback budgets: {FALLBACK_BUDGETS}", flush=True)
        print("  Run calibration first: PYTHONPATH=. python run_calibration.py", flush=True)
        print("=" * 70, flush=True)
        return None

    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)

    budgets_map = {}  # (dist, h_type) -> list of budgets
    for dist in DISTANCES:
        for h_type in HORIZON_TYPES:
            key = f"dist={dist}_{h_type}_ng={n_goals}_lr={lr}_grid={GRID_SIZE}"
            if key in calibration:
                budgets_map[(dist, h_type)] = calibration[key]['budgets']
            else:
                print(f"WARNING: No calibration for {key}. Using fallback.", flush=True)
                budgets_map[(dist, h_type)] = FALLBACK_BUDGETS

    return budgets_map


SAMPLE_CALIBRATION_PATH = 'results/calibration_sample.json'


def _load_sample_calibration(mode: str):
    """Load per-config sample calibration (LR, traj_per_update, budgets)."""
    n_goals = 1 if mode == 'zeta' else 3

    if not os.path.exists(SAMPLE_CALIBRATION_PATH):
        print("ERROR: Sample calibration not found!", flush=True)
        print(f"  Run: PYTHONPATH=. python run_calibration.py --mode sample", flush=True)
        return None

    with open(SAMPLE_CALIBRATION_PATH) as f:
        calibration = json.load(f)

    sample_config = {}  # (dist, h_type) -> {lr, tpu, budgets}
    for dist in DISTANCES:
        for h_type in HORIZON_TYPES:
            key = f"dist={dist}_{h_type}_ng={n_goals}_grid={GRID_SIZE}"
            if key in calibration:
                c = calibration[key]
                sample_config[(dist, h_type)] = {
                    'lr': c['best_lr'],
                    'trajectories_per_update': c['best_traj_per_update'],
                    'budgets': c['budgets'],
                }
            else:
                print(f"WARNING: No sample calibration for {key}.", flush=True)
                sample_config[(dist, h_type)] = {
                    'lr': 0.1,
                    'trajectories_per_update': 10,
                    'budgets': FALLBACK_BUDGETS,
                }

    return sample_config


# =========================================================================
# Sweep runner
# =========================================================================

def _run_single_experiment(args_tuple):
    """Worker function for multiprocessing. Returns result dict."""
    (mode, teacher_val, alpha, budget, h_type, dist, seed,
     horizons, exact_gradient, lr, traj_per_update) = args_tuple
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
        exact_gradient=exact_gradient,
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
              exact_gradient: bool = True) -> list:
    """Run all experiments for the given mode, save CSV, return result dicts."""
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

    if exact_gradient:
        budgets_map = _load_calibrated_budgets(mode)
        default_lr = 0.5
        default_tpu = 10
        sample_config = None
    else:
        sample_config = _load_sample_calibration(mode)
        budgets_map = {k: v['budgets'] for k, v in sample_config.items()} if sample_config else None
        default_lr = 0.1
        default_tpu = 10

    # Build configs with per-(dist, h_type) budgets
    configs = []
    all_budget_sets = set()
    for tv, alpha, h_type, dist, seed in itertools.product(
        teacher_values, ALPHA_VALUES, HORIZON_TYPES, DISTANCES, range(n_seeds)
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
        if sample_config and (dist, h_type) in sample_config:
            sc = sample_config[(dist, h_type)]
            lr = sc['lr']
            tpu = sc['trajectories_per_update']
        else:
            lr = default_lr
            tpu = default_tpu
        worker_args.append(
            (mode, tv, alpha, budget, h_type, dist, seed,
             horizons, exact_gradient, lr, tpu)
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
            # irrelevant at α=0, so we average all seeds across all teacher cells.
            vanilla_rows = sub[sub['alpha'] == 0.0][metric]
            vanilla_mean = float(vanilla_rows.mean()) if len(vanilla_rows) else np.nan

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
                        # Highlight cells whose mean matches/beats vanilla NPG.
                        # Skip the α=0 column itself (trivially equal) and the
                        # cells where vanilla baseline is undefined.
                        if (alpha != 0.0 and not np.isnan(vanilla_mean)
                                and val >= vanilla_mean - 1e-9):
                            ax.add_patch(plt.Rectangle(
                                (xi - 0.5, yi - 0.5), 1, 1,
                                fill=False, edgecolor='red',
                                linewidth=1.6, zorder=5))

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
            r'replicated across $\zeta$ for display. '
            r'Red outline: cell $\geq$ vanilla NPG ($\alpha=0$) baseline for the same (budget, horizon).',
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
        fig, axes = plt.subplots(n_rows, n_metrics,
                                 figsize=(5.5 * n_metrics, 3.5 * n_rows + 1.5),
                                 squeeze=False)

        for mi, (metric, mlabel) in enumerate(METRICS):
            for row, (budget, h_type, horizon) in enumerate(bh_pairs):
                ax = axes[row, mi]
                sub = dist_df[(dist_df['sample_budget'] == budget) &
                              (dist_df['horizon_type'] == h_type)]

                for ai, alpha in enumerate(ALPHA_VALUES):
                    alpha_sub = sub[sub['alpha'] == alpha]
                    means, sems = [], []
                    for tv in teacher_vals:
                        tv_data = alpha_sub[alpha_sub[tcol] == tv][metric]
                        means.append(tv_data.mean())
                        sems.append(tv_data.std() / np.sqrt(max(1, len(tv_data))))

                    means_arr = np.array(means)
                    sems_arr = np.array(sems)
                    c = colors[ai]
                    ax.plot(x_positions, means_arr, marker='o', markersize=3,
                            label=f'\u03b1={alpha}', color=c, linewidth=1.5)
                    ax.fill_between(x_positions, means_arr - sems_arr,
                                    means_arr + sems_arr, alpha=0.15, color=c)

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
        fig.suptitle(f'Performance vs Teacher — distance={dist}, goals={goals}\n'
                     f'{_mode_subtitle(mode)}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
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
            adv_grids_full = {}
            vmax_adv = 0
            for c in caps:
                for z in zetas:
                    A = teacher_advantages.get(f'cap={c}_z={z}')
                    if A is None:
                        continue
                    g = A.max(axis=1).reshape(env.grid_size, env.grid_size)
                    adv_grids_full[(c, z)] = g
                    vmax_adv = max(vmax_adv, np.abs(g).max())
            if vmax_adv == 0:
                vmax_adv = 1

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
                # Two side-by-side heatmaps per (cap, ζ) cell: visitation | advantage
                n_cols = 2 * n_zeta
                n_rows = n_budget * n_cap

                cell_size = 1.0
                fig_w = cell_size * n_cols + 4.0
                fig_h = cell_size * n_rows + 1.6
                fig, axes = plt.subplots(n_rows, n_cols,
                                          figsize=(fig_w, fig_h),
                                          squeeze=False)

                vis_mappables = {}  # budget -> im
                adv_mappable = None

                for bi, budget in enumerate(budget_vals):
                    vmax_vis = vmax_vis_per_budget.get(budget, 0) or 1
                    for ci, c in enumerate(caps):
                        global_row = bi * n_cap + ci
                        for zi, z in enumerate(zetas):
                            vis_col = 2 * zi
                            adv_col = 2 * zi + 1

                            # Left: visitation heatmap
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

                            # Right: advantage heatmap for THIS (cap, ζ) pair
                            ax_a = axes[global_row, adv_col]
                            adv_grid = adv_grids_full.get((c, z))
                            if adv_grid is not None:
                                im_a = ax_a.imshow(adv_grid, cmap='hot', origin='upper',
                                                   vmin=0, vmax=vmax_adv)
                                adv_mappable = im_a
                                _annotate_grid(ax_a, env, goals, compact=True)
                            else:
                                ax_a.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                          transform=ax_a.transAxes, fontsize=8, color='gray')
                                ax_a.set_xlim(0, 1)
                                ax_a.set_ylim(0, 1)
                            ax_a.set_xticks([])
                            ax_a.set_yticks([])

                            if global_row == 0:
                                # Column titles: a single ζ label spans the vis|adv pair,
                                # and each sub-column is annotated "vis" / "adv".
                                ax_v.set_title('visit', fontsize=7)
                                ax_a.set_title('adv', fontsize=7)

                fig.subplots_adjust(
                    left=0.13, right=0.86, bottom=0.04, top=0.92,
                    wspace=0.06, hspace=0.30)

                # Single ζ header centered over each (vis, adv) pair in the top row
                for zi, z in enumerate(zetas):
                    vis_col = 2 * zi
                    adv_col = 2 * zi + 1
                    ax_v = axes[0, vis_col]
                    ax_a = axes[0, adv_col]
                    bbox_v = ax_v.get_position()
                    bbox_a = ax_a.get_position()
                    x_center = (bbox_v.x0 + bbox_a.x1) / 2.0
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

                # Per-budget visit colorbars precisely aligned with each budget band
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
                    cax = fig.add_axes([0.88, y0, 0.010, height])
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_label('Visits', fontsize=8)
                    cb.ax.tick_params(labelsize=6)

                if adv_mappable is not None:
                    top_bbox = axes[0, 0].get_position()
                    bot_bbox = axes[-1, 0].get_position()
                    y0 = bot_bbox.y0
                    height = top_bbox.y1 - bot_bbox.y0
                    cax_adv = fig.add_axes([0.94, y0, 0.012, height])
                    cb_adv = fig.colorbar(adv_mappable, cax=cax_adv)
                    cb_adv.set_label(r'$\max_a A^{\mu}(s,a)$', fontsize=9)
                    cb_adv.ax.tick_params(labelsize=7)

                alpha_label = 'Vanilla NPG' if alpha == 0.0 else rf'$\alpha={alpha}$'
                fig.suptitle(
                    rf'Cap$\times\zeta$ visitation + teacher advantage — '
                    rf'distance={dist}, horizon={horizon_val} ({h_type}), {alpha_label}'
                    '\n'
                    r'Each $(c,\;\zeta)$ cell shows visitation (left) and '
                    r'$\max_a A^{\mu}(s,a)$ (right). '
                    r'Rows grouped by budget $T$ (per-budget visit colour scale).',
                    fontsize=11, fontweight='bold')

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
            n_cols = n_alpha + 1  # +1 for advantage column
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

            # Advantage shared across all rows (advantage doesn't depend on budget)
            adv_grids = {}
            vmax_adv = 0
            for rk in row_keys:
                A = teacher_advantages.get(rk)
                if A is not None:
                    g = A.max(axis=1).reshape(env.grid_size, env.grid_size)
                    adv_grids[rk] = g
                    vmax_adv = max(vmax_adv, np.abs(g).max())
            if vmax_adv == 0:
                vmax_adv = 1

            vis_mappables = {}  # budget -> matplotlib image (for per-budget colorbars)
            adv_mappable = None

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

                    # Advantage column (rightmost)
                    ax_adv = axes[global_row, n_alpha]
                    adv_grid = adv_grids.get(rk)
                    if adv_grid is not None:
                        im_adv = ax_adv.imshow(adv_grid, cmap='hot', origin='upper',
                                               vmin=0, vmax=vmax_adv)
                        adv_mappable = im_adv
                        _annotate_grid(ax_adv, env, goals, compact=True)
                    else:
                        ax_adv.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=ax_adv.transAxes, fontsize=9, color='gray')
                        ax_adv.set_xlim(0, 1)
                        ax_adv.set_ylim(0, 1)
                    ax_adv.set_xticks([])
                    ax_adv.set_yticks([])
                    if global_row == 0:
                        ax_adv.set_title(r'$\max_a A^{\mu}(s,a)$', fontsize=9)

            # Layout: leave room on left for band label + row labels, on right for
            # per-budget visit colorbars + one shared advantage colorbar.
            fig.subplots_adjust(
                left=0.14, right=0.86, bottom=0.04, top=0.93,
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

            # Per-budget visit colorbars precisely aligned with each budget band
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
                cax = fig.add_axes([0.88, y0, 0.010, height])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label('Visits', fontsize=8)
                cb.ax.tick_params(labelsize=6)

            if adv_mappable is not None:
                # Advantage colorbar spans full subplot grid height for consistent look
                top_bbox = axes[0, 0].get_position()
                bot_bbox = axes[-1, 0].get_position()
                y0 = bot_bbox.y0
                height = top_bbox.y1 - bot_bbox.y0
                cax_adv = fig.add_axes([0.94, y0, 0.012, height])
                cb_adv = fig.colorbar(adv_mappable, cax=cax_adv)
                cb_adv.set_label('Advantage', fontsize=9)
                cb_adv.ax.tick_params(labelsize=7)

            fig.suptitle(
                rf'State visitation + teacher advantage — distance={dist}, '
                rf'horizon={horizon_val} ({h_type})'
                '\n'
                rf'Rows grouped by budget $T\in\{{{",".join(str(b) for b in budget_vals)}\}}$. '
                r'Each budget band has its own visit colour scale.',
                fontsize=11, fontweight='bold')

            save_path = os.path.join(
                figures_dir,
                f'visitation_grid_dist{dist}_{h_type}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {save_path}")


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
                # would be unreadable at this density).
                for tv in teacher_vals:
                    if tv not in diags_by_tv:
                        continue
                    diags = diags_by_tv[tv]
                    steps = [d['step'] for d in diags]
                    dv_total = [d['delta_v_total'] for d in diags]
                    ax.plot(steps, dv_total, color=tv_color[tv],
                            linewidth=1.2, label=_teacher_label(mode, tv))
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            else:
                for tv in teacher_vals:
                    if tv not in diags_by_tv:
                        continue
                    diags = diags_by_tv[tv]
                    steps = [d['step'] for d in diags]
                    for mk_i, (metric, style) in enumerate(metric_keys):
                        vals = [d[metric] for d in diags]
                        # Only label the first metric line per teacher — the
                        # suptitle explains what each line style represents.
                        line_label = (_teacher_label(mode, tv)
                                      if mk_i == 0 else None)
                        ax.plot(steps, vals, color=tv_color[tv],
                                linewidth=1.2, linestyle=style,
                                label=line_label)

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
    parser.add_argument('--exact-gradient', type=str, default='true',
                        choices=['true', 'false'],
                        help='Use exact NPG (true, default) or sample-based (false)')
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
        exact_gradient = args.exact_gradient == 'true'
        all_results = run_sweep(args.mode, args.output_dir, args.n_seeds,
                                args.n_workers, exact_gradient=exact_gradient)

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
