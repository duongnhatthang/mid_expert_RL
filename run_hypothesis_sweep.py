"""
Comprehensive parameter sweep to test the mid-teacher-wins hypothesis.

Two modes:
  --mode zeta        Continuous mixture teacher: mu(zeta) = zeta*pi* + (1-zeta)*uniform
                     Uses 1 goal per distance.
  --mode capability  Discrete knowledge teacher: capacity in {-1, 0, 1, 2, 3}
                     Uses 3 goals per distance (so cap=1,2 are "mid-capacity").

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
    return ZETA_GOAL_POSITIONS if mode == 'zeta' else CAP_GOAL_POSITIONS


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


# =========================================================================
# Sweep runner
# =========================================================================

def _run_single_experiment(args_tuple):
    """Worker function for multiprocessing. Returns result dict."""
    mode, teacher_val, alpha, budget, h_type, dist, seed, horizons = args_tuple
    goal_pos = _goal_positions(mode)
    goals = goal_pos[dist]
    horizon = horizons[h_type]
    teacher_key = 'zeta' if mode == 'zeta' else 'teacher_capacity'

    kwargs = dict(
        grid_size=GRID_SIZE,
        goals=goals,
        horizon=horizon,
        sample_budget=budget,
        alpha=alpha,
        lr=0.5,
        trajectories_per_update=10,
        seed=seed,
        eval_interval=5,
        eval_n_episodes=50,
        exact_gradient=True,
    )
    if mode == 'zeta':
        kwargs['zeta'] = teacher_val
    else:
        kwargs['teacher_capacity'] = teacher_val

    result = run_experiment(**kwargs)
    result['distance'] = dist
    result['horizon_type'] = h_type
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


def run_sweep(mode: str, output_dir: str, n_seeds: int, n_workers: int = 1) -> list:
    """Run all experiments for the given mode, save CSV, return result dicts."""
    horizons = _get_horizons()
    csv_path = os.path.join(output_dir, f'{mode}_sweep_results.csv')

    if mode == 'zeta':
        teacher_values = ZETA_VALUES
    else:
        teacher_values = CAP_VALUES

    budgets_map = _load_calibrated_budgets(mode)

    # Build configs with per-(dist, h_type) budgets
    configs = []
    all_budget_sets = set()
    for tv, alpha, h_type, dist, seed in itertools.product(
        teacher_values, ALPHA_VALUES, HORIZON_TYPES, DISTANCES, range(n_seeds)
    ):
        if alpha == 0.0 and tv != teacher_values[0]:
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

    worker_args = [
        (mode, tv, alpha, budget, h_type, dist, seed, horizons)
        for tv, alpha, budget, h_type, dist, seed in configs
    ]

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

    # Replicate alpha=0 results for all teacher values (teacher is irrelevant at alpha=0)
    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    alpha_zero_results = [r for r in all_results if r.get('alpha') == 0.0]
    for r in list(alpha_zero_results):
        for tv in teacher_values:
            if tv != r[tcol]:
                r_copy = r.copy()
                r_copy[tcol] = tv
                r_copy.pop('visitation_counts', None)
                all_results.append(r_copy)

    # Save CSV
    if mode == 'zeta':
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
    return 'zeta' if mode == 'zeta' else 'teacher_capacity'


def _teacher_label(mode: str, val) -> str:
    if mode == 'zeta':
        return f'\u03b6={val:.2f}'
    return {-1: 'no teacher', 0: 'uniform'}.get(val, f'cap={val}')


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
    teacher_vals = sorted(df[tcol].unique(), key=float)

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


def plot_reward_vs_teacher(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per distance: one row per (budget, horizon) that has data, side-by-side metrics."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    tcol = _teacher_col(mode)
    goal_pos = _goal_positions(mode)
    teacher_vals = sorted(df[tcol].unique(), key=float)
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
                    ax.plot(teacher_vals, means_arr, marker='o', markersize=3,
                            label=f'\u03b1={alpha}', color=c, linewidth=1.5)
                    ax.fill_between(teacher_vals, means_arr - sems_arr,
                                    means_arr + sems_arr, alpha=0.15, color=c)

                ax.set_title(f'{mlabel}\n{_bh_title(budget, h_type, horizon)}',
                             fontsize=8)
                ax.set_ylabel(mlabel, fontsize=7)
                ax.grid(True, alpha=0.3)

                # Integer x-ticks for capability mode on ALL subplots
                if mode == 'zeta':
                    ax.set_xlabel('\u03b6 (teacher expertise)', fontsize=7)
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
    teacher_vals = sorted(df[tcol].unique(), key=float)
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


def _compute_teacher_advantages(mode, teacher_vals, env, goals, gamma):
    """Compute teacher advantage A^mu(s,a) for each teacher value.
    Returns dict: teacher_val -> (n_states, n_actions) array, or None if no teacher."""
    from tabular_prototype.teacher import compute_teacher_values_auto

    advantages = {}
    for tv in teacher_vals:
        if mode == 'zeta':
            Q_mu, V_mu, _ = compute_teacher_values_auto(
                env, env.goals, zeta=tv, gamma=gamma)
            advantages[tv] = Q_mu - V_mu[:, None]
        else:
            if tv == -1:
                advantages[tv] = None
            elif tv == 0:
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, env.goals, zeta=0.0, gamma=gamma)
                advantages[tv] = Q_mu - V_mu[:, None]
            else:
                known_goals = goals[:tv]
                Q_mu, V_mu, _ = compute_teacher_values_auto(
                    env, known_goals, zeta=1.0, gamma=gamma)
                advantages[tv] = Q_mu - V_mu[:, None]
    return advantages


def plot_visitation_grids(all_results: list, mode: str, figures_dir: str):
    """Visitation grids with an extra column showing teacher advantage A^μ."""
    from collections import defaultdict
    from tabular_prototype.config import compute_gamma_from_horizon
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    budget_vals = sorted(set(r['sample_budget'] for r in all_results))

    goal_pos = _goal_positions(mode)
    for dist in DISTANCES:
        goals = goal_pos[dist]
        horizons = _get_horizons()
        env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizons['large'])

        for budget in budget_vals:
            for h_type in HORIZON_TYPES:
                groups = defaultdict(list)
                for r in all_results:
                    if (r.get('distance') == dist and
                        r.get('sample_budget') == budget and
                        r.get('horizon_type') == h_type and
                        'visitation_counts' in r):
                        groups[(r[tcol], r['alpha'])].append(r['visitation_counts'])

                if not groups:
                    continue

                visitation_data = {k: np.mean(v, axis=0) for k, v in groups.items()}

                if mode == 'zeta':
                    row_keys = sorted(set(k[0] for k in visitation_data))
                    row_label_fn = lambda k: f'\u03b6={k:.2f}'
                else:
                    row_keys = sorted(set(k[0] for k in visitation_data))
                    cap_labels = {-1: 'no teacher', 0: 'uniform'}
                    row_label_fn = lambda k: cap_labels.get(k, f'cap={k}')

                col_keys = sorted(set(k[1] for k in visitation_data))
                col_label_fn = lambda a: 'Vanilla NPG' if a == 0.0 else f'\u03b1={a}'

                # Compute teacher advantages for the extra column
                horizon_val = horizons[h_type]
                gamma = compute_gamma_from_horizon(horizon_val)
                teacher_advantages = _compute_teacher_advantages(
                    mode, row_keys, env, goals, gamma)

                # Build the grid manually: visitation columns + 1 advantage column
                n_rows = len(row_keys)
                n_vis_cols = len(col_keys)
                n_total_cols = n_vis_cols + 1  # +1 for advantage
                cell_size = 1.8
                fig_w = cell_size * n_total_cols + 2.0
                fig_h = cell_size * n_rows + 1.0
                fig, axes = plt.subplots(
                    n_rows, n_total_cols,
                    figsize=(fig_w, fig_h))

                # Visitation shared vmax
                vmax_vis = 0
                vis_grids = {}
                for rk in row_keys:
                    for ck in col_keys:
                        vis = visitation_data.get((rk, ck))
                        if vis is not None:
                            grid = vis.sum(axis=1).reshape(env.grid_size, env.grid_size)
                            vis_grids[(rk, ck)] = grid
                            vmax_vis = max(vmax_vis, grid.max())
                if vmax_vis == 0:
                    vmax_vis = 1

                # Advantage shared vmax (absolute)
                adv_grids = {}
                vmax_adv = 0
                for rk in row_keys:
                    A = teacher_advantages.get(rk)
                    if A is not None:
                        # max_a A^μ(s,a) per state
                        adv_grid = A.max(axis=1).reshape(env.grid_size, env.grid_size)
                        adv_grids[rk] = adv_grid
                        vmax_adv = max(vmax_adv, np.abs(adv_grid).max())
                if vmax_adv == 0:
                    vmax_adv = 1

                # Ensure axes is 2D
                if n_rows == 1:
                    axes = axes[np.newaxis, :]
                if n_total_cols == 1:
                    axes = axes[:, np.newaxis]

                vis_mappable = None
                adv_mappable = None

                for i, rk in enumerate(row_keys):
                    # Visitation columns
                    for j, ck in enumerate(col_keys):
                        ax = axes[i, j]
                        grid = vis_grids.get((rk, ck))
                        if grid is not None:
                            im = ax.imshow(grid, cmap='hot', origin='upper',
                                           vmin=0, vmax=vmax_vis)
                            vis_mappable = im
                        _annotate_grid(ax, env, goals)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if j == 0:
                            ax.set_ylabel(row_label_fn(rk), fontsize=8)
                        if i == 0:
                            ax.set_title(col_label_fn(ck), fontsize=8)

                    # Advantage column (rightmost)
                    ax_adv = axes[i, n_vis_cols]
                    adv_grid = adv_grids.get(rk)
                    if adv_grid is not None:
                        im_adv = ax_adv.imshow(adv_grid, cmap='hot', origin='upper',
                                               vmin=0, vmax=vmax_adv)
                        adv_mappable = im_adv
                    else:
                        ax_adv.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    transform=ax_adv.transAxes, fontsize=9, color='gray')
                    _annotate_grid(ax_adv, env, goals)
                    ax_adv.set_xticks([])
                    ax_adv.set_yticks([])
                    if i == 0:
                        ax_adv.set_title('A\u03bc(s) = max_a A\u03bc(s,a)', fontsize=7)

                # Layout and colorbars
                fig.subplots_adjust(
                    left=0.06, right=0.85, bottom=0.04, top=0.90,
                    wspace=0.12, hspace=0.30)

                # Place colorbars in figure-coordinate axes
                if vis_mappable is not None:
                    cax1 = fig.add_axes([0.86, 0.15, 0.02, 0.70])
                    fig.colorbar(vis_mappable, cax=cax1, label='Visits')
                if adv_mappable is not None:
                    cax2 = fig.add_axes([0.93, 0.15, 0.02, 0.70])
                    fig.colorbar(adv_mappable, cax=cax2, label='Advantage')

                fig.suptitle(
                    f'Visitation + Teacher Advantage: dist={dist}, '
                    f'budget={budget}, H={horizon_val} ({h_type})',
                    fontsize=11, fontweight='bold')

                save_path = os.path.join(
                    figures_dir,
                    f'visitation_grid_dist{dist}_budget{budget}_{h_type}.png')
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Saved {save_path}")


def _annotate_grid(ax, env, goals):
    """Add start and goal markers to a grid subplot."""
    import matplotlib.patches as patches
    start = env.start
    ax.add_patch(patches.Circle(
        (start[1], start[0]), 0.25, fill=True, color='cyan', alpha=0.8, linewidth=0))
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            color='black', fontweight='bold', fontsize=6)
    if goals:
        for goal in goals:
            ax.add_patch(patches.Rectangle(
                (goal[1] - 0.35, goal[0] - 0.35), 0.7, 0.7,
                fill=False, edgecolor='lime', linewidth=1.5))
            ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                    color='lime', fontweight='bold', fontsize=6)


def _plot_sweep_diagnostics(all_results: list, mode: str, figures_dir: str):
    """Generate diagnostic plots from sweep results that contain diagnostics data."""
    from tabular_prototype.visualization import (
        plot_magnitude_decomposition,
        plot_delta_v_decomposition,
        plot_amu_distribution_evolution,
        plot_entropy_trajectory,
    )
    from collections import defaultdict

    tcol = _teacher_col(mode)
    diag_dir = os.path.join(figures_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    # Group results by (distance, horizon_type, budget, alpha)
    # For each group, average diagnostics across seeds per teacher value
    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if not r.get('diagnostics'):
            continue
        key = (r.get('distance'), r.get('horizon_type'),
               r.get('sample_budget'), r.get('alpha', 0.0))
        tv = r[tcol]
        groups[key][tv].append(r['diagnostics'])

    for (dist, h_type, budget, alpha), teacher_diags in groups.items():
        if alpha == 0.0:
            continue  # No teacher signal at alpha=0, diagnostics are trivial

        # Average diagnostics across seeds for each teacher value
        diagnostics_by_label = {}
        for tv, seed_diag_lists in teacher_diags.items():
            label = _teacher_label(mode, tv)
            # Average each metric across seeds at each step
            n_steps = min(len(d) for d in seed_diag_lists)
            avg_diags = []
            for step_i in range(n_steps):
                step_dicts = [d[step_i] for d in seed_diag_lists]
                avg = {k: np.mean([s[k] for s in step_dicts])
                       for k in step_dicts[0] if isinstance(step_dicts[0][k], (int, float))}
                avg['step'] = step_i
                avg_diags.append(avg)
            diagnostics_by_label[label] = avg_diags

        if not diagnostics_by_label:
            continue

        suffix = f'dist{dist}_budget{budget}_{h_type}_alpha{alpha}'

        plot_magnitude_decomposition(
            diagnostics_by_label,
            title=f'Signal Magnitude: dist={dist}, budget={budget}, H={h_type}, α={alpha}',
            save_path=os.path.join(diag_dir, f'magnitude_{suffix}.png'),
        )
        plot_delta_v_decomposition(
            diagnostics_by_label,
            title=f'Delta-V: dist={dist}, budget={budget}, H={h_type}, α={alpha}',
            save_path=os.path.join(diag_dir, f'delta_v_{suffix}.png'),
        )
        plot_amu_distribution_evolution(
            diagnostics_by_label,
            title=f'A^mu Stats: dist={dist}, budget={budget}, H={h_type}, α={alpha}',
            save_path=os.path.join(diag_dir, f'amu_stats_{suffix}.png'),
        )
        plot_entropy_trajectory(
            diagnostics_by_label,
            title=f'Entropy: dist={dist}, budget={budget}, H={h_type}, α={alpha}',
            save_path=os.path.join(diag_dir, f'entropy_{suffix}.png'),
        )

    print(f"Diagnostic plots saved to {diag_dir}/")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive hypothesis sweep (zeta or capability)')
    parser.add_argument('--mode', choices=['zeta', 'capability'], required=True,
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
        all_results = run_sweep(args.mode, args.output_dir, args.n_seeds, args.n_workers)

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
