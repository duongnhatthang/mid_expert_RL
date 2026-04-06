"""
Comprehensive parameter sweep to test the mid-teacher-wins hypothesis.

Two modes:
  --mode zeta        Continuous mixture teacher: mu(zeta) = zeta*pi* + (1-zeta)*uniform
  --mode capability  Discrete knowledge teacher: capacity in {-1, 0, 1}

Sweeps alpha, budget, horizon, and goal distance on a 9x9 grid.
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import time
import numpy as np
import pandas as pd

from tabular_prototype import (
    run_experiment,
    GridEnv,
    compute_exploration_thresholds,
    visualize_visitation_comparison_grid,
)
from tabular_prototype.environment import generate_equidistant_goals

# =========================================================================
# Constants
# =========================================================================

GRID_SIZE = 9

# Goal positions at varying Manhattan distances from start (4, 4)
GOAL_POSITIONS = {
    2: generate_equidistant_goals(9, 1, distance=2),
    4: generate_equidistant_goals(9, 1, distance=4),
    6: generate_equidistant_goals(9, 1, distance=6),
    7: generate_equidistant_goals(9, 1, distance=7),
}

ZETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
CAP_VALUES = [-1, 0, 1]

ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

ZETA_BUDGET_VALUES = [1000, 5000, 20000]
CAP_BUDGET_VALUES = [512, 2048, 5000, 20000]

HORIZON_TYPES = ['small', 'large']

DISTANCES = sorted(GOAL_POSITIONS.keys())


def _get_horizons():
    thresholds = compute_exploration_thresholds(GRID_SIZE)
    return {
        'small': thresholds['horizon_small'],
        'large': thresholds['horizon_large'],
    }


# =========================================================================
# Sweep runner
# =========================================================================

def _run_single_experiment(args_tuple):
    """Worker function for multiprocessing. Returns result dict."""
    mode, teacher_val, alpha, budget, h_type, dist, seed, horizons = args_tuple
    goals = GOAL_POSITIONS[dist]
    horizon = horizons[h_type]
    teacher_key = 'zeta' if mode == 'zeta' else 'teacher_capacity'

    kwargs = dict(
        grid_size=GRID_SIZE,
        goals=goals,
        horizon=horizon,
        sample_budget=budget,
        alpha=alpha,
        lr=0.1,
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
    return result


def run_sweep(mode: str, output_dir: str, n_seeds: int, n_workers: int = 1) -> list:
    """Run all experiments for the given mode, save CSV, return result dicts."""
    horizons = _get_horizons()
    csv_path = os.path.join(output_dir, f'{mode}_sweep_results.csv')

    if mode == 'zeta':
        teacher_values = ZETA_VALUES
        budget_values = ZETA_BUDGET_VALUES
    else:
        teacher_values = CAP_VALUES
        budget_values = CAP_BUDGET_VALUES

    configs = []
    for tv, alpha, budget, h_type, dist, seed in itertools.product(
        teacher_values, ALPHA_VALUES, budget_values, HORIZON_TYPES, DISTANCES, range(n_seeds)
    ):
        if alpha == 0.0 and tv != teacher_values[0]:
            continue  # skip redundant alpha=0 runs
        configs.append((tv, alpha, budget, h_type, dist, seed))
    total = len(configs)

    print("=" * 70)
    print(f"HYPOTHESIS SWEEP — mode={mode}")
    print("=" * 70)
    print(f"Teacher values: {teacher_values}")
    print(f"Alpha values:   {ALPHA_VALUES}")
    print(f"Budget values:  {budget_values}")
    print(f"Horizons:       small={horizons['small']}, large={horizons['large']}")
    print(f"Distances:      {DISTANCES}")
    print(f"Seeds:          {n_seeds}")
    print(f"Total experiments: {total}")
    print(f"Workers:        {n_workers}")
    print()

    # Build worker args
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
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    print(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                          f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
    else:
        all_results = []
        for i, args in enumerate(worker_args):
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = max(i, 1) / max(elapsed, 0.01)
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{total} ({100*i/total:.0f}%) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
            all_results.append(_run_single_experiment(args))

    elapsed = time.time() - t0
    print(f"  Completed {total} experiments in {elapsed:.1f}s ({total/elapsed:.1f} exp/s)")

    # Replicate alpha=0 results for all teacher values (teacher is irrelevant at alpha=0)
    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    alpha_zero_results = [r for r in all_results if r.get('alpha') == 0.0]
    for r in list(alpha_zero_results):
        for tv in teacher_values:
            if tv != r[tcol]:
                r_copy = r.copy()
                r_copy[tcol] = tv
                # Remove visitation_counts from copy to avoid memory bloat
                r_copy.pop('visitation_counts', None)
                all_results.append(r_copy)

    # Save CSV
    if mode == 'zeta':
        fieldnames = ['zeta', 'alpha', 'sample_budget', 'horizon', 'horizon_type',
                       'distance', 'seed', 'final_mean_reward', 'final_goal_rate',
                       'final_unique_sa', 'final_state_entropy']
    else:
        fieldnames = ['teacher_capacity', 'alpha', 'sample_budget', 'horizon', 'horizon_type',
                       'distance', 'seed', 'final_mean_reward', 'final_goal_rate',
                       'final_unique_sa', 'final_state_entropy']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    print(f"\nCSV saved to {csv_path}")
    return all_results


# =========================================================================
# Plotting
# =========================================================================

def _teacher_col(mode: str) -> str:
    return 'zeta' if mode == 'zeta' else 'teacher_capacity'


def _teacher_label(mode: str, val) -> str:
    if mode == 'zeta':
        return f'ζ={val:.2f}'
    return {-1: 'no teacher', 0: 'random'}.get(val, f'cap={val}')


def _budget_values(mode: str):
    return ZETA_BUDGET_VALUES if mode == 'zeta' else CAP_BUDGET_VALUES


def plot_heatmaps(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per distance: 3×2 grid of heatmaps (budget rows × horizon cols).
    Each heatmap: teacher param (y) vs alpha (x), cell color = mean reward."""
    import matplotlib.pyplot as plt

    tcol = _teacher_col(mode)
    teacher_vals = sorted(df[tcol].unique(), key=float)
    budget_vals = sorted(df['sample_budget'].unique())

    for dist in DISTANCES:
        dist_df = df[df['distance'] == dist]
        n_budgets = len(budget_vals)

        fig, axes = plt.subplots(n_budgets, 2, figsize=(10, 3 * n_budgets + 1))
        if n_budgets == 1:
            axes = axes[np.newaxis, :]

        for row, budget in enumerate(budget_vals):
            for col, h_type in enumerate(HORIZON_TYPES):
                ax = axes[row, col]
                sub = dist_df[(dist_df['sample_budget'] == budget) &
                              (dist_df['horizon_type'] == h_type)]

                pivot = sub.pivot_table(
                    values='final_mean_reward', index=tcol, columns='alpha', aggfunc='mean'
                )
                # Ensure consistent ordering
                pivot = pivot.reindex(index=teacher_vals, columns=ALPHA_VALUES)

                im = ax.imshow(pivot.values, aspect='auto', cmap='viridis',
                               vmin=0, vmax=max(0.01, dist_df['final_mean_reward'].max()))
                # Annotate cells
                for yi in range(pivot.shape[0]):
                    for xi in range(pivot.shape[1]):
                        val = pivot.values[yi, xi]
                        if not np.isnan(val):
                            ax.text(xi, yi, f'{val:.2f}', ha='center', va='center',
                                    fontsize=7, color='white' if val < pivot.values.max() * 0.7 else 'black')

                ax.set_xticks(range(len(ALPHA_VALUES)))
                ax.set_xticklabels(
                    ['NPG' if a == 0.0 else str(a) for a in ALPHA_VALUES],
                    fontsize=8,
                )
                ax.set_yticks(range(len(teacher_vals)))
                ax.set_yticklabels([_teacher_label(mode, v) for v in teacher_vals], fontsize=8)

                ax.set_title(f'budget={budget}, {h_type} horizon', fontsize=9)
                if col == 0:
                    ax.set_ylabel('Teacher')
                if row == n_budgets - 1:
                    ax.set_xlabel('Alpha')

        fig.suptitle(f'Mean Reward Heatmap — distance={dist}, goal={GOAL_POSITIONS[dist][0]}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(figures_dir, f'heatmap_dist{dist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_reward_vs_teacher(df: pd.DataFrame, mode: str, figures_dir: str):
    """Per distance: 3×2 grid of line plots. x=teacher param, one line per alpha."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    tcol = _teacher_col(mode)
    teacher_vals = sorted(df[tcol].unique(), key=float)
    budget_vals = sorted(df['sample_budget'].unique())
    n_budgets = len(budget_vals)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(ALPHA_VALUES)))

    for dist in DISTANCES:
        dist_df = df[df['distance'] == dist]
        fig, axes = plt.subplots(n_budgets, 2, figsize=(11, 3.5 * n_budgets + 1), sharex=True)
        if n_budgets == 1:
            axes = axes[np.newaxis, :]

        for row, budget in enumerate(budget_vals):
            for col, h_type in enumerate(HORIZON_TYPES):
                ax = axes[row, col]
                sub = dist_df[(dist_df['sample_budget'] == budget) &
                              (dist_df['horizon_type'] == h_type)]

                for ai, alpha in enumerate(ALPHA_VALUES):
                    alpha_sub = sub[sub['alpha'] == alpha]
                    means, sems = [], []
                    for tv in teacher_vals:
                        tv_data = alpha_sub[alpha_sub[tcol] == tv]['final_mean_reward']
                        means.append(tv_data.mean())
                        sems.append(tv_data.std() / np.sqrt(max(1, len(tv_data))))

                    ax.errorbar(teacher_vals, means, yerr=sems, marker='o', markersize=4,
                                capsize=3, label=f'α={alpha}', color=colors[ai], linewidth=1.5)

                ax.set_title(f'budget={budget}, {h_type} horizon', fontsize=9)
                ax.set_ylabel('Mean Reward', fontsize=8)
                ax.grid(True, alpha=0.3)

                if row == 0 and col == 1:
                    ax.legend(fontsize=7, loc='upper right')

        for ax in axes[-1]:
            if mode == 'zeta':
                ax.set_xlabel('ζ (teacher expertise)', fontsize=9)
            else:
                ax.set_xlabel('Teacher Capacity', fontsize=9)

        fig.suptitle(f'Reward vs Teacher — distance={dist}, goal={GOAL_POSITIONS[dist][0]}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(figures_dir, f'reward_vs_teacher_dist{dist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_distance_effect(df: pd.DataFrame, mode: str, figures_dir: str):
    """For select alpha values: reward vs distance, one line per teacher param."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    tcol = _teacher_col(mode)
    teacher_vals = sorted(df[tcol].unique(), key=float)
    budget_vals = sorted(df['sample_budget'].unique())
    n_budgets = len(budget_vals)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(teacher_vals)))

    for alpha in [0.0, 0.5, 1.0]:
        alpha_df = df[df['alpha'] == alpha]
        if len(alpha_df) == 0:
            continue

        fig, axes = plt.subplots(n_budgets, 2, figsize=(11, 3.5 * n_budgets + 1), sharex=True)
        if n_budgets == 1:
            axes = axes[np.newaxis, :]

        for row, budget in enumerate(budget_vals):
            for col, h_type in enumerate(HORIZON_TYPES):
                ax = axes[row, col]
                sub = alpha_df[(alpha_df['sample_budget'] == budget) &
                               (alpha_df['horizon_type'] == h_type)]

                for ti, tv in enumerate(teacher_vals):
                    tv_sub = sub[sub[tcol] == tv]
                    means, sems = [], []
                    for dist in DISTANCES:
                        d_data = tv_sub[tv_sub['distance'] == dist]['final_mean_reward']
                        means.append(d_data.mean() if len(d_data) > 0 else np.nan)
                        sems.append(d_data.std() / np.sqrt(max(1, len(d_data))) if len(d_data) > 0 else 0)

                    ax.errorbar(DISTANCES, means, yerr=sems, marker='o', markersize=4,
                                capsize=3, label=_teacher_label(mode, tv),
                                color=colors[ti], linewidth=1.5)

                ax.set_title(f'budget={budget}, {h_type} horizon', fontsize=9)
                ax.set_ylabel('Mean Reward', fontsize=8)
                ax.grid(True, alpha=0.3)

                if row == 0 and col == 1:
                    ax.legend(fontsize=7, loc='upper right')

        for ax in axes[-1]:
            ax.set_xlabel('Goal Distance (Manhattan)', fontsize=9)
            ax.set_xticks(DISTANCES)

        fig.suptitle(f'Distance Effect — α={alpha}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(figures_dir, f'distance_effect_alpha{alpha}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")


def plot_visitation_grids(all_results: list, mode: str, figures_dir: str):
    """Visitation comparison grids showing ALL baselines.
    Rows = teacher params, cols = all alpha values."""
    from collections import defaultdict

    tcol = 'zeta' if mode == 'zeta' else 'teacher_capacity'
    budget_vals = sorted(set(r['sample_budget'] for r in all_results))

    for dist in DISTANCES:
        goals = GOAL_POSITIONS[dist]
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
                    row_label_fn = lambda k: f'ζ={k:.2f}'
                else:
                    row_keys = sorted(set(k[0] for k in visitation_data))
                    cap_labels = {-1: 'no teacher', 0: 'random'}
                    row_label_fn = lambda k: cap_labels.get(k, f'cap={k}')

                col_keys = sorted(set(k[1] for k in visitation_data))
                col_label_fn = lambda a: 'Vanilla NPG' if a == 0.0 else f'α={a}'

                save_path = os.path.join(
                    figures_dir,
                    f'visitation_grid_dist{dist}_budget{budget}_{h_type}.png'
                )
                visualize_visitation_comparison_grid(
                    env, visitation_data, row_keys, col_keys,
                    row_label_fn, col_label_fn,
                    goals=goals,
                    suptitle=f'Visitation: dist={dist}, budget={budget}, {h_type} horizon',
                    save_path=save_path,
                )


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive hypothesis sweep (zeta or capability)')
    parser.add_argument('--mode', choices=['zeta', 'capability'], required=True,
                        help='Teacher parameterization to sweep')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/{mode}_sweep)')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds per config')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, use CPU count for max)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip experiments, only generate plots from existing CSV')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/{args.mode}_sweep'

    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, f'{args.mode}_sweep_results.csv')

    if args.skip_run:
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV not found at {csv_path}. Run without --skip-run first.")
            return
        all_results = None
    else:
        all_results = run_sweep(args.mode, args.output_dir, args.n_seeds, args.n_workers)

    # Load CSV for plotting
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}")

    print("\nGenerating plots...")
    plot_heatmaps(df, args.mode, figures_dir)
    plot_reward_vs_teacher(df, args.mode, figures_dir)
    plot_distance_effect(df, args.mode, figures_dir)

    if all_results is not None:
        print("\nGenerating visitation grids...")
        plot_visitation_grids(all_results, args.mode, figures_dir)
    else:
        print("\nSkipping visitation grids (no visitation data in CSV-only mode)")

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
