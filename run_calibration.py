"""
Budget calibration: find T_sat (vanilla NPG saturation step) per config.

For each (distance, horizon_type, n_goals, lr, grid_size), runs vanilla NPG
with a generous budget and finds the first step where mean_reward >= threshold.

Stores calibrated budgets in results/calibration.json. Skips configs that
already have entries.

Usage:
    PYTHONPATH=. python run_calibration.py [--n-seeds 5] [--max-budget 2000] [--threshold 0.95]
"""

import argparse
import json
import os
import numpy as np

from tabular_prototype import run_experiment, compute_exploration_thresholds
from tabular_prototype.environment import generate_equidistant_goals

GRID_SIZE = 9
DISTANCES = [4, 6, 7, 8]
HORIZON_TYPES = ['small', 'large']
LR = 0.5

# n_goals=1 for zeta mode, n_goals=3 for capability mode
N_GOALS_LIST = [1, 3]

CALIBRATION_PATH = 'results/calibration.json'

# Sample mode calibration parameters
SAMPLE_LR_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
SAMPLE_TRAJ_PER_UPDATE = [1, 5, 10, 20]
SAMPLE_CALIBRATION_PATH = 'results/calibration_sample.json'


def _config_key(distance, h_type, n_goals, lr, grid_size):
    return f"dist={distance}_{h_type}_ng={n_goals}_lr={lr}_grid={grid_size}"


def calibrate_single(distance, h_type, n_goals, lr, grid_size,
                     n_seeds, max_budget, threshold):
    """Run vanilla NPG and find T_sat."""
    goals = generate_equidistant_goals(grid_size, n_goals, distance=distance)
    horizons = compute_exploration_thresholds(grid_size)
    horizon = horizons[f'horizon_{h_type}']

    sat_steps = []
    for seed in range(n_seeds):
        result = run_experiment(
            grid_size=grid_size,
            goals=goals,
            teacher_capacity=0,
            horizon=horizon,
            sample_budget=max_budget,
            alpha=0.0,
            lr=lr,
            seed=seed,
            eval_interval=1,
            eval_n_episodes=50,
            exact_gradient=True,
        )
        # Find first step where mean_reward >= threshold
        t_sat = max_budget  # default if never saturates
        for entry in result['history']:
            if entry['mean_reward'] >= threshold:
                t_sat = entry['steps']
                break
        sat_steps.append(t_sat)

    t_sat_mean = int(np.ceil(np.mean(sat_steps)))
    t_sat_max = int(max(sat_steps))

    # Use the max across seeds (conservative: ensure ALL seeds saturate)
    t_sat = t_sat_max

    # Budget range: always exactly 4 values [T_sat//5, T_sat//3, T_sat, T_sat*2]
    # Note: in exact NPG mode, budget = update steps (full Bellman solve each),
    # NOT episode steps. So budget < horizon is valid and common for large horizons.
    # We only enforce a minimum of 3 steps and ensure all 4 are distinct.
    raw = [max(3, t_sat // 5), max(3, t_sat // 3), t_sat, t_sat * 2]
    # Ensure strictly increasing: bump duplicates up by 1
    budgets = [raw[0]]
    for v in raw[1:]:
        budgets.append(max(v, budgets[-1] + 1))

    return {
        'T_sat': t_sat,
        'T_sat_mean': t_sat_mean,
        'T_sat_max': t_sat_max,
        'per_seed': sat_steps,
        'budgets': budgets,
        'horizon': horizon,
        'n_goals': n_goals,
        'distance': distance,
        'horizon_type': h_type,
        'lr': lr,
        'grid_size': grid_size,
        'threshold': threshold,
        'max_budget': max_budget,
        'n_seeds': n_seeds,
    }


def calibrate_sample_single(distance, h_type, n_goals, grid_size,
                             n_seeds, max_budget, threshold):
    """Sweep LR x trajectories_per_update for sample mode."""
    goals = generate_equidistant_goals(grid_size, n_goals, distance=distance)
    horizons = compute_exploration_thresholds(grid_size)
    horizon = horizons[f'horizon_{h_type}']

    combo_results = {}

    for lr in SAMPLE_LR_VALUES:
        for tpu in SAMPLE_TRAJ_PER_UPDATE:
            sat_steps = []
            final_rewards = []

            for seed in range(n_seeds):
                result = run_experiment(
                    grid_size=grid_size,
                    goals=goals,
                    teacher_capacity=0,
                    horizon=horizon,
                    sample_budget=max_budget,
                    alpha=0.0,
                    lr=lr,
                    trajectories_per_update=tpu,
                    seed=seed,
                    eval_interval=max(1, max_budget // 200),
                    eval_n_episodes=50,
                    exact_gradient=False,
                )
                t_sat = max_budget
                for entry in result['history']:
                    if entry['mean_reward'] >= threshold:
                        t_sat = entry['steps']
                        break
                sat_steps.append(t_sat)
                if result['history']:
                    final_rewards.append(result['history'][-1]['mean_reward'])

            combo_results[(lr, tpu)] = {
                'lr': lr,
                'trajectories_per_update': tpu,
                'T_sat_max': int(max(sat_steps)),
                'T_sat_mean': int(np.ceil(np.mean(sat_steps))),
                'per_seed_sat': sat_steps,
                'final_reward_mean': float(np.mean(final_rewards)) if final_rewards else 0.0,
                'final_reward_std': float(np.std(final_rewards)) if final_rewards else 0.0,
            }

    # Selection: best final reward, then among 5% threshold, smallest T_sat
    best_final = max(cr['final_reward_mean'] for cr in combo_results.values())
    threshold_val = best_final * 0.95

    candidates = {k: v for k, v in combo_results.items()
                  if v['final_reward_mean'] >= threshold_val}
    if not candidates:
        candidates = combo_results

    best_key = min(candidates, key=lambda k: candidates[k]['T_sat_max'])
    best = candidates[best_key]
    t_sat = best['T_sat_max']

    raw = [max(3, t_sat // 5), max(3, t_sat // 3), t_sat, t_sat * 2]
    budgets = [raw[0]]
    for v in raw[1:]:
        budgets.append(max(v, budgets[-1] + 1))

    return {
        'T_sat': t_sat,
        'budgets': budgets,
        'best_lr': best['lr'],
        'best_traj_per_update': best['trajectories_per_update'],
        'best_final_reward': best['final_reward_mean'],
        'all_combos': {f'lr={k[0]}_tpu={k[1]}': v for k, v in combo_results.items()},
        'horizon': horizon,
        'n_goals': n_goals,
        'distance': distance,
        'horizon_type': h_type,
        'grid_size': grid_size,
        'threshold': threshold,
        'max_budget': max_budget,
        'n_seeds': n_seeds,
    }


def _plot_sample_calibration_curves(cal_result, config_key, output_dir):
    """Plot learning curves for all LR x traj combos for one config."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    combos = cal_result['all_combos']
    n_combos = len(combos)
    colors = cm.tab10(np.linspace(0, 1, max(10, n_combos)))[:n_combos]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (combo_key, combo_data) in enumerate(sorted(combos.items())):
        label = combo_key
        if combo_data['lr'] == cal_result['best_lr'] and \
           combo_data['trajectories_per_update'] == cal_result['best_traj_per_update']:
            label += ' (BEST)'
        ax.axhline(combo_data['final_reward_mean'], color=colors[i],
                   linestyle='--', alpha=0.3)
        ax.plot([], [], color=colors[i], label=f"{label}: final={combo_data['final_reward_mean']:.3f}, "
                f"T_sat={combo_data['T_sat_max']}")

    ax.set_xlabel('Observations')
    ax.set_ylabel('Mean Reward')
    ax.set_title(f'Sample Calibration: {config_key}\n'
                 f'Best: lr={cal_result["best_lr"]}, tpu={cal_result["best_traj_per_update"]}, '
                 f'T_sat={cal_result["T_sat"]}')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'calibration_{config_key}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _run_exact_calibration(args):
    """Run exact-mode calibration (original behavior)."""
    os.makedirs(os.path.dirname(CALIBRATION_PATH) or '.', exist_ok=True)

    # Load existing calibration
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            calibration = json.load(f)
    else:
        calibration = {}

    configs = [
        (dist, h_type, n_goals)
        for dist in DISTANCES
        for h_type in HORIZON_TYPES
        for n_goals in N_GOALS_LIST
    ]

    total = len(configs)
    print(f"Calibrating {total} configs (grid={GRID_SIZE}, lr={LR}, "
          f"threshold={args.threshold}, max_budget={args.max_budget})",
          flush=True)

    for i, (dist, h_type, n_goals) in enumerate(configs):
        key = _config_key(dist, h_type, n_goals, LR, GRID_SIZE)

        if key in calibration and not args.force:
            existing = calibration[key]
            print(f"  [{i+1}/{total}] {key}: CACHED "
                  f"(T_sat={existing['T_sat']}, budgets={existing['budgets']})",
                  flush=True)
            continue

        print(f"  [{i+1}/{total}] {key}: calibrating...", end=" ", flush=True)
        result = calibrate_single(
            dist, h_type, n_goals, LR, GRID_SIZE,
            args.n_seeds, args.max_budget, args.threshold,
        )
        calibration[key] = result
        print(f"T_sat={result['T_sat']}, budgets={result['budgets']}", flush=True)

        # Save after each config (crash-safe)
        with open(CALIBRATION_PATH, 'w') as f:
            json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to {CALIBRATION_PATH}", flush=True)

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Config':<45} {'T_sat':>6} {'Budgets'}")
    for key in sorted(calibration):
        c = calibration[key]
        print(f"  {key:<43} {c['T_sat']:>6} {c['budgets']}")


def _run_sample_calibration(args):
    """Run sample-mode calibration: sweep LR x traj_per_update."""
    cal_path = SAMPLE_CALIBRATION_PATH
    fig_dir = 'results/calibration_sample_figures'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cal_path) or '.', exist_ok=True)

    if os.path.exists(cal_path):
        with open(cal_path) as f:
            calibration = json.load(f)
    else:
        calibration = {}

    configs = [
        (dist, h_type, n_goals)
        for dist in DISTANCES
        for h_type in HORIZON_TYPES
        for n_goals in N_GOALS_LIST
    ]

    total = len(configs)
    print(f"Sample calibration: {total} configs (grid={GRID_SIZE}, "
          f"LRs={SAMPLE_LR_VALUES}, TPUs={SAMPLE_TRAJ_PER_UPDATE})",
          flush=True)
    print(f"  threshold={args.threshold}, max_budget={args.max_budget}", flush=True)

    for i, (dist, h_type, n_goals) in enumerate(configs):
        key = f"dist={dist}_{h_type}_ng={n_goals}_grid={GRID_SIZE}"

        if key in calibration and not args.force:
            existing = calibration[key]
            print(f"  [{i+1}/{total}] {key}: CACHED "
                  f"(lr={existing['best_lr']}, tpu={existing['best_traj_per_update']}, "
                  f"T_sat={existing['T_sat']})",
                  flush=True)
            continue

        print(f"  [{i+1}/{total}] {key}: calibrating...", flush=True)
        result = calibrate_sample_single(
            dist, h_type, n_goals, GRID_SIZE,
            args.n_seeds, args.max_budget, args.threshold,
        )
        calibration[key] = result
        print(f"    Best: lr={result['best_lr']}, tpu={result['best_traj_per_update']}, "
              f"T_sat={result['T_sat']}, budgets={result['budgets']}", flush=True)

        _plot_sample_calibration_curves(result, key, fig_dir)

        with open(cal_path, 'w') as f:
            json.dump(calibration, f, indent=2)

    print(f"\nSample calibration saved to {cal_path}", flush=True)
    print(f"Calibration plots saved to {fig_dir}/", flush=True)

    print("\n--- Summary ---")
    print(f"{'Config':<45} {'LR':>5} {'TPU':>4} {'T_sat':>6} {'Budgets'}")
    for key in sorted(calibration):
        c = calibration[key]
        print(f"  {key:<43} {c['best_lr']:>5} {c['best_traj_per_update']:>4} "
              f"{c['T_sat']:>6} {c['budgets']}")


def main():
    parser = argparse.ArgumentParser(description='Budget calibration')
    parser.add_argument('--mode', choices=['exact', 'sample'], default='exact',
                        help='Calibrate for exact NPG (default) or sample-based mode')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--max-budget', type=int, default=2000)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--force', action='store_true',
                        help='Recalibrate even if entry exists')
    args = parser.parse_args()

    if args.mode == 'sample':
        _run_sample_calibration(args)
    else:
        _run_exact_calibration(args)


if __name__ == '__main__':
    main()
