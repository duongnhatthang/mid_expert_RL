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


def main():
    parser = argparse.ArgumentParser(description='Budget calibration')
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--max-budget', type=int, default=2000)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--force', action='store_true',
                        help='Recalibrate even if entry exists')
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()
