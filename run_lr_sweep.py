"""
One-time learning rate calibration sweep.

Runs vanilla NPG (alpha=0) on a representative config to find optimal LR.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tabular_prototype import (
    run_experiment,
    generate_equidistant_goals,
    compute_exploration_thresholds,
    plot_learning_curves,
)

LR_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def run_lr_sweep(
    grid_size: int = 9,
    distance: int = 4,
    n_seeds: int = 5,
    budget: int = 500,
    output_dir: str = 'results/lr_sweep',
):
    """Sweep learning rates for vanilla NPG on a representative config."""
    os.makedirs(output_dir, exist_ok=True)

    goals = generate_equidistant_goals(grid_size, 1, distance=distance)
    thresholds = compute_exploration_thresholds(grid_size)
    horizon = thresholds['horizon_large']

    print(f"LR Sweep: grid={grid_size}, distance={distance}, "
          f"horizon={horizon}, budget={budget} update steps", flush=True)
    print(f"LR values: {LR_VALUES}", flush=True)

    histories = {}
    for lr in LR_VALUES:
        histories[lr] = []
        print(f"  lr={lr}...", end=" ", flush=True)
        rewards = []
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size,
                goals=goals,
                teacher_capacity=0,
                horizon=horizon,
                sample_budget=budget,
                alpha=0.0,
                lr=lr,
                seed=seed,
                eval_interval=2,
                eval_n_episodes=50,
                mode="exact",
            )
            histories[lr].append(result['history'])
            if result['history']:
                rewards.append(result['history'][-1]['mean_reward'])
        mean_r = np.mean(rewards) if rewards else 0
        print(f"final reward = {mean_r:.3f}")

    # Plot
    save_path = os.path.join(output_dir, 'lr_sweep_curves.png')
    plot_learning_curves(
        histories,
        title=f"LR Sweep (Vanilla NPG, grid={grid_size}, dist={distance})",
        ylabel="Mean Reward",
        metric="mean_reward",
        smooth_window=1,
        save_path=save_path,
        label_fn=lambda lr: f"lr={lr}",
    )
    print(f"\nFigure saved to {save_path}", flush=True)

    # Print summary
    print("\n--- LR Sweep Summary ---")
    for lr in LR_VALUES:
        final_rewards = []
        for h in histories[lr]:
            if h:
                final_rewards.append(h[-1]['mean_reward'])
        if final_rewards:
            print(f"  lr={lr:8.4f}: reward={np.mean(final_rewards):.4f} "
                  f"± {np.std(final_rewards)/np.sqrt(len(final_rewards)):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LR calibration sweep')
    parser.add_argument('--grid-size', type=int, default=9)
    parser.add_argument('--distance', type=int, default=4)
    parser.add_argument('--n-seeds', type=int, default=5)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='results/lr_sweep')
    args = parser.parse_args()

    run_lr_sweep(
        grid_size=args.grid_size,
        distance=args.distance,
        n_seeds=args.n_seeds,
        budget=args.budget,
        output_dir=args.output_dir,
    )
