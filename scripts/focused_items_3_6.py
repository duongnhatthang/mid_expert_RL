#!/usr/bin/env python
"""Focused runner for items 3 + 6 figures at default config.

Runs ONLY the (dist=6, h_type='small', α∈{1.0, 0.0}, B=budgets[-2]) cells
that plot_advantage_alignment and plot_mc_variance_curve actually filter
to. Skips the rest of the full-sweep cube. Output: pkl + figures dir.

Usage:
  python scripts/focused_items_3_6.py \\
      --mode {zeta,capability} \\
      --training-mode {exact,sample,hybrid} \\
      --output-dir results/items_3_6_<tag>
"""
import argparse
import json
import os
import pickle
import sys

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment
import run_hypothesis_sweep as sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['zeta', 'capability'], required=True)
    parser.add_argument('--training-mode',
                        choices=['exact', 'sample', 'hybrid'],
                        default='sample')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--distance', type=int, default=6)
    parser.add_argument('--horizon-type', type=str, default='small')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    n_goals = 1 if args.mode == 'zeta' else 3
    calib_path = sweep._calibration_path_for(args.training_mode)
    calib = json.load(open(calib_path))
    cell = sweep._find_calibration_cell(
        calib, args.distance, args.horizon_type, n_goals,
    )
    if cell is None:
        sys.exit(
            f"No calibration cell: distance={args.distance}, "
            f"horizon_type={args.horizon_type}, n_goals={n_goals}, "
            f"training_mode={args.training_mode}"
        )
    budget = cell['budgets'][-2]
    h_val = cell['horizon']
    lr = cell.get('lr', cell.get('best_lr', 0.5))
    tpu = cell.get('best_traj_per_update', 1)

    print(
        f"Cell: dist={args.distance}, h_type={args.horizon_type}, "
        f"ng={n_goals}, B={budget}, H={h_val}, lr={lr}, tpu={tpu}, "
        f"training_mode={args.training_mode}"
    )

    goals = generate_equidistant_goals(9, n_goals, distance=args.distance)

    if args.mode == 'zeta':
        teacher_vals_alpha1 = [0.0, 0.33, 0.67, 1.0]
        teacher_val_baseline = 0.0
    else:
        teacher_vals_alpha1 = [0, 1, 2, 3]
        teacher_val_baseline = 0

    def run_one(alpha, tv, seed):
        kwargs = dict(
            grid_size=9, goals=goals, lr=lr,
            horizon=h_val, sample_budget=budget,
            mode=args.training_mode, seed=seed,
            eval_interval=5, alpha=alpha,
            trajectories_per_update=tpu,
        )
        if args.mode == 'zeta':
            kwargs['teacher_capacity'] = 1
            kwargs['zeta'] = tv
        else:
            kwargs['teacher_capacity'] = tv
        r = run_experiment(**kwargs)
        r.update({
            'distance': args.distance,
            'alpha': alpha,
            'horizon_type': args.horizon_type,
            'horizon': h_val,
            'sample_budget': budget,
            'seed': seed,
            'mode': args.training_mode,
        })
        if args.mode == 'zeta':
            r['zeta'] = tv
        else:
            r['teacher_capacity'] = tv
        return r

    all_results = []
    n_total = (len(teacher_vals_alpha1) + 1) * args.n_seeds
    n_done = 0
    for tv in teacher_vals_alpha1:
        for seed in range(args.n_seeds):
            all_results.append(run_one(1.0, tv, seed))
            n_done += 1
            print(f"  [{n_done}/{n_total}] α=1.0, tv={tv}, seed={seed}")
    for seed in range(args.n_seeds):
        all_results.append(run_one(0.0, teacher_val_baseline, seed))
        n_done += 1
        print(
            f"  [{n_done}/{n_total}] α=0.0, tv={teacher_val_baseline}, "
            f"seed={seed}"
        )

    pkl_path = os.path.join(
        args.output_dir, f'{args.mode}_sweep_results.pkl',
    )
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved {pkl_path}")

    sweep.plot_advantage_alignment(all_results, args.mode, figures_dir)
    sweep.plot_mc_variance_curve(all_results, args.mode, figures_dir)
    print("FOCUSED_DONE")


if __name__ == '__main__':
    main()
