#!/usr/bin/env python
"""Focused runner for NPG update-direction diagnostics.

Runs ONLY the canonical single cell defined in
docs/superpowers/specs/2026-05-25-npg-update-direction-diagnostics-design.md:
  dist=6, h_type='small', alpha=1.0, B = calibrated budgets[-2].

For each sweep mode ∈ {zeta, capability}:
  - One α=0 baseline training run (any non-(-1) teacher).
  - For each teacher value in the mode's list, one α=1 training run
    per seed.

Emits three PNGs per mode in <output_dir>/{mode}/:
  npg_cosine.png, npg_var_trace.png, npg_var_s0.png.

Usage:
  python run_npg_diagnostics.py \\
      [--n-seeds 30] \\
      [--n-bootstrap 50] \\
      [--output-dir results/figures/npg_diagnostics_<timestamp>] \\
      [--training-mode sample] \\
      [--grid-size 9] \\
      [--override-budget N]
"""
import argparse
import datetime as dt
import json
import os
import pickle
import sys
from collections import defaultdict

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment
from tabular_prototype.visualization import (
    plot_npg_cosine, plot_npg_variance,
)
import run_hypothesis_sweep as sweep


ZETA_TEACHER_VALUES_ALPHA1 = [0.0, 0.33, 0.67, 1.0]
CAPABILITY_TEACHER_VALUES_ALPHA1 = [0, 1, 2, 3]


def _resolve_cell(args):
    n_goals_zeta = 1
    n_goals_cap = 3
    if args.override_budget is not None:
        # Test/dev path: bypass calibration entirely.
        from tabular_prototype.environment import compute_exploration_thresholds
        h_val = compute_exploration_thresholds(args.grid_size)['horizon_small']
        return {
            'zeta': dict(budget=args.override_budget, h_val=h_val,
                         lr=0.5, tpu=4, n_goals=n_goals_zeta),
            'capability': dict(budget=args.override_budget, h_val=h_val,
                               lr=0.5, tpu=4, n_goals=n_goals_cap),
        }
    calib_path = sweep._calibration_path_for(args.training_mode)
    try:
        calib = json.load(open(calib_path))
    except FileNotFoundError:
        sys.exit(
            f"Calibration JSON missing: {calib_path}. "
            f"Regenerate via run_calibration.py and retry."
        )
    out = {}
    for sweep_mode, ng in (('zeta', n_goals_zeta),
                           ('capability', n_goals_cap)):
        cell = sweep._find_calibration_cell(
            calib, args.distance, args.horizon_type, ng,
        )
        if cell is None:
            sys.exit(
                f"No calibration cell for distance={args.distance}, "
                f"horizon_type={args.horizon_type}, n_goals={ng}, "
                f"training_mode={args.training_mode}"
            )
        budgets = cell.get('budgets', [])
        if len(budgets) < 2:
            sys.exit(
                f"Calibration cell has fewer than 2 budgets: {budgets}"
            )
        out[sweep_mode] = dict(
            budget=budgets[-2],
            h_val=cell['horizon'],
            lr=cell.get('lr', cell.get('best_lr', 0.5)),
            tpu=cell.get('best_traj_per_update', 1),
            n_goals=ng,
        )
    return out


def _run_one(args, sweep_mode, cell, alpha, tv, seed):
    goals = generate_equidistant_goals(
        args.grid_size, cell['n_goals'], distance=args.distance,
    )
    kwargs = dict(
        grid_size=args.grid_size, goals=goals, lr=cell['lr'],
        horizon=cell['h_val'], sample_budget=cell['budget'],
        mode=args.training_mode, seed=seed,
        eval_interval=5, alpha=alpha,
        trajectories_per_update=cell['tpu'],
        n_bootstrap=args.n_bootstrap,
    )
    if sweep_mode == 'zeta':
        kwargs['teacher_capacity'] = 1
        kwargs['zeta'] = tv
    else:
        kwargs['teacher_capacity'] = tv
    return run_experiment(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=30)
    parser.add_argument('--n-bootstrap', type=int, default=50)
    default_out = os.path.join(
        'results', 'figures',
        f'npg_diagnostics_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    parser.add_argument('--output-dir', type=str, default=default_out)
    parser.add_argument('--training-mode', choices=['sample'],
                        default='sample')
    parser.add_argument('--distance', type=int, default=6)
    parser.add_argument('--horizon-type', type=str, default='small')
    parser.add_argument('--grid-size', type=int, default=9)
    parser.add_argument('--override-budget', type=int, default=None)
    args = parser.parse_args()

    cells = _resolve_cell(args)
    os.makedirs(args.output_dir, exist_ok=True)

    for sweep_mode in ('zeta', 'capability'):
        cell = cells[sweep_mode]
        if sweep_mode == 'zeta':
            tvs_alpha1 = ZETA_TEACHER_VALUES_ALPHA1
            baseline_tv = 0.0  # ζ=0 with α=0 is vanilla NPG
        else:
            tvs_alpha1 = CAPABILITY_TEACHER_VALUES_ALPHA1
            baseline_tv = 0  # cap=0 (random teacher); any non-(-1) works at α=0

        # Run α=1 per teacher value × seeds.
        histories_by_teacher: dict = defaultdict(list)
        all_records = []
        for tv in tvs_alpha1:
            for seed in range(args.n_seeds):
                r = _run_one(args, sweep_mode, cell, 1.0, tv, seed)
                histories_by_teacher[tv].append(r['history'])
                all_records.append({
                    'sweep_mode': sweep_mode, 'alpha': 1.0,
                    'teacher_value': tv, 'seed': seed,
                    'history': r['history'],
                })

        # Run α=0 baseline (one per seed).
        baseline_histories = []
        for seed in range(args.n_seeds):
            r = _run_one(args, sweep_mode, cell, 0.0, baseline_tv, seed)
            baseline_histories.append(r['history'])
            all_records.append({
                'sweep_mode': sweep_mode, 'alpha': 0.0,
                'teacher_value': baseline_tv, 'seed': seed,
                'history': r['history'],
            })

        mode_out = os.path.join(args.output_dir, sweep_mode)
        os.makedirs(mode_out, exist_ok=True)
        with open(os.path.join(mode_out, 'records.pkl'), 'wb') as fp:
            pickle.dump(all_records, fp)

        cell_info = {
            'distance': args.distance,
            'horizon': cell['h_val'],
            'horizon_type': args.horizon_type,
            'sample_budget': cell['budget'],
            'alpha': 1.0,
        }
        plot_npg_cosine(
            histories_by_teacher=dict(histories_by_teacher),
            mode=sweep_mode,
            out_path=os.path.join(mode_out, 'npg_cosine.png'),
            cell_info=cell_info,
        )
        plot_npg_variance(
            histories_by_teacher=dict(histories_by_teacher),
            baseline_history_alpha_zero=baseline_histories,
            mode=sweep_mode,
            out_dir=mode_out,
            cell_info=cell_info,
            n_bootstrap=args.n_bootstrap,
        )
        print(f'Wrote {mode_out}/*.png')

    print('NPG_DIAGNOSTICS_DONE')


if __name__ == '__main__':
    main()
