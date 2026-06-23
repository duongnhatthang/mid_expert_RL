#!/usr/bin/env python
"""Focused runner for PG / NPG update-direction diagnostics.

Runs the canonical single cell defined in
docs/superpowers/specs/2026-05-25-npg-update-direction-diagnostics-design.md:
  dist=6, h_type='small', alpha=1.0, B = calibrated budgets[-2].

For each training mode ∈ {sample, exact} and each sweep mode ∈ {zeta,
capability}:
  - One α=0 baseline training run per seed (any non-(-1) teacher).
  - For each teacher value in the mode's list, one α=1 training run
    per seed.

Emits per training mode and sweep mode:

  Sample mode (<output_dir>/sample/{mode}/):
      pg_cosine.png, pg_var_trace.png, pg_var_visited.png, pg_var_inner.png

  Exact mode (<output_dir>/exact/{mode}/):
      u_cosine_npg.png, u_cosine_pinv.png

Usage:
  python run_npg_diagnostics.py \\
      [--n-seeds 30] \\
      [--training-modes sample,exact] \\
      [--output-dir results/figures/npg_diagnostics_<timestamp>] \\
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

import numpy as np

from tabular_prototype.environment import generate_equidistant_goals
from tabular_prototype.experiments import run_experiment
from tabular_prototype.visualization import (
    plot_pg_cosine, plot_pg_variance, plot_u_cosine,
    plot_coverage_curves, plot_coverage_heatmaps,
)
import run_hypothesis_sweep as sweep


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

ZETA_TEACHER_VALUES_ALPHA1 = [0.0, 0.33, 0.67, 1.0]
CAPABILITY_TEACHER_VALUES_ALPHA1 = [0, 1, 2, 3]


def _resolve_cell(args, training_mode):
    """Resolve canonical cell parameters for both sweep modes.

    For the override-budget path, calibration is bypassed. Otherwise the
    appropriate calibration JSON is loaded based on training_mode:
      'sample' → results/calibration_sample.json
      'exact'  → results/calibration.json
    """
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
    calib_path = os.path.join(
        _REPO_ROOT, sweep._calibration_path_for(training_mode),
    )
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
                f"training_mode={training_mode}"
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


def _run_one(args, training_mode, sweep_mode, cell, alpha, tv, seed):
    goals = generate_equidistant_goals(
        args.grid_size, cell['n_goals'], distance=args.distance,
    )
    kwargs = dict(
        grid_size=args.grid_size, goals=goals, lr=cell['lr'],
        horizon=cell['h_val'], sample_budget=cell['budget'],
        mode=training_mode, seed=seed,
        eval_interval=1, alpha=alpha,
        trajectories_per_update=(cell['tpu'] if args.override_tpu == 0 else args.override_tpu),
        pg_diag_enabled=True,
        track_coverage=args.track_coverage,
    )
    if sweep_mode == 'zeta':
        kwargs['teacher_capacity'] = 1
        kwargs['zeta'] = tv
    else:
        kwargs['teacher_capacity'] = tv
    return run_experiment(**kwargs)


def _average_grids(list_of_coverage_grids, ref):
    """Mean d_pi / ratios and (identical) d_mu across seeds for one ref."""
    dicts = [g[ref] for g in list_of_coverage_grids if g is not None]
    keys = ('d_mu', 'd_pi', 'ratio_mu_over_pi', 'ratio_pi_over_mu')
    return {k: np.mean([d[k] for d in dicts], axis=0) for k in keys}


def _emit_coverage_plots(out_dir, mode, cell_info, histories_by_teacher,
                         baseline_histories, grids_by_teacher, baseline_grids,
                         top_tv, grid_size):
    """Write coverage curves (per ref) + heatmaps (top teacher & baseline)."""
    for ref in ('analytic', 'learned'):
        plot_coverage_curves(
            histories_by_teacher=histories_by_teacher,
            baseline_history_alpha_zero=baseline_histories,
            mode=mode, out_dir=out_dir, cell_info=cell_info, ref=ref,
        )
        plot_coverage_heatmaps(
            ref, _average_grids(grids_by_teacher[top_tv], ref), grid_size,
            os.path.join(out_dir, f'cov_{ref}_heatmap_top.png'),
            cell_info, student_label=f'top teacher (α=1)',
        )
        plot_coverage_heatmaps(
            ref, _average_grids(baseline_grids, ref), grid_size,
            os.path.join(out_dir, f'cov_{ref}_heatmap_baseline.png'),
            cell_info, student_label='α=0 (vanilla NPG)',
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=30)
    default_out = os.path.join(
        'results', 'figures',
        f'npg_diagnostics_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )
    parser.add_argument('--output-dir', type=str, default=default_out)
    parser.add_argument('--training-modes', type=str, default='sample,exact',
                        help='Comma-separated subset of {sample,exact}')
    parser.add_argument('--distance', type=int, default=6)
    parser.add_argument('--horizon-type', type=str, default='small')
    parser.add_argument('--grid-size', type=int, default=9)
    parser.add_argument('--override-budget', type=int, default=None)
    parser.add_argument(
        '--override-tpu', type=int, default=10,
        help='Override trajectories_per_update for diagnostic runs '
             '(default 10; calibrated value is typically 1 and makes '
             'per-trajectory variance meaningless). Set to 0 to use '
             'the calibrated value.',
    )
    parser.add_argument(
        '--track-coverage', action='store_true',
        help='Compute and plot coverage / distribution-mismatch diagnostics '
             '(d^pi vs analytic & learned reference occupancies).',
    )
    args = parser.parse_args()

    training_modes = [m.strip() for m in args.training_modes.split(',')
                      if m.strip()]
    valid = {'sample', 'exact'}
    bad = [m for m in training_modes if m not in valid]
    if bad:
        sys.exit(f"Unknown training mode(s): {bad}. Valid: {sorted(valid)}")
    os.makedirs(args.output_dir, exist_ok=True)

    for training_mode in training_modes:
        cells = _resolve_cell(args, training_mode)
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
            grids_by_teacher: dict = defaultdict(list)
            all_records = []
            for tv in tvs_alpha1:
                for seed in range(args.n_seeds):
                    r = _run_one(args, training_mode, sweep_mode, cell,
                                 1.0, tv, seed)
                    histories_by_teacher[tv].append(r['history'])
                    grids_by_teacher[tv].append(r.get('coverage_grids'))
                    all_records.append({
                        'training_mode': training_mode,
                        'sweep_mode': sweep_mode, 'alpha': 1.0,
                        'teacher_value': tv, 'seed': seed,
                        'history': r['history'],
                    })

            # Run α=0 baseline (one per seed).
            baseline_histories = []
            baseline_grids = []
            for seed in range(args.n_seeds):
                r = _run_one(args, training_mode, sweep_mode, cell,
                             0.0, baseline_tv, seed)
                baseline_histories.append(r['history'])
                baseline_grids.append(r.get('coverage_grids'))
                all_records.append({
                    'training_mode': training_mode,
                    'sweep_mode': sweep_mode, 'alpha': 0.0,
                    'teacher_value': baseline_tv, 'seed': seed,
                    'history': r['history'],
                })

            mode_out = os.path.join(args.output_dir, training_mode, sweep_mode)
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

            if training_mode == 'sample':
                plot_pg_cosine(
                    histories_by_teacher=dict(histories_by_teacher),
                    mode=sweep_mode,
                    out_path=os.path.join(mode_out, 'pg_cosine.png'),
                    cell_info=cell_info,
                )
                plot_pg_variance(
                    histories_by_teacher=dict(histories_by_teacher),
                    baseline_history_alpha_zero=baseline_histories,
                    mode=sweep_mode,
                    out_dir=mode_out,
                    cell_info=cell_info,
                )
            elif training_mode == 'exact':
                plot_u_cosine(
                    histories_by_teacher=dict(histories_by_teacher),
                    mode=sweep_mode,
                    out_path=os.path.join(mode_out, 'u_cosine_npg.png'),
                    cell_info=cell_info,
                    centering='npg',
                )
                plot_u_cosine(
                    histories_by_teacher=dict(histories_by_teacher),
                    mode=sweep_mode,
                    out_path=os.path.join(mode_out, 'u_cosine_pinv.png'),
                    cell_info=cell_info,
                    centering='pinv',
                )
            if args.track_coverage:
                _emit_coverage_plots(
                    out_dir=mode_out, mode=sweep_mode, cell_info=cell_info,
                    histories_by_teacher=dict(histories_by_teacher),
                    baseline_histories=baseline_histories,
                    grids_by_teacher=dict(grids_by_teacher),
                    baseline_grids=baseline_grids,
                    top_tv=tvs_alpha1[-1], grid_size=args.grid_size,
                )
            print(f'Wrote {mode_out}/*.png')

    print('NPG_DIAGNOSTICS_DONE')


if __name__ == '__main__':
    main()
