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
SAMPLE_LR_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SAMPLE_TRAJ_PER_UPDATE = [1, 4, 7, 10]
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
                             n_seeds, max_budget, threshold,
                             existing_combos=None):
    """Sweep LR x trajectories_per_update for sample mode.

    Args:
        existing_combos: dict of combo_key -> combo_data from a prior run.
            If provided, combos already in this dict are skipped (extend mode).
    """
    goals = generate_equidistant_goals(grid_size, n_goals, distance=distance)
    horizons = compute_exploration_thresholds(grid_size)
    horizon = horizons[f'horizon_{h_type}']

    combo_results = {}

    for lr in SAMPLE_LR_VALUES:
        for tpu in SAMPLE_TRAJ_PER_UPDATE:
            combo_key = f'lr={lr}_tpu={tpu}'
            if existing_combos and combo_key in existing_combos:
                combo_results[(lr, tpu)] = existing_combos[combo_key]
                continue
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

    # Check if the best combo actually saturated (reached threshold within budget)
    saturated = best['final_reward_mean'] >= threshold

    raw = [max(3, t_sat // 5), max(3, t_sat // 3), t_sat, t_sat * 2]
    budgets = [raw[0]]
    for v in raw[1:]:
        budgets.append(max(v, budgets[-1] + 1))

    return {
        'T_sat': t_sat,
        'saturated': saturated,
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


def _human_config_label(key):
    """Convert 'dist=4_small_ng=1_grid=9' into a readable label."""
    parts = {}
    for token in key.split('_'):
        if '=' in token:
            k, v = token.split('=', 1)
            parts[k] = v
        elif token in ('small', 'large'):
            parts['horizon'] = token
    dist = parts.get('dist', '?')
    h = parts.get('horizon', '?')
    ng = parts.get('ng', '?')
    horizons = compute_exploration_thresholds(GRID_SIZE)
    h_val = horizons.get(f'horizon_{h}', '?')
    goal_label = f'{ng} goal' if ng == '1' else f'{ng} goals'
    return f'd={dist}, H={h_val} ({h}), {goal_label}'


def _plot_sample_calibration_heatmaps(all_results, output_dir):
    """Consolidated sample calibration figures:

    1. (LR × TPU) heatmaps showing T_sat per combo (one subplot per config).
    2. T_sat comparison bar chart: exact vs sample across all configs, with
       the derived sweep budget breakpoints annotated.
    """
    import matplotlib.pyplot as plt

    configs = sorted(all_results.keys())
    if not configs:
        return
    lrs = sorted(SAMPLE_LR_VALUES)
    tpus = sorted(SAMPLE_TRAJ_PER_UPDATE)

    # --- Figure 1: one figure PER N_GOALS ---
    # Rows = distance (4, 6, 7, 8), cols = horizon (small, large).
    # Each subplot = (LR × TPU) heatmap of final mean reward.
    # Two figures total: ng=1 (zeta sweep), ng=3 (capability sweep).
    horizons = compute_exploration_thresholds(GRID_SIZE)
    h_vals = {'small': horizons['horizon_small'], 'large': horizons['horizon_large']}

    for ng in N_GOALS_LIST:
        n_rows = len(DISTANCES)
        n_cols = len(HORIZON_TYPES)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4.5 * n_cols + 1.5,
                                           3.2 * n_rows + 2.0),
                                  squeeze=False)
        im = None
        for ri, dist in enumerate(DISTANCES):
            for ci, h_type in enumerate(HORIZON_TYPES):
                ax = axes[ri, ci]
                key = f'dist={dist}_{h_type}_ng={ng}_grid={GRID_SIZE}'
                cal = all_results.get(key)
                if cal is None:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10, color='gray')
                    ax.axis('off')
                    continue

                combos = cal['all_combos']
                grid_reward = np.full((len(lrs), len(tpus)), np.nan)
                for i, lr in enumerate(lrs):
                    for j, tpu in enumerate(tpus):
                        ck = f'lr={lr}_tpu={tpu}'
                        if ck in combos:
                            grid_reward[i, j] = combos[ck]['final_reward_mean']

                im = ax.imshow(grid_reward, aspect='auto', cmap='viridis',
                                vmin=0, vmax=1, origin='lower')

                # Build T_sat grid for annotation
                grid_tsat = np.full((len(lrs), len(tpus)), np.nan)
                for i, lr in enumerate(lrs):
                    for j, tpu in enumerate(tpus):
                        ck = f'lr={lr}_tpu={tpu}'
                        if ck in combos:
                            grid_tsat[i, j] = combos[ck]['T_sat_max']

                for i in range(len(lrs)):
                    for j in range(len(tpus)):
                        v = grid_reward[i, j]
                        if np.isnan(v):
                            continue
                        color = 'white' if v < 0.6 else 'black'
                        # Reward on top line, T_sat below
                        ax.text(j, i - 0.15, f'{v:.2f}', ha='center', va='center',
                                fontsize=7, fontweight='bold', color=color)
                        ts = grid_tsat[i, j]
                        if not np.isnan(ts):
                            ts_label = str(int(ts)) if ts < 2000 else 'cap'
                            ax.text(j, i + 0.2, ts_label, ha='center', va='center',
                                    fontsize=5.5, color=color, alpha=0.8)
                        if lrs[i] == cal['best_lr'] and tpus[j] == cal['best_traj_per_update']:
                            ax.add_patch(plt.Rectangle(
                                (j - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2, zorder=5))

                ax.set_xticks(range(len(tpus)))
                ax.set_xticklabels([str(t) for t in tpus], fontsize=7)
                ax.set_yticks(range(len(lrs)))
                ax.set_yticklabels([str(l) for l in lrs], fontsize=7)
                if ri == n_rows - 1:
                    ax.set_xlabel('trajectories / update', fontsize=9)
                if ci == 0:
                    ax.set_ylabel(rf'$d={dist}$' + '\nlearning rate', fontsize=9)
                if ri == 0:
                    ax.set_title(rf'$H={h_vals[h_type]}$ ({h_type} horizon)',
                                 fontsize=10, fontweight='bold')

        if im is not None:
            fig.subplots_adjust(right=0.88)
            cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('Final mean reward', fontsize=9)

        goal_label = '1 goal (zeta sweep)' if ng == 1 else f'{ng} goals (capability sweep)'
        fig.suptitle(
            rf'Sample-mode calibration — {goal_label}'
            '\n'
            r'Each cell: final mean reward (bold) and $T_{\mathrm{sat}}$ '
            r'(small, or "cap" if $\geq$ 2000 obs) per (LR $\times$ TPU) combo.'
            '\n'
            r'Red outline = best combo chosen for sweep '
            r'(fastest $T_{\mathrm{sat}}$ with reward $\geq 0.95$). '
            r'Rows: goal distance $d$. Columns: episode horizon $H$.',
            fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.88, 0.90])
        save_path = os.path.join(output_dir,
                                  f'calibration_sample_ng{ng}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

    # --- Figure 2: T_sat heatmaps (exact and sample side by side) ---
    # One figure per n_goals. Two heatmaps: rows=distance, cols=horizon.
    # Left = exact T_sat (update steps), right = sample T_sat (observations).
    # Each cell shows T_sat (bold) and budget breakpoints (small).
    exact_path = CALIBRATION_PATH
    exact_cal = {}
    if os.path.exists(exact_path):
        with open(exact_path) as f:
            exact_cal = json.load(f)

    # Single figure: 2×2 grid. Top row = 1 goal (zeta), bottom = 3 goals (cap).
    # Left col = exact, right col = sample.
    n_d = len(DISTANCES)
    n_h = len(HORIZON_TYPES)
    fig, axes = plt.subplots(len(N_GOALS_LIST), 2,
                              figsize=(4.5 * 2 + 2.0, 3.0 * len(N_GOALS_LIST) + 2.0),
                              squeeze=False)

    for ngi, ng in enumerate(N_GOALS_LIST):
        exact_grid = np.full((n_d, n_h), np.nan)
        sample_grid = np.full((n_d, n_h), np.nan)
        exact_bud = {}
        sample_bud = {}
        for ri, dist in enumerate(DISTANCES):
            for ci, h_type in enumerate(HORIZON_TYPES):
                key = f'dist={dist}_{h_type}_ng={ng}_grid={GRID_SIZE}'
                exact_key = key.replace('_grid=9', f'_lr={LR}_grid=9')
                ecal = exact_cal.get(exact_key, {})
                cal = all_results.get(key)
                if ecal.get('T_sat'):
                    exact_grid[ri, ci] = ecal['T_sat']
                    exact_bud[(ri, ci)] = ecal.get('budgets', [])
                if cal:
                    sample_grid[ri, ci] = cal['T_sat']
                    sample_bud[(ri, ci)] = cal.get('budgets', [])

        goal_label = '1 goal (zeta sweep)' if ng == 1 else f'{ng} goals (capability sweep)'

        for mi, (ax, grid, bmap, mode_label) in enumerate([
            (axes[ngi, 0], exact_grid, exact_bud, 'Exact (update steps)'),
            (axes[ngi, 1], sample_grid, sample_bud, 'Sample (observations)'),
        ]):
            vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1
            im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', vmin=0,
                            vmax=vmax, origin='upper')
            for ri in range(n_d):
                for ci in range(n_h):
                    v = grid[ri, ci]
                    if np.isnan(v):
                        continue
                    color = 'black' if v < vmax * 0.7 else 'white'
                    ax.text(ci, ri, str(int(v)),
                            ha='center', va='center', fontsize=12,
                            fontweight='bold', color=color)

            ax.set_xticks(range(n_h))
            ax.set_xticklabels([rf'$H={h_vals[h]}$ ({h})' for h in HORIZON_TYPES],
                               fontsize=8)
            ax.set_yticks(range(n_d))
            ax.set_yticklabels([rf'$d={d}$' for d in DISTANCES], fontsize=9)
            if ngi == 0:
                ax.set_title(mode_label, fontsize=11, fontweight='bold')
            if mi == 0:
                ax.set_ylabel(goal_label + '\n', fontsize=10, fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.8, label=r'$T_{\mathrm{sat}}$',
                         pad=0.02)

    fig.suptitle(
        r'Vanilla NPG saturation budget $T_{\mathrm{sat}}$ — exact vs sample'
        '\n'
        r'$T_{\mathrm{sat}}$ = first step/obs where vanilla NPG ($\alpha=0$, '
        r'no teacher) reaches $\geq 0.95$ mean reward.'
        '\n'
        r'Rows: goal distance $d$. Columns: horizon $H$. '
        r'Sweep uses budget breakpoints $[T/5,\;T/3,\;T,\;2T]$.',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    save_path = os.path.join(output_dir, 'calibration_tsat.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def _plot_sample_calibration_curves(cal_result, config_key, output_dir):
    """Legacy per-config plot — kept for backward compat but superseded by
    _plot_sample_calibration_heatmaps which produces consolidated views.
    """
    pass


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

        if key in calibration and not args.force and not args.extend:
            existing = calibration[key]
            print(f"  [{i+1}/{total}] {key}: CACHED "
                  f"(lr={existing['best_lr']}, tpu={existing['best_traj_per_update']}, "
                  f"T_sat={existing['T_sat']})",
                  flush=True)
            continue

        # In extend mode, pass existing combos so only new LR/TPU values run
        existing_combos = None
        if args.extend and key in calibration:
            existing_combos = calibration[key].get('all_combos', {})
            n_existing = len(existing_combos)
            n_total = len(SAMPLE_LR_VALUES) * len(SAMPLE_TRAJ_PER_UPDATE)
            print(f"  [{i+1}/{total}] {key}: extending "
                  f"({n_existing} cached, {n_total - n_existing} new)...",
                  flush=True)
        else:
            print(f"  [{i+1}/{total}] {key}: calibrating...", flush=True)

        result = calibrate_sample_single(
            dist, h_type, n_goals, GRID_SIZE,
            args.n_seeds, args.max_budget, args.threshold,
            existing_combos=existing_combos,
        )
        calibration[key] = result
        print(f"    Best: lr={result['best_lr']}, tpu={result['best_traj_per_update']}, "
              f"T_sat={result['T_sat']}, budgets={result['budgets']}", flush=True)

        _plot_sample_calibration_curves(result, key, fig_dir)

        with open(cal_path, 'w') as f:
            json.dump(calibration, f, indent=2)

    print(f"\nSample calibration saved to {cal_path}", flush=True)
    print(f"Calibration plots saved to {fig_dir}/", flush=True)

    # Consolidated plots across all configs
    _plot_sample_calibration_heatmaps(calibration, fig_dir)

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
    parser.add_argument('--extend', action='store_true',
                        help='Keep existing combos, only run new LR/TPU values')
    parser.add_argument('--plot-only', action='store_true',
                        help='Regenerate plots from existing calibration JSON (no re-run)')
    args = parser.parse_args()

    if args.plot_only:
        if args.mode == 'sample':
            with open(SAMPLE_CALIBRATION_PATH) as f:
                calibration = json.load(f)
            fig_dir = 'results/calibration_sample_figures'
            os.makedirs(fig_dir, exist_ok=True)
            _plot_sample_calibration_heatmaps(calibration, fig_dir)
        else:
            print("--plot-only currently only supported for sample mode")
    elif args.mode == 'sample':
        _run_sample_calibration(args)
    else:
        _run_exact_calibration(args)


if __name__ == '__main__':
    main()
