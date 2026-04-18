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

                for i in range(len(lrs)):
                    for j in range(len(tpus)):
                        v = grid_reward[i, j]
                        if np.isnan(v):
                            continue
                        color = 'white' if v < 0.6 else 'black'
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                fontsize=7, color=color)
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

                # Best combo annotation
                best_text = (
                    rf'Best: LR={cal["best_lr"]}, TPU={cal["best_traj_per_update"]}'
                    f'\n'
                    rf'$T_{{\mathrm{{sat}}}}$={cal["T_sat"]}, '
                    f'budgets={cal["budgets"]}'
                )
                ax.text(0.02, 0.02, best_text,
                        transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=5.5, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', edgecolor='lightgray',
                                  alpha=0.9))

        if im is not None:
            fig.subplots_adjust(right=0.88)
            cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('Final mean reward', fontsize=9)

        goal_label = '1 goal (zeta sweep)' if ng == 1 else f'{ng} goals (capability sweep)'
        fig.suptitle(
            rf'Sample-mode calibration — {goal_label}'
            '\n'
            r'Each subplot: final mean reward of vanilla NPG at budget cap '
            r'(2000 obs) for each (LR $\times$ trajectories/update) combo.'
            '\n'
            r'Red outline = best combo chosen for the sample-mode sweep. '
            r'Rows: goal distance $d$. Columns: episode horizon $H$.',
            fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.88, 0.90])
        save_path = os.path.join(output_dir,
                                  f'calibration_sample_ng{ng}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

    # --- Figure 2: T_sat comparison (exact vs sample, grouped by config) ---
    exact_path = CALIBRATION_PATH
    exact_cal = {}
    if os.path.exists(exact_path):
        with open(exact_path) as f:
            exact_cal = json.load(f)

    # Group by distance with visual separators
    groups = {}  # dist -> [(label, exact_tsat, sample_tsat, exact_budgets, sample_budgets)]
    for key in configs:
        cal = all_results[key]
        exact_key = key.replace('_grid=9', f'_lr={LR}_grid=9')
        ecal = exact_cal.get(exact_key, {})
        # Parse distance from key
        dist = int(key.split('_')[0].split('=')[1])
        groups.setdefault(dist, []).append((
            _human_config_label(key),
            ecal.get('T_sat', 0),
            cal['T_sat'],
            ecal.get('budgets', []),
            cal['budgets'],
        ))

    # Flatten with gaps between distance groups
    labels, exact_vals, sample_vals = [], [], []
    exact_bud, sample_bud = [], []
    group_boundaries = []  # x positions where vertical separators go
    pos = 0
    for di, dist in enumerate(sorted(groups)):
        if di > 0:
            pos += 0.8  # gap between groups
            group_boundaries.append(pos - 0.4)
        for label, et, st, eb, sb in groups[dist]:
            labels.append(label)
            exact_vals.append(et)
            sample_vals.append(st)
            exact_bud.append(eb)
            sample_bud.append(sb)
            pos += 1

    x = np.arange(len(labels))
    # Shift x positions to include gaps
    x_pos = []
    pos = 0
    gi = 0
    for i in range(len(labels)):
        if gi < len(group_boundaries) and pos >= group_boundaries[gi]:
            pos += 0.8
            gi += 1
        x_pos.append(pos)
        pos += 1
    x_pos = np.array(x_pos)

    fig2, ax2 = plt.subplots(figsize=(max(14, 0.9 * len(labels)), 6))
    width = 0.35

    bars_exact = ax2.bar(x_pos - width / 2, exact_vals, width,
                          label='Exact (update steps)',
                          color='steelblue', alpha=0.8)
    bars_sample = ax2.bar(x_pos + width / 2, sample_vals, width,
                           label='Sample (observations)',
                           color='coral', alpha=0.8)

    for i, bar in enumerate(bars_exact):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 8,
                     str(int(h)),
                     ha='center', va='bottom', fontsize=7, fontweight='bold')
            if exact_bud[i]:
                bstr = ','.join(str(b) for b in exact_bud[i])
                ax2.text(bar.get_x() + bar.get_width() / 2, h * 0.5,
                         bstr, ha='center', va='center', fontsize=5,
                         rotation=90, color='white', alpha=0.9)
    for i, bar in enumerate(bars_sample):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 8,
                     str(int(h)),
                     ha='center', va='bottom', fontsize=7, fontweight='bold')
            if sample_bud[i]:
                bstr = ','.join(str(b) for b in sample_bud[i])
                ax2.text(bar.get_x() + bar.get_width() / 2, h * 0.5,
                         bstr, ha='center', va='center', fontsize=5,
                         rotation=90, color='white', alpha=0.9)

    # Distance group separators and labels
    for bx in group_boundaries:
        ax2.axvline(bx, color='gray', linewidth=1.0, linestyle='--', alpha=0.5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel(r'$T_{\mathrm{sat}}$', fontsize=10)
    ax2.set_title(
        r'Saturation budget $T_{\mathrm{sat}}$ — exact vs sample mode'
        '\n'
        r'$T_{\mathrm{sat}}$ = first step/obs where vanilla NPG reaches '
        r'$\geq 0.95$ mean reward. Bold = $T_{\mathrm{sat}}$; '
        r'vertical text = sweep budget breakpoints '
        r'$[T/5,\;T/3,\;T,\;2T]$.'
        '\n'
        r'Grouped by goal distance $d$. '
        r'$H$ = episode horizon (small=8 / large=36 for 9$\times$9 grid). '
        r'Goals = number of reward states.',
        fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'calibration_tsat_comparison.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {save_path2}")


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
