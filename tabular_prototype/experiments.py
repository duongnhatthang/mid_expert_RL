"""Experiment orchestration: suite, 2x2, and analysis."""

import csv
import os
import numpy as np
from typing import List, Dict, Tuple, Optional

from .config import compute_gamma_from_horizon
from .environment import GridEnv, generate_equidistant_goals, compute_exploration_thresholds
from .teacher import compute_teacher_values_auto
from .student import TabularSoftmaxPolicy, collect_trajectories
from .training import (
    compute_pav_rl_gradient,
    update_policy,
    exact_npg_update,
    evaluate_policy,
    compute_student_qvalues,
    compute_state_action_visitation,
    visitation_metrics,
)
from .visualization import visualize_state_visitation, visualize_visitation_comparison_grid


def run_experiment(
    grid_size: int = 10,
    goals: List[Tuple[int, int]] = [(9, 9), (0, 9), (9, 0)],
    teacher_capacity: int = 1,
    zeta: Optional[float] = None,
    horizon: int = 50,
    sample_budget: int = 10000,
    alpha: float = 0.5,
    lr: float = 0.5,
    trajectories_per_update: int = 10,
    seed: int = 0,
    eval_interval: int = 10,
    eval_n_episodes: int = 20,
    exact_gradient: bool = True,
) -> Dict:
    """Run a single experiment and return results dict.

    Args:
        exact_gradient: If True (default), use exact NPG update where budget
            = number of update steps. If False, use sample-based gradient
            where budget = number of observations.
    """
    rng = np.random.default_rng(seed)
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)

    gamma = compute_gamma_from_horizon(horizon)
    if teacher_capacity == -1 and zeta is None:
        Q_mu, V_mu = None, None
    elif zeta is not None:
        Q_mu, V_mu, gamma = compute_teacher_values_auto(
            env, env.goals, zeta=zeta, gamma=gamma)
    else:
        known_goals = goals[:teacher_capacity] if teacher_capacity > 0 else goals
        effective_zeta = 0.0 if teacher_capacity == 0 else 1.0
        Q_mu, V_mu, gamma = compute_teacher_values_auto(
            env, known_goals, zeta=effective_zeta, gamma=gamma)

    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)

    total_steps = 0
    update_count = 0
    history = []
    cumulative_visitation = np.zeros((env.n_states, env.n_actions))
    gamma_undiscounted = 1.0 - 1e-8

    if exact_gradient:
        # Exact NPG mode: budget = number of update steps
        for step in range(sample_budget):
            Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)
            exact_npg_update(policy, Q_pi, Q_mu, V_mu, alpha, lr)
            update_count += 1

            is_last_step = (step == sample_budget - 1)
            if update_count % eval_interval == 0 or is_last_step:
                eval_results = evaluate_policy(
                    env, policy, n_episodes=eval_n_episodes, rng=rng
                )
                start_idx = env.state_to_idx(env.start)
                _, V_pi_undiscounted = compute_student_qvalues(
                    env, policy, gamma_undiscounted
                )
                history.append({
                    'steps': update_count,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': 0,
                    'state_entropy': 0.0,
                })
    else:
        # Sample-based mode: budget = number of observations
        while total_steps < sample_budget:
            trajectories = collect_trajectories(
                env, policy, trajectories_per_update, rng
            )

            # Truncate to respect budget
            step_counts = [len(traj) for traj in trajectories]
            remaining = sample_budget - total_steps
            if sum(step_counts) > remaining:
                cumsum = np.cumsum(step_counts)
                keep = int(np.searchsorted(cumsum, remaining, side='right'))
                trajectories = trajectories[:max(1, keep)]

            total_steps += sum(len(traj) for traj in trajectories)

            step_vis = compute_state_action_visitation(
                trajectories, env.n_states, env.n_actions
            )
            cumulative_visitation += step_vis

            Q_pi, V_pi = compute_student_qvalues(env, policy, gamma)

            grad = compute_pav_rl_gradient(
                policy, trajectories, Q_mu, V_mu, alpha, gamma, Q_pi=Q_pi
            )
            update_policy(policy, grad, lr)
            update_count += 1

            if update_count % eval_interval == 0:
                eval_results = evaluate_policy(
                    env, policy, n_episodes=eval_n_episodes, rng=rng
                )
                vis_m = visitation_metrics(cumulative_visitation)
                start_idx = env.state_to_idx(env.start)
                _, V_pi_undiscounted = compute_student_qvalues(
                    env, policy, gamma_undiscounted
                )
                history.append({
                    'steps': total_steps,
                    'mean_reward': eval_results['mean_reward'],
                    'goal_rate': eval_results['goal_rate'],
                    'exact_V_start': float(V_pi[start_idx]),
                    'exact_V_start_undiscounted': float(
                        V_pi_undiscounted[start_idx]
                    ),
                    'unique_sa': vis_m['unique_sa'],
                    'state_entropy': vis_m['state_entropy'],
                })

    # In exact mode, collect visitation from the final policy via evaluation trajectories
    if exact_gradient and cumulative_visitation.sum() == 0:
        eval_trajs = collect_trajectories(env, policy, 100, rng)
        cumulative_visitation = compute_state_action_visitation(
            eval_trajs, env.n_states, env.n_actions
        )

    final_eval = evaluate_policy(env, policy, n_episodes=100, rng=rng)
    final_vis = visitation_metrics(cumulative_visitation)

    return {
        'seed': seed,
        'teacher_capacity': teacher_capacity if zeta is None else None,
        'zeta': zeta,
        'sample_budget': sample_budget,
        'horizon': horizon,
        'alpha': alpha,
        'lr': lr,
        'final_mean_reward': final_eval['mean_reward'],
        'final_std_reward': final_eval['std_reward'],
        'final_goal_rate': final_eval['goal_rate'],
        'final_unique_sa': final_vis['unique_sa'],
        'final_unique_states': final_vis['unique_states'],
        'final_state_entropy': final_vis['state_entropy'],
        'final_sa_entropy': final_vis['sa_entropy'],
        'final_total_visits': final_vis['total_visits'],
        'visitation_counts': cumulative_visitation,
        'history': history,
        'budget_mode': 'exact' if exact_gradient else 'sample',
        'exact_gradient': exact_gradient,
    }


def run_experiment_suite(
    grid_size: int = 10,
    goals: Optional[List[Tuple[int, int]]] = None,
    output_file: str = 'results/tabular_results.csv',
):
    """Run suite of experiments to test hypothesis."""
    if goals is None:
        goals = [(grid_size - 1, grid_size - 1), (0, grid_size - 1), (grid_size - 1, 0)]

    configs = []
    teacher_capacities = [-1] + list(range(len(goals) + 1))
    for teacher_capacity in teacher_capacities:
        for sample_budget in [1000, 5000, 10000]:
            for alpha in [0.0, 0.5, 1.0]:
                for seed in range(10):
                    configs.append({
                        'grid_size': grid_size,
                        'goals': goals,
                        'teacher_capacity': teacher_capacity,
                        'horizon': 50,
                        'sample_budget': sample_budget,
                        'alpha': alpha,
                        'lr': 0.5,
                        'seed': seed
                    })

    print(f"Running {len(configs)} experiments...")

    results = []
    for i, config in enumerate(configs):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(configs)}")
        result = run_experiment(**config)
        results.append(result)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['seed', 'teacher_capacity', 'sample_budget', 'horizon',
                      'alpha', 'lr', 'final_mean_reward', 'final_std_reward', 'final_goal_rate',
                      'final_unique_sa', 'final_unique_states',
                      'final_state_entropy', 'final_sa_entropy', 'final_total_visits']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Results saved to {output_file}")
    return results


def _save_visitation_heatmaps(
    all_results: List[Dict],
    env: GridEnv,
    goals: List[Tuple[int, int]],
    figures_dir: str,
    teacher_key: str = 'teacher_capacity',
):
    """
    Aggregate visitation counts across seeds and save one heatmap per
    (condition, teacher_setting) combination.

    Args:
        all_results: list of result dicts from run_experiment (must contain
                     'visitation_counts', 'budget_type', 'horizon_type', and
                     the column identified by *teacher_key*).
        env:         GridEnv used in the experiments.
        goals:       goal positions for annotation.
        figures_dir: directory to write PNG files into.
        teacher_key: 'teacher_capacity' for k-cap or 'zeta' for zeta runs.
    """
    vis_dir = os.path.join(figures_dir, 'visitation')
    os.makedirs(vis_dir, exist_ok=True)

    # Group results by (budget_type, horizon_type, teacher_setting)
    from collections import defaultdict
    groups: Dict[tuple, List[np.ndarray]] = defaultdict(list)
    for r in all_results:
        key = (r['budget_type'], r['horizon_type'], r[teacher_key])
        if 'visitation_counts' in r:
            groups[key].append(r['visitation_counts'])

    for (bt, ht, teacher_val), vis_list in groups.items():
        avg_vis = np.mean(vis_list, axis=0)

        if teacher_key == 'zeta':
            label = f"zeta{teacher_val:.2f}"
            title_label = f"\u03b6={teacher_val:.2f}"
        else:
            cap_labels = {-1: 'no_teacher', 0: 'random'}
            label = cap_labels.get(teacher_val, f"cap{teacher_val}")
            title_label = {-1: 'no teacher', 0: 'random'}.get(
                teacher_val, f"cap={teacher_val}"
            )

        cond_str = f"{bt}_budget_{ht}_horizon"
        save_path = os.path.join(vis_dir, f"visitation_{cond_str}_{label}.png")

        budget_val = int(np.mean([
            r['sample_budget'] for r in all_results
            if r['budget_type'] == bt and r['horizon_type'] == ht
        ]))
        horizon_val = int(np.mean([
            r['horizon'] for r in all_results
            if r['budget_type'] == bt and r['horizon_type'] == ht
        ]))

        visualize_state_visitation(
            env, avg_vis,
            title=(f"State Visitation: {title_label}\n"
                   f"({bt} budget={budget_val}, {ht} horizon={horizon_val})"),
            goals=goals,
            save_path=save_path,
        )

    print(f"Visitation heatmaps saved: {len(groups)} files in {vis_dir}/")


def _save_visitation_comparison_grid(
    all_results: List[Dict],
    env: GridEnv,
    goals: List[Tuple[int, int]],
    figures_dir: str,
    teacher_key: str = 'teacher_capacity',
):
    """
    Save a single multi-panel figure comparing visitation across teacher
    settings (rows) and budget×horizon conditions (columns).
    """
    from collections import defaultdict

    # Group and average visitation across seeds
    groups: Dict[tuple, List[np.ndarray]] = defaultdict(list)
    for r in all_results:
        key = (r[teacher_key], (r['budget_type'], r['horizon_type']))
        if 'visitation_counts' in r:
            groups[key].append(r['visitation_counts'])

    visitation_data = {}
    for (teacher_val, cond), vis_list in groups.items():
        visitation_data[(teacher_val, cond)] = np.mean(vis_list, axis=0)

    # Build ordered keys
    row_keys = sorted(set(k[0] for k in visitation_data.keys()),
                      key=lambda x: (isinstance(x, float), x))
    col_keys = [('low', 'small'), ('low', 'large'), ('high', 'small'), ('high', 'large')]

    if teacher_key == 'zeta':
        row_label_fn = lambda k: f'ζ={k:.2f}'
    else:
        cap_labels = {-1: 'no teacher', 0: 'random'}
        row_label_fn = lambda k: cap_labels.get(k, f'cap={k}')

    col_label_fn = lambda k: f'{k[0]} budget\n{k[1]} horizon'

    save_path = os.path.join(figures_dir, 'visitation_comparison_grid.png')
    visualize_visitation_comparison_grid(
        env, visitation_data, row_keys, col_keys,
        row_label_fn, col_label_fn,
        goals=goals,
        suptitle=f'State Visitation: {teacher_key} × Condition',
        save_path=save_path,
    )


def run_2x2_exploration_experiment(
    grid_size: int = 8,
    n_seeds: int = 10,
    n_goals: int = 3,
    alpha: float = 0.5,
    output_file: str = 'results/exploration_2x2_results.csv',
):
    """
    Run 2x2 experiment matrix: Budget (Low/High) x Horizon (Small/Large).

    Tests whether the mid-capacity teacher advantage depends on
    having limited overall samples (budget) and/or limited per-episode
    exploration (horizon).
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    goals = generate_equidistant_goals(grid_size, n_goals)
    teacher_capacities = [-1] + list(range(n_goals + 1))
    thresholds = compute_exploration_thresholds(grid_size)

    print("=" * 70)
    print("2x2 EXPLORATION EXPERIMENT")
    print("=" * 70)
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of goals: {n_goals}")
    print(f"Goals: {goals}")
    print(f"Teacher capacities: {teacher_capacities} (-1=no signal, 0=uniform-random, {n_goals}=best)")
    print(f"Total states: {thresholds['n_states']}")
    print(f"Total state-actions: {thresholds['n_state_actions']}")
    print()
    print("Thresholds:")
    print(f"  Budget LOW:    {thresholds['budget_low']:,} steps (~3 visits/state-action)")
    print(f"  Budget HIGH:   {thresholds['budget_high']:,} steps (~30 visits/state-action)")
    print(f"  Horizon SMALL: {thresholds['horizon_small']} steps (barely reach one corner)")
    print(f"  Horizon LARGE: {thresholds['horizon_large']} steps (visit multiple goals)")
    print()

    conditions = [
        {'name': 'Low Budget + Small Horizon', 'budget': thresholds['budget_low'],
         'horizon': thresholds['horizon_small'], 'budget_type': 'low', 'horizon_type': 'small'},
        {'name': 'Low Budget + Large Horizon', 'budget': thresholds['budget_low'],
         'horizon': thresholds['horizon_large'], 'budget_type': 'low', 'horizon_type': 'large'},
        {'name': 'High Budget + Small Horizon', 'budget': thresholds['budget_high'],
         'horizon': thresholds['horizon_small'], 'budget_type': 'high', 'horizon_type': 'small'},
        {'name': 'High Budget + Large Horizon', 'budget': thresholds['budget_high'],
         'horizon': thresholds['horizon_large'], 'budget_type': 'high', 'horizon_type': 'large'},
    ]

    all_results = []

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Running: {cond['name']}")
        print(f"  Budget: {cond['budget']:,}, Horizon: {cond['horizon']}")
        print(f"{'='*60}")

        for teacher_cap in teacher_capacities:
            print(f"  Teacher capacity {teacher_cap}...", end=" ")
            cap_results = []

            for seed in range(n_seeds):
                result = run_experiment(
                    grid_size=grid_size,
                    goals=goals,
                    teacher_capacity=teacher_cap,
                    horizon=cond['horizon'],
                    sample_budget=cond['budget'],
                    alpha=alpha,
                    lr=0.5,
                    trajectories_per_update=10,
                    seed=seed
                )
                result['budget_type'] = cond['budget_type']
                result['horizon_type'] = cond['horizon_type']
                result['condition'] = cond['name']
                result['n_goals'] = n_goals
                cap_results.append(result)
                all_results.append(result)

            rewards = [r['final_mean_reward'] for r in cap_results]
            mean_reward = np.mean(rewards)
            stderr = np.std(rewards) / np.sqrt(len(rewards))
            mean_unique_sa = np.mean([r['final_unique_sa'] for r in cap_results])
            mean_state_ent = np.mean([r['final_state_entropy'] for r in cap_results])
            print(f"reward = {mean_reward:.3f} ± {stderr:.3f}  |  "
                  f"unique_sa = {mean_unique_sa:.0f}, "
                  f"state_entropy = {mean_state_ent:.2f}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['seed', 'teacher_capacity', 'n_goals', 'sample_budget', 'horizon',
                      'alpha', 'lr', 'final_mean_reward', 'final_std_reward',
                      'final_goal_rate',
                      'final_unique_sa', 'final_unique_states',
                      'final_state_entropy', 'final_sa_entropy', 'final_total_visits',
                      'budget_type', 'horizon_type', 'condition']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to {output_file}")
    analyze_2x2_results(all_results, thresholds)

    # Generate visitation heatmaps
    figures_dir = os.path.dirname(output_file) or '.'
    figures_dir = os.path.join(figures_dir, 'figures')
    env = GridEnv(grid_size=grid_size, goals=goals,
                  horizon=thresholds['horizon_large'])
    _save_visitation_heatmaps(
        all_results, env, goals, figures_dir, teacher_key='teacher_capacity'
    )
    _save_visitation_comparison_grid(
        all_results, env, goals, figures_dir, teacher_key='teacher_capacity'
    )

    return all_results


def _detect_saturation(
    history: List[Dict],
    max_budget: int,
    window: int = 10,
    eps: float = 0.005,
    checks_needed: int = 3,
) -> int:
    """Detect when learning curve plateaus. Returns step count at saturation."""
    if len(history) < window:
        return max_budget

    rewards = [h['mean_reward'] for h in history]
    consecutive = 0

    for i in range(window, len(rewards)):
        window_vals = rewards[i - window:i]
        change = abs(max(window_vals) - min(window_vals))
        if change < eps:
            consecutive += 1
            if consecutive >= checks_needed:
                return history[i]['steps']
        else:
            consecutive = 0

    return max_budget


def run_learning_curve_experiment(
    grid_size: int = 10,
    goals: Optional[List[Tuple[int, int]]] = None,
    teacher_capacities: Optional[List[int]] = None,
    horizon: int = 50,
    sample_budget: Optional[int] = None,
    alpha: float = 0.5,
    lr: float = 0.5,
    n_seeds: int = 30,
    trajectories_per_update: int = 10,
    eval_interval: int = 2,
    eval_n_episodes: int = 50,
    exact_gradient: bool = True,
    max_budget: int = 10000,
    saturation_window: int = 10,
    saturation_eps: float = 0.005,
    saturation_checks: int = 3,
) -> Dict[int, list]:
    """
    Run learning curve experiments with optional saturation-based stopping.

    If sample_budget is None, runs vanilla NPG (alpha=0) first to find
    saturation point, then runs all baselines for that many steps.
    If sample_budget is provided, uses that fixed budget (legacy behavior).
    """
    if goals is None:
        goals = [(9, 9), (0, 9), (9, 0)]
    if teacher_capacities is None:
        teacher_capacities = [-1] + list(range(len(goals) + 1))

    if sample_budget is None:
        # Run vanilla NPG to find saturation
        print("  Finding saturation point for Vanilla NPG (α=0)...", flush=True)
        sat_result = run_experiment(
            grid_size=grid_size, goals=goals, teacher_capacity=0,
            horizon=horizon, sample_budget=max_budget, alpha=0.0, lr=lr,
            trajectories_per_update=trajectories_per_update, seed=0,
            eval_interval=eval_interval, eval_n_episodes=eval_n_episodes,
            exact_gradient=exact_gradient,
        )
        effective_budget = _detect_saturation(
            sat_result['history'], max_budget,
            saturation_window, saturation_eps, saturation_checks,
        )
        print(f"  Saturation at step {effective_budget} (max was {max_budget})")
    else:
        effective_budget = sample_budget

    histories: Dict[int, list] = {}

    for cap in teacher_capacities:
        histories[cap] = []
        print(f"  Teacher capacity {cap}...", end=" ", flush=True)
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size, goals=goals, teacher_capacity=cap,
                horizon=horizon, sample_budget=effective_budget,
                alpha=alpha, lr=lr,
                trajectories_per_update=trajectories_per_update,
                seed=seed, eval_interval=eval_interval,
                eval_n_episodes=eval_n_episodes,
                exact_gradient=exact_gradient,
            )
            histories[cap].append(result['history'])
        rewards = [h[-1]['mean_reward'] for h in histories[cap] if h]
        unique_sas = [h[-1].get('unique_sa', 0) for h in histories[cap] if h]
        state_ents = [h[-1].get('state_entropy', 0) for h in histories[cap] if h]
        print(f"final reward = {np.mean(rewards):.3f} ± "
              f"{np.std(rewards)/np.sqrt(len(rewards)):.3f}  |  "
              f"unique_sa = {np.mean(unique_sas):.0f}, "
              f"state_entropy = {np.mean(state_ents):.2f}")

    return histories


def analyze_2x2_results(results: List[Dict], thresholds: Dict):
    """Analyze 2x2 exploration experiment results."""
    import pandas as pd

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("2x2 EXPLORATION EXPERIMENT RESULTS")
    print("=" * 70)

    conditions = df['condition'].unique()

    print("\n" + "-" * 70)
    print("Mean Reward ± Stderr by Condition and Teacher Capacity")
    print("-" * 70)

    summary_data = []

    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        print(f"\n{cond}:")

        for cap in sorted(cond_df['teacher_capacity'].unique()):
            cap_data = cond_df[cond_df['teacher_capacity'] == cap]['final_mean_reward']
            mean = cap_data.mean()
            stderr = cap_data.std() / np.sqrt(len(cap_data))
            print(f"  Teacher cap {cap}: {mean:.3f} ± {stderr:.3f}")
            summary_data.append({
                'condition': cond,
                'teacher_capacity': cap,
                'mean_reward': mean,
                'stderr': stderr
            })

    all_capacities = sorted(df['teacher_capacity'].unique())
    max_cap = max(all_capacities)
    mid_cap = max_cap // 2 if max_cap > 1 else 1

    print("\n" + "=" * 70)
    print(f"HYPOTHESIS TEST: Does mid-teacher (cap={mid_cap}) outperform best-teacher (cap={max_cap})?")
    print("=" * 70)

    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        mid = cond_df[cond_df['teacher_capacity'] == mid_cap]['final_mean_reward']
        best = cond_df[cond_df['teacher_capacity'] == max_cap]['final_mean_reward']

        if len(mid) > 0 and len(best) > 0:
            mid_mean = mid.mean()
            best_mean = best.mean()
            diff = mid_mean - best_mean

            print(f"\n{cond}:")
            print(f"  Mid (cap={mid_cap}):  {mid_mean:.3f}")
            print(f"  Best (cap={max_cap}): {best_mean:.3f}")
            print(f"  Difference:   {diff:+.3f}")
            print(f"  Mid > Best:   {'YES' if mid_mean > best_mean else 'NO'}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The 2x2 experiment tests whether the mid-capacity teacher advantage depends on:

1. LOW BUDGET + SMALL HORIZON:
   - Most constrained setting
   - Hypothesis: Mid-teacher should be BEST (simple policy learnable in few samples)

2. LOW BUDGET + LARGE HORIZON:
   - Can explore widely per episode, but limited total samples
   - Hypothesis: Mid-teacher may still be better (limited total learning)

3. HIGH BUDGET + SMALL HORIZON:
   - Many samples, but each episode is short
   - Hypothesis: Effect may weaken (enough samples to learn despite complexity)

4. HIGH BUDGET + LARGE HORIZON:
   - Unconstrained setting
   - Hypothesis: Best-teacher should be BEST (enough samples to learn complex policy)

Expected pattern:
- Mid > Best advantage should be STRONGEST in Low Budget conditions
- Mid > Best advantage should WEAKEN or REVERSE in High Budget conditions
""")

    return summary_data


def run_2x2_exploration_experiment_zeta(
    grid_size: int = 8,
    n_seeds: int = 10,
    n_goals: int = 3,
    alpha: float = 0.5,
    zeta_values: Optional[List[float]] = None,
    output_file: str = 'results/exploration_2x2_zeta_results.csv',
):
    """
    2x2 experiment (Budget × Horizon) using the continuous zeta parameterisation.

    Each teacher is the mixture policy mu(zeta) = zeta*pi* + (1-zeta)*pi_random.
    zeta=0 → pure random, zeta=1 → pure optimal (best teacher).
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    if zeta_values is None:
        zeta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    goals = generate_equidistant_goals(grid_size, n_goals)
    thresholds = compute_exploration_thresholds(grid_size)

    print("=" * 70)
    print("2x2 EXPLORATION EXPERIMENT (zeta parameterisation)")
    print("=" * 70)
    print(f"Grid size: {grid_size}x{grid_size},  Goals: {n_goals},  Zeta values: {zeta_values}")
    print(f"Budget LOW:  {thresholds['budget_low']:,}    Budget HIGH: {thresholds['budget_high']:,}")
    print(f"Horizon SMALL: {thresholds['horizon_small']}   Horizon LARGE: {thresholds['horizon_large']}")
    print()

    conditions = [
        {'name': 'Low Budget + Small Horizon',  'budget': thresholds['budget_low'],
         'horizon': thresholds['horizon_small'], 'budget_type': 'low',  'horizon_type': 'small'},
        {'name': 'Low Budget + Large Horizon',  'budget': thresholds['budget_low'],
         'horizon': thresholds['horizon_large'], 'budget_type': 'low',  'horizon_type': 'large'},
        {'name': 'High Budget + Small Horizon', 'budget': thresholds['budget_high'],
         'horizon': thresholds['horizon_small'], 'budget_type': 'high', 'horizon_type': 'small'},
        {'name': 'High Budget + Large Horizon', 'budget': thresholds['budget_high'],
         'horizon': thresholds['horizon_large'], 'budget_type': 'high', 'horizon_type': 'large'},
    ]

    all_results = []

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Running: {cond['name']}")
        print(f"  Budget: {cond['budget']:,}, Horizon: {cond['horizon']}")
        print(f"{'='*60}")

        for zeta in zeta_values:
            print(f"  zeta={zeta:.2f}...", end=" ", flush=True)
            zeta_results = []

            for seed in range(n_seeds):
                result = run_experiment(
                    grid_size=grid_size,
                    goals=goals,
                    zeta=zeta,
                    horizon=cond['horizon'],
                    sample_budget=cond['budget'],
                    alpha=alpha,
                    lr=0.5,
                    trajectories_per_update=10,
                    seed=seed,
                )
                result['budget_type'] = cond['budget_type']
                result['horizon_type'] = cond['horizon_type']
                result['condition'] = cond['name']
                result['n_goals'] = n_goals
                zeta_results.append(result)
                all_results.append(result)

            rewards = [r['final_mean_reward'] for r in zeta_results]
            mean_r = np.mean(rewards)
            stderr = np.std(rewards) / np.sqrt(len(rewards))
            mean_unique_sa = np.mean([r['final_unique_sa'] for r in zeta_results])
            mean_state_ent = np.mean([r['final_state_entropy'] for r in zeta_results])
            print(f"reward = {mean_r:.3f} ± {stderr:.3f}  |  "
                  f"unique_sa = {mean_unique_sa:.0f}, "
                  f"state_entropy = {mean_state_ent:.2f}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['seed', 'zeta', 'n_goals', 'sample_budget', 'horizon',
                      'alpha', 'lr', 'final_mean_reward', 'final_std_reward',
                      'final_goal_rate',
                      'final_unique_sa', 'final_unique_states',
                      'final_state_entropy', 'final_sa_entropy', 'final_total_visits',
                      'budget_type', 'horizon_type', 'condition']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to {output_file}")
    _analyze_2x2_zeta_results(all_results, zeta_values)

    # Generate visitation heatmaps
    figures_dir = os.path.dirname(output_file) or '.'
    figures_dir = os.path.join(figures_dir, 'figures')
    env = GridEnv(grid_size=grid_size, goals=goals,
                  horizon=thresholds['horizon_large'])
    _save_visitation_heatmaps(
        all_results, env, goals, figures_dir, teacher_key='zeta'
    )
    _save_visitation_comparison_grid(
        all_results, env, goals, figures_dir, teacher_key='zeta'
    )

    return all_results


def _analyze_2x2_zeta_results(
    results: List[Dict], zeta_values: List[float]
):
    """Print summary for the 2x2 zeta experiment."""
    import pandas as pd

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("2x2 ZETA EXPERIMENT RESULTS")
    print("=" * 70)

    for cond in df['condition'].unique():
        cond_df = df[df['condition'] == cond]
        print(f"\n{cond}:")
        for z in sorted(cond_df['zeta'].unique()):
            cap_data = cond_df[cond_df['zeta'] == z]['final_mean_reward']
            mean = cap_data.mean()
            stderr = cap_data.std() / np.sqrt(len(cap_data))
            print(f"  zeta={z:.2f}: {mean:.3f} ± {stderr:.3f}")

    print("\n" + "=" * 70)
    print("HYPOTHESIS: Does a mid-zeta teacher outperform the best (zeta=1.0) teacher?")
    print("=" * 70)

    mid_zeta = zeta_values[len(zeta_values) // 2]
    best_zeta = 1.0 if 1.0 in zeta_values else max(zeta_values)

    for cond in df['condition'].unique():
        cond_df = df[df['condition'] == cond]
        mid = cond_df[cond_df['zeta'] == mid_zeta]['final_mean_reward']
        best = cond_df[cond_df['zeta'] == best_zeta]['final_mean_reward']
        if len(mid) > 0 and len(best) > 0:
            diff = mid.mean() - best.mean()
            print(f"\n{cond}:  mid(zeta={mid_zeta}) = {mid.mean():.3f},  "
                  f"best(zeta={best_zeta}) = {best.mean():.3f},  diff = {diff:+.3f}  "
                  f"{'✓ mid wins' if diff > 0 else '✗ best wins'}")


def run_learning_curve_experiment_zeta(
    grid_size: int = 10,
    goals: Optional[List[Tuple[int, int]]] = None,
    zeta_values: Optional[List[float]] = None,
    horizon: int = 50,
    sample_budget: Optional[int] = None,
    alpha: float = 0.5,
    lr: float = 0.5,
    n_seeds: int = 30,
    trajectories_per_update: int = 10,
    eval_interval: int = 2,
    eval_n_episodes: int = 50,
    exact_gradient: bool = True,
    max_budget: int = 10000,
    saturation_window: int = 10,
    saturation_eps: float = 0.005,
    saturation_checks: int = 3,
) -> Dict[float, list]:
    """
    Learning curve experiment using the continuous zeta parameterisation.

    If sample_budget is None, runs vanilla NPG (zeta=0, alpha=0) first to find
    saturation point, then runs all baselines for that many steps.
    If sample_budget is provided, uses that fixed budget (legacy behavior).

    Returns:
        Dict mapping zeta -> list of per-seed histories.
    """
    if goals is None:
        goals = generate_equidistant_goals(grid_size, 3)
    if zeta_values is None:
        zeta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    if sample_budget is None:
        # Run vanilla NPG to find saturation
        print("  Finding saturation point for Vanilla NPG (zeta=0, α=0)...", flush=True)
        sat_result = run_experiment(
            grid_size=grid_size, goals=goals, zeta=0.0,
            horizon=horizon, sample_budget=max_budget, alpha=0.0, lr=lr,
            trajectories_per_update=trajectories_per_update, seed=0,
            eval_interval=eval_interval, eval_n_episodes=eval_n_episodes,
            exact_gradient=exact_gradient,
        )
        effective_budget = _detect_saturation(
            sat_result['history'], max_budget,
            saturation_window, saturation_eps, saturation_checks,
        )
        print(f"  Saturation at step {effective_budget} (max was {max_budget})")
    else:
        effective_budget = sample_budget

    histories: Dict[float, list] = {}

    for zeta in zeta_values:
        histories[zeta] = []
        print(f"  zeta={zeta:.2f}...", end=" ", flush=True)
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size,
                goals=goals,
                zeta=zeta,
                horizon=horizon,
                sample_budget=effective_budget,
                alpha=alpha,
                lr=lr,
                trajectories_per_update=trajectories_per_update,
                seed=seed,
                eval_interval=eval_interval,
                eval_n_episodes=eval_n_episodes,
                exact_gradient=exact_gradient,
            )
            histories[zeta].append(result['history'])
        rewards = [h[-1]['mean_reward'] for h in histories[zeta] if h]
        unique_sas = [h[-1].get('unique_sa', 0) for h in histories[zeta] if h]
        state_ents = [h[-1].get('state_entropy', 0) for h in histories[zeta] if h]
        print(f"final reward = {np.mean(rewards):.3f} ± "
              f"{np.std(rewards)/np.sqrt(len(rewards)):.3f}  |  "
              f"unique_sa = {np.mean(unique_sas):.0f}, "
              f"state_entropy = {np.mean(state_ents):.2f}")

    return histories


def analyze_results(results_file: str = 'results/tabular_results.csv'):
    """Analyze and print summary of results."""
    import pandas as pd

    df = pd.read_csv(results_file)

    print("\n" + "=" * 60)
    print("ANALYSIS: Teacher Capacity vs Performance")
    print("=" * 60)

    print("\nMean Reward by Teacher Capacity and Sample Budget (alpha=0.5):")
    alpha_05 = df[df['alpha'] == 0.5]

    def stderr(x):
        return x.std() / np.sqrt(len(x))

    pivot = alpha_05.pivot_table(
        values='final_mean_reward',
        index='teacher_capacity',
        columns='sample_budget',
        aggfunc=['mean', stderr]
    )
    print(pivot)

    print("\n" + "=" * 60)
    print("KEY HYPOTHESIS TEST:")
    print("At low sample budget, does mid-teacher outperform best-teacher?")
    print("=" * 60)

    teacher_capacities = sorted(df['teacher_capacity'].unique())
    for budget in [1000, 5000, 10000]:
        print(f"\nSample Budget = {budget}, alpha = 0.5:")
        subset = df[(df['sample_budget'] == budget) & (df['alpha'] == 0.5)]
        for cap in teacher_capacities:
            cap_data = subset[subset['teacher_capacity'] == cap]['final_mean_reward']
            mean = cap_data.mean()
            se = cap_data.std() / np.sqrt(len(cap_data))
            print(f"  Teacher capacity {cap}: {mean:.3f} ± {se:.3f}")
