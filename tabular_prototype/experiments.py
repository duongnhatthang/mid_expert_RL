"""Experiment orchestration: suite, 2x2, and analysis."""

import csv
import numpy as np
from typing import List, Dict, Tuple, Optional

from .config import compute_gamma_from_horizon
from .environment import GridEnv, generate_equidistant_goals, compute_exploration_thresholds
from .teacher import (
    compute_teacher_values_auto,
    compute_uniform_random_teacher_values_auto,
    sample_uniform_random_teacher_knowledge,
)
from .student import TabularSoftmaxPolicy, collect_trajectories
from .training import compute_pav_rl_gradient, update_policy, evaluate_policy


def run_experiment(
    grid_size: int = 10,
    goals: List[Tuple[int, int]] = [(9, 9), (0, 9), (9, 0)],
    teacher_capacity: int = 1,
    horizon: int = 50,
    sample_budget: int = 10000,
    alpha: float = 0.5,
    lr: float = 0.1,
    trajectories_per_update: int = 10,
    seed: int = 0,
    eval_interval: int = 10,
    eval_n_episodes: int = 20,
) -> Dict:
    """Run a single experiment and return results dict."""
    rng = np.random.default_rng(seed)
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)

    if teacher_capacity == -1:
        known_goals = []
        Q_mu, V_mu = None, None
        gamma = compute_gamma_from_horizon(horizon)
    elif teacher_capacity == 0:
        known_goals, known_traps = sample_uniform_random_teacher_knowledge(env, rng=rng)
        Q_mu, V_mu, gamma = compute_uniform_random_teacher_values_auto(
            env, known_goals=known_goals, known_traps=known_traps
        )
    else:
        known_goals = goals[:teacher_capacity]
        Q_mu, V_mu, gamma = compute_teacher_values_auto(env, known_goals)

    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)

    total_steps = 0
    update_count = 0
    history = []

    while total_steps < sample_budget:
        trajectories = collect_trajectories(env, policy, trajectories_per_update, rng)
        total_steps += sum(len(traj) for traj in trajectories)

        grad = compute_pav_rl_gradient(policy, trajectories, Q_mu, V_mu, alpha, gamma)
        update_policy(policy, grad, lr)
        update_count += 1

        if update_count % eval_interval == 0:
            eval_results = evaluate_policy(env, policy, n_episodes=eval_n_episodes, rng=rng)
            history.append({
                'steps': total_steps,
                'mean_reward': eval_results['mean_reward'],
                'goal_rate': eval_results['goal_rate']
            })

    final_eval = evaluate_policy(env, policy, n_episodes=100, rng=rng)

    return {
        'seed': seed,
        'teacher_capacity': teacher_capacity,
        'sample_budget': sample_budget,
        'horizon': horizon,
        'alpha': alpha,
        'lr': lr,
        'final_mean_reward': final_eval['mean_reward'],
        'final_std_reward': final_eval['std_reward'],
        'final_goal_rate': final_eval['goal_rate'],
        'history': history
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
                        'lr': 0.1,
                        'seed': seed
                    })

    print(f"Running {len(configs)} experiments...")

    results = []
    for i, config in enumerate(configs):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(configs)}")
        result = run_experiment(**config)
        results.append(result)

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['seed', 'teacher_capacity', 'sample_budget', 'horizon',
                      'alpha', 'lr', 'final_mean_reward', 'final_std_reward', 'final_goal_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Results saved to {output_file}")
    return results


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
    import os
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
                    lr=0.1,
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
            print(f"reward = {mean_reward:.3f} ± {stderr:.3f}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['seed', 'teacher_capacity', 'n_goals', 'sample_budget', 'horizon',
                      'alpha', 'lr', 'final_mean_reward', 'final_std_reward',
                      'final_goal_rate', 'budget_type', 'horizon_type', 'condition']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nResults saved to {output_file}")
    analyze_2x2_results(all_results, thresholds)
    return all_results


def run_learning_curve_experiment(
    grid_size: int = 10,
    goals: Optional[List[Tuple[int, int]]] = None,
    teacher_capacities: Optional[List[int]] = None,
    horizon: int = 50,
    sample_budget: int = 10000,
    alpha: float = 0.5,
    lr: float = 0.1,
    n_seeds: int = 5,
    trajectories_per_update: int = 10,
    eval_interval: int = 2,
    eval_n_episodes: int = 50,
) -> Dict[int, list]:
    """
    Run experiments for each teacher capacity and collect per-seed learning histories.

    Returns:
        Dict mapping teacher_capacity -> list of per-seed histories.
        Each history is a list of {'steps', 'mean_reward', 'goal_rate'} dicts.
    """
    if goals is None:
        goals = [(9, 9), (0, 9), (9, 0)]
    if teacher_capacities is None:
        teacher_capacities = [-1] + list(range(len(goals) + 1))

    histories: Dict[int, list] = {}

    for cap in teacher_capacities:
        histories[cap] = []
        print(f"  Teacher capacity {cap}...", end=" ", flush=True)
        for seed in range(n_seeds):
            result = run_experiment(
                grid_size=grid_size,
                goals=goals,
                teacher_capacity=cap,
                horizon=horizon,
                sample_budget=sample_budget,
                alpha=alpha,
                lr=lr,
                trajectories_per_update=trajectories_per_update,
                seed=seed,
                eval_interval=eval_interval,
                eval_n_episodes=eval_n_episodes,
            )
            histories[cap].append(result['history'])
        rewards = [h[-1]['mean_reward'] for h in histories[cap] if h]
        print(f"final reward = {np.mean(rewards):.3f} ± "
              f"{np.std(rewards)/np.sqrt(len(rewards)):.3f}")

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
