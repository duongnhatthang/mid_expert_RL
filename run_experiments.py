"""
Main Experiment Runner for Mid-Capacity Teacher RL

Runs experiments on the tabular prototype gridworld.
"""

import argparse
import os
from typing import List, Optional

from tabular_prototype import (
    run_experiment,
    run_experiment_suite,
    run_2x2_exploration_experiment,
    run_2x2_exploration_experiment_zeta,
    run_learning_curve_experiment,
    run_learning_curve_experiment_zeta,
    plot_2x2_results,
    plot_2x2_results_zeta,
    plot_learning_curves,
    plot_learning_curves_improved,
    analyze_results,
    analyze_2x2_results,
    GridEnv,
    generate_equidistant_goals,
    compute_teacher_values_auto,
    compute_uniform_random_teacher_values_auto,
    compute_mixture_teacher_values_auto,
    sample_uniform_random_teacher_knowledge,
    compute_exploration_thresholds,
    visualize_teacher_policy,
    visualize_advantage_grid,
)


def visualize_2x2_policies(
    grid_size: int,
    n_goals: int,
    figures_dir: str = "results/figures",
) -> list:
    """
    Visualize teacher (expert) policies for 2x2 experiment setup.
    Saves one figure per teacher_capacity for capacities 1 and n_goals.

    Returns:
        List of saved figure paths.
    """
    os.makedirs(figures_dir, exist_ok=True)
    goals = generate_equidistant_goals(grid_size, n_goals)
    thresholds = compute_exploration_thresholds(grid_size)
    horizon = thresholds["horizon_large"]
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)
    saved_paths = []
    for teacher_cap in [1, n_goals]:
        known_goals = goals[:teacher_cap]
        Q_mu, V_mu, _ = compute_teacher_values_auto(env, known_goals)
        save_path = os.path.join(
            figures_dir,
            f"teacher_policy_grid{grid_size}_cap{teacher_cap}.png",
        )
        visualize_teacher_policy(
            env, Q_mu, V_mu, known_goals, save_path=save_path
        )
        saved_paths.append(save_path)
    return saved_paths


def visualize_suite_advantages(
    grid_size: int = 10,
    goals: Optional[List[tuple]] = None,
    horizon: Optional[int] = None,
    teacher_capacities: Optional[List[int]] = None,
    figures_dir: str = "results/figures",
) -> list:
    """
    Generate advantage-grid figures for the suite experiment configuration.

    For each teacher setting (-1 no signal, 0 random, 1..3 capacities) on
    the 10x10 grid with goals at
    (9,9), (0,9), (9,0), creates a wedge-colored grid showing
    A(s,a) = Q^mu(s,a) - V^mu(s) for every state-action pair.

    Returns:
        List of saved figure paths.
    """
    os.makedirs(figures_dir, exist_ok=True)

    if goals is None:
        goals = [(grid_size - 1, grid_size - 1), (0, grid_size - 1), (grid_size - 1, 0)]
    if horizon is None:
        thresholds = compute_exploration_thresholds(grid_size)
        horizon = thresholds["horizon_large"]
    if teacher_capacities is None:
        teacher_capacities = [-1] + list(range(len(goals) + 1))

    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)

    saved_paths = []
    for teacher_cap in teacher_capacities:
        save_path = os.path.join(
            figures_dir,
            f"advantage_grid_suite_cap{teacher_cap}.png",
        )
        if teacher_cap == -1:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5,
                "No teacher signal baseline (cap=-1)\nA(s,a) is undefined because Q_mu/V_mu are not used.",
                ha="center", va="center", fontsize=11
            )
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        elif teacher_cap == 0:
            sampled_goals, sampled_traps = sample_uniform_random_teacher_knowledge(env)
            print(f"[debug cap=0] sampled imagined goals: {sampled_goals}")
            print(f"[debug cap=0] sampled imagined traps: {sampled_traps}")
            Q_mu, V_mu, gamma = compute_uniform_random_teacher_values_auto(
                env,
                known_goals=sampled_goals,
                known_traps=sampled_traps,
                debug_print=True,
            )
            visualize_advantage_grid(
                env, Q_mu, V_mu,
                title=f"Teacher Advantage (uniform-random, cap={teacher_cap}, "
                      f"gamma={gamma:.3f})",
                goals=sampled_goals,
                traps=sampled_traps,
                save_path=save_path,
            )
        else:
            known_goals = goals[:teacher_cap]
            Q_mu, V_mu, gamma = compute_teacher_values_auto(env, known_goals)
            visualize_advantage_grid(
                env, Q_mu, V_mu,
                title=f"Teacher Advantage (capacity={teacher_cap}, "
                      f"gamma={gamma:.3f})",
                goals=goals,
                save_path=save_path,
            )
        saved_paths.append(save_path)

    return saved_paths


def visualize_suite_advantages_zeta(
    grid_size: int = 8,
    n_goals: int = 3,
    zeta_values: Optional[List[float]] = None,
    figures_dir: str = "results/figures",
) -> list:
    """
    Generate advantage-grid figures for each zeta value.

    For each mu(zeta) mixture teacher, draws A(s,a) = Q^mu(s,a) - V^mu(s)
    for every state-action pair as triangular wedges.
    """
    import os
    os.makedirs(figures_dir, exist_ok=True)

    if zeta_values is None:
        zeta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    goals = generate_equidistant_goals(grid_size, n_goals)
    thresholds = compute_exploration_thresholds(grid_size)
    horizon = thresholds["horizon_large"]
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)

    saved_paths = []
    for zeta in zeta_values:
        Q_mu, V_mu, gamma = compute_mixture_teacher_values_auto(env, zeta)
        save_path = os.path.join(
            figures_dir,
            f"advantage_grid_zeta{zeta:.2f}.png",
        )
        visualize_advantage_grid(
            env, Q_mu, V_mu,
            title=f"Teacher Advantage  ζ={zeta:.2f}  (gamma={gamma:.3f})",
            goals=goals,
            save_path=save_path,
        )
        saved_paths.append(save_path)

    return saved_paths


def _parse_teacher_capacities(raw_caps: Optional[str], n_goals: int) -> List[int]:
    """Parse comma-separated teacher capacities or build default range."""
    if not raw_caps:
        return [-1] + list(range(n_goals + 1))
    return [int(cap.strip()) for cap in raw_caps.split(",") if cap.strip()]


def _build_goals(grid_size: int, n_goals: int) -> List[tuple]:
    """Create evenly spread goals for a given grid setup."""
    return generate_equidistant_goals(grid_size, n_goals)


def main():
    parser = argparse.ArgumentParser(description='Run Mid-Capacity Teacher RL Experiments')
    parser.add_argument('--mode', choices=[
                            'quick', 'suite', '2x2', 'plot2x2', 'analyze', 'analyze2x2',
                            'learning_curve',
                            '2x2_zeta', 'plot2x2_zeta', 'learning_curve_zeta',
                        ],
                        default='2x2', help='Experiment mode')
    parser.add_argument('--grid-size', type=int, default=8,
                        help='Grid size for 2x2 experiment')
    parser.add_argument('--n-seeds', type=int, default=30,
                        help='Number of random seeds (30 for smooth learning curves)')
    parser.add_argument('--n-goals', type=int, default=3,
                        help='Number of goals in the environment')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for teacher advantage in PAV-RL')
    parser.add_argument('--output', type=str, default='results/exploration_2x2_results.csv',
                        help='Output file for results')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Large horizon value (default: 10 * grid_size)')
    parser.add_argument('--teacher-capacity', type=int, default=1,
                        help='Teacher capacity for quick mode')
    parser.add_argument('--teacher-capacities', type=str, default=None,
                        help='Comma-separated teacher capacities (e.g. "-1,0,1,2,3")')
    parser.add_argument('--sample-budget', type=int, default=12000,
                        help='Sample budget for quick/learning_curve modes')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for quick mode')
    parser.add_argument('--trajectories-per-update', type=int, default=10,
                        help='Trajectories per policy update')
    parser.add_argument('--eval-interval', type=int, default=2,
                        help='Evaluation interval (updates) for learning_curve mode')
    parser.add_argument('--eval-n-episodes', type=int, default=50,
                        help='Evaluation episodes per point for learning_curve mode')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Smoothing window for learning-curve plotting')
    parser.add_argument('--suite-output', type=str, default='results/tabular_results.csv',
                        help='Output CSV for suite mode')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                        help='Directory to save generated figures')
    parser.add_argument('--learning-curve-output', type=str, default='results/figures/learning_curves.png',
                        help='Output path for learning-curve figure')
    parser.add_argument('--zeta-values', type=str, default=None,
                        help='Comma-separated zeta values for zeta modes (e.g. "0.0,0.25,0.5,0.75,1.0")')
    parser.add_argument('--zeta-output', type=str, default='results/exploration_2x2_zeta_results.csv',
                        help='Output CSV for 2x2_zeta mode')
    parser.add_argument('--zeta-learning-curve-output', type=str,
                        default='results/figures/learning_curves_zeta.png',
                        help='Output path for zeta learning-curve figure')

    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    if os.path.dirname(args.learning_curve_output):
        os.makedirs(os.path.dirname(args.learning_curve_output), exist_ok=True)
    if os.path.dirname(args.suite_output):
        os.makedirs(os.path.dirname(args.suite_output), exist_ok=True)

    goals = _build_goals(args.grid_size, args.n_goals)
    teacher_capacities = _parse_teacher_capacities(args.teacher_capacities, args.n_goals)
    horizon = args.horizon if args.horizon else 10 * args.grid_size

    zeta_values = None
    if args.zeta_values:
        zeta_values = [float(z.strip()) for z in args.zeta_values.split(',') if z.strip()]

    if args.mode == 'quick':
        result = run_experiment(
            grid_size=args.grid_size,
            goals=goals,
            teacher_capacity=args.teacher_capacity,
            sample_budget=args.sample_budget,
            alpha=args.alpha,
            lr=args.lr,
            horizon=horizon,
            trajectories_per_update=args.trajectories_per_update,
            seed=args.seed,
        )
        print(f"Quick test result: {result['final_mean_reward']:.3f} reward, "
              f"{result['final_goal_rate']:.1%} goal rate")

    elif args.mode == 'suite':
        run_experiment_suite(
            grid_size=args.grid_size,
            goals=goals,
            output_file=args.suite_output,
        )
        analyze_results(args.suite_output)

        print("\nGenerating advantage grid figures for suite experiment...")
        saved = visualize_suite_advantages(
            grid_size=args.grid_size,
            goals=goals,
            horizon=horizon,
            teacher_capacities=teacher_capacities,
            figures_dir=args.figures_dir,
        )
        print(f"Advantage grid figures saved: {len(saved)} files in {args.figures_dir}/")

    elif args.mode == '2x2':
        results = run_2x2_exploration_experiment(
            grid_size=args.grid_size,
            n_seeds=args.n_seeds,
            n_goals=args.n_goals,
            alpha=args.alpha,
            output_file=args.output,
        )
        plot_2x2_results(args.output)

        print("\nVisualize policies...")
        saved = visualize_2x2_policies(
            args.grid_size, args.n_goals, figures_dir=args.figures_dir
        )
        print(f"Policy figures saved: {len(saved)} files in {args.figures_dir}/")

    elif args.mode == 'plot2x2':
        plot_2x2_results(args.output)

    elif args.mode == 'analyze':
        analyze_results(args.suite_output)

    elif args.mode == 'learning_curve':
        sample_budget = args.sample_budget

        print("Running learning curve experiments...")
        print(f"  Grid: {args.grid_size}x{args.grid_size}, Goals: {goals}")
        print(f"  Teacher capacities: {teacher_capacities}")
        print(f"  Budget: {sample_budget}, Alpha: {args.alpha}, Seeds: {args.n_seeds}")

        histories = run_learning_curve_experiment(
            grid_size=args.grid_size,
            goals=goals,
            teacher_capacities=teacher_capacities,
            horizon=horizon,
            sample_budget=sample_budget,
            alpha=args.alpha,
            lr=args.lr,
            n_seeds=args.n_seeds,
            trajectories_per_update=args.trajectories_per_update,
            eval_interval=args.eval_interval,
            eval_n_episodes=args.eval_n_episodes,
        )

        plot_learning_curves(
            histories,
            title=f"Student Learning Curves by Teacher Capacity\n"
                  f"(alpha={args.alpha}, grid={args.grid_size}x{args.grid_size}, "
                  f"budget={sample_budget})",
            smooth_window=args.smooth_window,
            save_path=args.learning_curve_output,
        )
        print(f"\nLearning curve figure saved to {args.learning_curve_output}")

        improved_path = args.learning_curve_output.replace('.png', '_improved.png')
        plot_learning_curves_improved(
            histories,
            title=f"Student Learning Curves by Teacher Capacity\n"
                  f"(alpha={args.alpha}, grid={args.grid_size}x{args.grid_size}, "
                  f"budget={sample_budget})",
            ema_span=20,
            save_path=improved_path,
        )
        print(f"Improved learning curve figure saved to {improved_path}")

    elif args.mode == '2x2_zeta':
        if os.path.dirname(args.zeta_output):
            os.makedirs(os.path.dirname(args.zeta_output), exist_ok=True)

        results = run_2x2_exploration_experiment_zeta(
            grid_size=args.grid_size,
            n_seeds=args.n_seeds,
            n_goals=args.n_goals,
            alpha=args.alpha,
            zeta_values=zeta_values,
            output_file=args.zeta_output,
        )
        plot_2x2_results_zeta(args.zeta_output)

        print("\nGenerating advantage grid figures for zeta teachers...")
        saved = visualize_suite_advantages_zeta(
            grid_size=args.grid_size,
            n_goals=args.n_goals,
            zeta_values=zeta_values,
            figures_dir=args.figures_dir,
        )
        print(f"Advantage grid figures saved: {len(saved)} files in {args.figures_dir}/")

    elif args.mode == 'plot2x2_zeta':
        plot_2x2_results_zeta(args.zeta_output)

    elif args.mode == 'learning_curve_zeta':
        if os.path.dirname(args.zeta_learning_curve_output):
            os.makedirs(os.path.dirname(args.zeta_learning_curve_output), exist_ok=True)

        print("Running zeta learning curve experiments...")
        print(f"  Grid: {args.grid_size}x{args.grid_size}, Goals: {goals}")
        print(f"  Zeta values: {zeta_values}, Budget: {args.sample_budget}, Alpha: {args.alpha}")

        histories = run_learning_curve_experiment_zeta(
            grid_size=args.grid_size,
            goals=goals,
            zeta_values=zeta_values,
            horizon=horizon,
            sample_budget=args.sample_budget,
            alpha=args.alpha,
            lr=args.lr,
            n_seeds=args.n_seeds,
            trajectories_per_update=args.trajectories_per_update,
            eval_interval=args.eval_interval,
            eval_n_episodes=args.eval_n_episodes,
        )

        plot_learning_curves(
            histories,
            title=f"Student Learning Curves by Teacher ζ\n"
                  f"(alpha={args.alpha}, grid={args.grid_size}x{args.grid_size}, "
                  f"budget={args.sample_budget})",
            smooth_window=args.smooth_window,
            save_path=args.zeta_learning_curve_output,
        )
        print(f"\nLearning curve figure saved to {args.zeta_learning_curve_output}")

        improved_path = args.zeta_learning_curve_output.replace('.png', '_improved.png')
        plot_learning_curves_improved(
            histories,
            title=f"Student Learning Curves by Teacher ζ\n"
                  f"(alpha={args.alpha}, grid={args.grid_size}x{args.grid_size}, "
                  f"budget={args.sample_budget})",
            ema_span=20,
            save_path=improved_path,
        )
        print(f"Improved learning curve figure saved to {improved_path}")

    elif args.mode == 'analyze2x2':
        import pandas as pd

        df = pd.read_csv(args.output)
        thresholds = compute_exploration_thresholds(args.grid_size)
        analyze_2x2_results(df.to_dict('records'), thresholds)


if __name__ == '__main__':
    main()
