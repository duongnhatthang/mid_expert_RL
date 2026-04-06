"""
Tabular Prototype for Mid-Capacity Teacher RL Hypothesis

A minimal, self-contained numpy implementation to build intuition before
committing to the PufferLib architecture.

PAV-RL Policy Gradient:
    grad J = sum_h grad log pi(a_h|s_h) * (Q^pi(s_h,a_h) + alpha * A^mu(s_h,a_h))

Where:
    - Q^pi(s,a) = action-value from environment rewards under student policy
    - A^mu(s,a) = Q^mu(s,a) - V^mu(s) = advantage under FIXED teacher policy
"""

from .config import compute_gamma_from_horizon

from .environment import (
    TrapPlacement,
    generate_traps_near_goals,
    generate_traps_on_paths,
    generate_traps_random,
    generate_traps,
    GridEnv,
    generate_equidistant_goals,
    compute_exploration_thresholds,
)

from .teacher import (
    evaluate_policy_values,
    build_optimal_policy,
    build_uniform_policy,
    build_mixture_policy,
    compute_teacher_values,
    compute_uniform_random_teacher_values,
    compute_teacher_values_auto,
    compute_uniform_random_teacher_values_auto,
    compute_mixture_teacher_values,
    compute_mixture_teacher_values_auto,
    get_teacher_advantage,
)

from .student import (
    NPGPolicy,
    TabularSoftmaxPolicy,
    Transition,
    collect_trajectory,
    collect_trajectories,
)

from .training import (
    estimate_returns,
    compute_pav_rl_gradient,
    update_policy,
    evaluate_policy,
    compute_student_qvalues,
    compute_state_action_visitation,
    visitation_metrics,
)

from .experiments import (
    run_experiment,
    run_experiment_suite,
    run_2x2_exploration_experiment,
    run_2x2_exploration_experiment_zeta,
    run_learning_curve_experiment,
    run_learning_curve_experiment_zeta,
    analyze_2x2_results,
    analyze_results,
)

from .visualization import (
    visualize_policy,
    visualize_teacher_policy,
    visualize_student_policy,
    compare_policies,
    visualize_q_values_per_action,
    visualize_advantage_grid,
    visualize_state_visitation,
    plot_2x2_results,
    plot_2x2_results_zeta,
    plot_learning_curves,
    plot_learning_curves_improved,
)

__all__ = [
    # config
    "compute_gamma_from_horizon",
    # environment
    "TrapPlacement", "generate_traps_near_goals", "generate_traps_on_paths",
    "generate_traps_random", "generate_traps", "GridEnv",
    "generate_equidistant_goals", "compute_exploration_thresholds",
    # teacher
    "evaluate_policy_values", "build_optimal_policy",
    "build_uniform_policy", "build_mixture_policy",
    "compute_teacher_values",
    "compute_uniform_random_teacher_values",
    "compute_teacher_values_auto", "compute_uniform_random_teacher_values_auto",
    "compute_mixture_teacher_values", "compute_mixture_teacher_values_auto",
    "get_teacher_advantage",
    # student
    "NPGPolicy", "TabularSoftmaxPolicy",
    "Transition", "collect_trajectory", "collect_trajectories",
    # training
    "estimate_returns", "compute_pav_rl_gradient", "update_policy", "evaluate_policy",
    "compute_student_qvalues", "compute_state_action_visitation", "visitation_metrics",
    # experiments
    "run_experiment", "run_experiment_suite",
    "run_2x2_exploration_experiment", "run_2x2_exploration_experiment_zeta",
    "run_learning_curve_experiment", "run_learning_curve_experiment_zeta",
    "analyze_2x2_results", "analyze_results",
    # visualization
    "visualize_policy", "visualize_teacher_policy", "visualize_student_policy",
    "compare_policies", "visualize_q_values_per_action", "visualize_advantage_grid",
    "visualize_state_visitation",
    "plot_2x2_results", "plot_2x2_results_zeta",
    "plot_learning_curves", "plot_learning_curves_improved",
]
