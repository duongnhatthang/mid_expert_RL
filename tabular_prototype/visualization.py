"""Visualization functions for policies, Q-values, advantages, and experiment results."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .environment import GridEnv
from .student import TabularSoftmaxPolicy


def _capacity_label(cap) -> str:
    """Human-readable label for teacher capacity or zeta setting."""
    if isinstance(cap, float):
        return f"ζ={cap:.2f}"
    if cap == -1:
        return "No teacher signal (cap=-1)"
    if cap == 0:
        return "Uniform-random teacher (cap=0)"
    return f"Teacher cap={cap}"


# =============================================================================
# Generic Policy Visualization
# =============================================================================

def visualize_policy(
    env: GridEnv,
    Q: np.ndarray,
    V: np.ndarray,
    title: str,
    goals: Optional[List[Tuple[int, int]]] = None,
    traps: Optional[List[Tuple[int, int]]] = None,
    timestep: Optional[int] = None,
    save_path: Optional[str] = None,
    ax: Optional[Any] = None,
    show_value_heatmap: bool = True
) -> Optional[Any]:
    """
    Heatmap of V(s) with arrows showing the greedy policy direction.

    When show_value_heatmap=False, draws only the policy arrows on a neutral grid.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if Q.ndim == 3:
        if timestep is None:
            timestep = 0
        Q_t = Q[timestep]
        V_t = V[timestep]
    else:
        Q_t = Q
        V_t = V

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    V_grid = V_t.reshape(env.grid_size, env.grid_size)

    if show_value_heatmap:
        im = ax.imshow(V_grid, cmap='viridis', origin='upper')
        plt.colorbar(im, ax=ax, label='V(s)')
    else:
        ax.imshow(np.ones((env.grid_size, env.grid_size)) * 0.08,
                  cmap='Greys', origin='upper', vmin=0.0, vmax=1.0)

    arrow_dirs = {
        0: (0, -0.3),   # up
        1: (0, 0.3),    # down
        2: (-0.3, 0),   # left
        3: (0.3, 0)     # right
    }

    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        row, col = state

        if goals and state in goals:
            continue
        if traps and state in traps:
            continue

        best_action = np.argmax(Q_t[s_idx])
        dx, dy = arrow_dirs[best_action]

        arrow_color = 'black' if not show_value_heatmap else 'white'
        ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1,
                 fc=arrow_color, ec=arrow_color, alpha=0.8)

    start = env.start
    ax.add_patch(patches.Circle((start[1], start[0]), 0.3,
                                fill=True, color='blue', alpha=0.7))
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            color='white', fontweight='bold', fontsize=10)

    if goals:
        for goal in goals:
            ax.add_patch(patches.Rectangle((goal[1]-0.4, goal[0]-0.4), 0.8, 0.8,
                                           fill=True, color='green', alpha=0.7))
            ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)

    if traps:
        for trap in traps:
            ax.add_patch(patches.Rectangle((trap[1]-0.4, trap[0]-0.4), 0.8, 0.8,
                                           fill=True, color='red', alpha=0.7))
            ax.text(trap[1], trap[0], 'T', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)

    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True, alpha=0.3)

    if save_path and created_fig:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if created_fig:
        return fig
    return None


# =============================================================================
# Teacher / Student Policy Wrappers
# =============================================================================

def visualize_teacher_policy(
    env: GridEnv,
    Q_mu: np.ndarray,
    V_mu: np.ndarray,
    known_goals: List[Tuple[int, int]],
    known_traps: Optional[List[Tuple[int, int]]] = None,
    timestep: Optional[int] = None,
    save_path: Optional[str] = None
) -> Any:
    """Visualize teacher (expert) policy only (no V heatmap)."""
    title = f"Teacher Policy (knows {len(known_goals)} goals)"
    if timestep is not None:
        title += f" [t={timestep}]"
    return visualize_policy(env, Q_mu, V_mu, title, known_goals, known_traps,
                            timestep, save_path, show_value_heatmap=False)


def visualize_student_policy(
    env: GridEnv,
    policy: TabularSoftmaxPolicy,
    goals: Optional[List[Tuple[int, int]]] = None,
    traps: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> Any:
    """Visualize student's learned policy (uses theta as Q approximation)."""
    Q_student = policy.theta
    V_student = Q_student.max(axis=1)
    return visualize_policy(env, Q_student, V_student, "Student Policy",
                            goals, traps, save_path=save_path)


def compare_policies(
    env: GridEnv,
    Q_teacher: np.ndarray,
    V_teacher: np.ndarray,
    policy_student: TabularSoftmaxPolicy,
    known_goals: List[Tuple[int, int]],
    all_goals: Optional[List[Tuple[int, int]]] = None,
    traps: Optional[List[Tuple[int, int]]] = None,
    timestep: Optional[int] = None,
    save_path: Optional[str] = None
) -> Any:
    """Side-by-side comparison of teacher vs student policy."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    teacher_title = f"Teacher (knows {len(known_goals)} goals)"
    if timestep is not None:
        teacher_title += f" [t={timestep}]"
    visualize_policy(env, Q_teacher, V_teacher, teacher_title,
                     known_goals, traps, timestep, ax=axes[0], show_value_heatmap=False)

    Q_student = policy_student.theta
    V_student = Q_student.max(axis=1)
    goals_to_show = all_goals if all_goals else known_goals
    visualize_policy(env, Q_student, V_student, "Student (learned)",
                     goals_to_show, traps, ax=axes[1])

    plt.suptitle("Teacher vs Student Policy Comparison", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")

    return fig


# =============================================================================
# Q-value Per-Action Visualization
# =============================================================================

def visualize_q_values_per_action(
    env: GridEnv,
    Q: np.ndarray,
    title: str = "Q-values by Action",
    timestep: Optional[int] = None,
    save_path: Optional[str] = None
) -> Any:
    """Visualize Q-values for each action as separate heatmaps."""
    import matplotlib.pyplot as plt

    if Q.ndim == 3:
        if timestep is None:
            timestep = 0
        Q_t = Q[timestep]
    else:
        Q_t = Q

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    action_names = ['Up', 'Down', 'Left', 'Right']
    vmin, vmax = Q_t.min(), Q_t.max()

    for a, (ax, name) in enumerate(zip(axes, action_names)):
        Q_action = Q_t[:, a].reshape(env.grid_size, env.grid_size)
        im = ax.imshow(Q_action, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
        ax.set_title(f"Q(s, {name})")
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)

    plt.suptitle(title + (f" [t={timestep}]" if timestep is not None else ""), fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Q-value figure saved to {save_path}")

    return fig


# =============================================================================
# Advantage Grid Visualization (NEW)
# =============================================================================

def visualize_advantage_grid(
    env: GridEnv,
    Q: np.ndarray,
    V: np.ndarray,
    title: str = "Advantage A(s,a) Grid",
    goals: Optional[List[Tuple[int, int]]] = None,
    traps: Optional[List[Tuple[int, int]]] = None,
    timestep: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Any:
    """
    Visualize advantage A(s,a) = Q(s,a) - V(s) for every state-action pair.

    Each cell is divided into 4 triangular wedges (one per action direction),
    colored by advantage value:
      - Top wedge    = Up action
      - Bottom wedge = Down action
      - Left wedge   = Left action
      - Right wedge  = Right action

    Color mapping: Green = high advantage (best action), Red = low (worst action).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    if Q.ndim == 3:
        if timestep is None:
            timestep = 0
        Q_t = Q[timestep]
        V_t = V[timestep]
    else:
        Q_t = Q
        V_t = V

    # A(s,a) = Q(s,a) - V(s)
    A = Q_t - V_t[:, np.newaxis]

    goals_set = set(goals) if goals else set()
    traps_set = set(traps) if traps else set()

    # Compute advantage range over non-absorbing states only
    non_absorbing_mask = np.ones(env.n_states, dtype=bool)
    for s_idx in range(env.n_states):
        if env.idx_to_state(s_idx) in goals_set or env.idx_to_state(s_idx) in traps_set:
            non_absorbing_mask[s_idx] = False

    if non_absorbing_mask.any():
        a_min = A[non_absorbing_mask].min()
        a_max = A[non_absorbing_mask].max()
    else:
        a_min, a_max = -1.0, 0.0

    # Ensure valid range for TwoSlopeNorm
    if a_min >= 0:
        a_min = -0.001
    if a_max <= 0:
        a_max = 0.001

    norm = mcolors.TwoSlopeNorm(vmin=a_min, vcenter=0, vmax=a_max)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(10, 10))

    h = 0.5  # half-cell size

    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        row, col = state

        if state in goals_set or state in traps_set:
            continue

        cx, cy = col, row

        # Triangular wedge vertices for each action:
        #   up(0):    center → top-left → top-right
        #   down(1):  center → bottom-left → bottom-right
        #   left(2):  center → top-left → bottom-left
        #   right(3): center → top-right → bottom-right
        wedge_vertices = {
            0: [(cx, cy), (cx - h, cy - h), (cx + h, cy - h)],
            1: [(cx, cy), (cx - h, cy + h), (cx + h, cy + h)],
            2: [(cx, cy), (cx - h, cy - h), (cx - h, cy + h)],
            3: [(cx, cy), (cx + h, cy - h), (cx + h, cy + h)],
        }

        for action, vertices in wedge_vertices.items():
            adv = A[s_idx, action]
            color = cmap(norm(adv))
            triangle = plt.Polygon(vertices, closed=True,
                                   facecolor=color, edgecolor='gray',
                                   linewidth=0.3)
            ax.add_patch(triangle)

    # Mark goals
    if goals:
        for goal in goals:
            rect = mpatches.FancyBboxPatch(
                (goal[1] - h, goal[0] - h), 2 * h, 2 * h,
                boxstyle="round,pad=0", facecolor='green', alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10, zorder=5)

    # Mark traps
    if traps:
        for trap in traps:
            rect = mpatches.FancyBboxPatch(
                (trap[1] - h, trap[0] - h), 2 * h, 2 * h,
                boxstyle="round,pad=0", facecolor='red', alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(trap[1], trap[0], 'T', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10, zorder=5)

    # Mark start
    start = env.start
    ax.add_patch(plt.Circle((start[1], start[0]), 0.2,
                             fill=True, color='blue', alpha=0.9, zorder=6))
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            color='white', fontweight='bold', fontsize=8, zorder=7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Advantage A(s,a)', shrink=0.8)

    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(env.grid_size - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Advantage grid saved to {save_path}")

    return fig


# =============================================================================
# State-Action Visitation Heatmap
# =============================================================================

def visualize_state_visitation(
    env: GridEnv,
    visitation_counts: np.ndarray,
    title: str = "State Visitation",
    goals: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
) -> Any:
    """
    Heatmap of state visitation (summed over actions).

    Args:
        visitation_counts: shape (n_states, n_actions) — raw visit counts.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    state_visits = visitation_counts.sum(axis=1)
    grid = state_visits.reshape(env.grid_size, env.grid_size)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap='hot', origin='upper')
    plt.colorbar(im, ax=ax, label='Visit Count')

    start = env.start
    ax.add_patch(patches.Circle((start[1], start[0]), 0.3,
                                fill=True, color='cyan', alpha=0.8))
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            color='black', fontweight='bold', fontsize=10)

    if goals:
        for goal in goals:
            ax.add_patch(patches.Rectangle((goal[1]-0.4, goal[0]-0.4), 0.8, 0.8,
                                           fill=False, edgecolor='lime',
                                           linewidth=2.5))
            ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                    color='lime', fontweight='bold', fontsize=10)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visitation figure saved to {save_path}")

    return fig


def visualize_visitation_comparison_grid(
    env: GridEnv,
    visitation_data: dict,
    row_keys: list,
    col_keys: list,
    row_label_fn,
    col_label_fn,
    goals: Optional[List[Tuple[int, int]]] = None,
    suptitle: str = "State Visitation Comparison",
    save_path: Optional[str] = None,
) -> Any:
    """
    Multi-panel heatmap comparing state visitation across teacher settings and conditions.

    Args:
        env:              GridEnv instance (for grid_size, start position).
        visitation_data:  Dict mapping (row_key, col_key) -> np.ndarray (n_states, n_actions).
        row_keys:         Ordered list of row keys (e.g. zeta values or teacher capacities).
        col_keys:         Ordered list of column keys (e.g. (budget_type, horizon_type) tuples).
        row_label_fn:     Callable: row_key -> display string for row labels.
        col_label_fn:     Callable: col_key -> display string for column headers.
        goals:            Goal positions for annotation.
        suptitle:         Figure title.
        save_path:        If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec

    n_rows = len(row_keys)
    n_cols = len(col_keys)
    cell_size = 2.5
    fig = plt.figure(
        figsize=(cell_size * n_cols + 2.0, cell_size * n_rows + 1.5)
    )
    gs = GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.15, hspace=0.25,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])

    # Compute shared color scale
    vmax = 0
    grids = {}
    for rk in row_keys:
        for ck in col_keys:
            vis = visitation_data.get((rk, ck))
            if vis is not None:
                state_visits = vis.sum(axis=1)
                grid = state_visits.reshape(env.grid_size, env.grid_size)
                grids[(rk, ck)] = grid
                vmax = max(vmax, grid.max())
    if vmax == 0:
        vmax = 1

    mappable = None
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            ax = axes[i, j]
            grid = grids.get((rk, ck))

            if grid is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(grid, cmap='hot', origin='upper', vmin=0, vmax=vmax)
                mappable = im

                # Mark start
                start = env.start
                ax.add_patch(patches.Circle(
                    (start[1], start[0]), 0.25,
                    fill=True, color='cyan', alpha=0.8, linewidth=0,
                ))
                ax.text(start[1], start[0], 'S', ha='center', va='center',
                        color='black', fontweight='bold', fontsize=7)

                # Mark goals
                if goals:
                    for goal in goals:
                        ax.add_patch(patches.Rectangle(
                            (goal[1] - 0.35, goal[0] - 0.35), 0.7, 0.7,
                            fill=False, edgecolor='lime', linewidth=1.5,
                        ))
                        ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                                color='lime', fontweight='bold', fontsize=7)

                ax.set_xticks([])
                ax.set_yticks([])

            # Row label on leftmost column
            if j == 0:
                ax.set_ylabel(row_label_fn(rk), fontsize=9)

            # Column header on top row
            if i == 0:
                ax.set_title(col_label_fn(ck), fontsize=9)

    # Shared colorbar
    if mappable is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        fig.colorbar(mappable, cax=cbar_ax, label='Visit Count')

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visitation comparison grid saved to {save_path}")

    plt.close(fig)
    return fig


def visualize_visitation_composite_grid(
    env: GridEnv,
    all_visitation_data: dict,
    outer_keys: list,
    inner_row_keys: list,
    inner_col_keys: list,
    outer_label_fn,
    inner_row_label_fn,
    inner_col_label_fn,
    goals: Optional[List[Tuple[int, int]]] = None,
    suptitle: str = "Visitation Composite",
    save_path: Optional[str] = None,
) -> Any:
    """
    Landscape composite figure: outer grid of configs, inner grid of heatmaps.

    Args:
        all_visitation_data: Dict mapping (outer_key, inner_row_key, inner_col_key)
                            -> np.ndarray (n_states, n_actions).
        outer_keys: List of outer grid keys (e.g., (budget, horizon) tuples).
        inner_row_keys: Row keys within each panel (e.g., teacher values).
        inner_col_keys: Column keys within each panel (e.g., alpha values).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    n_outer = len(outer_keys)
    n_inner_rows = len(inner_row_keys)
    n_inner_cols = len(inner_col_keys)

    # Landscape: arrange outer panels in a single row (or 2 rows if many)
    if n_outer <= 4:
        outer_rows, outer_cols = 1, n_outer
    else:
        outer_rows = 2
        outer_cols = (n_outer + 1) // 2

    cell = 1.8
    fig_w = cell * n_inner_cols * outer_cols + 2.5
    fig_h = cell * n_inner_rows * outer_rows + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_outer = GridSpec(
        outer_rows, outer_cols + 1,
        width_ratios=[1] * outer_cols + [0.03],
        wspace=0.3, hspace=0.4,
    )

    vmax = 0
    for key, vis in all_visitation_data.items():
        if vis is not None:
            vmax = max(vmax, vis.sum(axis=1).max())
    if vmax == 0:
        vmax = 1

    mappable = None
    for oi, ok in enumerate(outer_keys):
        o_row = oi // outer_cols
        o_col = oi % outer_cols

        gs_inner = GridSpecFromSubplotSpec(
            n_inner_rows, n_inner_cols,
            subplot_spec=gs_outer[o_row, o_col],
            wspace=0.05, hspace=0.15,
        )

        for ri, rk in enumerate(inner_row_keys):
            for ci, ck in enumerate(inner_col_keys):
                ax = fig.add_subplot(gs_inner[ri, ci])
                vis = all_visitation_data.get((ok, rk, ck))

                if vis is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    grid = vis.sum(axis=1).reshape(env.grid_size, env.grid_size)
                    im = ax.imshow(grid, cmap='hot', origin='upper',
                                   vmin=0, vmax=vmax)
                    mappable = im

                    start = env.start
                    ax.add_patch(mpatches.Circle(
                        (start[1], start[0]), 0.25,
                        fill=True, color='cyan', alpha=0.8,
                    ))
                    if goals:
                        for goal in goals:
                            ax.add_patch(mpatches.Rectangle(
                                (goal[1] - 0.35, goal[0] - 0.35), 0.7, 0.7,
                                fill=False, edgecolor='lime', linewidth=1.2,
                            ))

                    ax.set_xticks([])
                    ax.set_yticks([])

                if ci == 0:
                    ax.set_ylabel(inner_row_label_fn(rk), fontsize=6)
                if ri == 0:
                    ax.set_title(inner_col_label_fn(ck), fontsize=6)

    if mappable is not None:
        cbar_ax = fig.add_subplot(gs_outer[:, -1])
        fig.colorbar(mappable, cax=cbar_ax, label='Visit Count')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Composite visitation grid saved to {save_path}")

    plt.close(fig)
    return fig


# =============================================================================
# 2x2 Experiment Plot
# =============================================================================

def plot_2x2_results(results_file: str = 'results/exploration_2x2_results.csv'):
    """Create bar-chart visualization for 2x2 experiment with visitation metrics."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd

    df = pd.read_csv(results_file)

    conditions = [
        ('low', 'small'),
        ('low', 'large'),
        ('high', 'small'),
        ('high', 'large'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (budget_type, horizon_type) in zip(axes.flat, conditions):
        cond_df = df[(df['budget_type'] == budget_type) & (df['horizon_type'] == horizon_type)]

        if len(cond_df) == 0:
            title = f"{budget_type.title()} Budget + {horizon_type.title()} Horizon\n(No data)"
            ax.set_title(title)
            ax.set_visible(False)
            continue

        actual_budget = int(cond_df['sample_budget'].iloc[0])
        actual_horizon = int(cond_df['horizon'].iloc[0])
        title = (f"{budget_type.title()} Budget ({actual_budget:,}) + "
                 f"{horizon_type.title()} Horizon ({actual_horizon})")

        teacher_caps = sorted(cond_df['teacher_capacity'].unique())
        means, stderrs, usa_means, sent_means = [], [], [], []

        for cap in teacher_caps:
            cap_data = cond_df[cond_df['teacher_capacity'] == cap]
            reward_data = cap_data['final_mean_reward']
            if len(reward_data) > 0:
                means.append(reward_data.mean())
                stderrs.append(reward_data.std() / np.sqrt(len(reward_data)))
            else:
                means.append(0)
                stderrs.append(0)
            if 'final_unique_sa' in cap_data.columns:
                usa_means.append(cap_data['final_unique_sa'].mean())
            else:
                usa_means.append(0)
            if 'final_state_entropy' in cap_data.columns:
                sent_means.append(cap_data['final_state_entropy'].mean())
            else:
                sent_means.append(0)

        n_caps = len(teacher_caps)
        colors = cm.viridis(np.linspace(0.2, 0.8, n_caps))

        bars = ax.bar(teacher_caps, means, yerr=stderrs, capsize=5,
                      color=colors, alpha=0.8)

        # Annotate bars with visitation metrics
        for i, (bar, usa, sent) in enumerate(zip(bars, usa_means, sent_means)):
            y = bar.get_height() + stderrs[i] + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"SA={usa:.0f}\nH={sent:.2f}",
                    ha='center', va='bottom', fontsize=7, color='#333333')

        ax.set_xlabel('Teacher Setting')
        ax.set_ylabel('Mean Reward')
        ax.set_ylim(0, min(1.15, max(means) + 0.15) if means else 1)
        ax.set_title(title)
        ax.set_xticks(teacher_caps)
        ax.set_xticklabels([_capacity_label(cap) for cap in teacher_caps], rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)

        if means:
            best_idx = np.argmax(means)
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)

    plt.suptitle('2x2 Exploration Experiment: Budget x Horizon\n'
                 '(SA = unique state-actions visited, H = state entropy)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = results_file.replace('.csv', '.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {output_path}")
    return output_path


# =============================================================================
# 2x2 Zeta Experiment Plot
# =============================================================================

def plot_2x2_results_zeta(results_file: str = 'results/exploration_2x2_zeta_results.csv'):
    """Bar-chart visualization for 2x2 zeta experiment with visitation metrics."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd

    df = pd.read_csv(results_file)

    conditions = [
        ('low', 'small'),
        ('low', 'large'),
        ('high', 'small'),
        ('high', 'large'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, (budget_type, horizon_type) in zip(axes.flat, conditions):
        cond_df = df[(df['budget_type'] == budget_type) & (df['horizon_type'] == horizon_type)]

        if len(cond_df) == 0:
            ax.set_title(f"{budget_type.title()} Budget + {horizon_type.title()} Horizon\n(No data)")
            ax.set_visible(False)
            continue

        actual_budget = int(cond_df['sample_budget'].iloc[0])
        actual_horizon = int(cond_df['horizon'].iloc[0])
        title = (f"{budget_type.title()} Budget ({actual_budget:,}) + "
                 f"{horizon_type.title()} Horizon ({actual_horizon})")

        zeta_vals = sorted(cond_df['zeta'].unique())
        means, stderrs, usa_means, sent_means = [], [], [], []

        for z in zeta_vals:
            z_data = cond_df[cond_df['zeta'] == z]
            reward_data = z_data['final_mean_reward']
            means.append(reward_data.mean())
            stderrs.append(reward_data.std() / np.sqrt(len(reward_data)) if len(reward_data) > 1 else 0)
            if 'final_unique_sa' in z_data.columns:
                usa_means.append(z_data['final_unique_sa'].mean())
            else:
                usa_means.append(0)
            if 'final_state_entropy' in z_data.columns:
                sent_means.append(z_data['final_state_entropy'].mean())
            else:
                sent_means.append(0)

        n_z = len(zeta_vals)
        colors = cm.viridis(np.linspace(0.2, 0.8, n_z))
        x_pos = np.arange(n_z)

        bars = ax.bar(x_pos, means, yerr=stderrs, capsize=5, color=colors, alpha=0.8)

        # Annotate bars with visitation metrics
        for i, (bar, usa, sent) in enumerate(zip(bars, usa_means, sent_means)):
            y = bar.get_height() + stderrs[i] + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"SA={usa:.0f}\nH={sent:.2f}",
                    ha='center', va='bottom', fontsize=7, color='#333333')

        ax.set_xlabel('Teacher ζ')
        ax.set_ylabel('Mean Reward')
        ax.set_ylim(0, min(1.15, max(means) + 0.15) if means else 1)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"ζ={z:.2f}" for z in zeta_vals], rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)

        if means:
            best_idx = int(np.argmax(means))
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)

    plt.suptitle('2x2 Exploration Experiment (ζ parameterisation): Budget × Horizon\n'
                 '(SA = unique state-actions visited, H = state entropy)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = results_file.replace('.csv', '.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {output_path}")
    return output_path


# =============================================================================
# Learning Curve Plot
# =============================================================================

def plot_learning_curves(
    histories: Dict[int, List[List[Dict]]],
    title: str = "Student Learning Curves by Teacher Capacity",
    ylabel: str = "Mean Reward",
    metric: str = "mean_reward",
    smooth_window: int = 3,
    save_path: Optional[str] = None,
    dual_v_mode: bool = False,
    x_label: Optional[str] = None,
) -> Any:
    """
    Plot student learning curves for different teacher capacities on one figure.

    Args:
        histories: mapping from teacher_capacity -> list of per-seed histories,
                   where each history is a list of dicts with keys
                   'steps', 'mean_reward', 'goal_rate'.
        title: figure title.
        ylabel: y-axis label.
        metric: which key in the history dicts to plot ('mean_reward' or 'goal_rate').
        save_path: if provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import colorsys

    fig, ax = plt.subplots(figsize=(11, 6))

    capacities = sorted(histories.keys())
    n = len(capacities)

    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*",
                "8", "H", "d", "+", "x", "1", "2", "3", "4"]
    _LINESTYLES = ["solid", (0, (5, 3)), (0, (1, 1)), (0, (4, 2, 1, 2)),
                   (0, (3, 1, 1, 1, 1, 1))]

    def _generate_colors(count: int) -> list:
        """Generate *count* maximally separated, perceptually distinct colors."""
        if count <= 10:
            tab10 = plt.cm.tab10
            return [mcolors.to_hex(tab10(i)) for i in range(count)]
        if count <= 20:
            tab20 = plt.cm.tab20
            return [mcolors.to_hex(tab20(i)) for i in range(count)]
        colors = []
        for i in range(count):
            hue = i / count
            sat = 0.65 + 0.25 * (i % 3) / 2
            light = 0.45 + 0.15 * ((i // 3) % 3) / 2
            colors.append(mcolors.to_hex(colorsys.hls_to_rgb(hue, light, sat)))
        return colors

    palette = _generate_colors(n)

    styles = {}
    for i, cap in enumerate(capacities):
        ls_idx = 0
        if cap == -1:
            ls_idx = 1
        elif cap == 0:
            ls_idx = 2
        elif i >= len(_MARKERS):
            ls_idx = i // len(_MARKERS)
        styles[cap] = {
            "color": palette[i],
            "linestyle": _LINESTYLES[ls_idx % len(_LINESTYLES)],
            "marker": _MARKERS[i % len(_MARKERS)],
        }

    for cap in capacities:
        seed_histories = histories[cap]
        if not seed_histories or not seed_histories[0]:
            continue

        all_steps = sorted({
            entry['steps'] for h in seed_histories for entry in h
        })
        steps = np.array(all_steps)

        values_matrix = np.full((len(seed_histories), len(steps)), np.nan)
        for s_idx, h in enumerate(seed_histories):
            h_steps = np.array([e['steps'] for e in h])
            h_vals = np.array([e[metric] for e in h])
            interp_vals = np.interp(steps, h_steps, h_vals)
            values_matrix[s_idx, :] = interp_vals

        mean_vals = np.nanmean(values_matrix, axis=0)
        stderr_vals = np.nanstd(values_matrix, axis=0) / np.sqrt(
            np.sum(~np.isnan(values_matrix), axis=0)
        )

        if smooth_window > 1 and len(mean_vals) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            mean_vals = np.convolve(mean_vals, kernel, mode='valid')
            stderr_vals = np.convolve(stderr_vals, kernel, mode='valid')
            offset = (smooth_window - 1) // 2
            steps_plot = steps[offset:offset + len(mean_vals)]
        else:
            steps_plot = steps

        sty = styles[cap]
        label = _capacity_label(cap)
        marker_every = max(1, len(steps_plot) // 8)

        if dual_v_mode:
            # Undiscounted V (solid) from 'exact_V_start_undiscounted'
            undisc_matrix = np.full((len(seed_histories), len(steps)), np.nan)
            for s_idx, h in enumerate(seed_histories):
                h_steps = np.array([e['steps'] for e in h])
                h_vals = np.array([
                    e.get('exact_V_start_undiscounted', np.nan) for e in h
                ])
                if not np.all(np.isnan(h_vals)):
                    undisc_matrix[s_idx, :] = np.interp(steps, h_steps, h_vals)

            undisc_mean = np.nanmean(undisc_matrix, axis=0)
            if smooth_window > 1 and len(undisc_mean) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                undisc_mean = np.convolve(undisc_mean, kernel, mode='valid')

            # Solid = undiscounted
            ax.plot(
                steps_plot, undisc_mean,
                color=sty["color"], linestyle='solid',
                marker=sty["marker"], markevery=marker_every,
                markersize=6, linewidth=2, label=label,
            )
            # Dashed = discounted
            ax.plot(
                steps_plot, mean_vals,
                color=sty["color"], linestyle='dashed',
                linewidth=1.5, alpha=0.7,
            )
        else:
            ax.plot(
                steps_plot, mean_vals,
                color=sty["color"],
                linestyle=sty["linestyle"],
                marker=sty["marker"],
                markevery=marker_every,
                markersize=6,
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                steps_plot,
                mean_vals - stderr_vals,
                mean_vals + stderr_vals,
                color=sty["color"],
                alpha=0.12,
            )

    ax.set_xlabel(x_label if x_label else "Training Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curve figure saved to {save_path}")

    return fig


# =============================================================================
# Improved Learning Curve Plot
# =============================================================================

def _ema(vals: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


def plot_learning_curves_improved(
    histories: Dict[int, List[List[Dict]]],
    title: str = "Student Learning Curves (Improved)",
    metric: str = "mean_reward",
    ema_span: int = 20,
    save_path: Optional[str] = None,
) -> Any:
    """
    Two-panel learning curve plot with improved readability.

    Panel 1 – Normalized reward: (reward - baseline) / (1 - baseline),
              where baseline is the no-teacher (cap=-1) curve.
    Panel 2 – Cumulative mean evaluation reward over training.

    Applies EMA smoothing, sparse error bars, and a visual hierarchy
    (thicker lines for key conditions) instead of overlapping shaded bands.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # -- common step grid across all conditions --
    all_steps_set: set = set()
    for seed_histories in histories.values():
        for h in seed_histories:
            for entry in h:
                all_steps_set.add(entry['steps'])
    common_steps = np.sort(np.array(list(all_steps_set)))

    # -- interpolate each condition onto the common grid --
    raw_mean: Dict[int, np.ndarray] = {}
    raw_stderr: Dict[int, np.ndarray] = {}

    for cap in sorted(histories.keys()):
        seed_histories = histories[cap]
        if not seed_histories or not seed_histories[0]:
            continue
        mat = np.full((len(seed_histories), len(common_steps)), np.nan)
        for s_idx, h in enumerate(seed_histories):
            h_steps = np.array([e['steps'] for e in h])
            h_vals = np.array([e[metric] for e in h])
            mat[s_idx, :] = np.interp(common_steps, h_steps, h_vals)
        raw_mean[cap] = np.nanmean(mat, axis=0)
        n_valid = np.sum(~np.isnan(mat), axis=0).clip(min=1)
        raw_stderr[cap] = np.nanstd(mat, axis=0) / np.sqrt(n_valid)

    if not raw_mean:
        return None

    # -- EMA smoothing (suggestion 2) --
    smooth_mean = {c: _ema(v, ema_span) for c, v in raw_mean.items()}
    smooth_stderr = {c: _ema(v, ema_span) for c, v in raw_stderr.items()}

    # -- normalized reward (suggestion 4) --
    baseline_cap = -1 if -1 in smooth_mean else min(smooth_mean.keys())
    baseline = smooth_mean[baseline_cap]
    denom = np.maximum(1.0 - baseline, 1e-6)

    norm_mean: Dict[int, np.ndarray] = {}
    norm_stderr: Dict[int, np.ndarray] = {}
    for cap in smooth_mean:
        norm_mean[cap] = (smooth_mean[cap] - baseline) / denom
        norm_stderr[cap] = smooth_stderr[cap] / denom

    # -- cumulative mean reward (suggestion 5) --
    cum_mean: Dict[int, np.ndarray] = {}
    for cap in smooth_mean:
        cum_mean[cap] = np.cumsum(smooth_mean[cap]) / np.arange(
            1, len(smooth_mean[cap]) + 1
        )

    # -- styling with visual hierarchy (suggestion 7) --
    capacities = sorted(smooth_mean.keys())
    n = len(capacities)
    palette = [mcolors.to_hex(plt.cm.tab10(i)) for i in range(min(n, 10))]
    if n > 10:
        palette = [mcolors.to_hex(plt.cm.tab20(i)) for i in range(n)]

    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*"]
    _LINESTYLES = [
        "solid", (0, (5, 3)), (0, (1, 1)),
        (0, (4, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1)),
    ]

    final_vals = {c: smooth_mean[c][-1] for c in capacities}
    non_baseline = [c for c in capacities if c != baseline_cap]
    best_cap = max(non_baseline, key=lambda c: final_vals[c]) if non_baseline else baseline_cap

    styles: dict = {}
    for i, cap in enumerate(capacities):
        is_key = cap in (baseline_cap, best_cap)
        ls_idx = 0
        if cap == -1:
            ls_idx = 1
        elif cap == 0:
            ls_idx = 2
        styles[cap] = {
            "color": palette[i],
            "linestyle": _LINESTYLES[ls_idx % len(_LINESTYLES)],
            "marker": _MARKERS[i % len(_MARKERS)],
            "linewidth": 2.5 if is_key else 1.3,
            "alpha": 1.0 if is_key else 0.65,
            "markersize": 7 if is_key else 4,
        }

    errbar_every = max(1, len(common_steps) // 12)
    marker_every = max(1, len(common_steps) // 8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    # ---- Panel 1: Normalized reward ----
    for cap in capacities:
        sty = styles[cap]
        lbl = _capacity_label(cap)
        if cap == baseline_cap:
            ax1.axhline(0, color=sty["color"], linestyle="--", alpha=0.6,
                        linewidth=1.5, label=lbl + " (ref)")
            continue
        ax1.plot(
            common_steps, norm_mean[cap],
            color=sty["color"], linestyle=sty["linestyle"],
            linewidth=sty["linewidth"], alpha=sty["alpha"],
            label=lbl,
        )
        ax1.errorbar(
            common_steps[::errbar_every], norm_mean[cap][::errbar_every],
            yerr=norm_stderr[cap][::errbar_every],
            fmt='none', ecolor=sty["color"], alpha=0.35, capsize=2,
        )

    ax1.set_ylabel("Normalized Reward\n(fraction of gap to optimal)", fontsize=11)
    ax1.set_title("Normalized Improvement over No-Teacher Baseline", fontsize=13)
    ax1.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Cumulative mean reward ----
    for cap in capacities:
        sty = styles[cap]
        ax2.plot(
            common_steps, cum_mean[cap],
            color=sty["color"], linestyle=sty["linestyle"],
            marker=sty["marker"], markevery=marker_every,
            markersize=sty["markersize"],
            linewidth=sty["linewidth"], alpha=sty["alpha"],
            label=_capacity_label(cap),
        )

    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Cumulative Mean Reward", fontsize=11)
    ax2.set_title("Cumulative Average Evaluation Reward", fontsize=13)
    ax2.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Improved learning curve figure saved to {save_path}")

    return fig
