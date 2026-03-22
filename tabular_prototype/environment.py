"""Grid environment with sparse rewards, absorption states, and trap placement."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Trap Placement Strategies
# =============================================================================

class TrapPlacement(Enum):
    """Trap placement strategies."""
    NONE = "none"
    NEAR_GOALS = "near_goals"
    ON_PATHS = "on_paths"
    RANDOM = "random"


def generate_traps_near_goals(
    grid_size: int,
    goals: List[Tuple[int, int]],
    n_traps: int,
    start: Tuple[int, int],
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, int]]:
    """Generate traps adjacent to goal positions (risk/reward tradeoff near goals)."""
    rng = rng or np.random.default_rng()
    goals_set = set(goals)

    adjacent_cells = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for goal in goals:
        for dr, dc in directions:
            nr, nc = goal[0] + dr, goal[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                pos = (nr, nc)
                if pos not in goals_set and pos != start:
                    adjacent_cells.add(pos)

    adjacent_list = list(adjacent_cells)
    if len(adjacent_list) == 0:
        return []

    n_traps = min(n_traps, len(adjacent_list))
    selected_indices = rng.choice(len(adjacent_list), size=n_traps, replace=False)
    return [adjacent_list[i] for i in selected_indices]


def generate_traps_on_paths(
    grid_size: int,
    goals: List[Tuple[int, int]],
    n_traps: int,
    start: Tuple[int, int],
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, int]]:
    """Generate traps on optimal (shortest) paths from start to goals."""
    rng = rng or np.random.default_rng()
    goals_set = set(goals)

    path_cells = set()
    for goal in goals:
        total_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        for r in range(grid_size):
            for c in range(grid_size):
                pos = (r, c)
                dist_to_start = abs(r - start[0]) + abs(c - start[1])
                dist_to_goal = abs(r - goal[0]) + abs(c - goal[1])
                if dist_to_start + dist_to_goal == total_dist:
                    if pos not in goals_set and pos != start:
                        path_cells.add(pos)

    path_list = list(path_cells)
    if len(path_list) == 0:
        return []

    n_traps = min(n_traps, len(path_list))
    selected_indices = rng.choice(len(path_list), size=n_traps, replace=False)
    return [path_list[i] for i in selected_indices]


def generate_traps_random(
    grid_size: int,
    goals: List[Tuple[int, int]],
    n_traps: int,
    start: Tuple[int, int],
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, int]]:
    """Generate traps at random positions (avoiding start and goals)."""
    rng = rng or np.random.default_rng()
    excluded = set(goals) | {start}

    valid_positions = [
        (r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if (r, c) not in excluded
    ]

    if len(valid_positions) == 0:
        return []

    n_traps = min(n_traps, len(valid_positions))
    selected_indices = rng.choice(len(valid_positions), size=n_traps, replace=False)
    return [valid_positions[i] for i in selected_indices]


def generate_traps(
    grid_size: int,
    goals: List[Tuple[int, int]],
    n_traps: int,
    start: Tuple[int, int],
    placement: TrapPlacement = TrapPlacement.RANDOM,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[int, int]]:
    """Generate traps using the specified placement strategy."""
    if n_traps <= 0 or placement == TrapPlacement.NONE:
        return []

    rng = rng or np.random.default_rng()

    dispatch = {
        TrapPlacement.NEAR_GOALS: generate_traps_near_goals,
        TrapPlacement.ON_PATHS: generate_traps_on_paths,
        TrapPlacement.RANDOM: generate_traps_random,
    }
    fn = dispatch.get(placement)
    return fn(grid_size, goals, n_traps, start, rng) if fn else []


# =============================================================================
# Environment
# =============================================================================

@dataclass
class GridEnv:
    """
    Grid environment with sparse rewards and absorption states.

    Features:
    - Goals: Absorbing states with reward=1
    - Traps: Absorbing states with reward=0
    - Configurable stochastic dynamics (slip, wind, reward noise)
    """
    grid_size: int
    goals: List[Tuple[int, int]]
    horizon: int
    traps: List[Tuple[int, int]] = field(default_factory=list)
    absorbing_states: bool = True
    slip_prob: float = 0.0
    wind: Tuple[int, int] = (0, 0)
    reward_noise_std: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.n_states = self.grid_size * self.grid_size
        self.n_actions = 4  # up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['up', 'down', 'left', 'right']
        self.start = (self.grid_size // 2, self.grid_size // 2)
        self._rng = np.random.default_rng(self.seed) if self.seed is not None else None
        self._goals_set = set(self.goals)
        self._traps_set = set(self.traps)

    def state_to_idx(self, state: Tuple[int, int]) -> int:
        return state[0] * self.grid_size + state[1]

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return (idx // self.grid_size, idx % self.grid_size)

    def _apply_action(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Apply action to get next state (without stochastic effects)."""
        dr, dc = self.actions[action]
        new_r = int(np.clip(state[0] + dr, 0, self.grid_size - 1))
        new_c = int(np.clip(state[1] + dc, 0, self.grid_size - 1))
        return (new_r, new_c)

    def step(self, state: Tuple[int, int], action: int,
             rng: Optional[np.random.Generator] = None) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take action, return (next_state, reward, done).

        Args:
            state: Current position (row, col)
            action: Action index (0=up, 1=down, 2=left, 3=right)
            rng: Random generator for stochastic dynamics (optional)

        Returns:
            next_state, reward, done
        """
        rng = rng or self._rng

        if self.slip_prob > 0 and rng is not None:
            if rng.random() < self.slip_prob:
                action = rng.integers(0, self.n_actions)

        next_state = self._apply_action(state, action)

        if self.wind != (0, 0):
            wr, wc = self.wind
            next_state = (
                int(np.clip(next_state[0] + wr, 0, self.grid_size - 1)),
                int(np.clip(next_state[1] + wc, 0, self.grid_size - 1))
            )

        if next_state in self._goals_set:
            base_reward = 1.0
        elif next_state in self._traps_set:
            base_reward = 0.0
        else:
            base_reward = 0.0

        if self.reward_noise_std > 0 and rng is not None:
            reward = base_reward + rng.normal(0, self.reward_noise_std)
        else:
            reward = base_reward

        done = False
        if self.absorbing_states:
            done = next_state in self._goals_set or next_state in self._traps_set

        return next_state, reward, done

    def reset(self) -> Tuple[int, int]:
        return self.start

    def is_absorbing(self, state: Tuple[int, int]) -> bool:
        """Check if state is an absorbing state (goal or trap)."""
        return state in self._goals_set or state in self._traps_set

    def get_all_absorbing_states(self) -> List[Tuple[int, int]]:
        """Return list of all absorbing states."""
        return list(self._goals_set | self._traps_set)


# =============================================================================
# Goal Generation Utilities
# =============================================================================

def generate_equidistant_goals(
    grid_size: int,
    n_goals: int,
    distance: Optional[int] = None,
    start: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """
    Generate goals at equal Manhattan distance from the starting point.

    Goals form a diamond pattern around the starting point, making all goals
    equally attractive in terms of distance.

    Args:
        grid_size: Size of the grid
        n_goals: Number of goals to generate
        distance: Manhattan distance from start (default: reaches grid edges)
        start: Starting position (default: center of grid)

    Returns:
        List of (row, col) goal positions at equal Manhattan distance
    """
    if start is None:
        start = (grid_size // 2, grid_size // 2)

    if distance is None:
        distance = min(
            start[0],
            grid_size - 1 - start[0],
            start[1],
            grid_size - 1 - start[1]
        )

    equidistant_cells = [
        (r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if abs(r - start[0]) + abs(c - start[1]) == distance
    ]

    if len(equidistant_cells) == 0:
        raise ValueError(f"No cells at distance {distance} from start {start} in grid of size {grid_size}")

    if n_goals > len(equidistant_cells):
        raise ValueError(f"Cannot generate {n_goals} goals at distance {distance}. "
                         f"Only {len(equidistant_cells)} cells available.")

    def angle_from_start(pos):
        dr, dc = pos[0] - start[0], pos[1] - start[1]
        return np.arctan2(dr, dc)

    equidistant_cells.sort(key=angle_from_start)

    if n_goals == len(equidistant_cells):
        return equidistant_cells

    step = len(equidistant_cells) / n_goals
    return [equidistant_cells[int(i * step) % len(equidistant_cells)] for i in range(n_goals)]


def compute_exploration_thresholds(grid_size: int, n_actions: int = 4) -> dict:
    """
    Compute thresholds for "enough" exploration.

    Returns dict with budget and horizon thresholds.
    """
    n_states = grid_size * grid_size
    n_state_actions = n_states * n_actions
    center = grid_size // 2
    corner_dist = (grid_size - 1 - center) + (grid_size - 1 - center)

    return {
        'n_states': n_states,
        'n_state_actions': n_state_actions,
        'corner_distance': corner_dist,
        'budget_low': n_state_actions * 2,
        'budget_high': n_state_actions * 8,
        'horizon_small': corner_dist,
        'horizon_large': grid_size * 4,
    }
