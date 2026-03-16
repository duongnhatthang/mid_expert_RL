# Project Summary: Mid-Capacity Teacher RL

This document provides a comprehensive summary of the project for future AI agents and collaborators.

## Core Hypothesis

**In sparse reward MDPs with limited sample budgets, learning from a "mid-capacity" teacher outperforms both the best and worst teachers.**

The intuition:
- **Best teacher** has a complex policy (knows all goals) that requires many samples to learn
- **Worst teacher** provides no useful guidance (random policy)
- **Mid-capacity teacher** provides a simpler, more learnable signal within the sample budget

## Project Architecture

```
mid_expert_RL/
├── tabular_prototype.py      # Main implementation (self-contained)
├── run_experiments.py        # Experiment runner CLI
├── requirements.txt          # Dependencies: numpy, matplotlib, pandas
├── README.md                 # User documentation
├── SUMMARY.md               # This file (agent reference)
└── results/                  # Output directory
    └── figures/              # Generated plots
```

## Key Components

### 1. Environment: `GridEnv`

A configurable grid world with:
- **State space**: N×N grid (default 8×8)
- **Action space**: 4 actions (up, down, left, right)
- **Goals**: Absorbing states with reward=1
- **Traps**: Absorbing states with reward=0
- **Start**: Center of grid

**Key parameters:**
- `absorbing_states`: If True, goals/traps terminate episodes
- `slip_prob`: Stochastic action noise (prepared)
- `wind`: Directional drift (prepared)
- `reward_noise_std`: Reward noise (prepared)

### 2. Discount Factor

The discount factor uses the engineering trick `γ = 1 - 1/H` via `compute_gamma_from_horizon(horizon)`:

- Standard value iteration with effective horizon
- Q(s,a), V(s) are time-invariant
- Shape: Q[n_states, n_actions], V[n_states]

### 3. Teacher Value Computation

Teachers are defined by the number of goals they know:
- `capacity=0`: Random policy (no goals known)
- `capacity=k`: Knows first k goals
- `capacity=n_goals`: Best teacher (knows all)

**Functions:**
- `compute_teacher_values()`: Standard VI for γ < 1
- `compute_teacher_values_auto()`: Unified interface

### 4. PAV-RL Algorithm

Policy gradient with teacher advantage:

```
grad J = E[sum_h grad log π(a_h|s_h) * (Q^π(s_h,a_h) + α * A^μ(s_h,a_h))]
```

Where:
- `Q^π(s,a)`: Student's Q-value (Monte Carlo returns)
- `A^μ(s,a) = Q^μ(s,a) - V^μ(s)`: Teacher's advantage (FIXED)
- `α`: Weight for teacher guidance

### 5. Trap Placement Strategies

```python
class TrapPlacement(Enum):
    NONE = "none"
    NEAR_GOALS = "near_goals"      # Adjacent to goals
    ON_PATHS = "on_paths"          # On shortest paths
    RANDOM = "random"              # Random positions
```

Teacher trap awareness: Teacher knows `traps[:capacity]` (similar to goals).

### 6. Goal Generation

Two strategies:
- `generate_goals()`: Corners and edges (original)
- `generate_equidistant_goals()`: Equal Manhattan distance from start (new)

## Experiment Design

### 2x2 Matrix Experiment

Tests the hypothesis across:

|                  | Small Horizon | Large Horizon |
|------------------|---------------|---------------|
| **Low Budget**   | Most constrained | Wide exploration |
| **High Budget**  | Many samples | Unconstrained |

**Expected results:**
- Low budget: Mid-teacher > Best-teacher (simpler policy learnable)
- High budget: Best-teacher > Mid-teacher (enough samples for complex policy)

## Key Design Decisions

1. **Absorption states as default**: Goals/traps terminate episodes, aligning with theoretical finite-horizon MDP formulation

2. **Teacher trap awareness scales with capacity**: Teachers with higher capacity know more about both goals AND traps

3. **Manhattan distance for equidistant goals**: Natural for grid world (counts actual steps needed)

## API Quick Reference

### Creating Environment
```python
env = GridEnv(
    grid_size=8,
    goals=[(7, 7), (0, 7), (7, 0)],
    horizon=50,
    traps=[(3, 3), (4, 4)],
    absorbing_states=True
)
```

### Computing Teacher Values
```python
gamma = compute_gamma_from_horizon(env.horizon)
Q, V = compute_teacher_values(env, known_goals, gamma, known_traps=known_traps)
```

### Generating Goals and Traps
```python
goals = generate_equidistant_goals(grid_size=8, n_goals=4, distance=4)
traps = generate_traps(grid_size=8, goals=goals, n_traps=3, 
                       start=(4, 4), placement=TrapPlacement.NEAR_GOALS)
```

### Visualization
```python
visualize_teacher_policy(env, Q_mu, V_mu, known_goals, save_path='teacher.png')
visualize_student_policy(env, policy, goals, traps, save_path='student.png')
compare_policies(env, Q_teacher, V_teacher, policy_student, known_goals)
```

## Future Work Priorities

1. **Pufferfish integration** for faster experiments
2. **Gym/neural network extension** for real benchmarks
3. **Expert capability types** beyond "goals known"
4. **Curriculum learning over α**

## Common Pitfalls

1. **Forgetting to handle 3-tuple returns**: `env.step()` now returns `(next_state, reward, done)`

2. **Teacher advantage at terminal states**: Absorbing states have V=0, so A^μ computation must handle this

## Testing

```bash
# Quick test
python run_experiments.py --mode quick

# Full 2x2 experiment
python run_experiments.py --mode 2x2 --grid-size 8 --n-seeds 10 --n-goals 3
```

---

*Last updated: Based on implementation plan execution*
