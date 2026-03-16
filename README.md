# Mid-Capacity Teacher RL

Testing the hypothesis that in sparse reward MDPs with limited sample budgets, learning from a "mid-capacity" teacher outperforms both best and worst teachers.

## Hypothesis

In sparse reward MDPs with limited sample budgets:
- **Mid-capacity teacher** (knows partial information) > **Best teacher** (full knowledge)
- The mid-teacher provides a simple, learnable signal within the sample budget
- The best teacher has a more complex policy that requires more samples to learn

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test
python run_experiments.py --mode quick

# Run 2x2 exploration experiment (default)
python run_experiments.py --mode 2x2

# Run with custom parameters
python run_experiments.py --mode 2x2 --grid-size 10 --n-seeds 20 --n-goals 4

# Run full parameter sweep
python run_experiments.py --mode suite
```

## Project Structure

```
mid_expert_RL/
├── tabular_prototype.py      # Self-contained numpy implementation
├── run_experiments.py        # Main experiment runner
├── requirements.txt          # Dependencies (numpy, matplotlib, pandas)
├── README.md
└── results/                  # Output directory
    └── figures/              # Generated plots
```

## PAV-RL Objective

The PAV-RL (Policy with Advantage from teacher Value) objective:

```
Policy Gradient:
    sum_{h=1}^H grad_pi log pi(a_h | s_h) * (Q^pi(s_h, a_h) + alpha * A^mu(s_h, a_h))

Where:
    - Q^pi(s,a) = action-value from environment rewards under student policy
    - A^mu(s,a) = Q^mu(s,a) - V^mu(s) = advantage under FIXED teacher policy
```

**Key**: The teacher is FIXED during training. Only the student policy is updated.

## Teacher Capacities

Teachers are defined by the number of goals they know:
- `capacity=-1`: No teacher signal (student learns from environment rewards only)
- `capacity=0`: Worst teacher (teacher acts uniformly at random)
- `capacity=1`: Mid teacher (knows 1 goal)
- `capacity=3`: Best teacher (knows all 3 goals)

## 2x2 Exploration Experiment

The main experiment tests the hypothesis across a 2x2 matrix of conditions:

|                  | Small Horizon | Large Horizon |
|------------------|---------------|---------------|
| **Low Budget**   | Most constrained | Wide per-episode exploration |
| **High Budget**  | Many samples, short episodes | Unconstrained |

### Default Parameters

| Parameter | Value |
|-----------|-------|
| Grid size | 8x8 (configurable) |
| Goals | 3 corners (configurable) |
| Teacher capacity | 0 to n_goals |
| Budget LOW | ~3 visits per state-action |
| Budget HIGH | ~30 visits per state-action |
| Horizon SMALL | Just enough to reach one corner |
| Horizon LARGE | Enough to visit multiple goals |
| Seeds | 10 per configuration |

All results show mean ± standard error across seeds.

## Expected Results

At low sample budgets:
- Mid-teacher (cap=1) > Best-teacher (cap=3) > Worst-teacher (cap=0)

At high sample budgets:
- Best-teacher (cap=3) > Mid-teacher (cap=1) > Worst-teacher (cap=0)

The crossover point reveals the sample efficiency trade-off.

## Figures Generated

1. **2x2 Grid**: Bar charts showing mean reward by teacher capacity for each condition
   - Highlights the best performer in each condition
   - Saved to `results/exploration_2x2_results.png`

## New Features

### Discount Factor

The discount factor uses the engineering trick `γ = 1 - 1/H`:
- Standard infinite-horizon value iteration with effective horizon
- Q(s,a) and V(s) are time-invariant

### Absorption States

- **Goals**: Absorbing states with reward=1 (episode terminates)
- **Traps**: Absorbing states with reward=0 (episode terminates)

### Trap Placement Strategies

- `NEAR_GOALS`: Adjacent to goal positions (risk/reward tradeoff)
- `ON_PATHS`: On optimal paths from start to goals (tests navigation)
- `RANDOM`: Random positions avoiding start/goals

### Equidistant Goals

Generate goals at equal Manhattan distance from start:
```python
from tabular_prototype import generate_equidistant_goals
goals = generate_equidistant_goals(grid_size=8, n_goals=4, distance=4)
```

### Stochastic Dynamics (Prepared)

Environment supports configurable noise:
- `slip_prob`: Probability of random action
- `wind`: Consistent directional drift
- `reward_noise_std`: Gaussian noise on rewards

### Visualization

Visualize Q-values and policies:
```python
from tabular_prototype import visualize_teacher_policy, compare_policies
visualize_teacher_policy(env, Q_mu, V_mu, known_goals, save_path='teacher.png')
compare_policies(env, Q_teacher, V_teacher, student_policy, known_goals)
```

## Future Extensions

### Planned

#### Pufferfish Integration (Priority: High)
Use [pufferlib](https://github.com/PufferAI/PufferLib) for vectorized fast experiments:
- Parallel environment execution
- Efficient batched policy evaluation
- GPU-accelerated training

#### Gym/Neural Network Extension (Priority: High)
Extend to standard RL benchmarks:
- **Environments**: OpenAI Gym MDPs (CartPole, MountainCar, Atari)
- **Policy Gradient Teachers**: PPO, A2C with varying training budgets
- **Q-Learning Teachers**: DQN, SAC with different network architectures

#### Expert Capability Types (Priority: Medium)
Test different definitions of expert "capability":

1. **Function Class Capacity**
   - Simple: Linear policy, small MLP (1-2 layers)
   - Complex: Deep MLP, attention-based architectures
   
2. **Coverage Capacity**
   - Weak expert: Trained on limited state-action coverage
   - Strong expert: Full coverage during training
   
3. **Compute Budget Capacity**
   - Under-trained: Early-stopped models
   - Well-trained: Converged models with full compute

### Technical Improvements

- [ ] Absorption state handling in trajectory collection (edge cases)
- [ ] Comprehensive stochastic environment testing
- [ ] Parallel experiment execution with multiprocessing
- [ ] Experiment logging with wandb/tensorboard

### Research Directions

- Curriculum learning over α (teacher influence weight)
- Adaptive teacher selection during training
- Multi-teacher distillation (ensemble of different capacities)
- Theoretical analysis of sample complexity bounds
