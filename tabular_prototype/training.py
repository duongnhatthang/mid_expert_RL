"""PAV-RL policy gradient, updates, and evaluation."""

import numpy as np
from typing import List, Dict, Optional

from .environment import GridEnv
from .student import TabularSoftmaxPolicy, Transition
from .teacher import get_teacher_advantage


def estimate_returns(trajectory: List[Transition], gamma: float = 0.99) -> List[float]:
    """Compute Monte Carlo returns G_t for each timestep."""
    returns = []
    G = 0.0
    for t in reversed(trajectory):
        G = t.reward + gamma * G
        returns.append(G)
    return list(reversed(returns))


def compute_pav_rl_gradient(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Compute PAV-RL policy gradient.

    grad J = E[sum_h grad log pi(a_h|s_h) * (Q^pi(s_h,a_h) + alpha * A^mu(s_h,a_h))]

    For tabular softmax: grad log pi(a|s) w.r.t. theta[s,a'] = 1(a=a') - pi(a'|s)
    """
    grad = np.zeros_like(policy.theta)

    for traj in trajectories:
        returns = estimate_returns(traj, gamma)

        for trans, G_t in zip(traj, returns):
            if Q_mu is not None and V_mu is not None:
                A_mu = get_teacher_advantage(
                    Q_mu, V_mu, trans.state_idx, trans.action
                )
                effective_reward = G_t + alpha * A_mu
            else:
                effective_reward = G_t

            probs = policy.get_probs(trans.state_idx)
            for a in range(policy.n_actions):
                if a == trans.action:
                    grad[trans.state_idx, a] += effective_reward * (1 - probs[a])
                else:
                    grad[trans.state_idx, a] += effective_reward * (-probs[a])

    grad /= len(trajectories)
    return grad


def update_policy(policy: TabularSoftmaxPolicy, grad: np.ndarray, lr: float):
    """Gradient ascent update."""
    policy.theta += lr * grad


def evaluate_policy(
    env: GridEnv, policy: TabularSoftmaxPolicy,
    n_episodes: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Evaluate policy performance with absorption state support."""
    total_rewards = []
    goal_reached = []
    episode_lengths = []
    trap_reached = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        reached_goal = False
        reached_trap = False
        steps = 0

        for _ in range(env.horizon):
            state_idx = env.state_to_idx(state)
            action = policy.sample_action(state_idx, rng)

            next_state, reward, done = env.step(state, action, rng)

            episode_reward += reward
            steps += 1

            if next_state in env._goals_set:
                reached_goal = True
            if next_state in env._traps_set:
                reached_trap = True

            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        goal_reached.append(reached_goal)
        trap_reached.append(reached_trap)
        episode_lengths.append(steps)

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'goal_rate': np.mean(goal_reached),
        'trap_rate': np.mean(trap_reached),
        'mean_episode_length': np.mean(episode_lengths)
    }
