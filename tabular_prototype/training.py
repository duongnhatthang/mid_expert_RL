"""PAV-RL policy gradient, updates, and evaluation."""

import numpy as np
from typing import List, Dict, Optional, Tuple

from .environment import GridEnv
from .student import TabularSoftmaxPolicy, Transition
from .teacher import (
    get_teacher_advantage,
    evaluate_policy_values,
)


def estimate_returns(trajectory: List[Transition], gamma: float = 0.99) -> List[float]:
    """Compute Monte Carlo returns G_t for each timestep."""
    returns = []
    G = 0.0
    for t in reversed(trajectory):
        G = t.reward + gamma * G
        returns.append(G)
    return list(reversed(returns))


# =========================================================================
# Exact Q^π computation (replaces Monte Carlo returns)
# =========================================================================

def compute_student_qvalues(
    env: GridEnv,
    policy: TabularSoftmaxPolicy,
    gamma: float,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact Q^π(s,a) and V^π(s) for the student's current policy
    via Bellman policy evaluation.

        Q^π(s,a) = R(s,a) + γ V^π(T(s,a))
        V^π(s)   = Σ_a π(a|s) Q^π(s,a)

    Returns:
        Q_pi: shape (n_states, n_actions)
        V_pi: shape (n_states,)
    """
    policy_probs = np.array([policy.get_probs(s) for s in range(env.n_states)])
    return evaluate_policy_values(env, policy_probs, gamma, tol)


# =========================================================================
# State-action visitation tracking
# =========================================================================

def compute_state_action_visitation(
    trajectories: List[List[Transition]],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """
    Compute empirical state-action visitation counts from trajectories.

    Returns:
        counts: shape (n_states, n_actions) — raw visit counts.
    """
    counts = np.zeros((n_states, n_actions))
    for traj in trajectories:
        for trans in traj:
            counts[trans.state_idx, trans.action] += 1
    return counts


def visitation_metrics(counts: np.ndarray) -> Dict[str, float]:
    """
    Compute summary metrics from state-action visitation counts.

    Args:
        counts: shape (n_states, n_actions) — raw visit counts.

    Returns:
        Dict with keys: unique_sa, unique_states, sa_entropy,
        state_entropy, total_visits.
    """
    total = counts.sum()
    if total == 0:
        return {
            'unique_sa': 0,
            'unique_states': 0,
            'sa_entropy': 0.0,
            'state_entropy': 0.0,
            'total_visits': 0,
        }

    # State-action level
    sa_flat = counts.ravel()
    unique_sa = int(np.sum(sa_flat > 0))
    sa_probs = sa_flat / total
    sa_probs = sa_probs[sa_probs > 0]
    sa_entropy = float(-np.sum(sa_probs * np.log(sa_probs)))

    # State level
    state_counts = counts.sum(axis=1)
    unique_states = int(np.sum(state_counts > 0))
    state_probs = state_counts / total
    state_probs = state_probs[state_probs > 0]
    state_entropy = float(-np.sum(state_probs * np.log(state_probs)))

    return {
        'unique_sa': unique_sa,
        'unique_states': unique_states,
        'sa_entropy': sa_entropy,
        'state_entropy': state_entropy,
        'total_visits': int(total),
    }


# =========================================================================
# PAV-RL gradient (now supports exact Q^π)
# =========================================================================

def compute_pav_rl_gradient(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float = 0.99,
    Q_pi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute PAV-RL policy gradient.

    grad J = E[sum_h grad log pi(a_h|s_h) * (Q^pi(s_h,a_h) + alpha * A^mu(s_h,a_h))]

    For tabular softmax: grad log pi(a|s) w.r.t. theta[s,a'] = 1(a=a') - pi(a'|s)

    Args:
        Q_pi: If provided, use exact Q^π(s,a) instead of Monte Carlo returns.
              This eliminates variance from return estimation while keeping
              on-policy state-action sampling for the gradient.
    """
    grad = np.zeros_like(policy.theta)

    for traj in trajectories:
        if Q_pi is None:
            returns = estimate_returns(traj, gamma)

        for i, trans in enumerate(traj):
            if Q_pi is not None:
                G_t = Q_pi[trans.state_idx, trans.action]
            else:
                G_t = returns[i]

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
