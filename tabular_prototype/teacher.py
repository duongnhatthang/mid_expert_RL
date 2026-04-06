"""Teacher (expert) policy computation via value iteration."""

import numpy as np
from typing import Callable, List, Tuple, Optional, Set

from .config import compute_gamma_from_horizon
from .environment import GridEnv


def _build_transition_model(env: GridEnv) -> np.ndarray:
    """Build deterministic transition table T[s, a] -> next_state_idx."""
    T = np.zeros((env.n_states, env.n_actions), dtype=int)
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        for a in range(env.n_actions):
            next_state = env._apply_action(state, a)
            T[s_idx, a] = env.state_to_idx(next_state)
    return T


def _build_reward_matrix(env: GridEnv, known_goals: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build sparse reward matrix R[s, a] where reward=1 iff next_state is a known goal.
    """
    known_goals_set = set(known_goals)
    R = np.zeros((env.n_states, env.n_actions))
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        for a in range(env.n_actions):
            next_state = env._apply_action(state, a)
            if next_state in known_goals_set:
                R[s_idx, a] = 1.0
    return R


def _build_absorbing_state_indices(
    env: GridEnv,
    known_goals: List[Tuple[int, int]],
    known_traps: Optional[List[Tuple[int, int]]] = None,
) -> Set[int]:
    """Convert known goals/traps to absorbing-state indices."""
    known_traps = known_traps or []
    absorbing_states: Set[int] = set()
    for g in known_goals:
        absorbing_states.add(env.state_to_idx(g))
    for trap in known_traps:
        absorbing_states.add(env.state_to_idx(trap))
    return absorbing_states


def _solve_discounted_values(
    T: np.ndarray,
    R: np.ndarray,
    absorbing_states: Set[int],
    gamma: float,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic discounted Bellman solver with customizable state-value aggregator."""
    n_states, n_actions = T.shape
    Q = np.zeros((n_states, n_actions))
    V = np.zeros(n_states)

    for _iteration in range(10000):
        Q_old = Q.copy()
        for s_idx in range(n_states):
            if s_idx in absorbing_states:
                Q[s_idx, :] = 0.0
                continue
            for a in range(n_actions):
                next_idx = T[s_idx, a]
                Q[s_idx, a] = R[s_idx, a] + gamma * V[next_idx]
        V = aggregate_fn(Q)

        if np.abs(Q - Q_old).max() < tol:
            break

    return Q, V


def evaluate_policy_values(
    env: GridEnv,
    policy_probs: np.ndarray,
    gamma: float,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Q^pi and V^pi for an explicit policy matrix in the TRUE MDP.

    Args:
        env:          GridEnv instance (uses env.goals, env.traps for the true MDP).
        policy_probs: shape (n_states, n_actions) — pi(a|s) probability matrix.
        gamma:        discount factor.
        tol:          convergence tolerance.

    Returns:
        Q: shape (n_states, n_actions)
        V: shape (n_states,)
    """
    T = _build_transition_model(env)
    R_true = _build_reward_matrix(env, env.goals)
    absorbing_true = _build_absorbing_state_indices(env, env.goals, env.traps)

    def policy_aggregate(Q: np.ndarray) -> np.ndarray:
        return (policy_probs * Q).sum(axis=1)

    return _solve_discounted_values(T, R_true, absorbing_true, gamma, policy_aggregate, tol)


def build_optimal_policy(
    env: GridEnv,
    known_goals: List[Tuple[int, int]],
    gamma: float,
    tol: float = 1e-6,
    known_traps: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Build deterministic (one-hot) greedy policy optimal w.r.t. known_goals.

    Returns:
        policy_probs: shape (n_states, n_actions) — one-hot on argmax Q*.
    """
    known_traps = known_traps or []
    T = _build_transition_model(env)
    R_known = _build_reward_matrix(env, known_goals)
    absorbing_known = _build_absorbing_state_indices(env, known_goals, known_traps)
    Q_known, _ = _solve_discounted_values(
        T, R_known, absorbing_known, gamma,
        aggregate_fn=lambda q: q.max(axis=1),
        tol=tol,
    )
    optimal_actions = Q_known.argmax(axis=1)
    n_states, n_actions = Q_known.shape
    policy_probs = np.zeros((n_states, n_actions))
    policy_probs[np.arange(n_states), optimal_actions] = 1.0
    return policy_probs


def build_uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    """
    Build uniform random policy: pi(a|s) = 1/|A| for all s, a.

    Returns:
        policy_probs: shape (n_states, n_actions).
    """
    return np.full((n_states, n_actions), 1.0 / n_actions)


def build_mixture_policy(
    env: GridEnv,
    zeta: float,
    gamma: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Build mixture policy: pi(a|s) = zeta * I[a = a*(s)] + (1 - zeta) / |A|.

    a*(s) = argmax_a Q*(s,a) w.r.t. all TRUE goals (env.goals).

    Args:
        env:   GridEnv instance.
        zeta:  mixture weight in [0, 1]  (0 = pure random, 1 = pure optimal).
        gamma: discount factor.
        tol:   convergence tolerance.

    Returns:
        policy_probs: shape (n_states, n_actions).
    """
    n_actions = env.n_actions
    T = _build_transition_model(env)
    R_true = _build_reward_matrix(env, env.goals)
    absorbing_true = _build_absorbing_state_indices(env, env.goals, env.traps)
    Q_star, _ = _solve_discounted_values(
        T, R_true, absorbing_true, gamma,
        aggregate_fn=lambda q: q.max(axis=1),
        tol=tol,
    )
    optimal_actions = Q_star.argmax(axis=1)

    n_states = env.n_states
    policy_probs = np.full((n_states, n_actions), (1.0 - zeta) / n_actions)
    policy_probs[np.arange(n_states), optimal_actions] += zeta
    return policy_probs



def compute_teacher_values(
    env: GridEnv,
    known_goals: List[Tuple[int, int]],
    gamma: float = 0.99,
    tol: float = 1e-6,
    known_traps: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^mu and V^mu for teacher via value iteration (gamma < 1).

    The teacher's policy is optimal w.r.t. known_goals, but Q and V are
    evaluated by rolling out that policy in the true environment (env.goals,
    env.traps).

    Args:
        env: GridEnv instance
        known_goals: Goals the teacher knows about (used to derive policy)
        gamma: Discount factor
        tol: Convergence tolerance
        known_traps: Traps the teacher knows about (optional, used for policy)

    Returns:
        Q: shape (n_states, n_actions) — evaluated under true environment
        V: shape (n_states,)           — evaluated under true environment
    """
    policy_probs = build_optimal_policy(env, known_goals, gamma, tol, known_traps)
    return evaluate_policy_values(env, policy_probs, gamma, tol)


def compute_uniform_random_teacher_values(
    env: GridEnv,
    gamma: float = 0.99,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^mu and V^mu for a uniform-random teacher policy (gamma < 1).

    The teacher acts uniformly at random: pi(a|s) = 1/|A|.
    Q and V are evaluated in the TRUE MDP (env.goals, env.traps).
    """
    policy_probs = build_uniform_policy(env.n_states, env.n_actions)
    return evaluate_policy_values(env, policy_probs, gamma, tol)


def compute_teacher_values_auto(
    env: GridEnv,
    known_goals: List[Tuple[int, int]],
    known_traps: Optional[List[Tuple[int, int]]] = None,
    gamma: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute teacher values with automatic gamma from horizon.

    Returns:
        Q, V, effective_gamma
    """
    if gamma is None:
        gamma = compute_gamma_from_horizon(env.horizon)
    Q, V = compute_teacher_values(env, known_goals, gamma, known_traps=known_traps)
    return Q, V, gamma


def compute_uniform_random_teacher_values_auto(
    env: GridEnv,
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute uniform-random teacher values with automatic gamma from horizon.
    """
    if gamma is None:
        gamma = compute_gamma_from_horizon(env.horizon)
    Q, V = compute_uniform_random_teacher_values(env, gamma)
    return Q, V, gamma


def get_teacher_advantage(
    Q_mu: np.ndarray,
    V_mu: np.ndarray,
    state_idx: int,
    action: int,
) -> float:
    """
    Compute teacher advantage A^mu(s,a) = Q^mu(s,a) - V^mu(s).

    Q_mu shape: (n_states, n_actions), V_mu shape: (n_states,)
    """
    return Q_mu[state_idx, action] - V_mu[state_idx]



def compute_mixture_teacher_values(
    env: GridEnv,
    zeta: float,
    gamma: float = 0.99,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^mu and V^mu for a mixture teacher policy mu(zeta).

    mu(zeta)(a|s) = zeta * I[a = a*(s)] + (1 - zeta) / |A|

    where a*(s) = argmax_a Q*(s,a) w.r.t. all true goals.

    Returns true V^mu(s) = sum_a mu(a|s) Q^mu(s,a).

    Args:
        env:   GridEnv instance
        zeta:  mixture weight in [0, 1]  (0 = pure random, 1 = pure optimal)
        gamma: discount factor
        tol:   convergence tolerance

    Returns:
        Q: shape (n_states, n_actions)
        V: shape (n_states,)
    """
    policy_probs = build_mixture_policy(env, zeta, gamma, tol)
    return evaluate_policy_values(env, policy_probs, gamma, tol)


def compute_mixture_teacher_values_auto(
    env: GridEnv,
    zeta: float,
    gamma: Optional[float] = None,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute mixture teacher values with automatic gamma from horizon.

    Returns:
        Q, V, effective_gamma
    """
    if gamma is None:
        gamma = compute_gamma_from_horizon(env.horizon)
    Q, V = compute_mixture_teacher_values(env, zeta, gamma, tol)
    return Q, V, gamma
