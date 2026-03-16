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


def _sample_uniform_random_teacher_knowledge(
    env: GridEnv,
    n_goals: int,
    n_traps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Sample random known-goal and known-trap sets uniformly from non-start states.

    Counts are inferred from environment unless explicitly overridden by caller.
    """
    rng = rng or np.random.default_rng(0 if env.seed is None else env.seed)
    candidates = [
        env.idx_to_state(i)
        for i in range(env.n_states)
        if env.idx_to_state(i) != env.start
    ]
    total_needed = min(n_goals + n_traps, len(candidates))
    if total_needed <= 0:
        return [], []

    selected_idx = rng.choice(len(candidates), size=total_needed, replace=False)
    selected = [candidates[i] for i in selected_idx]
    goals = selected[:min(n_goals, len(selected))]
    traps = selected[min(n_goals, len(selected)):min(n_goals + n_traps, len(selected))]
    return goals, traps


def sample_uniform_random_teacher_knowledge(
    env: GridEnv,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Public helper to sample imagined goals/traps for cap=0 teachers.

    Uses counts inferred from the environment.
    """
    return _sample_uniform_random_teacher_knowledge(
        env,
        n_goals=len(env.goals),
        n_traps=len(env.traps),
        rng=rng,
    )


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
    known_traps = known_traps or []
    T = _build_transition_model(env)

    # Step 1: solve for optimal policy w.r.t. known goals
    R_known = _build_reward_matrix(env, known_goals)
    absorbing_known = _build_absorbing_state_indices(env, known_goals, known_traps)
    Q_known, _ = _solve_discounted_values(
        T,
        R_known,
        absorbing_known,
        gamma,
        aggregate_fn=lambda q: q.max(axis=1),
        tol=tol,
    )

    # Step 2: fix the greedy policy derived from Q_known
    optimal_actions = Q_known.argmax(axis=1)

    # Step 3: evaluate that fixed policy in the true environment
    R_true = _build_reward_matrix(env, env.goals)
    absorbing_true = _build_absorbing_state_indices(env, env.goals, env.traps)
    return _solve_discounted_values(
        T,
        R_true,
        absorbing_true,
        gamma,
        aggregate_fn=lambda q: q[np.arange(len(q)), optimal_actions],
        tol=tol,
    )


def compute_uniform_random_teacher_values(
    env: GridEnv,
    gamma: float = 0.99,
    tol: float = 1e-6,
    known_goals: Optional[List[Tuple[int, int]]] = None,
    known_traps: Optional[List[Tuple[int, int]]] = None,
    debug_print: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^mu and V^mu for a uniform-random teacher policy (gamma < 1).

    If known_goals/known_traps are not provided, sample random goals/traps
    uniformly from the state space using counts from the environment.
    """
    if known_goals is None or known_traps is None:
        sampled_goals, sampled_traps = sample_uniform_random_teacher_knowledge(env)
        goals = sampled_goals if known_goals is None else known_goals
        traps = sampled_traps if known_traps is None else known_traps
    else:
        goals = known_goals
        traps = known_traps

    if debug_print:
        print(f"[uniform_teacher] sampled/used goals: {goals}")
        print(f"[uniform_teacher] sampled/used traps: {traps}")

    T = _build_transition_model(env)
    R = _build_reward_matrix(env, goals)
    absorbing_states = _build_absorbing_state_indices(env, goals, traps)
    return _solve_discounted_values(
        T,
        R,
        absorbing_states,
        gamma,
        aggregate_fn=lambda q: q.max(axis=1),
        tol=tol,
    )


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
    known_goals: Optional[List[Tuple[int, int]]] = None,
    known_traps: Optional[List[Tuple[int, int]]] = None,
    gamma: Optional[float] = None,
    debug_print: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute uniform-random teacher values with automatic gamma from horizon.
    """
    if gamma is None:
        gamma = compute_gamma_from_horizon(env.horizon)
    Q, V = compute_uniform_random_teacher_values(
        env,
        gamma,
        known_goals=known_goals,
        known_traps=known_traps,
        debug_print=debug_print,
    )
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


def get_teacher_policy(Q_mu: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """Softmax policy from Q-values for imitation learning."""
    logits = Q_mu / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_mixture_teacher_values(
    env: GridEnv,
    zeta: float,
    gamma: float = 0.99,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^mu and V^mu for a mixture teacher policy mu(zeta).

    mu(zeta) takes the optimal action with probability zeta and a
    uniformly random action with probability (1-zeta) — i.e. a coin
    flip at each step:
        a ~ mu(zeta | s)  =>  a = a*(s) w.p. zeta, else Uniform(actions)

    Q^mu and V^mu are evaluated by rolling out mu(zeta) in the TRUE MDP
    (using env.goals / env.traps).  The Bellman aggregate is:
        V^mu(s) = zeta * Q^mu(s, a*(s)) + (1-zeta)/n_actions * sum_a Q^mu(s,a)

    Args:
        env:   GridEnv instance
        zeta:  mixture weight in [0, 1]  (0 = pure random, 1 = pure optimal)
        gamma: discount factor
        tol:   convergence tolerance

    Returns:
        Q: shape (n_states, n_actions)
        V: shape (n_states,)
    """
    T = _build_transition_model(env)
    R_true = _build_reward_matrix(env, env.goals)
    absorbing_true = _build_absorbing_state_indices(env, env.goals, env.traps)

    # Step 1: compute the optimal policy w.r.t. ALL true goals
    Q_star, _ = _solve_discounted_values(
        T, R_true, absorbing_true, gamma,
        aggregate_fn=lambda q: q.max(axis=1),
        tol=tol,
    )
    optimal_actions = Q_star.argmax(axis=1)  # shape (n_states,)

    # Step 2: evaluate the mixture policy in the true MDP.
    # The Bellman contraction uses the mixture policy's expected value so that
    # Q^mu(s,a) correctly reflects "take action a then follow mu(zeta) forever."
    def mixture_aggregate(Q: np.ndarray) -> np.ndarray:
        n_states = Q.shape[0]
        V_opt = Q[np.arange(n_states), optimal_actions]
        V_rand = Q.mean(axis=1)
        return zeta * V_opt + (1.0 - zeta) * V_rand

    Q, _ = _solve_discounted_values(
        T, R_true, absorbing_true, gamma,
        aggregate_fn=mixture_aggregate,
        tol=tol,
    )

    # Use the uniform-random baseline for the ADVANTAGE V so that
    # A^mu(s,a) = Q^mu(s,a) - mean_a Q^mu(s,a).
    #
    # Why: the mixture policy value V^mu(s) = zeta*Q(s,a*) + (1-zeta)*mean Q,
    # which makes A(s,a*) = (1-zeta)*[Q(s,a*) - mean Q] → 0 as zeta → 1.
    # At zeta=1 the optimal action has ZERO advantage — only bad actions are
    # penalised, never the right action rewarded — so the teacher signal is
    # entirely one-sided and weaker than the random teacher (zeta=0).
    #
    # With the random-policy baseline, A(s,a*) = Q^mu(s,a*) - mean Q^mu > 0
    # for all zeta in (0,1], and the spread grows with zeta (better policy →
    # larger Q* spread around the mean), giving stronger signal for better
    # teachers.
    V_baseline = Q.mean(axis=1)
    return Q, V_baseline


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
