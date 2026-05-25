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
                effective_reward = (1.0 - alpha) * G_t + alpha * A_mu
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


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis, returning 0.0 if variance is zero."""
    flat = arr.ravel()
    var = flat.var()
    if var == 0:
        return 0.0
    mean = flat.mean()
    centered = flat - mean
    return float((centered ** 4).mean() / (var ** 2) - 3.0)


def exact_npg_update(
    policy: TabularSoftmaxPolicy,
    Q_pi: np.ndarray,
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    lr: float,
) -> dict:
    """
    Exact NPG update for tabular softmax (mirror descent).

    θ[s,a] += lr · ((1-α)·Q^π(s,a) + α·A^μ(s,a))  for all (s,a)

    Derived from Lemma F.2 (Agarwal et al. 2021, extended for PAV-RL).
    The state-dependent offset ν cancels in the softmax normalization.

    When Q_mu/V_mu are None (no teacher), reduces to θ += lr · Q^π.

    Returns a diagnostics dict with per-step statistics.
    """
    q_component = (1.0 - alpha) * Q_pi

    if Q_mu is not None and V_mu is not None:
        A_mu = Q_mu - V_mu[:, None]
        a_component = alpha * A_mu
        policy.theta += lr * (q_component + a_component)
        # Cosine similarity between the two update directions, flattened over
        # all (s, a). Indicates whether the teacher is REDIRECTING the student
        # (negative/small cosine) or AMPLIFYING its own gradient (cosine→1).
        # A small-magnitude teacher can still steer the softmax if its
        # direction is persistently different from Q^π.
        q_flat = q_component.reshape(-1)
        a_flat = a_component.reshape(-1)
        q_norm = float(np.linalg.norm(q_flat))
        a_norm = float(np.linalg.norm(a_flat))
        if q_norm > 0 and a_norm > 0:
            cos_sim = float(q_flat @ a_flat / (q_norm * a_norm))
        else:
            cos_sim = 0.0
        diag = {
            'q_pi_l2': float(np.linalg.norm(q_component)),
            'q_pi_max': float(np.abs(q_component).max()),
            'a_mu_l2': float(np.linalg.norm(a_component)),
            'a_mu_max': float(np.abs(a_component).max()),
            'a_mu_mean': float(A_mu.mean()),
            'a_mu_var': float(A_mu.var()),
            'a_mu_kurtosis': _safe_kurtosis(A_mu),
            'a_mu_min_val': float(A_mu.min()),
            'a_mu_max_val': float(A_mu.max()),
            'cos_q_a': cos_sim,
        }
    else:
        policy.theta += lr * Q_pi
        diag = {
            'q_pi_l2': float(np.linalg.norm(q_component)),
            'q_pi_max': float(np.abs(q_component).max()),
            'a_mu_l2': 0.0,
            'a_mu_max': 0.0,
            'a_mu_mean': 0.0,
            'a_mu_var': 0.0,
            'a_mu_kurtosis': 0.0,
            'a_mu_min_val': 0.0,
            'a_mu_max_val': 0.0,
            'cos_q_a': 0.0,
        }

    return diag


# =========================================================================
# NPG update-direction diagnostic
# =========================================================================

def update_direction_diagnostics(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float,
    start_idx: int,
    rng: np.random.Generator,
    n_bootstrap: int = 50,
) -> dict:
    """Compute U = F̂⁺ ĝ_α and its trajectory-bootstrap variance.

    Returns six per-step scalars:
        cos_npg_dir       cos(U_α, U_{α=0})  at the same θ_t (same batch).
                          NaN if either norm is zero.
        var_U_trace       Σ_{s,a} Var_b(U_b[s,a])  across n_bootstrap trajectory-
                          level bootstrap resamples of the current batch.
        var_U_s0_a{0..3}  Var_b(U_b[s₀, a])  for each action a.
    """
    n_states, n_actions = policy.theta.shape

    # Per-trajectory cached arrays: (s_idx, a_idx, G_t, A_mu).
    per_traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    has_teacher = Q_mu is not None and V_mu is not None
    for traj in trajectories:
        if len(traj) == 0:
            per_traj.append((
                np.empty(0, dtype=int), np.empty(0, dtype=int),
                np.empty(0), np.empty(0),
            ))
            continue
        G = np.array(estimate_returns(traj, gamma), dtype=float)
        s_arr = np.array([t.state_idx for t in traj], dtype=int)
        a_arr = np.array([t.action for t in traj], dtype=int)
        if has_teacher:
            A_mu = np.array(
                [get_teacher_advantage(Q_mu, V_mu, int(s), int(a))
                 for s, a in zip(s_arr, a_arr)], dtype=float,
            )
        else:
            A_mu = np.zeros_like(G)
        per_traj.append((s_arr, a_arr, G, A_mu))

    def _compute_U(traj_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (U_α, U_{α=0}) for a (possibly resampled) set of trajectories."""
        if len(traj_indices) == 0:
            return (np.zeros((n_states, n_actions)),
                    np.zeros((n_states, n_actions)))
        s_all = np.concatenate([per_traj[i][0] for i in traj_indices])
        a_all = np.concatenate([per_traj[i][1] for i in traj_indices])
        G_all = np.concatenate([per_traj[i][2] for i in traj_indices])
        Amu_all = np.concatenate([per_traj[i][3] for i in traj_indices])
        if len(s_all) == 0:
            return (np.zeros((n_states, n_actions)),
                    np.zeros((n_states, n_actions)))

        A_eff = (1.0 - alpha) * G_all + alpha * Amu_all
        A_van = G_all  # α=0 baseline (vanilla NPG)

        U_alpha = np.zeros((n_states, n_actions))
        U_van = np.zeros((n_states, n_actions))

        # The 1/n in F̂ and ĝ cancels under pinv, so we drop it here.
        # Group by state via np.unique for efficiency.
        for s in np.unique(s_all):
            mask = s_all == s
            actions_s = a_all[mask]
            pi_s = policy.get_probs(int(s))  # shape (A,)

            # ψᵢ = e_{a_i} - π(.|s)  for each transition i visiting state s.
            # F̂_s = Σ_i ψᵢ ψᵢᵀ; ĝ_s = Σ_i A_i ψᵢ.
            psi_mat = np.zeros((len(actions_s), n_actions))
            psi_mat[np.arange(len(actions_s)), actions_s] = 1.0
            psi_mat -= pi_s[None, :]

            F_s = psi_mat.T @ psi_mat
            g_alpha_s = psi_mat.T @ A_eff[mask]
            g_van_s = psi_mat.T @ A_van[mask]

            F_s_pinv = np.linalg.pinv(F_s)
            U_alpha[int(s)] = F_s_pinv @ g_alpha_s
            U_van[int(s)] = F_s_pinv @ g_van_s

        return U_alpha, U_van

    # Point estimate on the full batch (for cosine).
    n_traj = len(trajectories)
    full_indices = np.arange(n_traj)
    U_alpha_full, U_van_full = _compute_U(full_indices)

    a_flat = U_alpha_full.reshape(-1)
    v_flat = U_van_full.reshape(-1)
    a_norm = float(np.linalg.norm(a_flat))
    v_norm = float(np.linalg.norm(v_flat))
    if a_norm > 0.0 and v_norm > 0.0:
        cos_npg_dir = float(a_flat @ v_flat / (a_norm * v_norm))
    else:
        cos_npg_dir = float('nan')

    # Bootstrap U_α only (we don't need variance of the α=0 direction).
    if n_traj > 0 and n_bootstrap > 0:
        U_boots = np.zeros((n_bootstrap, n_states, n_actions))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_traj, size=n_traj)
            U_b, _ = _compute_U(idx)
            U_boots[b] = U_b
        U_var = U_boots.var(axis=0)
    else:
        U_var = np.zeros((n_states, n_actions))

    result = {
        'cos_npg_dir': cos_npg_dir,
        'var_U_trace': float(U_var.sum()),
    }
    s0_var = U_var[int(start_idx)]
    for a in range(4):
        result[f'var_U_s0_a{a}'] = float(s0_var[a]) if a < n_actions else 0.0
    return result


def _update_direction_full_batch(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U_α, U_{α=0}) on the full batch.

    Exposed for testing the softmax-tangent property; production code
    should call `update_direction_diagnostics`.
    """
    # Build per_traj cache identically to update_direction_diagnostics
    n_states, n_actions = policy.theta.shape
    has_teacher = Q_mu is not None and V_mu is not None
    s_all_list, a_all_list, G_all_list, Amu_all_list = [], [], [], []
    for traj in trajectories:
        if not traj:
            continue
        G = np.array(estimate_returns(traj, gamma), dtype=float)
        s_arr = np.array([t.state_idx for t in traj], dtype=int)
        a_arr = np.array([t.action for t in traj], dtype=int)
        if has_teacher:
            A_mu = np.array(
                [get_teacher_advantage(Q_mu, V_mu, int(s), int(a))
                 for s, a in zip(s_arr, a_arr)], dtype=float,
            )
        else:
            A_mu = np.zeros_like(G)
        s_all_list.append(s_arr); a_all_list.append(a_arr)
        G_all_list.append(G); Amu_all_list.append(A_mu)
    if not s_all_list:
        return (np.zeros((n_states, n_actions)),
                np.zeros((n_states, n_actions)))
    s_all = np.concatenate(s_all_list)
    a_all = np.concatenate(a_all_list)
    G_all = np.concatenate(G_all_list)
    Amu_all = np.concatenate(Amu_all_list)
    A_eff = (1.0 - alpha) * G_all + alpha * Amu_all
    A_van = G_all

    U_alpha = np.zeros((n_states, n_actions))
    U_van = np.zeros((n_states, n_actions))
    for s in np.unique(s_all):
        mask = s_all == s
        actions_s = a_all[mask]
        pi_s = policy.get_probs(int(s))
        psi_mat = np.zeros((len(actions_s), n_actions))
        psi_mat[np.arange(len(actions_s)), actions_s] = 1.0
        psi_mat -= pi_s[None, :]
        F_s = psi_mat.T @ psi_mat
        g_alpha_s = psi_mat.T @ A_eff[mask]
        g_van_s = psi_mat.T @ A_van[mask]
        F_pinv = np.linalg.pinv(F_s)
        U_alpha[int(s)] = F_pinv @ g_alpha_s
        U_van[int(s)] = F_pinv @ g_van_s
    return U_alpha, U_van


def evaluate_policy(
    env: GridEnv, policy: TabularSoftmaxPolicy,
    n_episodes: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Evaluate policy performance with absorption state support."""
    from .config import compute_gamma_from_horizon
    gamma = compute_gamma_from_horizon(env.horizon)

    total_rewards = []
    discounted_returns = []
    goal_reached = []
    episode_lengths = []
    trap_reached = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        discounted_return = 0.0
        gamma_t = 1.0
        reached_goal = False
        reached_trap = False
        steps = 0

        for _ in range(env.horizon):
            state_idx = env.state_to_idx(state)
            action = policy.sample_action(state_idx, rng)

            next_state, reward, done = env.step(state, action, rng)

            episode_reward += reward
            discounted_return += gamma_t * reward
            gamma_t *= gamma
            steps += 1

            if next_state in env._goals_set:
                reached_goal = True
            if next_state in env._traps_set:
                reached_trap = True

            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        discounted_returns.append(discounted_return)
        goal_reached.append(reached_goal)
        trap_reached.append(reached_trap)
        episode_lengths.append(steps)

    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_reward_discounted': float(np.mean(discounted_returns)),
        'std_reward_discounted': float(np.std(discounted_returns)),
        'goal_rate': float(np.mean(goal_reached)),
        'trap_rate': float(np.mean(trap_reached)),
        'mean_episode_length': float(np.mean(episode_lengths)),
    }
