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
# PG / NPG update-direction diagnostics
# =========================================================================

def update_direction_diagnostics(
    policy: TabularSoftmaxPolicy,
    trajectories: List[List[Transition]],
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
    gamma: float,
    start_idx: int,
    Q_pi: Optional[np.ndarray] = None,
    V_pi: Optional[np.ndarray] = None,
) -> dict:
    """Sample-mode PG-direction diagnostics with per-trajectory normalization.

    Per-trajectory ĝ_τ,α = Σ_t A_t ψ_t (A_t = (1-α)G_t + α A^μ), normalized
    by ‖ĝ_τ,α‖ before averaging into ū_α.

    The α=0 reference is the *exact* π-centered advantage A^π = Q^π - V^π
    computed over all (s, a) from the student's policy — not a trajectory
    estimate. This matches the exact-mode reference and avoids the sparse-
    reward degeneracy where ĝ_τ,0 = 0 for trajectories that never hit a goal.

    Returns:
        pg_bias: -cos(ū_α, A^π). Higher → more rotated from the ideal PG
            direction. -1 = perfect alignment.
        var_g_trace: Σ_{s,a} Var_τ(û_τ,α[s,a]).
        var_g_visited: (1/n) Σ_τ Σ_t Var_τ'(û_τ',α[s_t, a_t]).
        var_inner_g_ref: Var_τ(⟨ĝ_τ,α/‖ĝ_τ,α‖, A^π/‖A^π‖⟩_F) — variance
            across rollouts of the *normalized* per-rollout cosine alignment
            with the α=0 reference. Pure rotational variability, scale removed
            on both sides. NaN-skip zero-norm rollouts.

    `start_idx` is accepted for API parity but unused.
    """
    del start_idx  # signature parity only
    n_states, n_actions = policy.theta.shape
    n_traj = len(trajectories)
    has_teacher = Q_mu is not None and V_mu is not None

    # Per-trajectory ĝ arrays (raw, pre-normalization).
    g_alpha = np.zeros((n_traj, n_states, n_actions))
    # Per-trajectory visit counts (multiset).
    visits = np.zeros((n_traj, n_states, n_actions), dtype=float)

    for tau_idx, traj in enumerate(trajectories):
        if not traj:
            continue
        G = estimate_returns(traj, gamma)
        for t, trans in enumerate(traj):
            s = trans.state_idx
            a = trans.action
            G_t = G[t]
            if has_teacher:
                A_mu = get_teacher_advantage(Q_mu, V_mu, s, a)
            else:
                A_mu = 0.0
            A_eff = (1.0 - alpha) * G_t + alpha * A_mu

            pi_s = policy.get_probs(s)
            # ψ at row s: e_a - π(·|s).
            psi_row = -pi_s.copy()
            psi_row[a] += 1.0

            g_alpha[tau_idx, s, :] += A_eff * psi_row

            visits[tau_idx, s, a] += 1.0

    # Normalize per trajectory; NaN-skip zero-norm trajectories.
    norms_alpha = np.linalg.norm(g_alpha.reshape(n_traj, -1), axis=1) if n_traj > 0 else np.zeros(0)

    def _normalize(g_arr, norms):
        out = np.full_like(g_arr, np.nan)
        nonzero = norms > 0
        out[nonzero] = g_arr[nonzero] / norms[nonzero, None, None]
        return out

    u_alpha = _normalize(g_alpha, norms_alpha)

    # Mean across non-NaN trajectories.
    import warnings
    if n_traj > 0:
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            u_bar_alpha = np.nanmean(u_alpha, axis=0)
    else:
        u_bar_alpha = np.zeros((n_states, n_actions))

    # α=0 reference: exact π-centered advantage A^π = Q^π - V^π over all (s,a).
    if Q_pi is not None:
        if V_pi is None:
            pi_all = np.stack([policy.get_probs(s) for s in range(n_states)])
            V_pi_arr = (pi_all * Q_pi).sum(axis=1)
        else:
            V_pi_arr = V_pi
        u_ref = Q_pi - V_pi_arr[:, None]
    else:
        u_ref = np.full((n_states, n_actions), np.nan)

    # Cosine (properly normalized on the outside).
    a_flat = u_bar_alpha.reshape(-1)
    v_flat = u_ref.reshape(-1)
    a_norm = float(np.linalg.norm(a_flat))
    v_norm = float(np.linalg.norm(v_flat))
    if (
        a_norm > 0.0
        and v_norm > 0.0
        and np.isfinite(a_norm)
        and np.isfinite(v_norm)
    ):
        cos_val = float(a_flat @ v_flat / (a_norm * v_norm))
    else:
        cos_val = float('nan')
    pg_bias = -cos_val if not np.isnan(cos_val) else float('nan')

    # Normalized per-cell variance (uses nanvar to ignore zero-norm τ's).
    if n_traj > 0:
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            u_var = np.nanvar(u_alpha, axis=0)
        u_var = np.nan_to_num(u_var, nan=0.0)
        mean_visits = visits.mean(axis=0)
    else:
        u_var = np.zeros((n_states, n_actions))
        mean_visits = np.zeros((n_states, n_actions))

    var_g_trace = float(u_var.sum())
    var_g_visited = float((u_var * mean_visits).sum())

    # Variance across rollouts of the normalized inner product
    # ⟨ĝ_τ,α/‖ĝ_τ,α‖, A^π/‖A^π‖⟩_F (i.e. per-rollout cosine vs. reference).
    if n_traj > 0 and Q_pi is not None and v_norm > 0 and np.isfinite(v_norm):
        ref_unit = v_flat / v_norm
        u_alpha_flat = u_alpha.reshape(n_traj, -1)
        inner_per_traj = u_alpha_flat @ ref_unit  # NaN for zero-norm τ
        var_inner_g_ref = float(np.nanvar(inner_per_traj))
        if not np.isfinite(var_inner_g_ref):
            var_inner_g_ref = float('nan')
    else:
        var_inner_g_ref = float('nan')

    return {
        'pg_bias': pg_bias,
        'var_g_trace': var_g_trace,
        'var_g_visited': var_g_visited,
        'var_inner_g_ref': var_inner_g_ref,
    }


def exact_direction_diagnostics(
    policy: TabularSoftmaxPolicy,
    Q_pi: np.ndarray,
    Q_mu: Optional[np.ndarray],
    V_mu: Optional[np.ndarray],
    alpha: float,
) -> dict:
    """Exact-mode NPG-direction diagnostics at θ_t.

    Computes A_eff(s, a) = (1-α) Q^π(s, a) + α A^μ(s, a) over all (s, a),
    then two centerings:
        U^(NPG)[s, a] = A_eff[s, a] - V_eff(s)      (π-weighted centering)
        U^(pinv)[s, a] = A_eff[s, a] - mean_a' A_eff[s, a']  (uniform centering;
                                                              strict F^† g)
    where V_eff(s) = Σ_a' π(a'|s) A_eff(s, a').

    Returns cos(U_α, U_{α=0}) for each centering. The α=0 baseline uses
    A_eff(α=0) = Q^π.
    """
    n_states, n_actions = policy.theta.shape
    pi = np.stack([policy.get_probs(s) for s in range(n_states)])  # (S, A)

    if Q_mu is not None and V_mu is not None:
        A_mu = Q_mu - V_mu[:, None]
    else:
        A_mu = np.zeros_like(Q_pi)

    A_eff_alpha = (1.0 - alpha) * Q_pi + alpha * A_mu
    A_eff_van = Q_pi.copy()  # α=0

    # π-centered (classical NPG / advantage form).
    V_eff_alpha = (pi * A_eff_alpha).sum(axis=1, keepdims=True)
    V_eff_van = (pi * A_eff_van).sum(axis=1, keepdims=True)
    U_npg_alpha = A_eff_alpha - V_eff_alpha
    U_npg_van = A_eff_van - V_eff_van

    # Uniform-centered (strict Moore-Penrose pinv of F̂ g).
    U_pinv_alpha = A_eff_alpha - A_eff_alpha.mean(axis=1, keepdims=True)
    U_pinv_van = A_eff_van - A_eff_van.mean(axis=1, keepdims=True)

    def _cos(x: np.ndarray, y: np.ndarray) -> float:
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        xn = float(np.linalg.norm(xf))
        yn = float(np.linalg.norm(yf))
        if xn > 0.0 and yn > 0.0:
            return float(xf @ yf / (xn * yn))
        return float('nan')

    cos_npg = _cos(U_npg_alpha, U_npg_van)
    cos_pinv = _cos(U_pinv_alpha, U_pinv_van)
    u_bias_npg = -cos_npg if not np.isnan(cos_npg) else float('nan')
    u_bias_pinv = -cos_pinv if not np.isnan(cos_pinv) else float('nan')

    return {
        'u_bias_npg': u_bias_npg,
        'u_bias_pinv': u_bias_pinv,
    }


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
