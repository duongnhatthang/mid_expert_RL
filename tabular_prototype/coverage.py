"""Coverage / distribution-mismatch diagnostics.

Exact discounted state-action occupancy and divergences between the student
policy's occupancy d^pi and fixed reference occupancies d^mu.
"""

import numpy as np

from .teacher import build_optimal_policy
from .student import TabularSoftmaxPolicy
from .training import compute_student_qvalues, exact_npg_update


def build_transition_table(env) -> np.ndarray:
    """Deterministic transition table T[s, a] -> next_state_idx."""
    T = np.zeros((env.n_states, env.n_actions), dtype=int)
    for s_idx in range(env.n_states):
        state = env.idx_to_state(s_idx)
        for a in range(env.n_actions):
            T[s_idx, a] = env.state_to_idx(env._apply_action(state, a))
    return T


def policy_to_matrix(policy) -> np.ndarray:
    """Row-softmax of policy.theta -> (n_states, n_actions) prob matrix."""
    logits = policy.theta - policy.theta.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_occupancy(transition, policy_probs, start_dist, gamma,
                      absorbing_states) -> np.ndarray:
    """Exact discounted state-action occupancy.

    d_state^T = (1 - gamma) * start_dist^T (I - gamma P_pi)^{-1}
    d(s, a)   = d_state(s) * policy_probs(s, a),   sum_{s,a} d = 1.

    Absorbing states self-loop (P[s,s]=1) so terminal mass accumulates there
    instead of leaking, matching episodic termination.
    """
    n_states, n_actions = policy_probs.shape
    absorbing = set(absorbing_states)
    P = np.zeros((n_states, n_states))
    for s in range(n_states):
        if s in absorbing:
            P[s, s] = 1.0
            continue
        for a in range(n_actions):
            P[s, transition[s, a]] += policy_probs[s, a]
    A = np.eye(n_states) - gamma * P
    d_state = (1.0 - gamma) * np.linalg.solve(A.T, np.asarray(start_dist, float))
    return d_state[:, None] * policy_probs


def _smooth(d, eps):
    d = np.asarray(d, float) + eps
    return d / d.sum()


def coverage_divergences(d_pi, d_mu, eps=1e-9, cap=1e3) -> dict:
    """Laplace-smoothed divergences between d_pi and d_mu (both summing to 1)."""
    dp = _smooth(d_pi, eps)
    dm = _smooth(d_mu, eps)
    r_mp = np.clip(dm / dp, 0.0, cap)
    r_pm = np.clip(dp / dm, 0.0, cap)
    max_pi_over_mu = float(r_pm.max())
    return {
        'max_mu_over_pi': float(r_mp.max()),
        'max_pi_over_mu': max_pi_over_mu,
        'chi2_pi_mu': float(np.sum((dp - dm) ** 2 / dm)),
        'kl_pi_mu': float(np.sum(dp * np.log(dp / dm))),
        'kl_mu_pi': float(np.sum(dm * np.log(dm / dp))),
        'tv': float(0.5 * np.sum(np.abs(dp - dm))),
        'renyi_inf': float(np.log(max_pi_over_mu)),
    }


def coverage_ratio_grids(d_pi, d_mu, eps=1e-9, cap=1e3):
    """Smoothed + capped per-(s,a) ratio grids (mu/pi, pi/mu)."""
    dp = _smooth(d_pi, eps)
    dm = _smooth(d_mu, eps)
    return np.clip(dm / dp, 0.0, cap), np.clip(dp / dm, 0.0, cap)


def _absorbing_indices(env):
    return {env.state_to_idx(s) for s in env.get_all_absorbing_states()}


def _start_dist(env):
    d = np.zeros(env.n_states)
    d[env.state_to_idx(env.start)] = 1.0
    return d


def build_reference_policy(env, gamma, ref_budget=2000, lr=0.5,
                           tol=1e-5, patience=10) -> np.ndarray:
    """Train an alpha=0 vanilla-NPG student to saturation; return its softmax.

    Exact NPG has no randomness, so the result is deterministic for a fixed
    (env, gamma). Early-stops when the policy matrix stops changing for
    `patience` consecutive steps, capped at ref_budget steps.
    """
    policy = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    prev = None
    stable = 0
    for _ in range(ref_budget):
        Q_pi, _ = compute_student_qvalues(env, policy, gamma)
        exact_npg_update(policy, Q_pi, None, None, alpha=0.0, lr=lr)
        cur = policy_to_matrix(policy)
        if prev is not None and np.abs(cur - prev).max() < tol:
            stable += 1
            if stable >= patience:
                break
        else:
            stable = 0
        prev = cur
    return policy_to_matrix(policy)


def build_reference_occupancies(env, gamma, ref_budget=2000, lr=0.5) -> dict:
    """d^mu for both references ('analytic' argmax-optimal, 'learned' alpha=0)."""
    T = build_transition_table(env)
    absorbing = _absorbing_indices(env)
    start = _start_dist(env)
    pi_analytic = build_optimal_policy(env, env.goals, gamma)
    pi_learned = build_reference_policy(env, gamma, ref_budget, lr)
    return {
        'analytic': compute_occupancy(T, pi_analytic, start, gamma, absorbing),
        'learned': compute_occupancy(T, pi_learned, start, gamma, absorbing),
    }


def compute_coverage_metrics(env, student_policy, references, gamma,
                             eps=1e-9, cap=1e3, want_grids=False):
    """Per-eval-tick coverage scalars (and optional final-tick grids)."""
    T = build_transition_table(env)
    absorbing = _absorbing_indices(env)
    start = _start_dist(env)
    d_pi = compute_occupancy(
        T, policy_to_matrix(student_policy), start, gamma, absorbing
    )
    scalars, grids = {}, {}
    for name, d_mu in references.items():
        for k, v in coverage_divergences(d_pi, d_mu, eps, cap).items():
            scalars[f'cov_{name}_{k}'] = v
        if want_grids:
            r_mp, r_pm = coverage_ratio_grids(d_pi, d_mu, eps, cap)
            grids[name] = {
                'd_mu': d_mu, 'd_pi': d_pi,
                'ratio_mu_over_pi': r_mp, 'ratio_pi_over_mu': r_pm,
            }
    return scalars, grids
