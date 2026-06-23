"""Coverage / distribution-mismatch diagnostics.

Exact discounted state-action occupancy and divergences between the student
policy's occupancy d^pi and fixed reference occupancies d^mu.
"""

import numpy as np


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
