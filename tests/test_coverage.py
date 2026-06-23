import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype import coverage


def _tiny_env():
    # 3x3 grid, single goal at a corner, short horizon.
    return GridEnv(grid_size=3, goals=[(0, 0)], horizon=10)


def test_transition_table_shape_and_values():
    env = _tiny_env()
    T = coverage.build_transition_table(env)
    assert T.shape == (env.n_states, env.n_actions)
    # From every state, moving 'up' (action 0) then mapping must match env.
    for s in range(env.n_states):
        state = env.idx_to_state(s)
        for a in range(env.n_actions):
            assert T[s, a] == env.state_to_idx(env._apply_action(state, a))


def test_policy_to_matrix_is_row_softmax():
    env = _tiny_env()
    from tabular_prototype.student import TabularSoftmaxPolicy
    pol = TabularSoftmaxPolicy(env.n_states, env.n_actions)
    pol.theta = np.arange(env.n_states * env.n_actions, dtype=float).reshape(
        env.n_states, env.n_actions
    )
    M = coverage.policy_to_matrix(pol)
    assert M.shape == (env.n_states, env.n_actions)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(M[0], pol.get_probs(0), atol=1e-12)


def test_occupancy_sums_to_one_and_self_loops_absorbing():
    env = _tiny_env()
    T = coverage.build_transition_table(env)
    n_s, n_a = env.n_states, env.n_actions
    probs = np.full((n_s, n_a), 1.0 / n_a)
    start = np.zeros(n_s)
    start[env.state_to_idx(env.start)] = 1.0
    absorbing = {env.state_to_idx(s) for s in env.get_all_absorbing_states()}
    d = coverage.compute_occupancy(T, probs, start, gamma=0.9,
                                   absorbing_states=absorbing)
    assert d.shape == (n_s, n_a)
    assert d.min() >= 0.0
    np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)


def test_occupancy_absorbing_only_chain_closed_form():
    # 2-state chain: state 0 -> state 1 (absorbing). Single action.
    # d(0) = (1-g)*1 ; remaining mass accumulates on absorbing state 1.
    T = np.array([[1], [1]])           # s0 -> s1 ; s1 self (absorbing anyway)
    probs = np.array([[1.0], [1.0]])
    start = np.array([1.0, 0.0])
    g = 0.8
    d = coverage.compute_occupancy(T, probs, start, gamma=g,
                                   absorbing_states={1})
    # d is (2,1); state occupancy = d[:,0].
    np.testing.assert_allclose(d[0, 0], 1.0 - g, atol=1e-9)
    np.testing.assert_allclose(d[1, 0], g, atol=1e-9)
    np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)
