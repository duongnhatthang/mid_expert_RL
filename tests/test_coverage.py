import numpy as np
import pytest

from tabular_prototype.environment import GridEnv
from tabular_prototype import coverage
from tabular_prototype.config import compute_gamma_from_horizon
from tabular_prototype.student import TabularSoftmaxPolicy


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


def _rand_dist(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.random(shape)
    return x / x.sum()


def test_divergences_zero_for_identical():
    d = _rand_dist((9, 4), seed=0)
    out = coverage.coverage_divergences(d, d)
    assert out['chi2_pi_mu'] == pytest.approx(0.0, abs=1e-9)
    assert out['kl_pi_mu'] == pytest.approx(0.0, abs=1e-9)
    assert out['kl_mu_pi'] == pytest.approx(0.0, abs=1e-9)
    assert out['tv'] == pytest.approx(0.0, abs=1e-9)
    assert out['max_pi_over_mu'] == pytest.approx(1.0, abs=1e-6)
    assert out['max_mu_over_pi'] == pytest.approx(1.0, abs=1e-6)
    assert out['renyi_inf'] == pytest.approx(0.0, abs=1e-6)


def test_divergences_finite_on_disjoint_support():
    n = (9, 4)
    d_pi = np.zeros(n); d_pi[0, 0] = 1.0
    d_mu = np.zeros(n); d_mu[1, 1] = 1.0   # disjoint support pre-smoothing
    out = coverage.coverage_divergences(d_pi, d_mu, eps=1e-6, cap=1e3)
    for v in out.values():
        assert np.isfinite(v)
    # Capped ratios never exceed cap.
    assert out['max_pi_over_mu'] <= 1e3 + 1e-6
    assert out['max_mu_over_pi'] <= 1e3 + 1e-6


def test_ratio_grids_shape_and_cap():
    d_pi = _rand_dist((9, 4), seed=1)
    d_mu = _rand_dist((9, 4), seed=2)
    r_mp, r_pm = coverage.coverage_ratio_grids(d_pi, d_mu, cap=50.0)
    assert r_mp.shape == (9, 4) and r_pm.shape == (9, 4)
    assert r_mp.max() <= 50.0 and r_pm.max() <= 50.0
    assert np.isfinite(r_mp).all() and np.isfinite(r_pm).all()


def test_build_reference_policy_is_valid_stochastic_and_deterministic():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    p1 = coverage.build_reference_policy(env, g, ref_budget=200)
    p2 = coverage.build_reference_policy(env, g, ref_budget=200)
    assert p1.shape == (env.n_states, env.n_actions)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-9)
    assert (p1 > 0).all()                      # softmax => full support
    np.testing.assert_allclose(p1, p2, atol=1e-12)  # reproducible


def test_build_reference_occupancies_keys_and_sums():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    refs = coverage.build_reference_occupancies(env, g, ref_budget=200)
    assert set(refs) == {'analytic', 'learned'}
    for d in refs.values():
        np.testing.assert_allclose(d.sum(), 1.0, atol=1e-9)


def test_compute_coverage_metrics_flat_keys_and_grids():
    env = _tiny_env()
    g = compute_gamma_from_horizon(env.horizon)
    refs = coverage.build_reference_occupancies(env, g, ref_budget=200)
    student = TabularSoftmaxPolicy(env.n_states, env.n_actions)  # uniform
    scalars, grids = coverage.compute_coverage_metrics(
        env, student, refs, g, want_grids=True
    )
    assert 'cov_analytic_max_pi_over_mu' in scalars
    assert 'cov_learned_chi2_pi_mu' in scalars
    for v in scalars.values():
        assert np.isfinite(v)
    assert set(grids) == {'analytic', 'learned'}
    assert grids['analytic']['d_pi'].shape == (env.n_states, env.n_actions)
    # want_grids=False -> empty grids dict.
    _, grids_off = coverage.compute_coverage_metrics(env, student, refs, g)
    assert grids_off == {}
