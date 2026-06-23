# tests/test_coverage_experiment.py
import numpy as np
from tabular_prototype.experiments import run_experiment


def _kwargs(**over):
    base = dict(grid_size=5, goals=[(0, 2), (4, 2), (2, 0)],
                teacher_capacity=3, horizon=12, sample_budget=8,
                alpha=1.0, lr=0.5, seed=0, eval_interval=2,
                eval_n_episodes=5, mode='exact')
    base.update(over)
    return base


def test_track_coverage_populates_history_and_grids():
    r = run_experiment(track_coverage=True, **_kwargs())
    last = r['history'][-1]
    for key in ('cov_analytic_max_pi_over_mu', 'cov_analytic_chi2_pi_mu',
                'cov_learned_kl_pi_mu', 'cov_learned_tv',
                'cov_analytic_renyi_inf'):
        assert key in last and np.isfinite(last[key])
    grids = r['coverage_grids']
    assert set(grids) == {'analytic', 'learned'}
    n_sa = (5 * 5, 4)
    assert grids['analytic']['d_pi'].shape == n_sa
    np.testing.assert_allclose(grids['learned']['d_mu'].sum(), 1.0, atol=1e-9)


def test_track_coverage_off_is_unchanged():
    r = run_experiment(track_coverage=False, **_kwargs())
    assert r['coverage_grids'] is None
    assert not any(k.startswith('cov_') for k in r['history'][-1])


def test_track_coverage_sample_mode():
    r = run_experiment(track_coverage=True,
                       **_kwargs(mode='sample', sample_budget=60))
    assert any(k.startswith('cov_') for k in r['history'][-1])
    assert r['coverage_grids'] is not None
