"""Smoke test for per-action advantage compass-rose + variance heatmap.

Builds minimal fake visitation data for one (distance, horizon, budget) and
invokes the plotting pipeline to render a visitation_grid figure with:
  - α-column visitations
  - per-action advantage (compass rose) column
  - action-variance column
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabular_prototype.environment import GridEnv, generate_equidistant_goals
from tabular_prototype.config import compute_gamma_from_horizon

from run_hypothesis_sweep import (
    GRID_SIZE, ALPHA_VALUES, DISTANCES,
    plot_visitation_grids,
    plot_visitation_grids_cap_zeta,
    _get_horizons,
    CAP_ZETA_CAPACITIES, CAP_ZETA_ZETAS,
)


def make_capability_results(dist=4, h_type='small', budget=50):
    horizons = _get_horizons()
    horizon = horizons[h_type]
    goals = generate_equidistant_goals(GRID_SIZE, 3, distance=dist)
    env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizon)

    results = []
    caps = [-1, 0, 1, 2, 3]
    rng = np.random.default_rng(0)
    for cap in caps:
        for alpha in ALPHA_VALUES:
            vis = rng.integers(0, 10, size=(env.n_states, env.n_actions)).astype(float)
            results.append({
                'distance': dist,
                'horizon_type': h_type,
                'sample_budget': budget,
                'teacher_capacity': cap,
                'alpha': alpha,
                'visitation_counts': vis,
            })
    return results, env


def make_cap_zeta_results(dist=4, h_type='small', budget=50):
    horizons = _get_horizons()
    horizon = horizons[h_type]
    goals = generate_equidistant_goals(GRID_SIZE, 3, distance=dist)
    env = GridEnv(grid_size=GRID_SIZE, goals=goals, horizon=horizon)

    results = []
    rng = np.random.default_rng(0)
    for cap in CAP_ZETA_CAPACITIES:
        for zeta in CAP_ZETA_ZETAS:
            for alpha in ALPHA_VALUES:
                vis = rng.integers(0, 10, size=(env.n_states, env.n_actions)).astype(float)
                results.append({
                    'distance': dist,
                    'horizon_type': h_type,
                    'sample_budget': budget,
                    'teacher_capacity': cap,
                    'zeta': zeta,
                    'alpha': alpha,
                    'visitation_counts': vis,
                })
    return results, env


def main():
    out_dir = 'results/smoke_adv_viz'
    os.makedirs(out_dir, exist_ok=True)

    print("[smoke] capability mode...")
    cap_results, env = make_capability_results(dist=4, h_type='small', budget=50)
    plot_visitation_grids(cap_results, 'capability', out_dir)

    print("[smoke] cap_zeta mode...")
    cz_results, env = make_cap_zeta_results(dist=4, h_type='small', budget=50)
    plot_visitation_grids_cap_zeta(cz_results, out_dir)

    print(f"[smoke] done — figures in {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
