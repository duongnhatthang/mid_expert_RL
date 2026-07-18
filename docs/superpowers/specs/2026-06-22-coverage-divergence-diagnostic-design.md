# Coverage / Distribution-Mismatch Diagnostic — Design

**Date:** 2026-06-22
**Branch:** `coverage-divergence-diagnostics`

## Goal

Add a diagnostic that measures the mismatch between the student policy's
state-action occupancy `d^π(s,a)` and a fixed reference policy's occupancy
`d^μ(s,a)`, tracked over the course of training. Beyond the headline
concentrability coefficients

```
max_{s,a} d^μ(s,a) / d^π(s,a)        and        max_{s,a} d^π(s,a) / d^μ(s,a)
```

we cache the full per-`(s,a)` densities and ratios (not just the maxima) and
report a panel of divergences (chi-square, KL both directions, total variation,
Rényi-∞). All quantities must be finite and plottable, with principled handling
of zero/near-zero denominators.

`μ` is a **fixed reference policy** (does not change during training); `π` is the
student being trained. We track how `d^π` drifts toward/away from `d^μ` as
training proceeds.

## Two reference policies

We compute the diagnostic against **two** reference policies, each built once per
seed (because the reference depends on the goal layout) and then frozen:

| ref id     | source                                                              | support              | interpretation |
|------------|---------------------------------------------------------------------|----------------------|----------------|
| `analytic` | `build_optimal_policy(env, env.goals, γ)` — exact argmax-optimal     | deterministic, sparse | the true optimum; Laplace smoothing + cap is load-bearing here |
| `learned`  | frozen, converged α=0 vanilla-NPG student (no teacher signal)        | softmax, full support | what plain RL converges to without a teacher; smoothing is a safety net |

The `learned` reference is trained with **exact NPG steps** to saturation (early-stop
when the policy matrix stops changing), capped at a large fixed step budget
(default 2000). The cap is a step count decoupled from the student's training
budget, because the student's budget may be measured in observations
(`sample`/`hybrid` modes) which is not comparable to NPG update steps.

Rationale: the repo has no PPO / GAE / separate baseline-policy implementation —
it is purely tabular softmax NPG plus a value-iteration teacher. The faithful
tabular analog of "a traditional RL-trained policy with no teacher signal" is a
converged α=0 vanilla-NPG student (`θ += lr·Q^π`, no teacher term). In a tabular
MDP this converges to the same softmax optimum PPO would reach. Keeping the
`analytic` argmax-optimal reference as well gives complementary information
(exact optimum vs. learned softmax).

## Exact occupancy (no sampling)

This is a tabular MDP with a deterministic transition table `T[s,a] → s'`
(see `tabular_prototype/teacher.py::_build_transition_model`). For any stochastic
policy `π(a|s)` (`policy_probs`, shape `(n_states, n_actions)`) and start
distribution `ρ₀` (point mass at the env start / center — **the same `ρ₀` for
both μ and π** so the comparison is fair), the discounted state-action occupancy
is computed exactly by a linear solve:

```
P_π[s, s'] = Σ_a π(a|s) · 1[T[s,a] = s']          # induced Markov chain
d(s)       = (1 - γ) · ρ₀ᵀ (I - γ P_π)⁻¹           # discounted state occupancy
d(s, a)    = d(s) · π(a|s)                          # state-action occupancy
```

`Σ_{s,a} d(s,a) = 1`, so `d` is a proper distribution and all divergences below
are well-defined. `(I - γ P_π)` is invertible for `γ = 1 - 1/H < 1`. Grid is
small (e.g. 81 states for 9×9), so a direct dense solve is fine. Absorbing states
(goals/traps) self-loop in `T` and accumulate occupancy mass, consistent with the
environment. No sampling noise; fully reproducible.

The same exact solve is used for `d^μ` (each reference policy) and for `d^π` at
every eval tick (current student softmax `policy_probs`).

## Divergences

`coverage_divergences(d_pi, d_mu, eps, cap)`:

1. **Laplace smoothing:** `d̃ = (d + eps) / Σ(d + eps)` for both `d_pi` and
   `d_mu` (a uniform pseudocount over all `(s,a)`), guaranteeing strictly
   positive denominators.
2. **Ratios:** `r_mu_over_pi = d̃_mu / d̃_pi`, `r_pi_over_mu = d̃_pi / d̃_mu`,
   each clipped to `[0, cap]` for plottability.
3. **Scalars returned:**
   - `max_mu_over_pi = clip(max r_mu_over_pi, cap)`
   - `max_pi_over_mu = clip(max r_pi_over_mu, cap)`  ← concentrability coefficient C
   - `chi2_pi_mu   = Σ (d̃_pi − d̃_mu)² / d̃_mu`
   - `kl_pi_mu     = Σ d̃_pi · log(d̃_pi / d̃_mu)`
   - `kl_mu_pi     = Σ d̃_mu · log(d̃_mu / d̃_pi)`
   - `tv           = 0.5 · Σ |d̃_pi − d̃_mu|`
   - `renyi_inf    = log(max_pi_over_mu)`   ← log of the concentrability coefficient

Defaults: `eps = 1e-9` (relative to a distribution summing to 1; tunable),
`cap = 1e3`.

## Full-array caching

In addition to the scalars above (cached per eval tick), the **full** grids are
cached at the **final** eval tick only (to bound history size):

```
coverage_grids[ref] = {
    'd_mu':             (n_states, n_actions),
    'd_pi':             (n_states, n_actions),
    'ratio_mu_over_pi': (n_states, n_actions),   # smoothed + capped
    'ratio_pi_over_mu': (n_states, n_actions),   # smoothed + capped
}
```

## Module layout

### `tabular_prototype/coverage.py` (new)

- `compute_occupancy(transition, policy_probs, start_dist, gamma, absorbing_states) -> np.ndarray`
  — exact discounted state-action occupancy, shape `(n_states, n_actions)`.
- `build_reference_policy(env, gamma, ref_budget) -> np.ndarray`
  — train an α=0 vanilla-NPG student to saturation (reuse the existing saturation
  auto-detection, capped at `ref_budget`), return its frozen softmax
  `policy_probs`. This is the `learned` reference. The `analytic` reference comes
  from the existing `teacher.build_optimal_policy(env, env.goals, gamma)`.
- `coverage_divergences(d_pi, d_mu, eps, cap) -> dict` — scalars listed above.
- A small helper to assemble both references' `d^μ` once and to compute the
  per-tick scalar dict for both refs, returning flattened keys
  `cov_<ref>_<metric>` (e.g. `cov_analytic_max_pi_over_mu`,
  `cov_learned_chi2_pi_mu`).

### `tabular_prototype/experiments.py`

- New `track_coverage: bool = False` parameter on `run_experiment` (and threaded
  through the runners that need it). Default `False` → **zero cost** on normal
  sweeps.
- When `True`:
  - Build both reference occupancies `d^μ_analytic`, `d^μ_learned` **once** at the
    start of the run (the `learned` ref requires one converged α=0 sub-run on the
    same env).
  - At each eval tick: compute `d^π` from the current student `policy_probs`, call
    `coverage_divergences` for each ref, and merge the flattened `cov_<ref>_<metric>`
    scalars into the per-step diagnostics dict (flows into `history` exactly like
    the existing `pg_bias`, `var_*` fields).
  - At the final eval tick: also stash `coverage_grids[ref]` into the result dict.

### `tabular_prototype/visualization.py`

For **each** reference (`analytic`, `learned`):

- **Curves:** reuse `_generic_metric_figure` (mean + SEM bands, cell-config
  annotation, α=0 baseline overlay, one line per teacher value). One figure per
  scalar metric: `cov_<ref>_max_pi_over_mu`, `cov_<ref>_max_mu_over_pi`,
  `cov_<ref>_chi2_pi_mu`, `cov_<ref>_kl_pi_mu`, `cov_<ref>_kl_mu_pi`,
  `cov_<ref>_tv`, `cov_<ref>_renyi_inf`.
- **Heatmaps (final tick):** state-marginal `d^μ(s)` and `d^π(s)` (sum over
  actions), plus the two state-marginal ratio grids, reshaped to the N×N grid and
  log-color-scaled. Full `(s,a)` arrays remain cached for later recomputation;
  the visuals use state marginals to keep the panel count manageable.

Figure titles and filenames are tagged with the reference id (`analytic` /
`learned`) and carry the standard `(dist, H, B, α)` cell annotation.

### `run_npg_diagnostics.py`

- Add a `--track-coverage` flag. When set, pass `track_coverage=True` into the
  per-teacher α=1 runs and the α=0 baseline runs at the canonical diagnostic cell
  (`dist=6`, `H=small`, `B=budgets[-2]`, α=1, plus the α=0 baseline), then call the
  new coverage plot functions. Single diagnostics entry point; reuses the existing
  per-teacher history collection.

## Testing

- `compute_occupancy`: on a tiny hand-checkable MDP, verify (a) `Σ d = 1`,
  (b) closed-form occupancy for a known deterministic policy / chain matches the
  linear solve, (c) higher γ pushes mass further from the start.
- `coverage_divergences`: identical distributions → all divergences ≈ 0, ratios
  ≈ 1; disjoint-support (pre-smoothing) inputs → finite, capped outputs (no inf /
  nan); KL/chi-square asymmetry sanity checks.
- `build_reference_policy`: converged α=0 policy is a valid stochastic matrix
  (rows sum to 1, full support) and is reproducible for a fixed seed.
- Integration: `run_experiment(..., track_coverage=True)` on a small grid
  populates `cov_*` fields at every eval tick and `coverage_grids` at the final
  tick, for both references, with no inf/nan.
- Regression: `run_experiment(..., track_coverage=False)` (default) is unchanged.

## Known limitations

- **Analytic-reference `π/μ` ratio saturation.** Because the `analytic` reference
  is deterministic (one-hot), its off-path occupancy smooths to ~`eps`, so any
  student exploration drives `max d^π/d^μ` (and `renyi_inf = log` of it) past the
  `cap`, leaving those two curves pinned at `log(cap)` with little training signal.
  This is the deterministic-support pathology the `learned` reference was added to
  avoid — the `learned`-reference panel gives a clean concentrability signal, and
  for the `analytic` reference the `tv`, `kl_mu_pi`, and `chi2` metrics remain
  informative. Left as-is by decision (2026-06-22); a future refinement could use a
  larger per-reference `eps` (uniform pseudocount `1/(n_states·n_actions)`) for the
  analytic reference to turn its ratio into a regularized concentrability.

## Non-goals

- No PPO / GAE implementation (the `learned` ref uses the project's native α=0
  NPG learner).
- No per-action heatmaps in the figures (full per-action arrays are cached, but
  visuals use state marginals).
- No changes to the default sweep behavior — coverage is opt-in via
  `track_coverage` / `--track-coverage`.
