# NPG Update Direction Diagnostics — Design

**Date:** 2026-05-25
**Branch:** `npg-update-direction-diagnostics`
**Status:** Draft

## Motivation

The default hypothesis sweep shows a "mid-capacity bump": student
performance is best at intermediate teacher signals (zeta ∈ {0.33, 0.67}
or cap ∈ {1, 2}), worse at both α=0 (vanilla NPG) and α=1 with the
best teacher. Existing per-step diagnostics (`adv_product_s0`,
`cos_q_a`, `mc_var_*`) tell us things about the policy state and the
return distribution but say nothing about the actual NPG update
direction the student takes each step.

For tabular softmax NPG, the policy update is

    π_{t+1} = π_t + γ · F_ρ(π_t)^† · ∇L_PAV-RL(π_t)|_{π_t},

where the Fisher matrix

    F_ρ(π) = E_{s ~ d_ρ^π} E_{a ~ π(·|s)} [ ∇ log π(a|s) ∇ log π(a|s)^T ]

is rank-deficient (the softmax tangent space lives in a per-state
codim-1 subspace), so the Moore–Penrose pseudo-inverse is the natural
choice. The empirical, sample-mode version of the per-step update is

    U = F̂^† ĝ,   F̂ = (1/n) Σᵢ ψᵢ ψᵢᵀ,
                  ĝ = (1/n) Σᵢ Aᵢ ψᵢ,

with ψᵢ = ∇ log π(aᵢ|sᵢ) and Aᵢ = (1-α) Gᵢ + α A^μ(sᵢ, aᵢ).

This spec adds, **for the canonical single cell only**:

1. An inline per-update-step diagnostic that computes U and a
   trajectory-bootstrap distribution of U, emitting six scalars per
   step into `history`.
2. A cosine-similarity figure cos(U_α, U_{α=0}) over training, with
   one curve per teacher value and an α=0 reference at 1.0.
3. A bootstrap-variance trace figure (`Σ_{s,a} Var_b(U_b)`).
4. A 4-panel figure showing `Var_b(U_b[s₀, a])` per action, where s₀ is
   the student's start state.

The intent is to explain the bump: if mid-capacity teachers give an
update direction that is either (a) more strongly rotated away from
vanilla NPG, or (b) less noisy under finite-sample MC, that is
directly visible here.

## Goals

- Add `update_direction_diagnostics(...)` in
  `tabular_prototype/training.py` — pure function, no policy mutation.
- Plumb the six new scalars (`cos_npg_dir`, `var_U_trace`,
  `var_U_s0_a0..a3`) through the sample-mode branch of `run_experiment`.
- Add `plot_npg_cosine` and `plot_npg_variance` to
  `tabular_prototype/visualization.py`.
- Add a focused runner `run_npg_diagnostics.py` that drives the
  canonical cell across both `zeta` and `capability` sweep modes,
  including one α=0 baseline run per mode for the variance overlay.
- Add tests pinning the mathematical claims the figures rest on.

## Non-Goals

- No new diagnostic for `exact` or `hybrid` training modes.
- No sweep-wide rollout over all (dist, H, B, α) cells. **Only the
  canonical cell** (dist=6, H=small, B=second-largest calibrated
  budget, α=1.0) — saved in
  `~/.claude/.../project_canonical_diagnostic_cell.md`.
- No changes to existing diagnostics (`cos_q_a`, `mc_var_*`,
  `adv_product_s0`, advantage_alignment, learning_curve).
- No new sweep modes; the focused runner is invoked directly.
- Not added to the hypothesis sweep's `cap_zeta` mode.

## Quantities Computed Per Update Step

At each update step, before the policy update is applied, with θ = θ_t
and the batch of trajectories already collected for that step:

1. **ψᵢ** for each transition (sᵢ, aᵢ). Sparse: only entries at row
   sᵢ are non-zero. Closed form for tabular softmax:
   `ψᵢ[sᵢ, a] = 1(a = aᵢ) - π(a|sᵢ)`.

2. **F̂** = (1/n) Σ ψᵢ ψᵢᵀ. Block-diagonal across states; per-state
   block has shape (A, A), rank ≤ A-1.

3. **ĝ_α** = (1/n) Σ Aᵢ ψᵢ with Aᵢ = (1-α) Gᵢ + α A^μ(sᵢ, aᵢ).
   Gᵢ is the Monte-Carlo return from transition i to end of trajectory
   using the same `estimate_returns` helper as `compute_pav_rl_gradient`.

4. **ĝ_0** = (1/n) Σ Gᵢ ψᵢ — vanilla-NPG variant on the same batch
   (effectively re-uses Gᵢ from step 3).

5. **U_α** = F̂^† ĝ_α, **U_0** = F̂^† ĝ_0. Pseudo-inverse applied
   per-state block (each is an A × A matrix); unvisited states get
   zero rows in U.

6. **Bootstrap.** Draw B trajectory-level index resamples
   (with replacement, each draw of size `len(trajectories)`). For each
   resample b ∈ 1..B, rebuild F̂_b, ĝ_{α,b}, compute
   U_b = F̂_b^† ĝ_{α,b}. Store the B values of U_b per (s, a).

7. **Emitted scalars** into `history` per step:
   - `cos_npg_dir` = cos(U_α, U_0)   (flatten both, dot / (‖·‖ ‖·‖)).
     `NaN` if either norm is zero.
   - `var_U_trace` = Σ_{s,a} Var_b(U_b[s, a]).
   - `var_U_s0_a` = Var_b(U_b[s₀, a])  for a ∈ {0, 1, 2, 3}.

## Architecture

### New helper — `tabular_prototype/training.py`

```python
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
    """
    Compute U = F̂^† ĝ_α and its trajectory-bootstrap variance.

    Returns:
        {
          'cos_npg_dir': float,         # cos(U_α, U_{α=0})
          'var_U_trace': float,         # Σ_{s,a} Var_b(U_b)
          'var_U_s0_a0': float, 'var_U_s0_a1': float,
          'var_U_s0_a2': float, 'var_U_s0_a3': float,
        }
    """
```

Implementation notes:

- Iterate trajectories once to collect per-transition tuples
  (s_idx, a_idx, G_i, A_mu_i). Convert G_i and A_mu_i via the
  existing `estimate_returns` and `get_teacher_advantage` helpers.
- Build per-state lists of (a_idx, G_i, A_mu_i). For each state s,
  compute the per-state block F̂_s from its observations, then U at
  state s as `pinv(F̂_s) @ ĝ_{α,s}` and `pinv(F̂_s) @ ĝ_{0,s}` using
  the closed-form ψ for tabular softmax. Unvisited states leave U_α
  and U_0 at zero.
- Bootstrap is at the trajectory level — resample trajectory indices,
  not transitions, preserving within-trajectory correlation.
- All RNG draws use the passed-in `rng`.

### Sample-mode wiring — `tabular_prototype/experiments.py`

Insert one call between collecting trajectories and computing the
gradient, in the existing `else:` branch (sample/hybrid path).

```python
if mode == "sample":
    diag = update_direction_diagnostics(
        policy, trajectories, Q_mu, V_mu, alpha, gamma,
        start_idx=start_idx, n_bootstrap=n_bootstrap, rng=rng,
    )
else:
    diag = _zeros_for_npg_diag()  # so history schema is stable
```

The six new keys are appended to the per-step `history` record
alongside the existing diagnostic scalars.

A new constructor parameter `n_bootstrap: int = 50` is added to
`run_experiment` and threaded through `run_learning_curve_experiment`
so the focused runner can override it.

### Visualization — `tabular_prototype/visualization.py`

Two new functions, modeled on `plot_advantage_alignment` and
`plot_mc_variance_curve`:

```python
def plot_npg_cosine(
    histories_by_teacher: Dict[Any, List[dict]],
    out_path: str,
    cell_info: dict,
) -> None
```

- One axes; x = update step; y = `cos_npg_dir`.
- One curve per teacher value, color cycled like existing plots.
- Dashed horizontal reference at y = 1.0 labeled "α=0 (reference)".
- Title shows numeric horizon (per `feedback_horizon_titles`) and the
  (dist, H, B, α) cell (per `feedback_figure_config_annotation`).
- Sub-title or in-figure text annotation:
  "U = F̂⁺ĝ; cos taken at same θ_t between U_{α=1} and U_{α=0}".

```python
def plot_npg_variance(
    histories_by_teacher: Dict[Any, List[dict]],
    baseline_history_alpha_zero: List[dict],
    out_dir: str,
    cell_info: dict,
) -> None
```

Writes two PNGs into `out_dir`:

1. **`npg_var_trace.png`** — single axes; y = `var_U_trace`. Curves
   per teacher value + a separate `α=0 (vanilla NPG)` baseline curve
   sourced from `baseline_history_alpha_zero`. Annotation:
   "Var across B={B} trajectory-level bootstrap resamples of the
   per-step n={n_traj} trajectories".

2. **`npg_var_s0.png`** — 2×2 panel grid, one subplot per action
   a ∈ {0, 1, 2, 3}. y = `var_U_s0_a{a}`. Curves per teacher value
   + α=0 baseline. Same annotation as the trace figure. Subplots
   share x-axis and use the action label from `GridEnv` (or "action
   {a}" if no human label is available).

Both figures honor `feedback_figure_config_annotation` (cell info in
title) and `feedback_alpha_zero_baseline` (α=0 overlay).

### Focused runner — `run_npg_diagnostics.py` (new top-level script)

Modeled on the focused runner introduced in commit `5e44405`.

```
usage: python run_npg_diagnostics.py
    [--n-seeds 30]
    [--n-bootstrap 50]
    [--output-dir results/figures/npg_diagnostics_<timestamp>]
    [--training-mode sample]   # locked; CLI flag for symmetry only
    [--grid-size 9]            # for tests/dev runs on smaller grids
    [--override-budget N]      # bypass calibration; for tests/dev only
```

Behavior:

1. Load calibrated budgets via the same helper
   `run_hypothesis_sweep._load_calibrated_budgets` uses for sample
   mode. Pick the canonical cell:
   `dist=6, h_type='small'`. Use the second-largest budget for that
   `(dist, h_type, n_goals)` from the calibrated list (after the
   sorted-ascending convention used elsewhere). Hard-fail with a
   clear message if calibration is missing.
2. For each `mode ∈ {'zeta', 'capability'}`:
   - Determine teacher value list (`ZETA_VALUES` minus 0.0 / `CAP_VALUES`
     minus -1, since 0.0 zeta and -1 cap are equivalent to "no teacher"
     which is α=0).
   - Run one **α=0 baseline** training run (any non-(-1) teacher chosen
     for compatibility with the rest of the cell — same convention as
     commit `5e44405`).
   - For each teacher value in the mode's list, run one α=1 training
     run with `n_seeds` seeds. Each run's history is collected.
3. Emit figures into `{output_dir}/{mode}/`:
   `npg_cosine.png`, `npg_var_trace.png`, `npg_var_s0.png`.

Per-seed history aggregation (mean + ±1 std band across seeds) matches
the existing diagnostic figures' convention.

## Data Flow

```
collect_trajectories ─┐
                      ├─► update_direction_diagnostics(θ_t, trajs, Q_μ, V_μ, α)
compute Q^π,V^π ──────┘    │
                           ├─► {cos_npg_dir, var_U_trace, var_U_s0_a*}
compute PAV-RL gradient    │
       │                   ▼
       │              history[t].update(...)
       ▼
update_policy(...)
```

No new rollouts; bootstrap is over indices into the already-collected
batch. Per-step cost: B × S × O(A³) for the per-state pinv, negligible
for S=81, A=4, B=50.

## Edge Cases

- **Zero-norm direction.** `cos_npg_dir` set to `NaN`. Plotters use
  `np.nanmean` / `np.nanstd` across seeds.
- **States never visited in any bootstrap draw.** Their per-state F̂_b
  block stays zero in every draw → U_b is zero there → variance is
  zero (correct).
- **Small trajectory batch (e.g., default 10).** Bootstrap with B=50 is
  slightly over-saturated but cheap. Configurable via `--n-bootstrap`.
- **α=0 baseline curves.**
  - Cosine: U_{α=0} vs U_0 = 1.0 by construction, drawn as a dashed
    reference line. No extra training run needed.
  - Variance: a real α=0 training run is required for the overlay.
    The focused runner runs exactly one α=0 run per sweep mode for
    this purpose.
- **Determinism.** Bootstrap RNG is the seed-derived `rng` already
  used by the training loop, so per-seed reproducibility holds.
- **Calibration missing.** The runner errors out with a precise
  instruction to regenerate `results/calibration_sample.json` rather
  than silently falling back.

## Testing Strategy

### `tests/test_update_direction_diagnostics.py`

1. **Output shape & types** — 3×3 grid, one step, six expected scalar
   keys present and finite (non-degenerate setup).
2. **Cosine identity at α=0** — `update_direction_diagnostics` with
   α=0.0 returns `cos_npg_dir == 1.0` exactly (float tolerance).
   **Locking test.**
3. **Variance non-negativity & trace consistency** — every
   `var_U_s0_a*` ≥ 0; `var_U_trace` ≥ Σ_a var_U_s0_a* (because trace
   sums over all states, not just s₀).
4. **Softmax tangent space** — for a fixed batch, per-state action
   sum of U_α is ≈ 0.
5. **Reproducibility under seed** — same seeded `rng` ⇒ same returned
   scalars bit-for-bit.
6. **Single-state corner** — trajectories that visit only one state
   produce finite scalars (or `NaN` for cos only if degenerate), no
   exceptions.

### Visualization smoke tests

7. `plot_npg_cosine` writes a non-empty PNG; the dashed reference at
   y=1 is drawn.
8. `plot_npg_variance` writes both `npg_var_trace.png` and
   `npg_var_s0.png`. The s₀ figure has 4 subplots. Both PNGs contain
   the cell-info annotation and the bootstrap-method annotation.

### Integration smoke test

9. `tests/test_run_npg_diagnostics.py`:
   `python run_npg_diagnostics.py --n-seeds 1 --n-bootstrap 5
   --grid-size 3 --override-budget 30 --output-dir <tmp>` produces a
   complete `<tmp>/{zeta,capability}/` directory with 3 PNGs each.
   `@pytest.mark.slow` if total runtime exceeds a few seconds.

## Files Added / Modified

**Added:**
- `tabular_prototype/training.py` — new function `update_direction_diagnostics` (~120 LOC).
- `tabular_prototype/visualization.py` — `plot_npg_cosine`, `plot_npg_variance` (~200 LOC).
- `run_npg_diagnostics.py` — new top-level focused runner (~200 LOC).
- `tests/test_update_direction_diagnostics.py` — unit tests for the new helper (~150 LOC).
- `tests/test_run_npg_diagnostics.py` — integration smoke test (~50 LOC).

**Modified:**
- `tabular_prototype/experiments.py` — call helper in sample-mode branch; thread `n_bootstrap` through `run_experiment` / `run_learning_curve_experiment` (~30 LOC delta).

**Unchanged:**
- `run_experiments.py`, `run_hypothesis_sweep.py`, all existing
  diagnostic modes. The focused runner is a standalone top-level
  script.

## Rollout

1. Land the helper + tests (TDD: tests 1–6 written first).
2. Wire into sample-mode training; verify existing tests still pass.
3. Land visualization functions + smoke tests.
4. Land the focused runner + integration smoke test.
5. Run end-to-end on the canonical cell to produce real figures on
   the dev machine and confirm the figures answer the bump question.
