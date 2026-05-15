# Learning Curves by Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auto-generated set of learning-curve figures to `run_hypothesis_sweep.py` that overlay per-step `V^π(s₀)` trajectories of every teacher baseline (ζ values for zeta sweep, capability values for capability sweep) within each `(distance, alpha)` slice, sourced from existing pickled sweep results.

**Architecture:** New function `plot_learning_curves(all_results, mode, figures_dir)` lives in `run_hypothesis_sweep.py` alongside `plot_visitation_grids`. It groups pickled per-seed `history` lists by `(distance, alpha, horizon_type, sample_budget, teacher_value)`, averages `exact_V_start` across seeds, and writes one PNG per `(distance, alpha)` to `<figures_dir>/learning_curves/`. It is wired into the post-run / `--skip-run` block immediately after `plot_visitation_grids`. `cap_zeta` mode is an early-return no-op.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest. Reuses existing `_sort_teacher_vals` and `_teacher_label` helpers in `run_hypothesis_sweep.py` for consistent ordering and legend formatting.

---

## File Structure

**Modified:**
- `run_hypothesis_sweep.py` — add `plot_learning_curves` function (around line 1466, just before `plot_visitation_grids`), and one call site in `main()` (around line 2293).

**Created:**
- `tests/test_run_hypothesis_sweep_plots.py` — new pytest module with synthetic `all_results` fixtures and three tests (zeta, capability, cap_zeta no-op).

---

## Task 1: Set up test scaffolding and first failing test (zeta sweep)

**Files:**
- Create: `tests/test_run_hypothesis_sweep_plots.py`
- Reference: `run_hypothesis_sweep.py` (for the function to be added in Task 2)

- [ ] **Step 1.1: Create the test file with a synthetic-results helper and a zeta test**

```python
# tests/test_run_hypothesis_sweep_plots.py
"""Tests for plot_learning_curves in run_hypothesis_sweep.

Builds a small synthetic all_results list (the same shape produced by the
sweep pipeline and pickled to <mode>_sweep_results.pkl) and asserts the
expected PNGs are written to <figures_dir>/learning_curves/.

No image-diff testing — only file existence and non-trivial size.
"""

from pathlib import Path

import pytest

import run_hypothesis_sweep as sweep


def _make_history(n_points: int, base_v: float):
    """Synthetic history list of length n_points, exact_V_start increasing
    linearly from base_v toward base_v + 0.5.
    """
    return [
        {
            'steps': 5 * (i + 1),
            'mean_reward': 0.0,
            'goal_rate': 0.0,
            'exact_V_start': base_v + 0.5 * (i / max(1, n_points - 1)),
            'exact_V_start_undiscounted': 0.0,
            'unique_sa': 0,
            'state_entropy': 0.0,
        }
        for i in range(n_points)
    ]


def _zeta_results():
    """2 distances × 2 alphas × 2 horizons × 2 budgets × 2 zetas × 2 seeds = 64 entries."""
    out = []
    for dist in [4, 6]:
        for alpha in [0.0, 0.5]:
            for h_type, h_val in [('small', 8), ('large', 16)]:
                for budget in [10, 30]:
                    for zeta in [0.0, 1.0]:
                        for seed in [0, 1]:
                            out.append({
                                'distance': dist,
                                'alpha': alpha,
                                'horizon_type': h_type,
                                'horizon': h_val,
                                'sample_budget': budget,
                                'zeta': zeta,
                                'seed': seed,
                                'history': _make_history(4, base_v=0.1 * zeta + 0.01 * seed),
                            })
    return out


def test_plot_learning_curves_zeta(tmp_path):
    figures_dir = tmp_path
    sweep.plot_learning_curves(_zeta_results(), mode='zeta',
                               figures_dir=str(figures_dir))

    out_dir = figures_dir / 'learning_curves'
    assert out_dir.is_dir(), "learning_curves subdir not created"

    # 2 distances × 2 alphas = 4 PNGs
    expected = {
        f'learning_curve_dist{d}_alpha{a:.2f}.png'
        for d in [4, 6] for a in [0.0, 0.5]
    }
    actual = {p.name for p in out_dir.glob('*.png')}
    assert actual == expected, f"PNG set mismatch: {actual} vs {expected}"

    for p in out_dir.glob('*.png'):
        assert p.stat().st_size > 1000, f"{p.name} suspiciously small ({p.stat().st_size} B)"
```

- [ ] **Step 1.2: Run the test and verify it fails because the function does not exist yet**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_zeta -v`
Expected: FAIL with `AttributeError: module 'run_hypothesis_sweep' has no attribute 'plot_learning_curves'`

- [ ] **Step 1.3: Commit the failing test**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: add failing test for plot_learning_curves (zeta mode)"
```

---

## Task 2: Implement plot_learning_curves (minimal, supports zeta)

**Files:**
- Modify: `run_hypothesis_sweep.py` — add `plot_learning_curves` function just above the existing `plot_visitation_grids` (currently at line 1468).

- [ ] **Step 2.1: Add the function**

Insert the following block immediately before the `def plot_visitation_grids(...)` line (search for `def plot_visitation_grids` to locate the insertion point):

```python
def plot_learning_curves(all_results: list, mode: str, figures_dir: str):
    """Per-(distance, alpha) learning curves overlaid by teacher baseline.

    Reads per-seed `history` lists from `all_results` (the in-memory form of
    `<mode>_sweep_results.pkl`), groups by (distance, alpha, horizon_type,
    sample_budget, teacher_value), averages `exact_V_start` across seeds,
    and writes one PNG per (distance, alpha) to
    `<figures_dir>/learning_curves/`.

    Layout: 2 rows (horizon_type small/large) × N cols (budgets ascending).
    Lines = teacher values. Mean ± 1σ band across seeds.

    `cap_zeta` mode is a no-op (16-line legend reads poorly on a learning
    curve; cap_zeta has its own dedicated visualisations).
    """
    if mode == 'cap_zeta':
        print("plot_learning_curves: cap_zeta mode — skipping")
        return

    from collections import defaultdict
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    tcol = _teacher_col(mode)
    out_dir = os.path.join(figures_dir, 'learning_curves')
    os.makedirs(out_dir, exist_ok=True)

    # Group results: (dist, alpha, h_type, budget, teacher_val) -> list of history lists
    groups = defaultdict(list)
    for r in all_results:
        if 'history' not in r or not r['history']:
            continue
        # Defensive: skip entries missing exact_V_start (legacy pkls)
        if 'exact_V_start' not in r['history'][0]:
            continue
        key = (r['distance'], r['alpha'], r['horizon_type'],
               r['sample_budget'], r[tcol])
        groups[key].append(r['history'])

    if not groups:
        print("plot_learning_curves: no usable history entries — skipping")
        return

    # Unique (distance, alpha) pairs → one figure each
    da_pairs = sorted({(k[0], k[1]) for k in groups})
    h_types = ['small', 'large']

    for dist, alpha in da_pairs:
        # Budget set per horizon type for this (dist, alpha)
        budgets_by_h = {
            h: sorted({k[3] for k in groups
                       if k[0] == dist and k[1] == alpha and k[2] == h})
            for h in h_types
        }
        n_cols = max(len(budgets_by_h[h]) for h in h_types)
        if n_cols == 0:
            continue

        fig, axes = plt.subplots(
            len(h_types), n_cols,
            figsize=(4 * n_cols, 3.2 * len(h_types)),
            squeeze=False,
        )

        # Teacher values present anywhere in this (dist, alpha) slice
        teacher_vals_set = {k[4] for k in groups
                            if k[0] == dist and k[1] == alpha}
        # Use _sort_teacher_vals via a faux pandas Series-compatible iterable
        import pandas as pd
        sorted_teachers = _sort_teacher_vals(
            pd.Series(list(teacher_vals_set)), mode
        )

        for row_idx, h_type in enumerate(h_types):
            budgets = budgets_by_h[h_type]
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                if col_idx >= len(budgets):
                    ax.set_visible(False)
                    continue
                budget = budgets[col_idx]

                for tv in sorted_teachers:
                    histories = groups.get(
                        (dist, alpha, h_type, budget, tv), []
                    )
                    if not histories:
                        continue
                    lengths = {len(h) for h in histories}
                    if len(lengths) > 1:
                        raise ValueError(
                            f"Inconsistent history lengths {lengths} for "
                            f"(dist={dist}, alpha={alpha}, h={h_type}, "
                            f"budget={budget}, teacher={tv}) — pkl may "
                            f"mix runs."
                        )
                    steps = [h['steps'] for h in histories[0]]
                    values = np.stack([
                        [h['exact_V_start'] for h in seed_hist]
                        for seed_hist in histories
                    ], axis=0)
                    mean = values.mean(axis=0)
                    std = values.std(axis=0)
                    label = _teacher_label(mode, tv)
                    ax.plot(steps, mean, label=label, linewidth=1.5)
                    ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

                ax.set_title(f'H={h_type}, B={budget}', fontsize=9)
                ax.grid(True, alpha=0.3)
                if col_idx == 0:
                    ax.set_ylabel(r'$V^\pi(s_0)$', fontsize=9)
                if row_idx == len(h_types) - 1:
                    ax.set_xlabel('update step', fontsize=9)
                # Legend on rightmost visible cell of each row
                if col_idx == len(budgets) - 1:
                    ax.legend(fontsize=7, loc='best')

        fig.suptitle(
            f'Learning curve ({mode} sweep) — '
            f'dist={dist}, ' + rf'$\alpha={alpha}$',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(
            out_dir, f'learning_curve_dist{dist}_alpha{alpha:.2f}.png'
        )
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'Saved {out_path}')
```

- [ ] **Step 2.2: Run the zeta test and verify it passes**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_zeta -v`
Expected: PASS

- [ ] **Step 2.3: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: add plot_learning_curves (zeta mode)"
```

---

## Task 3: Add capability-mode test and confirm it passes (no code change expected)

**Files:**
- Modify: `tests/test_run_hypothesis_sweep_plots.py`

- [ ] **Step 3.1: Add the capability test**

Append to `tests/test_run_hypothesis_sweep_plots.py`:

```python
def _capability_results():
    """Capability sweep: 2 distances × 2 alphas × 2 horizons × 2 budgets ×
    capability values × 2 seeds.

    At alpha=0, includes c=-1 (no teacher) and c=0 (uniform). At alpha>0,
    only c in {1, 2, 3} — matches sweep behavior where redundant teacher
    variations are skipped at alpha=0.
    """
    out = []
    for dist in [4, 6]:
        for alpha in [0.0, 0.5]:
            caps = [-1, 0, 1, 2, 3] if alpha == 0.0 else [1, 2, 3]
            for h_type, h_val in [('small', 8), ('large', 16)]:
                for budget in [10, 30]:
                    for cap in caps:
                        for seed in [0, 1]:
                            out.append({
                                'distance': dist,
                                'alpha': alpha,
                                'horizon_type': h_type,
                                'horizon': h_val,
                                'sample_budget': budget,
                                'teacher_capacity': cap,
                                'seed': seed,
                                'history': _make_history(
                                    4, base_v=0.05 * (cap + 1) + 0.01 * seed
                                ),
                            })
    return out


def test_plot_learning_curves_capability(tmp_path):
    figures_dir = tmp_path
    sweep.plot_learning_curves(_capability_results(), mode='capability',
                               figures_dir=str(figures_dir))

    out_dir = figures_dir / 'learning_curves'
    expected = {
        f'learning_curve_dist{d}_alpha{a:.2f}.png'
        for d in [4, 6] for a in [0.0, 0.5]
    }
    actual = {p.name for p in out_dir.glob('*.png')}
    assert actual == expected, f"PNG set mismatch: {actual} vs {expected}"

    for p in out_dir.glob('*.png'):
        assert p.stat().st_size > 1000
```

- [ ] **Step 3.2: Run the capability test and verify it passes**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_capability -v`
Expected: PASS

- [ ] **Step 3.3: Commit**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: cover capability mode in plot_learning_curves"
```

---

## Task 4: Add cap_zeta no-op test

**Files:**
- Modify: `tests/test_run_hypothesis_sweep_plots.py`

- [ ] **Step 4.1: Add the cap_zeta test**

Append to `tests/test_run_hypothesis_sweep_plots.py`:

```python
def test_plot_learning_curves_cap_zeta_noop(tmp_path):
    """cap_zeta mode is intentionally unsupported — function early-returns
    and writes nothing.
    """
    figures_dir = tmp_path
    # Pass an arbitrary non-empty list; should not be inspected.
    sweep.plot_learning_curves(
        [{'distance': 4, 'alpha': 0.0, 'horizon_type': 'small',
          'sample_budget': 10, 'cap_zeta': 'cap=1_z=0.5', 'seed': 0,
          'history': []}],
        mode='cap_zeta',
        figures_dir=str(figures_dir),
    )
    out_dir = figures_dir / 'learning_curves'
    # Either dir doesn't exist or it exists but is empty.
    if out_dir.exists():
        assert list(out_dir.glob('*.png')) == [], \
            "cap_zeta mode must not write learning-curve PNGs"
```

- [ ] **Step 4.2: Run the test and verify it passes**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_cap_zeta_noop -v`
Expected: PASS (function early-returns before `os.makedirs`).

- [ ] **Step 4.3: Run the full new test module to confirm all three tests pass**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py -v`
Expected: 3 passed.

- [ ] **Step 4.4: Commit**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: cover cap_zeta no-op for plot_learning_curves"
```

---

## Task 5: Wire plot_learning_curves into the sweep pipeline

**Files:**
- Modify: `run_hypothesis_sweep.py:2293` — call `plot_learning_curves` immediately after `plot_visitation_grids`.

- [ ] **Step 5.1: Add the call site**

Locate the block that currently reads (around line 2291–2298):

```python
    if all_results is not None:
        print("\nGenerating visitation grids...")
        plot_visitation_grids(all_results, args.mode, figures_dir)

        print("\nGenerating diagnostic plots...")
        _plot_sweep_diagnostics(all_results, args.mode, figures_dir)
    else:
        print("\nSkipping visitation grids (no pickle cache with visitation data)")
```

Replace with:

```python
    if all_results is not None:
        print("\nGenerating visitation grids...")
        plot_visitation_grids(all_results, args.mode, figures_dir)

        print("\nGenerating learning curves...")
        plot_learning_curves(all_results, args.mode, figures_dir)

        print("\nGenerating diagnostic plots...")
        _plot_sweep_diagnostics(all_results, args.mode, figures_dir)
    else:
        print("\nSkipping visitation grids (no pickle cache with visitation data)")
```

- [ ] **Step 5.2: Confirm full test suite still passes**

Run: `PYTHONPATH=. pytest tests/ -v -m 'not slow'`
Expected: all collected tests pass (including the three new ones). Slow integration tests stay skipped.

- [ ] **Step 5.3: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: emit learning curves alongside visitation grids in sweep"
```

---

## Task 6: Smoke test against a real cached pkl

**Files:**
- No code changes. Manual verification.

- [ ] **Step 6.1: Pick an existing zeta_exact output directory**

Run: `ls results/full_suite_20260419_105552/zeta_exact/`
Expected output includes `zeta_sweep_results.pkl` and `figures/`.

If `zeta_sweep_results.pkl` is missing, fall back to any other `results/*/zeta_*` directory that has the pkl. If none exists, skip this task and rely on the unit tests — note the skip in the commit message at Task 7.

- [ ] **Step 6.2: Replot from cache**

Run:

```bash
PYTHONPATH=. python run_hypothesis_sweep.py \
    --mode zeta --skip-run \
    --output-dir results/full_suite_20260419_105552/zeta_exact
```

Expected stdout contains lines like `Saved .../learning_curves/learning_curve_dist4_alpha0.00.png` (12 lines total: 3 distances × 4 alphas).
Do NOT pass `--all-plots` (per CLAUDE.md, distance_effect_alpha and heatmap_dist must stay off).

- [ ] **Step 6.3: Inspect the new figures directory**

Run: `ls results/full_suite_20260419_105552/zeta_exact/figures/learning_curves/`
Expected: 12 PNGs named `learning_curve_dist{4,6,8}_alpha{0.00,0.33,0.67,1.00}.png`. Each file should be at least a few KB.

- [ ] **Step 6.4: Spot-check one figure visually**

Open one PNG (e.g. `learning_curve_dist6_alpha0.67.png`). Verify:
- 2 rows of subplots (horizon `small` on top, `large` on bottom).
- Multiple budget columns per row.
- Each subplot shows up to 4 lines with a shaded ±σ band, legend in math-formatted ζ labels, monotone-ish increasing curves.

If anything looks broken (empty axes, wrong axis labels, lines missing), debug before proceeding to Task 7.

- [ ] **Step 6.5: Repeat for capability mode if a capability pkl is available**

Run: `ls results/full_suite_20260419_105552/capability_exact/capability_sweep_results.pkl` to confirm presence, then:

```bash
PYTHONPATH=. python run_hypothesis_sweep.py \
    --mode capability --skip-run \
    --output-dir results/full_suite_20260419_105552/capability_exact
```

Inspect `results/full_suite_20260419_105552/capability_exact/figures/learning_curves/`. Expected: 12 PNGs. At α=0.00 figures, the legend should include `no teacher` and `uniform`; at α>0 figures, only `$c=1$, $c=2$, $c=3$`.

---

## Task 7: Final review and PR-ready commit

**Files:**
- No new code changes. Recap and finalise.

- [ ] **Step 7.1: Review the diff**

Run: `git log --oneline origin/main..HEAD`
Expected commits (in order):
1. `Add design spec: learning curves by baseline`
2. `test: add failing test for plot_learning_curves (zeta mode)`
3. `feat: add plot_learning_curves (zeta mode)`
4. `test: cover capability mode in plot_learning_curves`
5. `test: cover cap_zeta no-op for plot_learning_curves`
6. `feat: emit learning curves alongside visitation grids in sweep`

Run: `git diff origin/main..HEAD --stat`
Expected: `run_hypothesis_sweep.py` and `tests/test_run_hypothesis_sweep_plots.py` and `docs/superpowers/specs/2026-04-27-learning-curves-by-baseline-design.md` modified/created. No other files.

- [ ] **Step 7.2: Run the full non-slow test suite once more**

Run: `PYTHONPATH=. pytest tests/ -v -m 'not slow'`
Expected: all green.

- [ ] **Step 7.3: (Optional) Push branch and open PR**

Only if the user explicitly asks. Leave the branch local otherwise.

```bash
git push -u origin learning-curves-by-baseline
gh pr create --title "Learning-curve plots per (distance, alpha) for zeta and capability sweeps" --body "$(cat <<'EOF'
## Summary
- Adds `plot_learning_curves(all_results, mode, figures_dir)` to `run_hypothesis_sweep.py`.
- Auto-generates 12 PNGs per sweep (3 distances × 4 alphas) from the existing pickled `history` lists; no re-runs.
- Wired into the post-run / `--skip-run` block alongside `plot_visitation_grids`.
- Skipped for `cap_zeta` mode (legend would have 16 entries).

## Test plan
- [x] `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py -v` (3 new tests)
- [x] Smoke test: `--mode zeta --skip-run` against an existing `zeta_exact` output dir
- [x] Smoke test: `--mode capability --skip-run` against an existing `capability_exact` output dir

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** every spec section maps to a task — Layout/aggregation → Task 2; capability mode legend trimming → Task 3; cap_zeta no-op → Task 4; pipeline wiring → Task 5; acceptance criteria (12 PNGs each, `--skip-run` regenerates, cap_zeta unaffected) → Task 6.
- **Placeholder scan:** all code blocks contain real implementations; commands have explicit expected output; file paths use real LOC anchors in the live `run_hypothesis_sweep.py` (verified at write-time: `plot_visitation_grids` at 1468, call site near 2293, `_sort_teacher_vals` at 606, `_teacher_label` at 587).
- **Type consistency:** function signature `plot_learning_curves(all_results: list, mode: str, figures_dir: str)` matches `plot_visitation_grids` exactly. Helpers `_sort_teacher_vals` and `_teacher_label` referenced only with their existing signatures.
- **Edge cases handled inline:** missing `exact_V_start` (legacy pkl), empty groups, mismatched history lengths (raises with config tuple), unequal budget counts per horizon (hidden axes).
