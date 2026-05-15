# Learning Curves — Explanation Note + V*(s_0) Skyline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed explanation note to every learning-curve suptitle, and draw a dashed `V*(s_0)` skyline on every subplot (with the value in the legend), by modifying only `plot_learning_curves` in `run_hypothesis_sweep.py` and adding three tests.

**Architecture:** Single-function change. Inside `plot_learning_curves`, build a `dict[(distance, horizon_type), float]` of optimal start-state values using existing helpers `build_optimal_policy` + `evaluate_policy_values` from `tabular_prototype/teacher.py`, then call `ax.axhline(...)` per subplot. Add `fig.text(...)` for the explanation note and shrink `rect` to make room.

**Tech Stack:** Python 3, NumPy, Matplotlib, pytest. Uses the project's existing tabular-RL helpers; no new dependencies.

---

## File Map

- **Modify:** `run_hypothesis_sweep.py` — `plot_learning_curves` function only (lines 1468–1622). No other code paths touched.
- **Modify:** `tests/test_run_hypothesis_sweep_plots.py` — add three new tests; keep existing four tests unchanged.
- **No new files.**

## Test Strategy

- New tests use **figure capture via monkey-patching `matplotlib.figure.Figure.savefig`** rather than image diff. This lets us assert on `Line2D` / `Text` artists without comparing pixels.
- Each new test mirrors the synthetic-data helpers (`_zeta_results`, `_make_history`) that already exist in the test file.
- Use `dist in {4, 6}` for synthetic test data (subset of `DISTANCES = [4, 6, 8]`) so V* lookups land on valid `_goal_positions` keys.

---

## Task 1: V*(s_0) sanity unit test

Establishes that the existing teacher.py + GridEnv composition gives a finite, sensible V*(s_0) for a representative env. This is a pure regression test against the underlying helpers we're about to call from `plot_learning_curves`. Runs in milliseconds; no plotting involved.

**Files:**
- Test: `tests/test_run_hypothesis_sweep_plots.py` (append new test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_hypothesis_sweep_plots.py`:

```python
def test_v_star_at_s0_is_finite_and_in_range():
    """Sanity-check the V*(s_0) computation we wire into learning curves.

    Build a GridEnv with the same parameters used by the sweep (grid_size=9,
    1 goal at distance 4, horizon=8 i.e. the 'small' bucket for 9x9), compute
    pi* and V*, and assert 0 < V*(s_0) <= 1.

    Lower bound: the goal is reachable in 4 steps (Manhattan distance), well
    within H=8, so the optimal discounted return is strictly positive.
    Upper bound: maximum undiscounted return is 1 (single absorbing goal
    with reward=1); discounting shrinks it further. Values above 1 would
    indicate a sign / state-index bug.
    """
    from tabular_prototype.environment import GridEnv, generate_equidistant_goals
    from tabular_prototype.teacher import build_optimal_policy, evaluate_policy_values
    from tabular_prototype.config import compute_gamma_from_horizon

    grid_size = 9
    horizon = 8  # 'small' bucket for 9x9 (corner_dist = 8)
    goals = generate_equidistant_goals(grid_size, n_goals=1, distance=4)
    env = GridEnv(grid_size=grid_size, goals=goals, horizon=horizon)
    gamma = compute_gamma_from_horizon(horizon)

    pi_star = build_optimal_policy(env, env.goals, gamma)
    _, V_star = evaluate_policy_values(env, pi_star, gamma)
    v_star_s0 = float(V_star[env.state_to_idx(env.start)])

    assert 0.0 < v_star_s0 <= 1.0, (
        f"V*(s_0) out of expected (0, 1] range: {v_star_s0}"
    )
```

- [ ] **Step 2: Run test to verify it passes immediately (no implementation needed)**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_v_star_at_s0_is_finite_and_in_range -v`

Expected: PASS. This is a regression test on existing helpers — no implementation needed yet. If it fails, the underlying `teacher.py` / `GridEnv` composition is broken and the rest of the plan is invalid.

- [ ] **Step 3: Commit**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: pin V*(s_0) computation for learning-curve skyline" -m "Sanity-checks that build_optimal_policy + evaluate_policy_values on a 9x9 GridEnv with 1 goal at distance 4 yields a finite V*(s_0) in (0, 1]. Foundation for the upcoming V* skyline in plot_learning_curves."
```

---

## Task 2: V*(s_0) skyline — failing test

Write the figure-capture test that asserts a black dashed horizontal line appears on every visible subplot. The test will fail because `plot_learning_curves` doesn't draw one yet. We implement in Task 3.

**Files:**
- Test: `tests/test_run_hypothesis_sweep_plots.py` (append new test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_hypothesis_sweep_plots.py`:

```python
def test_plot_learning_curves_draws_v_star_skyline(tmp_path, monkeypatch):
    """Every visible subplot must carry a black dashed V*(s_0) horizontal line."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_learning_curves(_zeta_results(), mode='zeta',
                               figures_dir=str(tmp_path))

    assert captured_figures, "expected at least one figure saved"

    for fig in captured_figures:
        for ax in fig.get_axes():
            if not ax.get_visible():
                continue
            dashed_black = [
                line for line in ax.get_lines()
                if line.get_linestyle() == '--' and line.get_color() == 'black'
            ]
            assert dashed_black, (
                f"axes titled {ax.get_title()!r} missing a black dashed "
                f"V*(s_0) line"
            )

    plt.close('all')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_draws_v_star_skyline -v`

Expected: FAIL with `AssertionError: axes titled '...' missing a black dashed V*(s_0) line` (no `axhline` has been added yet).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: add failing test for V*(s_0) skyline (no implementation yet)"
```

---

## Task 3: V*(s_0) skyline — implementation

Build the per-(distance, horizon_type) V* cache and draw the dashed line. Keys come from the groups dict so synthetic test data with a subset of `DISTANCES` doesn't trip on missing `_goal_positions` lookups.

**Files:**
- Modify: `run_hypothesis_sweep.py:1468–1622` (the `plot_learning_curves` function body)

- [ ] **Step 1: Add imports inside `plot_learning_curves`**

In `run_hypothesis_sweep.py`, locate the function (line 1468). Inside the function body, immediately after the existing `import matplotlib.pyplot as plt` line (around line 1488), add:

```python
    from tabular_prototype.teacher import (
        build_optimal_policy, evaluate_policy_values,
    )
    from tabular_prototype.config import compute_gamma_from_horizon
```

Rationale for in-function import: matches the existing import style inside this function (`from collections import defaultdict`, `import matplotlib.pyplot as plt`). Keeps optional teacher.py / config.py loading lazy.

- [ ] **Step 2: Build the V* cache after the `groups` dict is populated**

After the existing block that builds `groups` (ends around line 1511 with the `training_modes.add(...)` line) and before the existing `if training_modes == {'exact'}:` block (around line 1515), insert:

```python
    # V*(s_0) cache, keyed by (distance, horizon_type). V* depends only on
    # those two — γ = 1 - 1/H, and the env has deterministic goals + no
    # traps in the sweep paths. Computed once per unique (dist, h_type)
    # actually present in the data, not over DISTANCES × HORIZON_TYPES,
    # so synthetic test data with a subset of distances works without
    # adjustment.
    horizons = _get_horizons()
    goal_pos = _goal_positions(mode)
    v_star_cache: dict = {}
    for (dist_k, _alpha_k, h_type_k, _budget_k, _tv_k) in groups:
        cache_key = (dist_k, h_type_k)
        if cache_key in v_star_cache:
            continue
        h_val = horizons[h_type_k]
        gamma = compute_gamma_from_horizon(h_val)
        vstar_env = GridEnv(
            grid_size=GRID_SIZE, goals=goal_pos[dist_k], horizon=h_val,
        )
        pi_star = build_optimal_policy(vstar_env, vstar_env.goals, gamma)
        _, V_star = evaluate_policy_values(vstar_env, pi_star, gamma)
        v_star_cache[cache_key] = float(
            V_star[vstar_env.state_to_idx(vstar_env.start)]
        )
```

- [ ] **Step 3: Draw the dashed line inside each subplot**

In the existing inner loop (around line 1600, immediately after the existing `ax.set_title(f'{h_label}, B={budget}', fontsize=9)` line), add:

```python
                v_star = v_star_cache[(dist, h_type)]
                ax.axhline(
                    v_star, linestyle='--', color='black', linewidth=1.2,
                    label=rf'$V^*(s_0)={v_star:.3f}$',
                )
```

Place this BEFORE the `ax.grid(...)` / `ax.set_ylabel(...)` / `ax.legend(...)` block so the axhline label is registered before the legend is drawn.

- [ ] **Step 4: Run the skyline test to verify it now passes**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_draws_v_star_skyline -v`

Expected: PASS. Every visible axes has at least one black dashed `Line2D`.

- [ ] **Step 5: Run the full plots test file to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py -v`

Expected: ALL pass (the four pre-existing tests + the two new ones from Tasks 1 and 2).

- [ ] **Step 6: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: draw V*(s_0) skyline on every learning-curve subplot" -m "Computes V*(s_0) once per (distance, horizon_type) via build_optimal_policy + evaluate_policy_values from teacher.py, caches it in plot_learning_curves, and draws a black dashed axhline per subplot labeled with the numeric value. Cache is keyed on (dist, h_type) tuples actually present in the data so synthetic test data with a subset of distances doesn't break."
```

---

## Task 4: Explanation suptitle note — failing test

Test that the figure has a `Text` artist containing the substring `budget < eval_interval` (the diagnostic phrase from the note). Will fail until Task 5 lands.

**Files:**
- Test: `tests/test_run_hypothesis_sweep_plots.py` (append new test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_hypothesis_sweep_plots.py`:

```python
def test_plot_learning_curves_includes_explanation_note(tmp_path, monkeypatch):
    """Every figure must carry the suptitle note explaining missing cells
    and per-baseline x-position differences."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    captured_figures = []
    original_savefig = Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capturing_savefig)

    sweep.plot_learning_curves(_zeta_results(), mode='zeta',
                               figures_dir=str(tmp_path))

    assert captured_figures, "expected at least one figure saved"

    for fig in captured_figures:
        note_texts = [
            t.get_text() for t in fig.texts
            if 'budget < eval_interval' in t.get_text()
        ]
        assert note_texts, (
            "figure missing explanation note containing "
            "'budget < eval_interval'"
        )

    plt.close('all')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_includes_explanation_note -v`

Expected: FAIL with `AssertionError: figure missing explanation note containing 'budget < eval_interval'`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_run_hypothesis_sweep_plots.py
git commit -m "test: add failing test for suptitle explanation note"
```

---

## Task 5: Explanation suptitle note — implementation

Adjust the existing `fig.suptitle` y-coordinate, add `fig.text` for the note, and shrink `tight_layout` rect to make vertical room.

**Files:**
- Modify: `run_hypothesis_sweep.py:1610–1615` (the suptitle + tight_layout block at end of the per-figure loop)

- [ ] **Step 1: Replace the existing `fig.suptitle` + `fig.tight_layout` block**

In `run_hypothesis_sweep.py`, find the existing block (currently around lines 1610–1615):

```python
        fig.suptitle(
            f'Learning curve ({mode} sweep) — '
            f'dist={dist}, ' + rf'$\alpha={alpha}$',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
```

Replace with:

```python
        fig.suptitle(
            f'Learning curve ({mode} sweep) — '
            f'dist={dist}, ' + rf'$\alpha={alpha}$',
            fontsize=11, y=0.985,
        )
        fig.text(
            0.5, 0.945,
            "Cells may be missing when no seed completed ≥1 evaluation "
            "tick (budget < eval_interval). Lines for different baselines "
            "start at different update steps because trajectory length "
            "varies stochastically: equal env-step budgets yield different "
            "update counts per baseline.",
            ha='center', va='top', fontsize=7.5, color='dimgray', wrap=True,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
```

Two changes vs. current:
1. `fig.suptitle` gets `y=0.985` (was default ~0.98) to pin it to the top.
2. New `fig.text` call places the dim-gray explanation note below the suptitle.
3. `tight_layout` rect top bound shrinks from `0.96` to `0.92` so the note has room.

- [ ] **Step 2: Run the explanation test to verify it now passes**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py::test_plot_learning_curves_includes_explanation_note -v`

Expected: PASS.

- [ ] **Step 3: Run the full plots test file to verify no regressions**

Run: `PYTHONPATH=. pytest tests/test_run_hypothesis_sweep_plots.py -v`

Expected: ALL pass (4 pre-existing + 3 new = 7 tests).

- [ ] **Step 4: Commit**

```bash
git add run_hypothesis_sweep.py
git commit -m "feat: add explanation note to learning-curve suptitle" -m "Dim-gray multi-line note placed below the main suptitle explains: (a) why some subplot cells may be missing at tiny budgets, (b) why baselines' lines start/end at different update-step values in trajectory-based modes. Wording is fixed across all figures; shrinks tight_layout rect from 0.96 to 0.92 to make vertical room for the note."
```

---

## Task 6: Regenerate one sample figure manually to verify visual

Before opening the PR, regenerate one figure from cached pkl data and eyeball it. This is a manual smoke test — the automated tests assert the right artists are present, but the visual readability of the new note + skyline still needs a human glance.

**Files:**
- No code changes.

- [ ] **Step 1: Find an existing sweep output directory with cached pkl**

Run: `ls results/full_suite_*/zeta_sample/zeta_sweep_results.pkl 2>/dev/null | head -1`

Expected: prints one path (e.g. `results/full_suite_20260419_105552/zeta_sample/zeta_sweep_results.pkl`). If empty, fall back to `ls results/full_suite_*/capability_sample/capability_sweep_results.pkl 2>/dev/null | head -1` and adjust the next step accordingly.

- [ ] **Step 2: Regenerate the figures from cached pkl**

Substitute `<DIR>` with the directory from step 1 (the parent containing the pkl):

```bash
python run_hypothesis_sweep.py --mode zeta --skip-run --output-dir <DIR>
```

Expected: prints `Saved <DIR>/figures/learning_curves/learning_curve_dist*_alpha*.png` for each (distance, alpha) combo, plus the existing visitation / heatmap outputs.

- [ ] **Step 3: Manually open one PNG and visually verify**

Run: `open <DIR>/figures/learning_curves/learning_curve_dist4_alpha0.00.png` (or any one of them).

Verify by eye:
- A black dashed horizontal line appears on every visible subplot.
- The rightmost subplot of each row has `V*(s_0)=0.xxx` in its legend, alongside the baseline curve labels.
- A dim-gray multi-line note appears immediately under the main title, starting with "Cells may be missing when no seed...".
- Main title is still readable and not visually overlapped by the note.

- [ ] **Step 4: No commit needed** — this task verifies the visual; no code changes.

---

## Task 7: Final sanity sweep

Run the entire `tests/` directory to catch any unrelated breakage, then check `git status` is clean.

**Files:**
- No code changes.

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v`

Expected: ALL tests pass. New ones from this plan added in Tasks 1, 2, 4. Existing tests unchanged in behavior.

- [ ] **Step 2: Verify clean working tree**

Run: `git status`

Expected: `nothing to commit, working tree clean` apart from any pre-existing untracked log files that were present at branch start (`results_*.log`).

- [ ] **Step 3: Show the commit log delta**

Run: `git log --oneline origin/main..HEAD`

Expected: 5 new commits on top of the existing branch history — one per Tasks 1, 2, 3, 4, 5 — plus the spec commit already on the branch.

---

## Spec coverage check

- Suptitle expansion with explanation note (spec §"Suptitle + explanation note") → Tasks 4 + 5.
- V*(s_0) skyline (spec §"V*(s_0) skyline") → Tasks 2 + 3, with Task 1 as foundation.
- V*(s_0) sanity test in (0, 1] range (spec §Tests → test_v_star_finite_and_in_range) → Task 1.
- Suptitle note assertion test (spec §Tests → test_suptitle_has_explanation_note) → Task 4.
- Axhline-drawn-on-each-subplot test (spec §Tests → test_v_star_drawn_on_each_subplot) → Task 2.
- Cache strategy: inline in `plot_learning_curves`, keyed by `(distance, horizon_type)` actually present in `groups` (spec §"Computation strategy") → Task 3 Step 2.
- No changes to `cap_zeta` mode (spec §Out of scope) → unchanged; the early-return at the top of `plot_learning_curves` still handles cap_zeta.

No spec gaps remain.
