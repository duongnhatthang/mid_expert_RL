---
name: visualize-rl
description: Run the mid-capacity teacher RL visualization experiments: advantage grid, learning curve, and 2x2 exploration matrix
---

Run three visualization experiments in sequence from the project root `/Users/thangduong/Desktop/mid_expert_RL`.

Teacher capacity is now parameterised by **ζ ∈ [0,1]**: the teacher policy is
μ(ζ) = ζ·π* + (1−ζ)·π_random (coin-flip at each step).  ζ=0 is pure random,
ζ=1 is the optimal teacher, and ζ∈(0,1) are mid-capacity teachers.

1. **Advantage grid + 2x2 exploration matrix** — runs the full budget×horizon grid
   and also saves per-ζ advantage heatmaps:
   ```
   python run_experiments.py --mode 2x2_zeta --grid-size 8 --n-seeds 5 --n-goals 3 --alpha 0.5 --figures-dir results/figures --zeta-output results/exploration_2x2_zeta_results.csv
   ```

2. **Learning curve** — shows how different ζ teachers affect learning speed:
   ```
   python run_experiments.py --mode learning_curve_zeta --grid-size 8 --n-seeds 5 --n-goals 3 --sample-budget 12000 --alpha 0.5 --figures-dir results/figures --zeta-learning-curve-output results/figures/learning_curves_zeta.png
   ```

Run each command using the Bash tool and report:
- Any errors encountered
- Where output figures were saved (list paths from `results/figures/`)
- A brief summary of what each experiment produced, including whether a mid-ζ teacher outperformed the best (ζ=1) teacher
