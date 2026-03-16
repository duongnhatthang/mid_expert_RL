---
name: visualize-rl
description: Run the mid-capacity teacher RL visualization experiments: advantage grid, learning curve, and 2x2 exploration matrix
---

Run three visualization experiments in sequence from the project root `/Users/thangduong/Desktop/mid_expert_RL`:

1. **Advantage grid** — suite mode generates per-capacity advantage heatmaps:
   ```
   python run_experiments.py --mode suite --grid-size 10 --n-goals 3 --figures-dir results/figures
   ```

2. **Learning curve** — shows how different teacher capacities learn over time:
   ```
   python run_experiments.py --mode learning_curve --grid-size 8 --n-seeds 5 --n-goals 3 --sample-budget 12000 --alpha 0.5 --figures-dir results/figures --learning-curve-output results/figures/learning_curves.png
   ```

3. **2x2 exploration matrix** — budget × horizon grid comparing teacher capacities:
   ```
   python run_experiments.py --mode 2x2 --grid-size 8 --n-seeds 5 --n-goals 3 --alpha 0.5 --figures-dir results/figures
   ```

Run each command using the Bash tool and report:
- Any errors encountered
- Where output figures were saved (list paths from `results/figures/`)
- A brief summary of what each experiment produced
