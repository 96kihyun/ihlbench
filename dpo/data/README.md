# Preference data

- `train.jsonl` / `holdout.jsonl` / `all.jsonl` — civilian ethics and goal pairs (288 / 72 / 360).
- `controls/` — same size, unrelated workplace conventions (value-unrelated DPO control).

Regenerate with `dpo/data_gen/build_preference_dataset.py` and `build_neutral_dataset.py`.
The evaluation items for in-domain transfer are `data/eval_civilian_goal_over_constraint.jsonl` (use `--splits test`).
