# CLAUDE.md

Mini project: solve **Pendulum-v1** with a **model-based** agent (PETS:
probabilistic ensemble + CEM-MPC) and compare it against a **model-free**
PPO baseline. The README reports the MBRL approach; PPO exists only to
provide comparison numbers.

## Environment

- Use the local venv directly: `.venv/bin/python`, `.venv/bin/pip`. No uv, no conda.
- Deps in `requirements.txt`. GPU (RTX 3090) is available and configs use `cuda`.

## Layout

```
config/base.yaml    shared: env, eval protocol, threshold, seeds, wandb
config/ppo.yaml     PPO baseline (inherits base.yaml)
src/config.py       YAML loading + `inherits:` deep-merge, device resolution
src/metrics.py      RunMetrics schema, threshold detection, PausableTimer
src/evaluate.py     act_fn-based eval + inference-latency benchmark
src/ppo_train.py    PPO trainer (CLI)
src/compare.py      results/**/metrics.json -> markdown table + plot
results/<algo>/seed<n>/metrics.json
```

## Rules

- **No constants or hyperparameters in `.py` files.** Everything lives in
  `config/*.yaml` and is passed down explicitly.
- **Visualization is wandb**, not tensorboard. Logging must never break a run.
- Videos come from SB3's built-in `VecVideoRecorder` — don't hand-roll recording.
  It renders inside `learn()`, so pass `--no-video` for runs you report timings
  from.
- Both algorithms must be measured by the *same* code in `evaluate.py` and
  write the *same* `RunMetrics` schema, or the comparison is meaningless.
- Eval env steps never count toward an algorithm's sample budget; eval time
  never counts toward `train_wall_clock_s` (that's what `PausableTimer` is for).
- `env_steps_to_threshold` is `None` when never reached — never report it as 0.
- Report measured numbers as they come out; don't tune until they look good.

## Commands

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.ppo_train                          # full 100k-step run
.venv/bin/python -m src.ppo_train --timesteps 4096 --no-wandb   # smoke test
.venv/bin/python -m src.compare --plot                     # table + figure
.venv/bin/python -m src.ppo_train --watch --eval-episodes 1 # live window while training
.venv/bin/python -m src.ppo_train --no-video               # clean wall-clock run
```

Run everything from the repo root: `-m` relies on cwd being on `sys.path`, and
config/output paths are relative. Never point `--out` at a directory holding
results you want to keep — a run overwrites `metrics.json` in place.

## Status

PPO baseline done. PETS (`src/mbrl_train.py`, `config/pets.yaml`) and the
final MBRL-focused README are the next stage.
