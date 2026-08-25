# CLAUDE.md

Mini project: solve **Pendulum-v1** with **PETS** (probabilistic ensemble +
CEM-MPC), compared against **model-free PPO**, from either state or pixels. The
README reports the MBRL approach; PPO only supplies comparison numbers. Use the
local venv directly (`.venv/bin/python`) — no uv, no conda; configs use `cuda`.

## Layout

```
config/base.yaml          shared: env, obs, eval protocol, threshold, video, wandb
config/{pets,ppo}.yaml    per-algo; a `pixels:` block merged in by --obs-type pixels
src/config.py, cli.py     YAML load + `inherits:` merge; flags shared by both trainers
src/metrics.py            RunMetrics schema, threshold detection, PausableTimer
src/obs.py, evaluate.py   state/pixel observation pipeline; eval + latency benchmark
src/dynamics.py           probabilistic ensemble + NLL / multi-step training
src/encoder.py            pixel PETS: conv autoencoder, reward + transition heads
src/planner.py            CEM-MPC; reward_fn injected, not hard-coded
src/{pets,ppo}_train.py   trainers (CLI); src/compare.py -> table + plot
results/<algo>/<obs_type>/seed<n>/   metrics.json (committed), model, videos
```

## Rules

- **No constants or hyperparameters in `.py` files** — all in `config/*.yaml`.
  **wandb**, not tensorboard; logging must never break a run.
- Integrated wrappers only (SB3 `VecVideoRecorder`, gymnasium `RecordVideo`, the
  pixel wrappers); video renders inside training, so time `--no-video` runs.
- Both algorithms must use the *same* `evaluate.py` and `RunMetrics` schema, or the
  comparison is meaningless. Eval steps never count toward the sample budget; eval
  time never counts toward `train_wall_clock_s` (`PausableTimer`).
  `env_steps_to_threshold` is `None` when unreached — never report it as 0.
- State PETS plans with the **known** reward, the pixel agent learns one; only
  dynamics are learned either way, and the README must keep saying so.
- Report numbers as measured; don't tune until they look good.

## Commands

```bash
.venv/bin/python -m src.pets_train --no-video                 # ~20 min (reported)
.venv/bin/python -m src.ppo_train  --no-video                 # ~2 min
.venv/bin/python -m src.{pets,ppo}_train --obs-type pixels --no-video   # 40-60 min
.venv/bin/python -m src.compare --plot                        # table + figure
```

Run from the repo root; `--out` overwrites `metrics.json` in place. Time reported
runs sequentially on an idle GPU. Smoke test: `--timesteps 600
--eval-episodes 1 --final-episodes 2 --no-wandb --out <scratch>`. Debugging PETS:
what predicts whether planning works is the correlation between predicted and true
return over the horizon (≈0.98 state, ≈0.09 pixels) — one-step holdout error does
**not** predict it.
