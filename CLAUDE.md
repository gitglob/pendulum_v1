# CLAUDE.md

Solve **Pendulum-v1** with model-based RL (**PETS**, **DreamerV3**) against a
**model-free PPO** baseline, from state or pixels. The README reports the MBRL
approaches; PPO only supplies comparison numbers. Use the local venv directly
(`.venv/bin/python`) — no uv, no conda; configs use `cuda`.

## Layout

```
config/base.yaml    shared: env, obs, eval protocol, threshold, video, wandb
config/<algo>.yaml  per-algo; a `pixels:` block merged in by --obs-type pixels
src/common/         config+cli (YAML `inherits:` merge, shared flags), metrics,
                    obs (state/pixel pipeline), evaluate, wandb_utils, compare
src/pets/           dynamics (ensemble), planner (CEM-MPC, injected reward_fn),
                    encoder (pixel latent + reward/transition heads), train
src/dreamer/        nets (symlog/twohot/CNN), rssm, agent (world model +
                    imagination actor-critic), train
src/ppo/train.py    SB3 PPO
results/<algo>/<obs_type>/seed<n>/  metrics.json (committed), model.pt, videos
```

## Rules

- **No constants or hyperparameters in `.py` files** — all in `config/*.yaml`;
  **wandb**, not tensorboard, and logging must never break a run.
- Integrated wrappers only (SB3 `VecVideoRecorder`, gymnasium `RecordVideo`,
  pixel wrappers); video renders inside training, so time `--no-video` runs.
- Every algorithm must use the *same* `common/evaluate.py` and `RunMetrics`
  schema, or the comparison is meaningless. Eval steps never count toward the
  sample budget, eval time never toward `train_wall_clock_s`, and
  `env_steps_to_threshold` is `None` when unreached — never 0.
- State PETS plans with the **known** reward; pixel agents learn theirs — the
  README must keep saying so. Report numbers as measured, don't tune to taste.

## Commands

```bash
.venv/bin/python -m src.pets.train --no-video                  # ~20 min
.venv/bin/python -m src.ppo.train  --no-video                  # ~2 min
.venv/bin/python -m src.{pets,ppo}.train --obs-type pixels --no-video  # 40-60 min
.venv/bin/python -m src.dreamer.train --no-video               # pixels only, ~2 h
.venv/bin/python -m src.common.compare --plot                  # table + figure
```

Run from repo root; `--out` overwrites `metrics.json` in place. Time reported runs
sequentially on an idle GPU. Smoke: `--timesteps 600 --eval-episodes 1
--final-episodes 2 --no-wandb --out <scratch>`. Stateful agents expose `reset()`,
called each episode start. **Debugging model-based:** what predicts whether
planning/imagination works is predicted-vs-true return correlation over the
horizon (≈0.98 state PETS, ≈0.09 pixel PETS, ≈0.90 Dreamer) — not one-step error.
