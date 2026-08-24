# CLAUDE.md

Mini project: solve **Pendulum-v1** with **PETS** (probabilistic ensemble +
CEM-MPC) and compare it against **model-free PPO**. The README reports the MBRL
approach; PPO only supplies comparison numbers. Use the local venv directly
(`.venv/bin/python`) — no uv, no conda. GPU available; configs use `cuda`.

## Layout

```
config/base.yaml          shared: env, eval protocol, threshold, seeds, video, wandb
config/{pets,ppo}.yaml    per-algo settings; `inherits: base.yaml`
src/config.py, cli.py     YAML load + `inherits:` merge; flags shared by both trainers
src/metrics.py            RunMetrics schema, threshold detection, PausableTimer
src/evaluate.py           act_fn-based eval + inference-latency benchmark
src/dynamics.py           probabilistic ensemble (batched members) + NLL training
src/planner.py            CEM-MPC + Pendulum's analytic reward
src/{pets,ppo}_train.py   trainers (CLI); src/compare.py -> table + plot
```

## Rules

- **No constants or hyperparameters in `.py` files** — all in `config/*.yaml`.
- **wandb**, not tensorboard; logging must never break a run.
- Integrated video wrappers only (SB3 `VecVideoRecorder`, gymnasium
  `RecordVideo`); they render inside training, so time `--no-video` runs.
- Both algorithms must use the *same* `evaluate.py` and `RunMetrics` schema, or
  the comparison is meaningless. Eval steps never count toward the sample budget;
  eval time never counts toward `train_wall_clock_s` (`PausableTimer`).
- `env_steps_to_threshold` is `None` when unreached — never report it as 0.
- PETS plans with the **known** reward (`planner.pendulum_reward`); only dynamics
  are learned, and the README must keep saying so.
- Report numbers as measured; don't tune until they look good.

## Commands

Run from the repo root. `--out` overwrites `metrics.json` in place.

```bash
.venv/bin/python -m src.pets_train --no-video   # PETS, ~20 min (reported run)
.venv/bin/python -m src.ppo_train --no-video    # PPO baseline, ~2 min
.venv/bin/python -m src.compare --plot          # table + figure
```

Smoke test: `--timesteps 600 --eval-episodes 1 --final-episodes 2 --no-wandb
--out <scratch>`. Live window: `--watch --eval-episodes 1`.
Debugging PETS: isolate planner from model by running CEM against a hand-written
*perfect* Pendulum model (should score ≈ −150 or better) — if that works, blame the
learned model (`model/holdout_mse`); if not, the reward sign or the
particle-to-member reshape (every candidate must be scored by *all* members).
