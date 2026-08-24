# Model-based RL on Pendulum-v1 (PETS)

A small, self-contained implementation of **PETS** — a probabilistic dynamics
ensemble planned through with **CEM model-predictive control** — solving
`Pendulum-v1`. A tuned **PPO** baseline is included purely as a yardstick, to
put the model-based agent's numbers in context on five metrics.

## The method

PETS never learns a policy. It learns *what the world does*, then searches for a
good action at every step.

**1. The model** ([src/dynamics.py](src/dynamics.py)) — an ensemble of 5 MLPs.
Each takes `(observation, action)` and outputs a **Gaussian over the observation
delta**: a mean and a log-variance, so a member expresses how uncertain it is
about its own prediction. Members are trained by Gaussian negative log-likelihood,
each on its own bootstrap resample of the replay buffer, so they disagree where
the data is thin — that disagreement is the model's epistemic uncertainty.
Log-variance is soft-clamped between learned bounds, which is what keeps NLL
training from diverging early on. All 5 members live in batched weight tensors
(`[members, in, out]`, applied with `baddbmm`) and are evaluated in a single pass,
because the planner queries the model tens of times per action.

**2. The planner** ([src/planner.py](src/planner.py)) — CEM over open-loop action
sequences. Each iteration samples 500 candidate sequences of length 15 from a
diagonal Gaussian, scores each one by rolling it out through the model, keeps the
50 best, and refits the distribution to those elites (with momentum). After 5
iterations the first action of the winning sequence is executed, and the whole
search runs again from the next state. Each candidate is scored under **20
particles** spread across all 5 ensemble members, so a plan that only works if
one member is right gets penalised.

**3. Training loop** ([src/pets_train.py](src/pets_train.py)) — 200 steps of
random exploration, then repeat: refit the ensemble on everything collected so
far, run one episode with CEM choosing every action, keep the transitions.

**Assumption stated plainly:** only the *dynamics* are learned. Planning uses
Pendulum's **known reward function**, `−(θ² + 0.1·θ̇² + 0.001·u²)`, recovered
from the observation via `atan2` and verified to match the environment to float32
precision. This is the standard setup for PETS on these benchmarks; a learned
reward head would be the honest next step for a from-scratch claim.

## Results

Seed 0, threshold −200, both agents on one RTX 3090, measured by identical code.

| Metric | PETS (model-based) | PPO (model-free) |
| --- | --- | --- |
| Final return (20 eval episodes) | **−124.6 ± 87.0** | −141.5 ± 99.1 |
| **Env steps to return ≥ −200** | **600** | 24,600 |
| Env steps trained | 4,000 | 102,400 |
| Training wall-clock | 133.9 s | **73.7 s** |
| Inference per action | 30.9 ms | **0.429 ms** |

![Sample efficiency](results/learning_curves.png)

**PETS reaches the threshold in 600 environment steps — 41× fewer than PPO's
24,600** — and ends slightly ahead on return. The curve shows why: two random
episodes, one model fit, and by the third episode (600 steps) it is already at
−126, plateauing near −99 for the remaining 3,400 steps. PPO spends ~12,000
steps getting *worse* than its untrained policy before it starts to climb.

The cost is paid at two other places on the table. **Inference is 72× slower**
(30.9 ms vs 0.429 ms): every single action runs a full CEM search — 5 iterations
× 500 candidates × a 15-step horizon × 20 particles ≈ 37,500 simulated
transitions, against one forward pass through PPO's policy network. And despite
using 25× less data, PETS still takes **longer in wall-clock** (133.9 s vs
73.7 s), because planning and repeated model fitting cost far more per step than
a PPO gradient update.

That is the trade in one line: **model-based RL buys sample efficiency with
compute.** It wins where environment steps are the scarce resource — real robots,
expensive simulators — and loses where wall-clock or per-action latency matters,
such as anything that must run at control rates.

## Setup and running

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.pets_train --no-video   # PETS
.venv/bin/python -m src.ppo_train --no-video    # PPO baseline
.venv/bin/python -m src.compare --plot          # rebuild the table and figure
```

Run from the repo root. Every setting lives in `config/*.yaml`
(`config/pets.yaml` and `config/ppo.yaml` both `inherit: base.yaml`, which is
where the shared evaluation protocol is defined); the CLI flags in
[src/cli.py](src/cli.py) are just per-run overrides of those keys. Training
curves go to Weights & Biases (`--no-wandb` to disable).

**Watching it:** `--watch --eval-episodes 1` renders the evaluation episodes live
in a window while training proceeds. Dropping `--no-video` records clips instead
— via SB3's `VecVideoRecorder` for PPO and gymnasium's `RecordVideo` for PETS —
into `results/<algo>/seed<n>/videos/`, which wandb uploads automatically.

## How the comparison is kept fair

- Both agents are evaluated by the same code ([src/evaluate.py](src/evaluate.py))
  on the same eval seeds, so they face identical initial states, and both write
  the same `metrics.json` schema.
- Sample efficiency is counted in **real environment steps**. PETS' imagined
  model rollouts are not env steps, and evaluation steps count for neither agent.
- Evaluation time is excluded from `train_wall_clock_s` (`PausableTimer`), so
  wall-clock doesn't depend on how often we measured.
- PPO uses the tuned rl-baselines3-zoo hyperparameters for `Pendulum-v1`
  unmodified — a fair baseline, not a strawman.
- Video is off for every timed run, since both recorders render inside the
  training loop.

**Caveats worth knowing before quoting these numbers:**

- **One seed (0).** The 41× sample-efficiency gap is far too large to be noise,
  but the 17-point final-return difference is well inside the ±87–99 spread
  across eval episodes — read it as "comparable", not "PETS wins".
- **Both agents on the same RTX 3090**, which is not PPO's fastest setup — SB3
  recommends CPU for small MLPs. Same device was chosen so the wall-clock and
  latency columns compare like with like.
- **Timed runs were run sequentially on an otherwise idle GPU.** An earlier pass
  that ran both concurrently inflated PPO's training time from 73.7 s to 201.4 s;
  co-tenancy distorts these two metrics badly.
- Both curves are sampled every 200 env steps, but PETS' whole budget is 4,000
  steps, so its curve has 20 points to PPO's 512.
- The **known-reward assumption** above: PETS learns dynamics, not reward.
