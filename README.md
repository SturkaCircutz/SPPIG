# SPPIG Parking Programmatic Policy

This repository contains a local implementation study inspired by
**Synthesizing Programmatic Policies that Inductively Generalize**.  The active
benchmark is a **10 m** parking task where loop-free teacher traces are distilled
into a compact programmatic state machine.

The parking action is:

```text
[velocity, steering]
```

> [!IMPORTANT]
> There is no direct lateral-rate action channel.  The parking
> controller uses the paper-style velocity/steering interface.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick Smoke Run

Use this first to confirm the code works on your machine:

```bash
.venv/bin/python scripts/run_parking_reproduction.py \
  --train-n 24 --test-n 24 \
  --teacher-iters 2 --outer-iters 2 \
  --outdir artifacts/parking_policy_smoke \
  --verify
```

## Final Report Run

This is the main scale used for the polished local report:

```bash
.venv/bin/python scripts/run_parking_reproduction.py \
  --train-n 2000 --test-n 100 \
  --teacher-iters 2 --outer-iters 2 --seed 0 \
  --outdir artifacts/parking_policy_2000x100 \
  --verify
```

> [!IMPORTANT]
> The final reported parking run uses **2000** training tasks and **100**
> held-out test tasks.  Keep `--verify` enabled so the CLI fails if the learned
> student does not satisfy the basic result checks.

Run the small PPO diagnostic on the same task scale:

```bash
.venv/bin/python scripts/run_parking_ppo.py \
  --train-n 2000 --test-n 100 --seed 0 \
  --outdir artifacts/parking_ppo_2000x100 \
  --verify
```

> [!IMPORTANT]
> The PPO command is a local diagnostic baseline.  It is not a paper-scale PPO
> training run.

## Outputs

Each run writes metrics, trajectories, a repository manifest, and summary plots
under the selected `artifacts/` directory.  These files are ignored by Git
because the full traces can be large.

## Main Files

```text
src/parking_env.py              parking task generation and dynamics
src/programmatic_policy.py      parking state-machine policy and parameters
src/adaptive_teaching_sim.py    teacher search and student distillation
src/train_parking_psm.py        parking PSM training CLI
scripts/run_parking_reproduction.py
scripts/run_parking_ppo.py
tests/test_parking_training_cli.py
tests/test_parking_ppo_cli.py
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Scope

This is a local implementation and artifact bundle.  It is reinforcement
learning related because it studies programmatic policies for continuous-control
tasks and includes a PPO diagnostic, but it does not claim a new RL algorithm or
a full paper-scale reproduction.
