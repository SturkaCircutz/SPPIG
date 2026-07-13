# SPPIG Parking Programmatic Policy

This repository now uses the parking benchmark as the active training path. The
parking code trains a compact programmatic state machine from loop-free teacher
traces, evaluates it on train and test parking-task distributions, and writes
auditable metrics plus full trajectory sidecars.

The parking environment uses the paper-style continuous action interface:
`[velocity, steering]`. There is no direct lateral-rate action channel.

## Main Command

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run a small verified parking training job:

```bash
.venv/bin/python src/train_parking_psm.py \
  --train-n 24 \
  --test-n 24 \
  --teacher-iters 3 \
  --outer-iters 2 \
  --outdir artifacts/parking_policy \
  --metrics-output artifacts/parking_policy/metrics.json \
  --traces-output artifacts/parking_policy/traces.json \
  --verify
```

Equivalent script entry point:

```bash
.venv/bin/python scripts/run_parking_reproduction.py --verify
```

Run the PPO baseline on the same parking action interface:

```bash
.venv/bin/python scripts/run_parking_ppo.py \
  --train-n 8 \
  --test-n 8 \
  --updates 4 \
  --rollouts-per-update 8 \
  --outdir artifacts/parking_ppo \
  --metrics-output artifacts/parking_ppo/metrics.json \
  --traces-output artifacts/parking_ppo/traces.json \
  --verify
```

## Outputs

The trainer writes:

```text
artifacts/parking_policy/metrics.json       training, evaluation, learned thresholds
artifacts/parking_policy/traces.json        teacher and student trajectories
artifacts/parking_policy/repo_manifest.json reusable state-machine scan
artifacts/parking_policy/trajectories.png   train/test rollout plot, when matplotlib is installed
artifacts/parking_policy/success_rates.png  success-rate summary, when matplotlib is installed
artifacts/parking_ppo/metrics.json          PPO baseline metrics
artifacts/parking_ppo/traces.json           PPO baseline trajectories
artifacts/parking_ppo/ppo_trajectories.png  PPO rollout plot, when matplotlib is installed
```

`metrics.json` includes baseline, teacher, student-train, and student-test
summaries. `traces.json` includes parking task geometry, teacher trajectories,
student train trajectories, and student test trajectories.

## Important Files

```text
src/parking_env.py              parking task generation and dynamics
src/programmatic_policy.py      parking state-machine policy and parameters
src/adaptive_teaching_sim.py    parking teacher/student training loop
src/train_parking_psm.py        parking training CLI
scripts/run_parking_reproduction.py
scripts/run_parking_ppo.py
tests/test_parking_training_cli.py
tests/test_parking_ppo_cli.py
```

## Tests

Run all tests:

```bash
.venv/bin/python -m unittest discover -s tests
```
