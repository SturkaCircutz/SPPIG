# SPPIG CartPole Reproduction Roadmap

This repository is a partial reproduction of the CartPole experiment from
*Synthesizing Programmatic Policies that Inductively Generalize*.

The goal is to compare neural reinforcement-learning policies with compact
programmatic policies on continuous-force CartPole. This is an essay-scale
implementation study, not a full paper-scale reproduction.

## What We Have Done

### 1. CartPole Environment

- Implemented continuous-force CartPole.
- Matched the paper's main CartPole split:
  - training: 5 seconds, pole length 0.5
  - testing: 300 seconds, pole length 1.0
- Kept old wrapper files so older commands still work.
- Added tests for the environment and CLI compatibility.

### 2. PPO Baselines

- Implemented feed-forward PPO.
- Implemented PPO-LSTM.
- Fixed important RL bugs:
  - train episodes now truncate at the 5-second training horizon
  - PPO stores raw sampled actions before clipping environment force
  - PPO-LSTM reuses the rollout recurrent state during updates
- Ran a medium local sweep:
  - PPO MLP, 2 seeds
  - PPO-LSTM, 2 seeds
  - 1,000,000 timesteps per job
  - full 300-second test evaluation
- Saved trained checkpoints under:

```text
artifacts/ppo_sweep_cuda_medium_core/checkpoints/
```

### 3. Programmatic Policy Work

- Implemented a programmatic state-machine policy representation.
- Implemented local PSM training and evaluation commands.
- Added Direct-Opt diagnostic search.
- Added a fixed Programmatic PSM baseline.
- Added a selected SPPIG-style PSM diagnostic that reaches the full 300-second
  CartPole horizon in local evaluation.
- Kept the current Synthesized PSM diagnostic as a failure case so the project
  does not overclaim full synthesis success.

### 4. Metrics, Figures, and Essay Evidence

- Metrics are saved as JSON files under:

```text
artifacts/results/metrics/
```

- Main figures are saved under:

```text
essay/figures/
```

- The essay PDF is generated from LaTeX source:

```text
essay/project.tex
essay/project.pdf
```

- Added stronger graph evidence to the essay:
  - rollout survival distribution
  - train-vs-test generalization scatter
  - compactness-vs-performance Pareto plot
  - selected PSM switch-boundary rollout overlay

### 5. Verification

- Added an essay-scale verifier:

```bash
.venv/bin/python scripts/verify_essay_scale_reproduction.py
```

- The verifier checks that:
  - result tables match metrics artifacts
  - key figures exist
  - stale result values are not used
  - paper-scale claims remain false
  - the essay stays consistent with local diagnostic evidence

## Current Local Results

These are local diagnostic results, not the original paper's full five-seed,
10,000,000-timestep, 1000-rollout protocol.

| Policy | Train success | Test success | Train reward | Test reward |
| --- | ---: | ---: | ---: | ---: |
| PPO MLP | 1.00 | 0.00 | 250.0 | 910.6 |
| PPO-LSTM warm started | 1.00 | 0.00 | 250.0 | 912.2 |
| Direct-Opt diagnostic | 1.00 | 0.10 | 250.0 | 4311.0 |
| Programmatic PSM | 1.00 | 0.00 | 250.0 | 1560.5 |
| SPPIG selected PSM | 1.00 | 1.00 | 250.0 | 15000.0 |
| Synthesized PSM diagnostic | 0.00 | 0.00 | 48.0 | 59.9 |

The strongest local result is the selected SPPIG-style PSM. It reaches the full
300-second test horizon while remaining a compact, readable controller. PPO MLP
and PPO-LSTM checkpoints can also reach the reward ceiling in the medium sweep,
so the main advantage shown here is compactness and interpretability, not a
higher capped reward.

## Important Files

```text
src/cartpole/env.py                  CartPole environment
src/cartpole/ppo/                    PPO and PPO-LSTM code
src/cartpole/psm/                    Programmatic policy code
src/cartpole/direct_opt/             Direct-Opt diagnostic code
scripts/run_cartpole_reproduction.py Reproduction runner
scripts/make_paper_figures.py        Table and figure generator
scripts/verify_essay_scale_reproduction.py
                                      Essay/result verifier
essay/project.tex                    LaTeX source
essay/project.pdf                    Compiled essay
```

## How To Run The Main Checks

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Verify essay artifacts:

```bash
.venv/bin/python scripts/verify_essay_scale_reproduction.py
```

Re-evaluate a saved PPO-LSTM checkpoint without retraining:

```bash
.venv/bin/python scripts/evaluate_cartpole_checkpoint.py \
  --checkpoint artifacts/ppo_sweep_cuda_medium_core/checkpoints/00002_lstm_seed0.pt \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --metrics-output artifacts/results/metrics/manual_lstm_seed0_reeval_20.json
```

## What Is Still Left

- Run the full paper-scale PPO/PPO-LSTM protocol:
  - 5 seeds
  - 10,000,000 timesteps
  - 1000 evaluation rollouts
  - larger hyperparameter search
- Finish the full probabilistic adaptive-teaching SPPIG learner.
- Improve the synthesized PSM so it can discover the strong selected PSM
  automatically.
- Extend beyond CartPole to other tasks from the paper.
- Run larger-scale GPU training if more compute is available.

## Current Stage

The project is past the basic implementation stage. The main CartPole
environment, PPO baselines, PSM diagnostics, metrics, essay figures, and verifier
are in place.

The biggest remaining gap is full paper-scale training and full automatic SPPIG
synthesis. The current repository should be read as a strong local reproduction
study, not as a complete reproduction of every result in the original paper.
