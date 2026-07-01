# Reproducing Programmatic Policy Generalization on CartPole

This repository is a reproduction-oriented engineering study of the CartPole
benchmark from *Synthesizing Programmatic Policies that Inductively
Generalize*. It implements:

- continuous-force CartPole with the paper's train/test split,
- PyTorch PPO with a feed-forward policy,
- PyTorch PPO with an LSTM policy,
- a compact two-mode programmatic state-machine learner,
- tests and audit notes documenting what matches the paper and what does not.

This is not a full reproduction of the original paper. The programmatic learner
is a compact trace-based reconstruction, and the PPO baselines have not yet been
run with the paper's full `10^7` timestep, five-seed, hyperparameter-search
protocol.

## Current Verified Results

Local diagnostic evaluation backed by `artifacts/results/cartpole_results.csv`
and per-row metrics artifacts:

| Policy | Train success | Test success | Train reward | Test reward |
| --- | ---: | ---: | ---: | ---: |
| PPO MLP | 1.00 | 0.00 | 250.0 | 910.6 |
| PPO-LSTM, warm started | 1.00 | 0.00 | 250.0 | 912.2 |
| Programmatic state machine | 1.00 | 0.20 | 250.0 | 6275.4 |

The test split is the full paper horizon: 300 seconds, or 15,000 simulator
steps. Pure PPO-LSTM is implemented, but it did not solve the training split
within the local diagnostic budget. The displayed programmatic row is a fixed
two-mode policy reevaluation; current synthesis metrics are tracked separately
and should not be read as a completed probabilistic adaptive-teaching result.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Run Experiments

Regenerate the CartPole result table, summary, and manifest:

```bash
.venv/bin/python scripts/run_cartpole_reproduction.py \
  --seeds 0,1,2,3,4 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --outdir artifacts/results
```

Use `--quick` for a small diagnostic run, and add `--include-ppo` to include
PPO/PPO-LSTM. Without `--quick`, PPO uses the paper-scale `10^7` timestep
budget per seed; the runner still does not perform the paper's hyperparameter
search. The runner writes raw per-seed rows to `cartpole_results.csv`, grouped
mean/std plus the best training seed to `cartpole_summary.csv`, and full
configs/provenance to `cartpole_manifest.json`. Each PSM row records a metrics
JSON path with the fitted probabilistic student and teacher-trace provenance.
When PPO is included, each PPO row also records its checkpoint path and metrics
JSON path under the output directory. Use `--ppo-eval-interval N` to record
intermediate train/test evaluations in each PPO metrics JSON; quick runs default
to interval `32`, while full runs default to final-result-only metrics unless an
interval is supplied. PPO metrics also include compact per-update training
diagnostics such as rollout reward, horizon truncations, and failure
terminations.

Programmatic state machine:

```bash
.venv/bin/python src/train_cartpole_psm.py \
  --num-initial-states 64 \
  --segment-steps 8 \
  --segments-per-trace 32 \
  --teacher-reward-lambda 100 \
  --teacher-top-rho 10 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --metrics-output artifacts/cartpole_psm_metrics.json
```

The PSM metrics JSON records the deterministic policy plus the fitted
probabilistic student summary: Gaussian constant-action distributions, switch
parameter distributions, latent responsibility totals, and compact teacher
trace examples with reward, length, gains, segment durations, switches, and
boundary observations.
The CLI exposes the current teacher gain, teacher/student iteration, reward
scale, regularization, top-rho, and local-refinement settings, and the metrics
JSON records their exact values under `config`.
It also records fixed local synthesis constants, including EM count, minimum
Gaussian standard deviation, switch-timing scale, switch-search grids, and
teacher-search refinement schedule, under `algorithm_provenance`.
The switch threshold means are locally refined against the current discrete
Eq. (12)-style timing likelihood, and the teacher regularizer scores both
action likelihood and switch timing under the current student's Gaussian switch
distributions. The teacher objective uses the paper-reported reward scale
`lambda = 100` by default. This is provenance for the current partial student
implementation, not evidence that the full probabilistic adaptive-teaching
algorithm has been completed.

PPO MLP:

```bash
.venv/bin/python src/train_cartpole_ppo.py \
  --policy mlp \
  --timesteps 131072 \
  --rollout-steps 128 \
  --num-envs 8 \
  --update-epochs 8 \
  --minibatches 8 \
  --learning-rate 0.0003 \
  --entropy-coef 0.01 \
  --initial-log-std -1 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --eval-interval 16384 \
  --verbose \
  --output artifacts/cartpole_ppo_mlp.pt \
  --metrics-output artifacts/cartpole_ppo_mlp_metrics.json
```

PPO-LSTM diagnostic:

```bash
.venv/bin/python src/train_cartpole_ppo.py \
  --policy lstm \
  --timesteps 262144 \
  --rollout-steps 128 \
  --num-envs 8 \
  --update-epochs 8 \
  --learning-rate 0.0003 \
  --entropy-coef 0.01 \
  --initial-log-std -1 \
  --eval-interval 32768 \
  --verbose \
  --output artifacts/cartpole_ppo_lstm.pt \
  --metrics-output artifacts/cartpole_ppo_lstm_metrics.json
```

When `--eval-interval` is positive, the PPO trainer records each train/test
evaluation in `eval_history`, plus the selected checkpoint result and config in
the metrics JSON file. Each metrics file also records `update_history` rows for
local rollout rewards and train-horizon termination counts. This is
training-curve provenance for local diagnostics; it is not a substitute for the
missing paper-scale `10^7` timestep, five-seed hyperparameter search.

PPO hyperparameter sweep plan/execution:

```bash
.venv/bin/python scripts/run_cartpole_ppo_sweep.py \
  --policies mlp,lstm \
  --seeds 0,1,2,3,4 \
  --outdir artifacts/ppo_sweep
```

Use `--dry-run` to write only `cartpole_ppo_sweep_plan.csv` and the manifest,
or `--quick --max-configs 1` for a smoke execution. Executed sweeps also write
`cartpole_ppo_sweep_results.csv` and `cartpole_ppo_sweep_summary.csv`; the
summary selects the best completed config per policy by train success, then
train reward. The sweep enumerates the paper's reported `nminibatches`,
`ent_coef`, `noptepochs`, and `cliprange` ranges, with PPO-LSTM fixed to
`nminibatches=1`. The extracted paper text gives a learning-rate interval
rather than exact samples, so the runner records the explicit sampled values in
the manifest.

## Paper and Audit

- Paper draft: `essay/project.tex`
- Generated abstract result fragment: `essay/cartpole_abstract_results.tex`
- Generated result table fragment: `essay/cartpole_results_table.tex`
- Generated PSM policy fragment: `essay/cartpole_policy_fragment.tex`
- arXiv source manifest: `essay/00README.json`
- Figures: `essay/figures/`
- Figure generation script: `scripts/make_paper_figures.py` (uses
  `cartpole_summary.csv` when present, otherwise raw result rows, and rewrites
  the abstract result, table, and PSM policy fragments; if PSM metrics with a
  linear switch exist, it writes the switch-boundary figure from that artifact,
  and if PPO metrics JSON files exist, it also writes a training-curve figure)
- Paper fidelity audit: `docs/cartpole_paper_audit.md`
- Result table: `artifacts/results/cartpole_results.csv`
- Result metrics: `artifacts/results/metrics/`
- Result summary: `artifacts/results/cartpole_summary.csv`
- Programmatic policy metrics: `artifacts/cartpole_psm*_metrics.json`
- PPO training metrics: `artifacts/cartpole_ppo_*_metrics.json`,
  `artifacts/results/metrics/*.json`, and `artifacts/ppo_sweep/metrics/*.json`
- PPO sweep plan/results: `artifacts/ppo_sweep/`
- PPO training-curve figure: `essay/figures/cartpole_ppo_training_curves.png`

## Resume Framing

Recommended resume bullet:

> Implemented a PyTorch reproduction study of programmatic policy
> generalization on continuous-force CartPole, including PPO, PPO-LSTM, and a
> two-mode state-machine learner; audited paper train/test fidelity, fixed PPO
> horizon/log-probability/recurrent-state bugs, and evaluated long-horizon
> generalization over a 300-second test split.

## Interview Talking Points

- The paper asks whether structured programmatic policies generalize better
  than neural policies on changed test distributions.
- The hardest bug was training PPO on the wrong objective because rollouts were
  not truncated at the paper's 5-second training horizon.
- Continuous-action PPO needed careful handling of raw sampled actions versus
  clipped environment actions.
- LSTM PPO needed recurrent state replay during policy updates.
- The feed-forward PPO solved the short training split but failed full
  300-second generalization; the programmatic policy survived much longer.

## Remaining Work Before a Strong arXiv Submission

- Run PPO/PPO-LSTM for `10^7` timesteps.
- Run five random seeds and report mean/std.
- Run the paper's PPO hyperparameter search.
- Add training curves and survival-time plots.
- Either tune pure PPO-LSTM until it solves train or report it as a carefully
  bounded negative result.
- Replace the compact trace-based programmatic learner with the full
  probabilistic adaptive-teaching algorithm if exact reproduction is required.
