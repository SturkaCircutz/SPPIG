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
| Direct-Opt diagnostic | 1.00 | 0.10 | 250.0 | 4311.0 |
| Programmatic state machine | 1.00 | 0.00 | 250.0 | 1560.6 |

The test split is the full paper horizon: 300 seconds, or 15,000 simulator
steps. Pure PPO-LSTM is implemented, but it did not solve the training split
within the local diagnostic budget. The displayed programmatic row is a fixed
two-mode policy reevaluation; current synthesis metrics are tracked separately
and should not be read as a completed probabilistic adaptive-teaching result.
The checked-in synthesized PSM diagnostic artifact has been regenerated under
the current mode-order semantics, but remains a bounded local diagnostic.
Programmatic policies execute the current mode's action before applying the
switch predicate to update the next mode, matching the paper's state-machine
semantics.

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
  --include-ppo \
  --include-direct-opt \
  --outdir artifacts/results
```

Use `--quick` for a small diagnostic run, add `--include-ppo` to include
PPO/PPO-LSTM, and add `--include-direct-opt` to include the bounded Direct-Opt
diagnostic baseline. Without `--quick`, PPO uses the paper-scale `10^7`
timestep budget per seed; the runner still does not perform the paper's
hyperparameter search. The Direct-Opt path is a local bounded search over linear
switches, Boolean-tree CartPole switch candidates, and bounded Appendix B.3-style
continuous one-hot leaf/depth-2 feature-mixture candidates, not the paper's full two-hour
parallel direct optimization protocol. The runner writes
raw per-seed rows to `cartpole_results.csv`, grouped mean/std plus the best training seed to
`cartpole_summary.csv`, and full configs/provenance to `cartpole_manifest.json`.
The manifest includes a top-level `paper_protocol_status` block that records
selected seeds, paper rollout/horizon coverage, PPO/Direct-Opt inclusion, and
the remaining blockers to a paper-scale result claim.
Those rows and summaries report mean survived simulator steps and survival
seconds explicitly, rather than using reward as an implicit survival-time proxy.
Paper-scale result claims also require the paper's `1000` evaluation rollouts;
local examples in this README often pass `--eval-rollouts 20` only to keep
diagnostics cheap.
Each PSM row records a metrics JSON path with the fitted probabilistic student
and per-iteration teacher-trace provenance, plus a `traces_output` sidecar with
the full selected teacher traces and per-iteration teacher-trace history for that seed.
The checked-in fixed-program reevaluation metrics instead record
`paper_protocol_status` with `synthesized_by_current_algorithm` false, so that
the full-horizon fixed PSM row is not confused with a current synthesis result.
Current synthesis metrics record the same flag as true.
When PPO is included, each PPO row also records its checkpoint path and metrics
JSON path under the output directory. Use `--ppo-eval-interval N` to record
intermediate train/test evaluations in each PPO metrics JSON; quick runs default
to interval `32`, while full runs default to final-result-only metrics unless an
interval is supplied. PPO metrics also include compact per-update training
diagnostics such as rollout reward, horizon truncations, and failure
terminations. CartPole metrics and manifests include a `reward_spec` block
recording the standard OpenAI classic-control reward used here: `+1` per
survived simulator step, with no extra terminal bonus or penalty.
They also include a `space_spec` block. Its action/observation dimensions come
from the paper's CartPole row (`#A = 1`, `#O = 4`), while force bounds,
feature names, and the numeric independent-uniform reset range `[-0.05, 0.05]`
are recorded as local implementation provenance rather than paper-specified
constants.

Programmatic state machine:

```bash
.venv/bin/python src/train_cartpole_psm.py \
  --num-initial-states 4 \
  --candidate-rollouts 8 \
  --teacher-reward-lambda 100 \
  --teacher-top-rho 2 \
  --teacher-refinement-steps 1 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --metrics-output artifacts/cartpole_psm_metrics.json
```

The PSM metrics JSON records the deterministic policy plus the fitted
probabilistic student summary: Gaussian constant-action distributions, switch
parameter distributions, latent responsibility totals, hard/ambiguous latent
mode counts, responsibility entropy, and compact teacher trace examples with
reward, length, gains, segment durations, switches, and boundary observations.
Trace examples include the loop-free segment action
sequence, duration sequence, and per-segment time increments used by the teacher. It also records
`synthesis_history`, a compact row for each teacher/student iteration with
trace summaries, fitted student parameters, local switch-fit diagnostics,
per-EM student-fit history, adaptive-teacher objective summaries, and train/test
evaluation under the requested diagnostic rollout budget. Each
`student_fit_history` row records the EM iteration, responsibility pass, fit
phase, Gaussian action/switch summaries, latent responsibility confidence, and
compact adjacent switch-pair posterior mass for the bounded switch M-step.
The top-level
`adaptive_teacher_summary` array records each iteration's teacher sampling
model, teacher-source counts, reward summary, recorded student log-probability
coverage, and the recorded reward-plus-student-likelihood objective components
when available.
Pass `--traces-output path/to/traces.json` to write the full selected teacher
traces and per-iteration teacher-trace history as a sidecar artifact;
orchestrated reproduction runs write this sidecar for each PSM row automatically.
It also records `switch_fit_diagnostics`, which compares the selected switch's
responsibility-weighted label loss and bounded Eq. (12)-style distribution
timing loss against a fixed local reference switch, while also retaining hard
trace-label mistakes and the older deterministic timing comparator. That block
is intended to explain current synthesis failures; it is not a paper-scale
result claim.
The CLI exposes the current teacher gain, teacher/student iteration, reward
scale, regularization, top-rho, and local-refinement settings, and the metrics
JSON records their exact values under `config`.
The current CartPole PSM defaults use one-step loop-free teacher segments over
the full 250-step training horizon (`segment_steps=1`, `segments_per_trace=250`);
this is a local teacher hyperparameter profile selected for CartPole
diagnostics, not a paper-reported constant.
It also records local synthesis defaults, including student EM count, per-EM
switch-timing responsibility-refinement passes, minimum Gaussian standard deviation,
switch-timing scale, switch-search grids, bounded switch-parameter coordinate
refinement plus finite-difference gradient polishing with backtracking, and teacher-search
refinement schedule, under `algorithm_provenance`;
the actual configured EM schedule is recorded under `config` and
`paper_protocol_status`.
The metrics JSON also includes `paper_protocol_status`, which records the
matched CartPole train/test horizons and the remaining algorithmic gaps. That
block deliberately keeps `full_probabilistic_adaptive_teaching`,
`full_continuous_switch_m_step`, `full_cem_teacher_optimizer`, and
`paper_scale_result` false for the current bounded diagnostic implementation.
The probabilistic student likelihood and EM responsibility refinement are
conditioned on the executable CartPole PSM's fixed initial mode `0`, matching
the paper's fixed initial memory-state assumption.
The first teacher iteration uses an explicit probabilistic student prior, then
later teacher candidate pools are sampled from the current probabilistic
student before top-rho local refinement, matching the paper's sampled-teacher
phase more closely than the earlier gain-sampled bootstrap search. Trace
summaries record the selected source and sampled-trace log-probability when
available. Selected teacher traces also record the direct Eq. (8)-style
teacher objective and the bounded top-rho refinement objective used for local
selection. If a sampled closed-loop rollout is projected back into the loop-free
teacher budget, that likelihood is recomputed on the projected trace before
teacher-objective ranking. Teacher scoring also recomputes likelihoods against
the current probabilistic student whenever raw trace actions are available, so
cached likelihoods from an earlier student cannot drive a later adaptive-
teaching objective.
Local refinement can vary teacher gains, one segment duration, one segment
time increment, or one
constant-action segment at a time, accepting only improvements under the
current teacher objective or, after the first student fit, a top-rho
elite-distance kernel approximation of the paper's second teacher optimization
phase; that kernel now includes teacher gains plus normalized action,
duration, and time-increment segment schedules.
Student-sampled traces can also be locally refined through
duration/time-increment/action coordinate search plus one bounded
finite-difference teacher-gain candidate, one bounded finite-difference action
candidate, one bounded finite-difference integer-duration candidate, and one
bounded finite-difference time-increment candidate, plus one bounded joint
finite-difference gain/action/duration/time-increment schedule candidate per
refinement iteration, each with a short backtracking line search.
The teacher also evaluates one deterministic
centroid recombination of the top-rho loop-free action/duration/time-increment schedules
and configurable bounded rounds that fit a Gaussian schedule distribution over
teacher gains plus per-segment actions, durations, and time increments from the
current top-rho set. Each round evaluates the fitted mean and samples from that
distribution, refreshes the top-rho set, and refits before the next round. This
is still only a bounded CEM-style approximation, not the paper's full CEM plus
gradient optimizer.
The student starts with action-likelihood responsibilities, then each configured
EM iteration alternates bounded forward-backward refinements using the learned
switch-timing likelihood with action-distribution and switch-parameter refits.
The first segment of each trace is fixed to mode `0` in these responsibility
updates, rather than using a uniform latent initial-mode prior.
That timing likelihood now treats selector-off to
selector-on and selector-on to selector-off transitions as separate directed
events, and it treats loop-free segment durations as elapsed time normalized to
the CartPole simulator step, so per-segment time increments influence the
bounded Eq. (12)-style timing terms. The final observed segment contributes
no-transition-before-duration evidence, so a trace that stays in a mode is not
scored only by its action likelihood. Switch refits consume adjacent pair
posteriors from this forward-backward pass for transition/stay weights instead
of reconstructing them only from independent neighboring segment marginals.
The switch threshold Gaussian means and standard deviations are locally refined
against the current Eq. (12)-style timing likelihood using a grid initializer
plus bounded coordinate steps and finite-difference gradient polishing with backtracking. Depth-2 Boolean-tree
expansions and final switch candidates are prefiltered by a cheaper hard-label/timing
objective, then bounded top-32 subsets are ranked first by
responsibility-weighted expected label loss over non-boundary segment
observations and then by this bounded distribution-timing objective.
The teacher regularizer scores both action
likelihood and switch timing under the current student's Gaussian switch
distributions. The teacher objective uses the
paper-reported reward scale `lambda = 100` by default. This is provenance for
the current partial student implementation, not evidence that the full
probabilistic adaptive-teaching algorithm has been completed.

Paper Figure 19 CartPole reference policy:

```bash
.venv/bin/python scripts/evaluate_cartpole_program.py \
  --paper-figure19 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --metrics-output artifacts/results/metrics/figure19_cartpole_reference.json
```

This reevaluates the manually transcribed Figure 19 CartPole state machine from
the rendered paper PDF. The metrics mark it as
`paper_figure19_manual_transcription` and keep
`synthesized_by_current_algorithm` false.

Direct-Opt diagnostic:

```bash
.venv/bin/python src/train_cartpole_direct_opt.py \
  --num-train-states 10 \
  --random-candidates 256 \
  --batch-size 10 \
  --batch-refinement-rounds 1 \
  --local-refinement-steps 2 \
  --restart-candidates-on-stall 1 \
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --metrics-output artifacts/results/metrics/direct_opt_seed0_full_horizon.json
```

This baseline searches a bounded two-mode constant-action CartPole PSM directly
on the 5-second training split, including the previous linear switch grid,
bounded depth-1/depth-2 Boolean-tree switch predicates with explicit one-hot
feature, relation, and tree-operator metadata, and a bounded Appendix B.3-style
continuous one-hot leaf/depth-2 feature-mixture candidate family. It then applies a bounded
batch/restart local refinement seeded from the best candidate so far, optimizing
mean reward over all selected finite initial states, not the full initial-state
distribution, before reevaluating the selected program on the full paper test
horizon.
Its metrics JSON records the exact grid, Boolean/continuous one-hot leaf/depth-2 counts,
candidate-evaluation-call counts, train-rollout-evaluation counts,
batch/restart diagnostics, configurable local parallel-candidate evaluation and
time-limit metadata, selected program, and limitation note. This is still not
the paper's two-hour, ten-thread Direct-Opt protocol over the optimized
continuous one-hot switching grammar. The metrics JSON also includes
`paper_protocol_status`, which keeps the full Direct-Opt protocol flag false
unless the paper batch size, ten-thread/two-hour optimization budget, full
continuous one-hot grammar, full test horizon, and `1000`-rollout evaluation
are actually satisfied.

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

The standalone PPO CLI defaults to the paper `10^7` timestep budget when
`--timesteps` is omitted. The shorter commands above are local diagnostic
examples that override the default explicitly.

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
local rollout rewards and train-horizon termination counts. PPO training metrics
also include a `paper_protocol_status` block that marks whether the run used the
paper `10^7` timestep budget, 300s test horizon, and `1000` evaluation rollouts,
while keeping the full five-seed baseline protocol claim false for standalone runs. This is
training-curve provenance for local diagnostics; it is not a substitute for the
missing paper-scale `10^7` timestep, five-seed hyperparameter search.
Checkpoint reevaluation metrics from `scripts/evaluate_cartpole_checkpoint.py`
also include `paper_protocol_status`, separating the checkpoint's original
training/evaluation settings from the later reevaluation horizon and rollout
count. Warm-start checkpoint rows also record whether the checkpoint config
itself proves the pretraining teacher policy and PSM mode-update order; older
checked-in warm-start artifacts are marked as missing that provenance instead
of assuming current teacher semantics.

PPO hyperparameter sweep plan/execution:

```bash
.venv/bin/python scripts/run_cartpole_ppo_sweep.py \
  --policies mlp,lstm \
  --seeds 0,1,2,3,4 \
  --outdir artifacts/ppo_sweep
```

Use `--dry-run` to write only `cartpole_ppo_sweep_plan.csv` and the manifest,
or `--quick --max-configs 1` for a smoke execution. Use `--resume` to continue
an interrupted sweep; it skips only completed rows whose plan fields still
match and whose checkpoint plus metrics artifacts still exist. Use
`--continue-on-error` only when a long sweep should record failed jobs to
`cartpole_ppo_sweep_failures.csv` and continue; by default, the first failed job
stops the sweep. Executed sweeps also write `cartpole_ppo_sweep_results.csv`,
`cartpole_ppo_sweep_summary.csv`, and `cartpole_ppo_sweep_hyperparam_summary.csv`;
the first summary selects the best completed single job per policy by train
success, then train reward, while the hyperparameter summary aggregates completed
seeds for each sampled config, records selected-seed coverage and missing seeds,
and marks the best config per policy only after preferring complete selected-seed
coverage before mean training success. Executed sweep rows and summaries also include explicit
mean survived steps and survival seconds for train/test evaluation. The manifest records
both the jobs actually planned and the uncapped job count for the selected
search space, the concrete sampled hyperparameter configs, plus
`paper_protocol_status` flags showing whether the plan is paper-scale, whether
all planned jobs completed with zero failures, whether it is quick/truncated or
dry-run only, the selected and distinct seed/policy lists, and whether both PPO
MLP and PPO-LSTM are included. The status block validates the actual generated
hyperparameter configs against the paper's reported discrete ranges, learning-rate
interval, and PPO-LSTM `nminibatches=1` rule before allowing a paper-scale plan
claim. By default the
sweep now uses `--hyperparam-mode paper-random`, which
plans 10 uniformly sampled PPO hyperparameter configs per policy from the
reported ranges and evaluates each config for every selected seed, with
PPO-LSTM fixed to `nminibatches=1`.
Use `--hyperparam-mode grid` for the older explicit Cartesian-grid diagnostic.
The full-plan flag requires paper-random mode, 10 samples per policy, five
seeds, both PPO MLP and PPO-LSTM, the `10^7` timestep budget, and the full
15,000-step/300-second test horizon, plus the paper's `1000` evaluation
rollouts; grid mode is documented as a local extension rather than the paper's
sampled search.

## Paper and Audit

- Paper draft: `essay/project.tex`
- Generated abstract result fragment: `essay/cartpole_abstract_results.tex`
- Generated result table fragment: `essay/cartpole_results_table.tex`
- Generated PSM policy fragment: `essay/cartpole_policy_fragment.tex`
- Generated Figure 19 reference fragment:
  `essay/cartpole_figure19_reference_fragment.tex`
- arXiv source manifest: `essay/00README.json`
- Figures: `essay/figures/`
- Figure generation script: `scripts/make_paper_figures.py` (uses
  `cartpole_summary.csv` when present, otherwise raw result rows, and rewrites
  the abstract result, table, PSM policy, and Figure 19 reference fragments; if PSM metrics with a
  linear switch exist, it writes the switch-boundary figure from that artifact,
  and if PPO metrics JSON files exist, it also writes a training-curve figure;
  survival-reward plots prefer explicit survived-step fields when present;
  generated result fragments carry a local-diagnostic limitation note and reject
  rows whose explicit `test_horizon_steps` is not the paper 300-second horizon)
- Paper fidelity audit: `docs/cartpole_paper_audit.md`
- Result table: `artifacts/results/cartpole_results.csv`
- Result metrics: `artifacts/results/metrics/`
- Result PSM trace sidecars: `artifacts/results/traces/`
- Result summary: `artifacts/results/cartpole_summary.csv`
- Result manifest: `artifacts/results/cartpole_manifest.json`, including a
  bundle-level `paper_protocol_status` block for the checked-in diagnostics
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
- Evaluate paper result claims over `1000` rollouts.
- Run the paper's PPO hyperparameter search.
- Regenerate training curves and survival-time plots from completed paper-scale
  five-seed runs rather than local diagnostic artifacts.
- Either tune pure PPO-LSTM until it solves train or report it as a carefully
  bounded negative result.
- Replace the compact trace-based programmatic learner with the full
  probabilistic adaptive-teaching algorithm if exact reproduction is required.
