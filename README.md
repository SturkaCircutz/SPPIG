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

## Repository Structure

- `src/cartpole/env.py`: CartPole environment, reward/space provenance, and fixed/manual PSM evaluators.
- `src/cartpole/ppo/`: PPO/PPO-LSTM runtime and training CLI implementation.
- `src/cartpole/psm/`: current probabilistic-programmatic synthesis, PSM training CLI, and evaluator.
- `src/cartpole/direct_opt/`: bounded Direct-Opt diagnostic baseline and CLI implementation.
- Legacy entry points such as `src/train_cartpole_ppo.py`, `src/train_cartpole_psm.py`,
  `src/train_cartpole_direct_opt.py`, `src/cartpole_env.py`, `src/cartpole_synthesis.py`,
  and `src/cartpole_direct_opt.py` are retained as compatibility wrappers, so older commands in this
  README and existing artifact provenance still work.

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
timestep budget per seed for fixed-config PPO rows, and the PSM runner profile
defaults to the paper-reported 10 teacher workers and 10 student-switch workers.
Pass
`--ppo-sweep-manifest artifacts/ppo_sweep/cartpole_ppo_sweep_manifest.json` to
attach PPO/PPO-LSTM hyperparameter-search evidence from the sweep runner; the
runner requires the typed sweep manifest and only marks that evidence present when the sweep reports
`paper_scale_execution = true` with matching top-level policy, seed, and sample-count
fields plus embedded best-hyperparameter rows that cover every selected seed for both PPO policies.
This can be used with or without rerunning fixed PPO rows in the same bundle. The Direct-Opt path is a local bounded search over linear
switches, Boolean-tree CartPole switch candidates, bounded Appendix B.3-style
continuous one-hot leaf/depth-2 feature-mixture candidates, and bounded continuous
one-hot random restarts, not the paper's full two-hour
parallel direct optimization protocol; without `--quick`, its runner profile
defaults to the paper-reported 10 candidate-evaluation threads and 7200-second
time limit while keeping the protocol flag false until the remaining Direct-Opt
requirements are actually satisfied. The runner writes
raw per-seed rows to `cartpole_results.csv`, grouped mean/std plus the best training seed to
`cartpole_summary.csv`, and full configs/provenance to `cartpole_manifest.json`.
The manifest includes a top-level `paper_protocol_status` block that records
selected seeds, paper rollout/horizon coverage, PPO/Direct-Opt inclusion, and
the remaining blockers to a paper-scale result claim.
Direct-Opt evidence in that block is derived from the Direct-Opt rows and their
metrics JSON artifacts; the runner requires command/config provenance and a
matching per-row `paper_protocol_status`, so current bounded diagnostics stay
separated from a future full Direct-Opt protocol artifact.
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
  --candidate-rollouts 10 \
  --teacher-reward-lambda 100 \
  --teacher-top-rho 10 \
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
phase, local trace log likelihood, Gaussian action/switch summaries, latent
responsibility confidence, and compact adjacent switch-pair posterior mass for
the bounded switch M-step.
Rows after a directed switch M-step also record the fitted transition-specific
`0->1` and `1->0` switch snapshots used by later timing-responsibility passes.
The top-level
`adaptive_teacher_summary` array records each iteration's teacher sampling
model, teacher-source counts, reward summary, recorded student log-probability
coverage, and the recorded reward-plus-student-likelihood objective components
when available, including reward-term, student-regularizer-term, direct-objective,
and refinement-objective summaries. It also records the configured teacher
candidate rollout count, effective top-rho value, the paper's `rho=10`
reference, and whether the local candidate pool actually covers that paper
top-rho setting. Selected teacher traces also record bounded candidate-pool
diagnostics: sampled rollout count, top-rho elite count, recombination and
distribution-candidate count, refinement seed/refined candidate counts, selected
source, raw/effective candidate-rollout counts, and objective summaries for the
sampled and final selection pools.
For selected teacher traces it also summarizes the refreshed top-rho elite set
used by the bounded refinement objective:
elite counts, source counts, nearest-elite distances, and kernel log-probability
terms, including normalized elite probability weights and distance-weighted
kernel component weights when a probabilistic student is available.
Pass `--traces-output path/to/traces.json` to write the full selected teacher
traces and per-iteration teacher-trace history as a sidecar artifact;
orchestrated reproduction runs write this sidecar for each PSM row automatically.
It also records `switch_fit_diagnostics`, which compares the selected switch's
responsibility-weighted label loss and bounded Eq. (12)-style distribution
timing loss against a fixed local reference switch, while also retaining hard
trace-label mistakes and the older deterministic timing comparator. The fitted
probabilistic student also records separate bounded `0->1` and `1->0`
transition switch conditions and their Gaussian parameter distributions; these
are executed by the deterministic and sampled PSM projections, while the legacy
selector switch is retained as a fallback/provenance comparator. That block is
intended to explain current synthesis failures; it is not a paper-scale result
claim.
The CLI exposes the current teacher gain, teacher/student iteration, reward
scale, regularization, top-rho, and local-refinement settings, and the metrics
JSON records their exact values under `config`. `--parallel-trace-workers 10`
selects the paper-reported worker limit for independent loop-free teacher
optimization across initial states; status fields separately report the active
parallel trace slots, so runs with fewer than 10 initial states do not claim the
paper's 10-thread execution. `--parallel-switch-workers 2` can evaluate the two
directed CartPole student transition-switch fits concurrently; the paper's
10-thread student-side claim still remains false because this bounded two-mode
grammar has only two directed transition fits. Standalone PSM CLI runs and
`--quick` reproduction diagnostics remain serial unless these flags are
provided explicitly; non-quick reproduction-runner PSM rows default the two
worker counts to the paper-reported value of 10.
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
block derives `full_probabilistic_adaptive_teaching` and `paper_scale_result`
from named requirement maps, so the current bounded diagnostic explicitly lists
the unsatisfied paper requirements instead of relying on a hard-coded false
flag. The current artifacts still leave `full_continuous_switch_m_step`,
`full_cem_teacher_optimizer`, paper-scale worker coverage, five-seed selection,
and/or `1000`-rollout evaluation unsatisfied, depending on the run. The same
status block separately records whether the teacher CEM-style sampling phase
used the paper's `rho=10` top-elite setting, whether local teacher trace
optimization actually had `10` active trace slots, and how many student
transition-switch fit slots were active for the local bounded switch M-step.
The student status distinguishes a configured `10`-worker limit from actual
`10`-slot switch optimizer coverage; the bounded implementation can now use
`parallel_switch_workers` for candidate-level switch rescoring, but still keeps
only a bounded depth-2/transition switch M-step instead of the paper's full
continuous switch optimizer.
The probabilistic student likelihood and EM responsibility refinement are
conditioned on the executable CartPole PSM's fixed initial mode `0`, matching
the paper's fixed initial memory-state assumption. After the first bounded
switch M-step, later EM timing-responsibility passes use the latest fitted
transition-specific `0->1` and `1->0` switching conditions rather than the
bootstrap selector fallback.
The first teacher iteration uses an explicit probabilistic student prior, then
later teacher candidate pools are sampled from the current probabilistic
student before top-rho local refinement, matching the paper's sampled-teacher
phase more closely than the earlier gain-sampled bootstrap search. Trace
summaries record the selected source and sampled-trace log-probability when
available. Selected teacher traces also record the direct Eq. (8)-style
teacher objective and the bounded top-rho refinement objective used for local
selection, plus a compact summary of the refreshed elite set that defined that
top-rho kernel approximation. If a sampled closed-loop rollout is projected back into the loop-free
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
distribution, refreshes the top-rho set, and refits before the next round.
Distribution-generated teacher traces serialize the fitted source weights,
source objectives, and Gaussian schedule parameters used to produce them. This
is still only a bounded CEM-style approximation, not the paper's full CEM plus
gradient optimizer.
The student starts with action-likelihood responsibilities, then each configured
EM iteration repeats bounded fixed-switch forward-backward refinements with
action-distribution refits before one bounded Eq. (12)-style switch-parameter
M-step.
The first segment of each trace is fixed to mode `0` in these responsibility
updates, rather than using a uniform latent initial-mode prior.
That timing likelihood now treats selector-off to selector-on and selector-on to
selector-off transitions as separate directed events, then fits separate bounded
transition conditions for `0->1` and `1->0` before projecting the student back to
an executable PSM. It treats loop-free segment durations as elapsed time
normalized to the CartPole simulator step, so per-segment time increments
influence the bounded Eq. (12)-style timing terms. The final observed segment
contributes no-transition-before-duration evidence, so a trace that stays in a
mode is not scored only by its action likelihood. The per-iteration switch M-step
consumes adjacent pair posteriors from the latest forward-backward pass plus
final-segment stay weights from final segment marginals for transition/stay weights
instead of reconstructing them only from independent
neighboring segment marginals.
The switch threshold Gaussian means and standard deviations are locally refined
against the current Eq. (12)-style timing likelihood using a grid initializer
plus bounded coordinate steps and finite-difference gradient polishing with backtracking. Depth-2 Boolean-tree
expansions and final switch candidates are prefiltered by a cheaper hard-label/timing
objective, then bounded top-32 subsets are ranked first by
responsibility-weighted expected label loss over non-boundary segment
observations and then by this bounded distribution-timing objective.
The teacher regularizer scores both action
likelihood and switch timing under the current student's Gaussian switch
distributions, using the fitted directed `0->1` and `1->0` transition switches
when they are available and falling back to the legacy selector otherwise. The
teacher objective uses the
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
continuous one-hot leaf/depth-2 feature-mixture candidate family, with bounded continuous
one-hot random restarts when random or stalled-batch restart phases are reached. It then applies a bounded
batch/restart local refinement over forces, thresholds, and continuous one-hot `alpha_s`/feature weights
seeded from the best candidate so far when no earlier candidate has solved all selected training
states, optimizing
mean reward over all selected finite initial states, not the full initial-state
distribution, before reevaluating the selected program on the full paper test
horizon.
Its metrics JSON records the exact selected training initial states, grid,
Boolean/continuous one-hot leaf/depth-2 counts reached before a training solution,
candidate-evaluation-call counts, the solution-found phase,
train-rollout-evaluation counts, batch/restart diagnostics with compact
per-batch seed/local/restart/full-train reevaluation trace, bounded continuous
`alpha_s`/feature-weight refinement,
optional sampled train-distribution rerank diagnostics and rerank rollout counts,
configurable local parallel-candidate evaluation and
time-limit metadata, selected program, and limitation note. Standalone and
`--quick` diagnostics remain serial with no wall-clock cap unless flags are
provided explicitly; non-quick reproduction-runner Direct-Opt rows default to
the paper's 10-thread, 7200-second budget. This is still not the paper's full
Direct-Opt protocol over the optimized continuous one-hot switching grammar. The metrics JSON also includes
`paper_protocol_status`, which keeps the full Direct-Opt protocol flag false
unless the paper batch size with batch rounds, local refinement, restart-on-stall,
ten-thread/two-hour optimization budget, full continuous one-hot grammar,
full initial-state distribution, full test horizon,
and `1000`-rollout evaluation are actually satisfied. The optional train-distribution rerank is
sampled evidence only and does not satisfy the full initial-state distribution requirement. Reproduction manifests require these Direct-Opt
protocol rows to be backed by matching metrics JSON command/config/provenance artifacts. The status also lists the named Direct-Opt protocol
requirements that remain unsatisfied for each diagnostic run.

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
  --device cuda \
  --verbose \
  --output artifacts/ppo_gpu_diagnostics/cartpole_ppo_mlp_131k_cuda.pt \
  --metrics-output artifacts/ppo_gpu_diagnostics/cartpole_ppo_mlp_131k_cuda_metrics.json
```

The standalone PPO CLI defaults to the paper `10^7` timestep budget when
`--timesteps` is omitted. The shorter commands above are local diagnostic
examples that override the default explicitly. PPO accepts `--device auto`,
`--device cpu`, or `--device cuda[:index]` and records the selected torch device
in metrics. A CUDA-capable PyTorch install is required before local GPU runs can
use the visible NVIDIA GPU; CPU-only smoke runs remain valid launch diagnostics
but are not paper-scale training evidence.

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
  --eval-rollouts 20 \
  --test-max-steps 15000 \
  --eval-interval 32768 \
  --device cuda \
  --verbose \
  --output artifacts/ppo_gpu_diagnostics/cartpole_ppo_lstm_262k_cuda.pt \
  --metrics-output artifacts/ppo_gpu_diagnostics/cartpole_ppo_lstm_262k_cuda_metrics.json
```

Current CUDA diagnostic artifacts from those bounded commands record
`requested=cuda`, `selected=cuda`. The MLP run selected the 98,304-step
checkpoint with train success `1.00`, test success `0.00`, train reward
`250.0`, and test reward `448.65`. The pure PPO-LSTM run selected the
32,768-step checkpoint with train success `0.00`, test success `0.00`, train
reward `43.75`, and test reward `59.15`. These artifacts are local runtime
diagnostics, not paper-scale result evidence.

When `--eval-interval` is positive, the PPO trainer records each train/test
evaluation in `eval_history`, plus the selected checkpoint result and config in
the metrics JSON file. Each metrics file also records `update_history` rows for
local rollout rewards, train-horizon termination counts, and optimizer-side PPO
diagnostics such as loss means, entropy, approximate KL, clip fraction, and
minibatch-update count. PPO training metrics also include a
`paper_protocol_status` block that marks whether the run used the
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
the manifest embeds the summary and hyperparameter-summary rows used as downstream evidence.
Every sweep manifest also writes a dependency-light `runtime_preflight` block with
PyTorch import/CUDA availability, `nvidia-smi` GPU metadata when available, planned
job counts, requested training timesteps, requested evaluation rollouts, and the
full paper-scale reference size. This is launch provenance only; it does not turn
a dry-run or partial run into paper-scale execution evidence.
The first summary selects the best completed single job per policy by train
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
claim. A completed sweep manifest can be passed to
`scripts/run_cartpole_reproduction.py --ppo-sweep-manifest ...`; the
reproduction bundle records that evidence but still keeps the full paper-scale
result flag false until the full probabilistic adaptive-teaching and Direct-Opt
protocols are also complete. By default the
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
  `artifacts/ppo_gpu_diagnostics/*_metrics.json`, `artifacts/results/metrics/*.json`,
  and `artifacts/ppo_sweep/metrics/*.json`
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
- LSTM PPO needed recurrent state replay plus done-aligned resets during policy updates.
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
