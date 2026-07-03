# Cartpole Paper Audit

Source: `/home/jiawen/Downloads/1321_synthesizing_programmatic_poli.pdf`.

## Verified Details

- Benchmark: Cartpole.
- Action dimension: `#A = 1`.
- Observation dimension: `#O = 4`.
- Observation text: position `x`, cart velocity `v`, pole angle `theta`, pole angular velocity `omega`.
- Training distribution: `time = 5s, len = 0.5`.
- Test distribution: `time = 300s, len = 1.0`.
- Programmatic state-machine modes: `2`.
- Action grammar: `Constant`.
- Switch grammar: `Boolean tree (depth 2)`.
- Baselines: PPO feed-forward neural policy, PPO-LSTM, and Direct-Opt.
- RL implementation in paper: PPO2 from OpenAI Baselines.
- RL training budget in paper: `10^7` timesteps.
- PPO hyperparameter search in paper:
  - `nminibatches` in `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}`.
  - LSTM uses `nminibatches = 1`.
  - `ent_coef` in `{0.0, 0.01, 0.05, 0.1}`.
  - `noptepochs` in `{3, ..., 36}`.
  - `cliprange` in `{0.1, 0.2, 0.3}`.
  - learning rate in `[5e-6, 0.003]`.
- Classic control reward functions: standard OpenAI environment rewards.
- Adaptive-teaching reward scale: `lambda = 100`.
- Evaluation metric: fraction of rollouts out of `1000` that satisfy the benchmark metric.

## Not Verified From Extracted Text

- Exact numerical Cartpole train/test success rates from Figure 4. The PDF text extraction exposes the
  graphical comparison but not the Cartpole bar values.
- Figure 19's Cartpole state-machine formula is not exposed by text extraction, but the rendered PDF
  page is readable and has been manually transcribed as reference provenance.

## Implementation Mapping

- `src/cartpole_env.py`: continuous-force Cartpole with the train/test pole length and horizon split.
  It exposes a machine-readable reward spec for the standard OpenAI CartPole reward used by the
  paper's classic-control baselines: `+1` per survived simulator step and no extra terminal bonus or
  penalty. It also includes `PaperFigure19CartpolePSM`, a manual visual transcription of the
  CartPole policy diagram in paper Figure 19 for reference reevaluation and exact-policy comparison;
  this is not produced by the current synthesizer.
- `src/ppo_cartpole.py`: local PyTorch PPO implementation with MLP and LSTM policy classes.
- `src/train_cartpole_ppo.py`: CLI for PPO and PPO-LSTM experiments; with `--eval-interval`, it can
  persist per-evaluation train/test metrics to JSON for checkpoint provenance. Its default
  `--timesteps` value is the paper PPO budget `10^7`; local diagnostic commands override that
  default explicitly.
- `src/evaluate_cartpole_psm.py`: two-mode constant-action/depth-2-switch programmatic policy evaluator.
- `src/cartpole_direct_opt.py` and `src/train_cartpole_direct_opt.py`: bounded diagnostic Direct-Opt
  baseline over a two-mode constant-action Cartpole PSM, with a linear-switch grid plus explicit
  bounded depth-1/depth-2 Boolean-tree switch candidates that record one-hot feature, relation, and
  tree-operator metadata, plus a local batch/restart refinement seeded from the best candidate so far.
  Candidate selection optimizes mean train-horizon reward over the selected initial states, then
  success as a tie-breaker. This records exact search grids, search diagnostics, and selected program
  provenance, and can evaluate candidate pools with a configurable local thread count and optional
  wall-clock stop. It is still not the paper's full two-hour parallel direct optimization protocol. Direct-Opt metrics include
  `paper_protocol_status` flags for the paper batch size, ten-thread/two-hour budget, full continuous
  one-hot grammar, combined-reward optimization over all selected finite training states, full test horizon, and `1000`-rollout evaluation;
  the full Direct-Opt protocol flag remains false for the bounded diagnostic.
- `src/train_cartpole_psm.py`: CLI for synthesizing and evaluating the Cartpole programmatic state
  machine; it exposes the current teacher gain, teacher/student iteration, reward-scale,
  regularization, top-rho, and local-refinement settings, and can persist config, policy description,
  fixed local synthesis constants, probabilistic-student parameters, trace count, and train/test
  metrics to JSON. Its default evaluation rollout count is the paper's `1000`, and metrics record
  whether a run actually used that count. It also persists teacher candidate-source counts, loop-free segment action and
  duration schedules, latent responsibility confidence/entropy summaries, sampled-trace log-probability provenance, and switch-fit diagnostics comparing
  the selected switch objective tuple to a fixed local reference switch; this is failure-analysis
  provenance, not a controller selection rule. PSM metrics also include `paper_protocol_status`, a
  compact machine-readable block that records matched CartPole horizons and keeps the full
  probabilistic adaptive-teaching, full continuous switch-M-step, full CEM teacher optimizer, and
  paper-scale result flags false for the current bounded implementation. The metrics now include a
  compact `adaptive_teacher_summary` for each teacher/student iteration, recording the teacher
  sampling model, selected trace-source counts, reward summary, student log-probability coverage, and
  the recorded reward-plus-student-likelihood objective components when available. Each
  `synthesis_history` row also records `student_fit_history`, a compact trace of the inner
  action-likelihood initialization and switch-timing responsibility/refit passes that produced that
  iteration's probabilistic student.
- `src/cartpole_env.py::cartpole_space_spec`: records CartPole action/observation space provenance.
  The paper-derived claims are limited to Figure 8's `#A = 1` and `#O = 4` plus Appendix B.4's
  statement that RL baselines used the same action spaces, observation spaces, and set of initial
  states as the programmatic-policy approach. Force bounds, feature names, and the local
  independent-uniform reset range `[-0.05, 0.05]` are explicitly tagged as local implementation
  provenance, not as separately paper-specified numeric details.
- `src/cartpole_synthesis.py`: trace-based synthesis of a two-mode constant-action policy, plus a
  partial probabilistic Cartpole student with Gaussian action-parameter distributions and Boolean-tree
  switch candidates.
- `scripts/evaluate_cartpole_program.py`: reevaluates fixed two-mode CartPole programs and writes a
  `paper_protocol_status` block that keeps manual programs distinct from the current synthesis
  algorithm. With `--paper-figure19`, it reevaluates the manually transcribed Figure 19 reference
  policy while still marking `synthesized_by_current_algorithm = false`.
- `scripts/run_cartpole_reproduction.py`: orchestrated Cartpole runner that writes
  `cartpole_results.csv`, `cartpole_summary.csv`, and `cartpole_manifest.json` for selected seeds
  and settings. Its manifest records the evaluation rollout count, whether the run used the paper's
  `1000`-rollout metric, the PSM teacher overrides, and fixed local synthesis constants,
  plus a top-level `paper_protocol_status` block that records selected seed coverage, paper
  rollout/horizon coverage, PPO/Direct-Opt inclusion, and the still-false paper-scale result flag,
  and each PSM row links to a per-seed metrics JSON with the fitted probabilistic student and
  teacher-trace provenance plus a full selected-teacher-trace and per-iteration trace-history sidecar.
  When PPO is included, it also writes per-row PPO checkpoints and metrics
  JSON under the requested output directory; `--ppo-eval-interval` controls whether those metrics
  contain intermediate train/test `eval_history` entries or only the selected final result.
  PPO metrics also contain compact `update_history` rows with rollout reward means and
  train-horizon termination counts. PPO manifest rows mirror the metrics JSON `paper_protocol_status`
  block, so local diagnostic runs and single fixed-config runs are not mistaken for the full
  five-seed paper baseline protocol. Result rows, summaries, and metrics JSON explicitly record mean
  survived steps and survival seconds so long-horizon plots do not rely on reward as an implicit
  survival-time proxy.
- `scripts/evaluate_cartpole_checkpoint.py`: reevaluates existing PPO/PPO-LSTM checkpoints and writes
  a `paper_protocol_status` block that separates the checkpoint's original training/evaluation budget
  from the later full-horizon reevaluation budget, keeping paper-scale checkpoint-result claims false
  for short or warm-started local checkpoints. For warm-started checkpoints it also records whether
  the checkpoint config itself proves the pretraining teacher policy and PSM mode-update order.
- `scripts/run_cartpole_ppo_sweep.py`: PPO/PPO-LSTM hyperparameter sweep runner that defaults to 10
  uniformly sampled hyperparameter configs from the reported ranges per policy, evaluates each
  config for every selected seed, writes a plan/manifest that includes the concrete sampled
  hyperparameter configs, and can execute jobs with per-config checkpoints and metrics JSON. It also
  supports an explicit Cartesian-grid diagnostic mode and writes both a single-best-job summary and a
  per-hyperparameter summary aggregating completed seeds for each sampled config, including
  survived-step, survival-second, and evaluation-rollout
  provenance for executed rows. Its paper-scale plan/execution flags require the paper's
  `1000` evaluation rollouts and its manifest records the standard CartPole reward spec. This is
  search infrastructure; the full paper-scale sweep has not been run.
- `scripts/make_paper_figures.py`: figure/table generator that prefers grouped summary rows when
  available and falls back to raw per-seed result rows for older artifacts. It also writes the
  generated abstract-result, LaTeX table, PSM policy, and Figure 19 reference fragments consumed by `essay/project.tex`,
  plots the PSM switch-boundary figure from a linear-switch PSM metrics artifact when available, and
  plots PPO training curves when metrics JSON artifacts with `eval_history` are present. Its
  survival plot uses explicit survived-step fields when available and falls back to reward only for
  older artifacts.
- `artifacts/results/cartpole_summary.csv` and `artifacts/results/cartpole_manifest.json`: checked-in
  local diagnostic provenance for the current result bundle. The manifest records the command behind
  each metrics artifact and explicitly keeps `paper_scale_result` false. Its bundle-level
  `paper_protocol_status` records that the checked-in diagnostics use one seed, 20 evaluation
  rollouts, PPO/PPO-LSTM checkpoint reevaluations, a warm-started PPO-LSTM row, bounded Direct-Opt,
  and an incomplete synthesized PSM diagnostic rather than paper-scale reproduced results.

## Current Status

- Implemented and tested: Cartpole dynamics, train/test split, PPO MLP, PPO-LSTM, a bounded
  Direct-Opt diagnostic baseline, and Cartpole programmatic policy synthesis.
- Partially complete against the paper: the Cartpole programmatic policy is synthesized from
  model-based teacher traces into a two-mode constant-action/depth-2-switch policy. The student now
  fits Gaussian distributions over constant action parameters and latent mode responsibilities. Those
  responsibilities now start from action likelihoods and then alternate bounded forward-backward
  switch-timing refinements with action-distribution and switch-parameter refits inside each
  configured EM iteration, but the learner still approximates switch timing and does not implement
  the full probabilistic adaptive-teaching objective from the paper. The switch grammar now includes decision
  stumps plus depth-2 conjunction and disjunction candidates over observation inequalities via a
  bounded greedy leaf-expansion step. Switch threshold Gaussian means and standard deviations are locally refined
  against an elapsed-time-normalized Eq. (12)-style timing likelihood with a bounded grid initializer plus coordinate
  refinement, but the learner still does not fully optimize Eq. (12).
- Complete as a local diagnostic baseline: feed-forward PPO reaches 100% success on the paper's
  5-second training split.
- Not complete against the paper: PPO/PPO-LSTM have not been run for `10^7` timesteps or selected
  from the paper's 5-run/hyperparameter-search protocol. Pure PPO-LSTM is executable but did not
  solve the train split within the local diagnostic budget.

## Local Diagnostic Results

These are implementation diagnostics, not paper-scale reproduced results.

- Fixed PSM reevaluation command:
  `python scripts/evaluate_cartpole_program.py --theta-weight 10 --omega-weight 1 --threshold 0 --eval-rollouts 20 --test-max-steps 15000 --metrics-output artifacts/results/metrics/psm_seed0_fixed_program_full_horizon.json`
- Fixed PSM policy:
  `m0 action=-10.000; m1 action=10.000; mode=1 if 10.000*theta + 1.000*omega >= 0.000, else mode=0`
- Fixed PSM output:
  train success `1.000`, test success over the full 15000-step/300-second horizon `0.000`,
  train reward mean `250.0`, test reward mean `1560.55`; the same artifact records
  train/test survived-step means `250.0` and `1560.55`, or `5.0s` and `31.211s`.
  Its `paper_protocol_status` marks this as a fixed two-mode program reevaluation with
  `synthesized_by_current_algorithm = false` and `paper_scale_fixed_program_result = false`;
  it is not evidence that the current synthesis implementation reproduced the paper result.
- Paper Figure 19 reference policy, manually transcribed from rendered PDF page 21:
  start chooses `m1` when `omega >= 0.02` and `m2` when `omega < 0.02`; `m1` uses constant action
  `-3.3` and switches to `m2` when `omega >= 0.46 and theta >= -0.06`; `m2` uses constant action
  `3.98` and switches to `m1` when `omega < -0.49`. It can be reevaluated with
  `python scripts/evaluate_cartpole_program.py --paper-figure19 ...`, and its metrics mark
  `policy_source = paper_figure19_manual_transcription` and `synthesized_by_current_algorithm = false`.
- Current synthesizer diagnostic command:
  `python src/train_cartpole_psm.py --num-initial-states 4 --candidate-rollouts 8 --teacher-top-rho 2 --teacher-refinement-steps 1 --eval-rollouts 20 --test-max-steps 15000 --metrics-output artifacts/results/metrics/psm_seed0_full_horizon.json --traces-output artifacts/results/traces/psm_seed0_full_horizon_teacher_traces.json`
- Current synthesizer diagnostic output:
  train success `0.000`, test success over the full 15000-step/300-second horizon `0.000`,
  train reward mean `47.1`, test reward mean `60.05`; the same artifact records train/test
  survived-step means `47.1` and `60.05`, or `0.942s` and `1.201s`. The tracked artifact was
  regenerated with the full selected-teacher-trace sidecar, inner student fit history, fixed initial-mode likelihood, and
  `mode_update_order = act_with_current_mode_then_update_next_mode`. It uses the CartPole PSM loop-free teacher profile
  (`segment_steps = 1`, `segments_per_trace = 250`)
  so the teacher can span the full 250-step training horizon with one-step segments. Its metadata
  records `rollout_parameter_resampling = on_mode_entry`,
  `initial_mode_prior = fixed_mode_0`,
  `bootstrap_source = probabilistic_student_prior`, fitted teacher-gain sampling in the bounded
  elite-distribution refresh, first-iteration source counts
  `{"bootstrap_elite_centroid": 1, "bootstrap_student_sample_refined": 3}`,
  final-iteration source counts `{"student_sample": 3, "student_sample_refined": 1}`, and policy
  `m0 action=-0.018; m1 action=1.124; mode=1 if o[1] <= -0.334 or o[1] >= 0.713, else mode=0`; it also records
  `student_sample_segment_budget =
  preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood`.
  This remains a local synthesis diagnostic and still demonstrates a full-horizon programmatic-policy
  gap, not a paper-level reproduction result.
- Direct-Opt diagnostic command:
  `python src/train_cartpole_direct_opt.py --seed 0 --num-train-states 10 --random-candidates 256 --batch-size 10 --batch-refinement-rounds 1 --local-refinement-steps 2 --restart-candidates-on-stall 1 --eval-rollouts 20 --test-max-steps 15000 --metrics-output artifacts/results/metrics/direct_opt_seed0_full_horizon.json`
- Direct-Opt diagnostic output:
  train success `1.000`, test success over the full 15000-step/300-second horizon `0.100`,
  train reward mean `250.0`, test reward mean `4311.0`. The selected bounded two-mode policy is
  `m0 action=-10.000; m1 action=10.000; mode=1 if 1.000*theta + 0.250*omega >= 0.000, else mode=0`.
  This is an executable local baseline artifact, not the paper's full Direct-Opt protocol. The local
  implementation optimizes mean reward over all selected finite initial states and records bounded
  Boolean-tree switch-candidate one-hot metadata, Appendix B.3 continuous one-hot vertex fields, and
  batch/restart diagnostics to mirror part of the paper baseline's grammar and batch seeding
  structure. Candidate pools can now be evaluated with configurable local parallel threads and an
  optional wall-clock stop, and diagnostics record the selected thread count and time-limit status.
  Its diagnostics separate candidate evaluation calls from individual selected-state train rollout
  evaluations, while keeping `not_paper_scale` true.
- PPO MLP command:
  `python src/train_cartpole_ppo.py --policy mlp --timesteps 131072 --rollout-steps 128 --num-envs 8 --update-epochs 8 --minibatches 8 --learning-rate 0.0003 --entropy-coef 0.01 --initial-log-std -1 --seed 0 --eval-rollouts 20 --test-max-steps 1000 --eval-interval 16384 --verbose --output artifacts/progress_mlp_128k_seed0.pt`
- PPO MLP selected checkpoint:
  train success `1.000`, diagnostic 1000-step test success `0.350`,
  train reward mean `250.0`, diagnostic test reward mean `861.0`.
- PPO MLP full-horizon reevaluation:
  train success `1.000`, test success `0.000`, train reward mean `250.0`,
  test reward mean `910.6`.
- Pure PPO-LSTM diagnostics:
  with corrected recurrent state and MLP heads, `262144` timesteps did not solve the training split;
  best observed train success `0.000`, train reward mean `45.0`.
- PPO-LSTM warm-start diagnostic:
  supervised pretraining from the two-mode controller followed by PPO fine-tuning preserves train
  success `1.000`, but full-horizon test success remains `0.000`. The checked-in warm-start
  checkpoint predates explicit teacher-policy and teacher-order metadata, so its reevaluation status
  marks `checkpoint_pretrain_teacher_policy_status = missing_from_checkpoint_config` and
  `checkpoint_pretrain_teacher_mode_order_status = missing_from_checkpoint_config` rather than
  claiming the checkpoint proves the current `BangBangCartpolePSM` and
  `act_with_current_mode_then_update_next_mode` pretraining semantics.

The PPO diagnostics now verify that the feed-forward PPO baseline can solve the paper's training
split locally. They still do not reproduce the paper-scale PPO/PPO-LSTM protocol.

## Bugs Fixed During Audit

- PPO rollout collection originally reset only on pole/cart failure. It now truncates and resets at
  `env.cfg.max_steps = 250` for the paper's 5-second training horizon.
- PPO now stores raw sampled continuous actions for log-probability calculations and clips only the
  action sent to the environment.
- Vectorized rollouts were added so short local runs get more PPO updates with stable batch shapes.
- PPO now caps the final vectorized rollout so a configured timestep budget is not exceeded when
  `total_timesteps` is not divisible by `rollout_steps * num_envs`.
- LSTM PPO now preserves recurrent state across rollout chunks and replays the same initial state
  during the update.
- Test evaluation defaults now use `15000` steps, matching the paper's 300-second test horizon.
- Programmatic-state-machine synthesis can now write metrics JSON containing the synthesis config,
  policy description, fitted Gaussian action/switch distributions, latent responsibility summary,
  compact teacher-trace examples with segment-duration and time-increment schedules, per-teacher/student-iteration
  `synthesis_history`, compact adaptive-teacher objective summaries, number of teacher traces,
  optional full selected-teacher-trace/per-iteration trace-history sidecar paths, evaluation settings,
  switch-fit diagnostics, and train/test metrics.
- The Cartpole deterministic and probabilistic PSM executors, plus the local bang-bang evaluator and
  PPO warm-start teacher policy, now act with the current mode before applying the switch predicate to
  update the next mode, matching the paper's state-machine semantics `an = Hsn(on), s0 = ms,
  sn+1 = ...`; sampled teacher traces label the mode that produced each action.
- The Cartpole switch learner now performs bounded local grid, coordinate refinement, and
  finite-difference gradient polishing of selected switch-threshold Gaussian means and standard
  deviations against a discrete Eq. (12)-style
  likelihood, while using responsibility-weighted expected label loss over non-boundary segment
  observations as the primary structure/refinement label objective when soft EM responsibilities are
  available. This moves the switch M-step closer to the paper's latent-responsibility objective, but
  remains a diagnostic approximation: second-predicate Boolean-tree expansions and final switch
  structures are prefiltered by a cheaper hard-label/timing objective before bounded top-32
  distribution rescoring, depth-2 Boolean-tree probabilities use a
  shared-threshold rectangle-union calculation, and this is not the paper's full continuous
  switch-parameter optimizer.
- The first Cartpole teacher iteration now samples from an explicit probabilistic student prior, and
  later teacher candidate pools are sampled from the current probabilistic student before top-rho
  selection. These sampled rollouts now start from fixed mode `0` and resample action and switch
  parameters whenever execution enters a mode segment, matching the paper's probabilistic PSM
  execution model more closely. The
  teacher locally refines top sampled loop-free traces by duration/time-increment/action coordinate search under a
  top-rho elite-distance kernel approximation with action differences normalized by the larger sampled
  force magnitude, adds one bounded finite-difference teacher-gain
  candidate, one bounded finite-difference action candidate, one bounded finite-difference
  integer-duration candidate, and one bounded finite-difference time-increment candidate per
  refinement iteration with a short backtracking line search, evaluates one centroid recombination of the
  elite action/duration/time-increment schedules plus configurable bounded rounds that fit a Gaussian
  schedule distribution over teacher gains and per-segment actions, durations, and time increments
  from the current top-rho set. Each bounded round evaluates the fitted distribution mean, samples
  from that distribution, refreshes the top-rho set, and refits before the next round; local refinement
  then uses the refreshed top-rho set for its objective. The teacher also records
  selected trace sources plus sampled-trace log probabilities in metrics JSON. When a sampled
  closed-loop rollout is projected back into the loop-free teacher budget, its student likelihood is
  recomputed on the projected trace before teacher-objective ranking. This moves
  toward the sampled-teacher and local-optimization
  phases in Section 4.2, but it is not the paper's full CEM plus gradient-based trajectory optimizer.
- The Cartpole teacher regularizer now scores candidate traces with both Gaussian action likelihood
  and the student's discrete Eq. (12)-style switch timing likelihood, marginalizing over the latent
  mode sequence with a two-state forward pass. The bounded two-mode timing model now distinguishes
  selector-off to selector-on transitions from selector-on to selector-off transitions instead of using
  one symmetric "mode changed" probability. For scalar-threshold switches, that timing likelihood
  uses the learned Gaussian switch-parameter distribution with one sampled threshold shared across a
  segment, matching the paper's probabilistic-state-machine sampling model. Loop-free segment
  durations are interpreted as elapsed time normalized to the CartPole simulator step, so the
  teacher's per-segment time increments affect the bounded switch-timing likelihood. The final
  observed segment now contributes no-transition-before-duration evidence, so single-segment traces
  and final teacher segments are not scored only by action likelihood.
- The Cartpole student now initializes latent segment responsibilities from action likelihoods, then
  alternates the configured bounded forward-backward switch-timing passes with action-distribution and
  switch-parameter refits inside each EM iteration. The first segment of each trace is conditioned on
  the executable PSM's fixed initial mode `0` instead of a uniform latent start prior. The E-step pair potentials and bounded switch
  timing loss use directed 0-to-1 and 1-to-0 selector events plus final-segment stay evidence, closer
  to the transition and no-transition terms in Eq. (12). This moves Eq. (10) closer to the paper by
  using both `H` and `G` evidence throughout the bounded EM loop, but it remains a local bounded
  approximation rather than the paper's full EM/M-step optimizer.
- The loop-free Cartpole teacher now records its segment-duration schedule and locally refines one
  integer segment duration at a time during bounded coordinate search. It also records the
  corresponding constant-action sequence, duration-only refinement preserves that action sequence
  while varying durations, and bounded action refinement can take local continuous force steps for
  one constant-action segment at a time. This moves toward the paper's loop-free
  action-function-plus-duration teacher parameterization, but is not the continuous gradient-based
  action/duration optimization from Section 4.2.
- The Cartpole teacher objective now uses the paper-reported reward scale `lambda = 100` by default
  when trading off reward against student likelihood.
- The PSM training CLI now exposes the current configurable teacher/adaptive-teaching settings and
  records their exact values in metrics JSON.
- PSM metrics and reproduction-runner manifests now record fixed local synthesis constants such as
  default EM iterations, Gaussian floors, switch-timing scale, switch-search grids, and teacher-search
  refinement schedule. The actual configured student EM iterations and per-EM switch-responsibility
  passes are recorded under `config` and the compact PSM `paper_protocol_status` block, which also
  distinguishes matched CartPole environment settings from the still-missing full probabilistic
  adaptive-teaching optimizer and paper-scale result reproduction. These values document the current
  partial implementation; they are not claimed as paper-specified constants.
- PPO training runs can now write metrics JSON containing the full evaluation history, compact
  per-update rollout diagnostics, selected result, config, checkpoint-selection rule, and explicit
  survived-step/survival-second means for train/test evaluation.
- The orchestrated reproduction runner now persists PPO/PPO-LSTM checkpoints and metrics JSON for
  `--include-ppo` rows, tying those local diagnostic results to concrete artifacts. Runner rows and
  summaries also report survived steps and survival seconds explicitly.
- The orchestrated reproduction runner can include a bounded Direct-Opt diagnostic row through
  `--include-direct-opt`, writing a per-seed metrics JSON with the selected program, searched
  candidate count, exact search grids, bounded batch/restart diagnostics, and limitation note.
- The orchestrated reproduction runner now also writes per-seed PSM metrics JSON and links it from
  `cartpole_results.csv` and `cartpole_manifest.json`, so synthesized PSM rows are tied to concrete
  student/teacher-trace provenance artifacts, including per-teacher/student-iteration
  `synthesis_history` with local diagnostic train/test evaluations. The checked-in result table
  currently separates the fixed two-mode programmatic diagnostic from the current
  synthesized-student diagnostic because the current
  synthesizer does not reproduce the fixed-program row.
- `scripts/make_paper_figures.py` can turn those PPO metrics JSON files into
  `essay/figures/cartpole_ppo_training_curves.png`. Current smoke metrics are local diagnostics only,
  not paper-scale learning curves. It discovers standalone PPO metrics, reproduction-runner metrics
  under `artifacts/results/metrics/`, and PPO sweep metrics.
- `scripts/make_paper_figures.py` now writes abstract result claims from result artifacts, and parses
  linear Cartpole PSM switch boundaries from PSM metrics artifacts before writing
  `essay/cartpole_policy_fragment.tex` and plotting
  `programmatic_switch_boundary.png`; it writes an explicit fallback fragment and skips that figure
  when only non-linear/Boolean-tree switch descriptions are available instead of drawing a hard-coded
  boundary. It also writes `essay/cartpole_figure19_reference_fragment.tex` only from a metrics
  artifact whose protocol status marks `policy_source = paper_figure19_manual_transcription`, keeping
  the paper reference policy separate from synthesized local diagnostics. Generated result fragments now carry an explicit local-diagnostic limitation note and
  refuse rows whose explicit `test_horizon_steps` is not the paper 300-second horizon. Synthesized
  PSM rows are also rejected unless their full-trace sidecar contains per-iteration trace history
  whose recorded trace counts match the serialized trace lists.
- PPO hyperparameter search can now be planned or executed through
  `scripts/run_cartpole_ppo_sweep.py`; the runner records the paper search ranges, reproducible
  `paper-random` hyperparameter sample IDs, the concrete sampled hyperparameter configs, and the
  uncapped job count for the selected search space in a manifest. It writes machine-readable
  `paper_protocol_status` flags to distinguish full
  paper-scale plans from quick/truncated/grid-diagnostic or dry-run diagnostics and to mark
  paper-scale execution only when all planned jobs complete with zero failures. The status block
  records selected and distinct seed/policy lists so duplicate entries cannot masquerade as the
  paper's five-seed/two-baseline protocol. The full-plan flag
  now requires `paper-random` mode with 10 samples per policy, five seeds, both PPO MLP and PPO-LSTM,
  `10^7` timesteps, and the full 15,000-step/300-second test horizon. Grid mode remains available as
  a local diagnostic extension. Paper-scale execution additionally requires the planned job count to
  match the uncapped selected search space. The runner writes a best-config summary for completed
  jobs, can resume interrupted sweeps by reusing only matching completed rows with existing checkpoint
  and metrics artifacts, writes a per-hyperparameter summary that marks the best completed sampled
  config per policy after preferring complete selected-seed coverage, records selected-seed coverage
  and missing seeds for each sampled hyperparameter config, records survived-step/survival-second
  columns for executed rows and summaries, and can optionally record failed jobs while continuing a long sweep.

## Verified PPO Invariants

These checks are unit-level correctness guards for the local PPO implementation; they do not replace
paper-scale PPO2 runs.

- `tests/test_cartpole_paper.py::test_ppo_rollout_truncates_at_paper_training_horizon` verifies that
  rollout collection treats the 5-second/250-step training horizon as terminal and resets the vector
  environment counter.
- `tests/test_cartpole_paper.py::test_ppo_training_does_not_exceed_configured_timestep_budget` and
  `tests/test_cartpole_paper.py::test_ppo_metrics_record_partial_final_rollout` verify exact
  configured timestep accounting for vectorized PPO rollouts, including a one-transition final update.
- `tests/test_cartpole_paper.py::test_cartpole_reward_matches_openai_classic_control_step_reward`
  verifies that the local CartPole environment returns the paper's standard classic-control reward:
  `+1` per survived simulator step, with no terminal bonus or penalty.
- `tests/test_cartpole_paper.py::test_ppo_stores_raw_continuous_actions_for_log_probs` verifies that
  PPO stores the raw sampled Gaussian action for log-probability replay while clipping only the force
  applied to the continuous Cartpole environment.
- `tests/test_cartpole_paper.py::test_lstm_update_replays_rollout_initial_state` verifies that the
  PPO-LSTM update replays the rollout's stored initial recurrent state instead of silently starting
  updates from zeros.
- `tests/test_cartpole_paper.py::test_ppo_writes_eval_history_metrics_json` verifies that PPO
  interval evaluations and per-update rollout diagnostics are persisted to JSON instead of existing
  only in stdout.
- `tests/test_cartpole_paper.py::test_ppo_protocol_status_distinguishes_single_run_from_full_baseline`
  verifies that PPO metrics can mark a single configured run as matching the paper timestep/test
  horizon budget while still marking the full five-seed paper baseline protocol as incomplete.
- `tests/test_evaluate_cartpole_checkpoint.py::test_checkpoint_reevaluation_protocol_status_distinguishes_checkpoint_from_reeval`
  verifies that checkpoint reevaluation metrics do not conflate the checkpoint's original diagnostic
  eval horizon with a later full-horizon reevaluation.
- `tests/test_cartpole_paper.py::test_ppo_config_defaults_to_paper_timestep_budget` and
  `tests/test_cartpole_ppo_cli.py::test_cli_defaults_to_paper_timestep_budget_without_running`
  verify that the standalone PPO config and CLI default to the paper's `10^7` timestep budget
  without making smoke tests execute that budget.
- `tests/test_cartpole_paper.py::test_cartpole_space_spec_records_action_observation_and_reset_contract`
  verifies the CartPole action/observation dimensions and local reset bounds, including provenance
  tags that prevent the local numeric reset distribution from being overclaimed as paper-specified.
- `tests/test_cartpole_psm_cli.py::test_cli_writes_metrics_json` verifies that synthesized
  programmatic-policy metrics are persisted to JSON and that the file records the full paper test
  horizon even when a quick test cap is supplied. It also verifies that exposed teacher
  hyperparameters, fixed local synthesis constants, the fitted probabilistic student summary, and
  per-iteration bounded teacher-trace provenance are persisted, plus PSM protocol-status flags that
  keep the full probabilistic adaptive-teaching and paper-scale result claims false. The metrics also
  record the paper's `1000` evaluation-rollout target and whether the current run used it.
- `tests/test_cartpole_ppo_sweep.py::test_build_jobs_uses_paper_minibatch_rule_for_lstm` verifies
  that the sweep includes the paper's feed-forward minibatch grid while forcing PPO-LSTM to
  `nminibatches = 1` in grid-diagnostic mode.
- `tests/test_cartpole_ppo_sweep.py::test_build_jobs_defaults_to_paper_random_hyperparameter_samples`
  verifies that default sweep planning uses 10 reproducible paper-random hyperparameter samples per
  policy, evaluates them for each selected seed, and keeps PPO-LSTM minibatches fixed to one.
- `tests/test_cartpole_ppo_sweep.py::test_sampled_hyperparameter_manifest_records_each_policy_config_once`
  verifies that sampled PPO sweep manifest entries record each policy-level hyperparameter config
  once and match every per-seed job generated from that config.
- `tests/test_cartpole_ppo_sweep.py::test_dry_run_writes_plan_and_manifest` verifies that dry-run
  sweep planning writes a CSV plan and manifest with paper-space metadata.
- `tests/test_cartpole_ppo_sweep.py::test_summarize_hyperparameter_configs_aggregates_completed_seeds`
  verifies that completed PPO sweep rows are grouped by policy and sampled hyperparameter config,
  with seed-level mean/std metrics and a best-config flag.
- `tests/test_cartpole_ppo_sweep.py::test_summarize_hyperparameter_configs_prefers_complete_seed_coverage`
  verifies that a partial-seed hyperparameter result is not selected over a complete selected-seed
  result only because its incomplete mean training score is higher.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_identifies_full_dry_run_plan`
  verifies that the manifest status flags distinguish a full paper-scale dry-run plan from completed
  paper-scale execution.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_grid_mode_as_paper_scale_plan`
  verifies that the explicit Cartesian-grid diagnostic is not tagged as the paper's sampled
  hyperparameter protocol.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_requires_ten_random_hyperparameter_samples`
  verifies that the paper-scale plan flag requires the paper's 10 random hyperparameter samples.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_requires_full_test_horizon`
  verifies that the paper-scale plan flag requires the paper's full 15,000-step/300-second test
  horizon.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_requires_1000_eval_rollouts`
  verifies that the paper-scale plan flag requires the paper's 1000-rollout evaluation metric.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_duplicate_seed_list`
  verifies that five repeated seed entries are not tagged as the paper five-seed protocol.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_requires_completed_jobs_for_execution`
  verifies that paper-scale execution is not marked true unless the planned job count matches the
  uncapped selected search space, all planned jobs completed, and no job failed.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_empty_learning_rate_list`
  verifies that an empty learning-rate list is not tagged as a paper-scale plan.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_reduced_learning_rate_grid`
  verifies that a smaller non-empty learning-rate subset is not tagged as the runner's full plan.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_partial_policy_set`
  verifies that a PPO MLP-only plan is not tagged as the full paper baseline protocol.
- `tests/test_cartpole_ppo_sweep.py::test_paper_protocol_status_rejects_duplicate_policy_entries`
  verifies that duplicate policy entries are not tagged as the full two-baseline protocol.
- `tests/test_cartpole_ppo_sweep.py::test_summarize_results_selects_best_train_per_policy` verifies
  the sweep summary selection rule.
- `tests/test_cartpole_ppo_sweep.py::test_quick_execution_writes_results_summary_and_manifest`
  verifies that quick sweep execution writes results, summary, and manifest artifacts.
- `tests/test_cartpole_ppo_sweep.py::test_resume_skips_matching_completed_jobs_with_artifacts`
  verifies that resumable sweep execution skips a completed matching job and records skipped/run
  counts in the manifest.
- `tests/test_cartpole_ppo_sweep.py::test_resume_rejects_rows_without_artifacts` verifies that resume
  does not trust result rows unless their checkpoint and metrics artifacts still exist.
- `tests/test_cartpole_ppo_sweep.py::test_continue_on_error_records_failed_jobs` verifies that
  opt-in sweep continuation records failed jobs to a failure artifact and manifest counters.
- `tests/test_cartpole_ppo_sweep.py::test_default_job_failure_stops_sweep` verifies that job failures
  still stop the sweep by default.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_returns_policy_and_provenance` verifies that
  the bounded Direct-Opt diagnostic baseline selects a Cartpole PSM and records explicit
  non-paper-scale provenance, including local batch/restart diagnostics, Appendix B.3 one-hot vertex
  metadata, candidate-call versus train-rollout accounting, configurable parallel-candidate
  evaluation/time-limit metadata, and Direct-Opt protocol-status flags.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_protocol_status_marks_quick_diagnostic_limits`
  verifies that quick Direct-Opt diagnostics do not claim the paper batch size, full test horizon,
  `1000`-rollout metric, restart/batch optimization, or paper-scale Direct-Opt protocol.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_can_disable_batch_refinement_for_grid_random_diagnostic`
  verifies that the grid/random/Boolean-tree diagnostic can still be isolated when batch refinement is
  disabled.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_boolean_tree_candidates_use_cartpole_switch_grammar`
  verifies that the Direct-Opt diagnostic evaluates serializable depth-2 Boolean-tree switch
  candidates from the Cartpole switch grammar and records bounded one-hot feature/relation/operator
  metadata for those candidates.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_cli_writes_metrics_json` verifies that the
  Direct-Opt CLI writes config, selected candidate, train/test metrics, search diagnostics, and
  provenance JSON.

## Verified Programmatic-Student Invariants

These checks cover the partial probabilistic Cartpole student, not the complete paper algorithm.

- `tests/test_cartpole_paper.py::test_cartpole_probabilistic_student_uses_gaussian_modes` verifies
  that Cartpole student fitting produces two Gaussian constant-action distributions, positive standard
  deviations, Gaussian switch-parameter distributions, and normalized latent mode responsibilities
  over loop-free teacher segments.
- `tests/test_cartpole_paper.py::test_cartpole_responsibility_refinement_uses_switch_timing` verifies
  that switch-timing likelihood can shift ambiguous latent segment responsibilities away from the
  action-only posterior while preserving normalization.
- `tests/test_cartpole_paper.py::test_cartpole_initial_segment_responsibility_is_fixed_to_mode_zero`
  verifies that the first segment of each trace is conditioned on the CartPole PSM's fixed initial
  mode rather than a free latent start mode.
- `tests/test_cartpole_paper.py::test_cartpole_switch_timing_responsibilities_are_directed_by_next_mode`
  verifies that the bounded two-mode responsibility likelihood distinguishes selector-off to
  selector-on transitions from selector-on to selector-off transitions.
- `tests/test_cartpole_paper.py::test_cartpole_student_switch_responsibility_passes_are_configurable`
  verifies that the configured number of switch-timing responsibility passes changes the fitted
  student responsibilities and action distributions.
- `tests/test_cartpole_paper.py::test_cartpole_student_alternates_switch_responsibility_passes_per_em_iteration`
  verifies that switch-timing responsibility refinements are applied inside each configured EM
  iteration rather than only after action-only EM has completed.
- `tests/test_cartpole_paper.py::test_cartpole_student_fit_history_records_inner_em_steps`
  verifies that the fitted Cartpole student can expose a compact per-EM/pass training history whose
  final row matches the returned probabilistic student.
- `tests/test_cartpole_paper.py::test_cartpole_synthesis_can_return_probabilistic_student` verifies
  that synthesis can expose the fitted probabilistic student directly for metrics/provenance without
  re-fitting from traces.
- `tests/test_cartpole_paper.py::test_cartpole_probabilistic_student_projects_to_policy` verifies
  that the probabilistic student can be projected to a deterministic two-mode Cartpole policy for
  train/test evaluation.
- `tests/test_cartpole_paper.py::test_cartpole_switch_timing_loss_prefers_segment_boundary`
  verifies that the Cartpole switch candidate scoring penalizes switches that fire before the observed
  loop-free segment boundary. This is a narrow guard for the paper's switch-duration likelihood
  objective, not the full Eq. (12) implementation.
- `tests/test_cartpole_paper.py::test_cartpole_eq12_likelihood_rewards_transition_at_duration`
  verifies the transition-at-duration term in the discrete Eq. (12)-style switch likelihood.
- `tests/test_cartpole_paper.py::test_cartpole_eq12_likelihood_uses_elapsed_time_increment_duration`
  verifies that the discrete Eq. (12)-style switch likelihood uses loop-free segment elapsed time,
  normalized to the CartPole simulator step, rather than only raw simulator-step counts.
- `tests/test_cartpole_paper.py::test_cartpole_eq12_likelihood_penalizes_early_transition_when_staying`
  verifies the no-transition-before-duration term in the discrete Eq. (12)-style switch likelihood.
- `tests/test_cartpole_paper.py::test_cartpole_switch_timing_loss_penalizes_final_segment_early_transition`,
  `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_timing_loss_penalizes_final_segment_early_transition`,
  and `tests/test_cartpole_paper.py::test_cartpole_trace_log_probability_penalizes_final_segment_early_transition`
  verify that final observed segments contribute no-transition-before-duration evidence to the
  deterministic timing comparator, Gaussian switch-parameter timing loss, and teacher trace
  log-probability.
- `tests/test_cartpole_paper.py::test_cartpole_eq12_likelihood_is_directed_for_selector_off_transition`
  verifies that the deterministic Eq. (12)-style timing approximation scores selector-on to
  selector-off transitions separately from selector-off to selector-on transitions.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_improves_timing_likelihood`
  verifies that bounded switch-threshold refinement can improve the current timing likelihood
  without increasing hard segment-label mistakes.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_timing_loss_uses_elapsed_duration`
  verifies that the Gaussian switch-parameter timing loss uses elapsed loop-free segment duration.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_can_improve_probabilistic_std`
  verifies that bounded Gaussian standard-deviation refinement can improve the current
  probabilistic timing objective.
- `tests/test_cartpole_paper.py::test_cartpole_switch_coordinate_refinement_polishes_grid_solution`
  verifies that the bounded coordinate pass can improve beyond the discrete std-candidate grid when
  finite-difference gradient polishing is disabled.
- `tests/test_cartpole_paper.py::test_cartpole_switch_gradient_refinement_polishes_coordinate_solution`
  verifies that finite-difference gradient polishing can improve the Eq. (12)-style timing objective
  beyond the bounded coordinate pass.
- `tests/test_cartpole_paper.py::test_cartpole_switch_mistake_cache_key_preserves_submillithresholds`
  verifies that hard-label mistake caching keys exact switch thresholds rather than rounded
  descriptions, so tiny gradient-polish moves cannot bypass the mistake guard.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_keeps_std_finite`
  verifies that refined switch Gaussian standard deviations remain finite and above the local
  Gaussian floor.
- `tests/test_cartpole_paper.py::test_cartpole_switch_std_refinement_uses_boundary_variance_candidate`
  verifies that the bounded std grid includes transition-boundary variance evidence.
- `tests/test_cartpole_paper.py::test_cartpole_switch_parameter_refinement_rejects_more_label_mistakes`
  verifies that probabilistic timing refinement does not accept a switch mean that worsens the
  current responsibility-weighted label objective.
- `tests/test_cartpole_paper.py::test_cartpole_switch_structure_cost_uses_soft_responsibility_label_loss`
  verifies that the switch structure objective uses soft EM responsibilities even when hard trace
  labels favor a different selector.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_timing_loss_rejects_responsibility_mismatch`
  verifies that malformed segment/responsibility inputs do not silently produce a zero timing loss.
- `tests/test_cartpole_paper.py::test_cartpole_switch_probability_uses_gaussian_threshold_distribution`
  verifies that switch-enable probabilities are computed from Gaussian threshold distributions.
- `tests/test_cartpole_paper.py::test_cartpole_switch_transition_probability_uses_shared_threshold_sample`
  verifies that scalar switch timing probability treats the same sampled threshold as shared over the
  segment rather than resampling independently at each simulator step.
- `tests/test_cartpole_paper.py::test_cartpole_deterministic_psm_acts_before_mode_transition`,
  `tests/test_cartpole_paper.py::test_cartpole_probabilistic_rollout_acts_before_detected_mode_transition`,
  `tests/test_cartpole_paper.py::test_cartpole_student_sampled_trace_labels_action_mode_before_transition`,
  and `tests/test_cartpole_paper.py::test_bangbang_cartpole_psm_acts_before_mode_transition`
  verify that Cartpole PSM execution uses the current mode's action before updating the next mode,
  and that sampled teacher traces label the mode that produced each action.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_cumulative_probability_matches_prefix_union`
  verifies that depth-2 Boolean-tree cumulative switch probabilities match the independent
  shared-threshold rectangle-union semantics used by the bounded Eq. (12)-style timing model.
- `tests/test_cartpole_paper.py::test_cartpole_sampled_depth2_switch_preserves_predicate_count`
  verifies that sampling a depth-2 Boolean-tree switch preserves both learned predicates.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_objective_defaults_to_reward` verifies that
  the Cartpole teacher objective uses the paper-reported `lambda = 100` reward scale and reduces to
  reward-only candidate selection when no previous student exists.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_reward_lambda_is_configurable` verifies that
  the reward scale is explicit in the Cartpole synthesis config.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_objective_uses_student_regularizer` verifies
  that, once a previous student exists, the teacher objective can prefer a lower-reward loop-free trace
  that has higher probability under the student's Gaussian action distributions.
- `tests/test_cartpole_paper.py::test_cartpole_trace_log_probability_uses_fixed_initial_mode`
  verifies that the teacher regularizer marginalizes over latent segment modes after conditioning the
  first segment on the fixed initial mode instead of treating posterior responsibilities as extra priors.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_regularizer_uses_switch_timing_likelihood`
  verifies that the teacher regularizer also prefers traces with switch timing that the current
  student explains better. These regularizer tests cover a partial implementation of the probability
  regularizer in Eq. (8), not the paper's full CEM plus
  gradient-based trajectory optimizer.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_regularizer_uses_switch_distribution_uncertainty`
  verifies that the teacher regularizer uses switch-distribution uncertainty when scoring a trace.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distance_matches_loop_free_parameters`
  verifies that the top-rho refinement distance compares loop-free action and duration parameters.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_kernel_uses_normalized_top_rho_distance`
  verifies that the refinement objective uses the paper-style normalized elite-distance kernel
  approximation for trace probability.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distance_includes_teacher_gains`
  verifies that the elite-distance kernel includes loop-free teacher gains as well as segment
  action, duration, and time-increment schedules.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distance_normalizes_actions`
  verifies that action differences are scaled by the compared traces' maximum absolute segment
  action before contributing to the loop-free elite distance.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_centroid_recombines_loop_free_schedules`
  verifies that the teacher can evaluate one deterministic top-rho centroid of segment actions and
  durations before local refinement. This is only a bounded recombination approximation, not the
  paper's full CEM distribution update.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_sample_uses_top_rho_statistics`
  verifies that the teacher can sample one loop-free candidate from per-segment action/duration
  statistics fit to the top-rho traces. This is a bounded CEM-like sample, not an iterative CEM
  optimizer.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_sample_fits_gain_statistics`
  verifies that the bounded elite-distribution sample also fits and samples teacher-gain
  statistics from the top-rho traces.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_mean_uses_fitted_statistics`
  verifies that the teacher can evaluate the deterministic per-segment fitted-distribution mean
  candidate before local refinement.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_fits_schedule_parameters`
  verifies that the bounded CEM-style step fits a reusable schedule distribution over teacher gains,
  per-segment actions, durations, time increments, and majority modes.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_resample_count_is_configurable`
  verifies that the bounded elite distribution sample count is configured rather than hard-coded.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_rounds_refresh_elites`
  verifies that bounded elite distribution rounds refresh the top-rho set between sampling rounds.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_elite_distribution_rounds_refit_distribution`
  verifies that each bounded distribution round samples from a distribution refit on the refreshed
  top-rho set, rather than repeatedly sampling around the original elites.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_refinement_uses_refreshed_distribution_elites`
  verifies that local refinement uses the refreshed top-rho elite set produced by the bounded
  distribution rounds.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_can_sample_candidates_from_probabilistic_student`
  verifies that the bounded teacher candidate pool can include rollouts sampled from the current
  probabilistic student after the first student fit.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_bootstrap_uses_probabilistic_student_prior`
  verifies that the first teacher iteration samples from an explicit Gaussian PSM prior and records
  sampled-trace log probabilities.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_candidate_pool_uses_student_samples_after_first_iteration`
  verifies that the teacher candidate pool is sampled from the current probabilistic student after the
  first student fit.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_optimization_bootstrap_returns_prior_sample`
  verifies that first-iteration optimized teacher traces retain bootstrap-sampling provenance.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_optimization_records_selected_refinement_objective`
  verifies that the selected loop-free teacher trace records both the Eq. (8)-style teacher objective
  and the bounded top-rho refinement objective used for selection.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_can_refine_student_sampled_trace` verifies
  that a sampled-student loop-free trace can be locally refined without reducing the current
  elite-kernel refinement objective.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_refinement_does_not_reduce_objective`
  verifies that local refinement of Cartpole loop-free teacher gains does not reduce the teacher
  objective after top-candidate sampling. This is a bounded coordinate refinement over the diagnostic
  teacher gains, not the paper's continuous gradient-based trajectory optimizer.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_gain_gradient_uses_central_differences`
  verifies that bounded teacher-gain refinement can estimate a central finite-difference direction.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_gain_gradient_backtracks_to_improving_step`
  verifies that teacher-gain finite-difference refinement can backtrack to a smaller improving step
  when the full normalized step is worse.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_gain_gradient_refinement_can_be_accepted`
  verifies that the finite-difference gain update is only accepted when it does not reduce the
  current teacher objective.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_rollout_records_segment_durations` verifies
  that loop-free teacher traces persist the segment-action, segment-duration, and time-increment
  schedules used to generate them.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_rollout_uses_segment_time_increments`
  verifies that the loop-free teacher can vary per-segment integration increments without changing
  the global CartPole environment timestep.
- `tests/test_cartpole_paper.py::test_cartpole_student_segments_use_elapsed_time_increment_duration`
  verifies that teacher trace segments expose normalized elapsed duration to the student timing
  likelihood while preserving raw simulator-step duration for schedule accounting.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_refinement_preserves_action_sequence`
  verifies that duration-only refinement preserves the loop-free teacher's constant-action sequence.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_refinement_changes_one_action_at_a_time`
  verifies that bounded action refinement mutates one loop-free action-function segment at a time.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_refinement_uses_continuous_local_steps`
  verifies that bounded action refinement proposes local continuous force steps rather than only
  bang-bang action flips.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_gradient_uses_central_differences`
  verifies that bounded action-schedule refinement can estimate a central finite-difference direction.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_gradient_backtracks_to_improving_step`
  verifies that action-schedule finite-difference refinement can backtrack to a smaller improving
  step when the full normalized step is worse.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_gradient_refinement_can_be_accepted`
  verifies that the finite-difference action update is only accepted when it does not reduce the
  current teacher objective. This remains a bounded local approximation, not the paper's full
  gradient trajectory optimizer.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_gradient_uses_central_differences`
  verifies that bounded duration-schedule refinement can estimate a central finite-difference
  direction over integer segment durations.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_gradient_backtracking_rejects_worse_integer_step`
  verifies that duration finite-difference refinement rejects backtracked integer candidates that do
  not improve the current teacher objective.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_gradient_refinement_can_be_accepted`
  verifies that the finite-difference duration update is only accepted when it does not reduce the
  current teacher objective.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_time_increment_gradient_uses_central_differences`
  verifies that bounded time-increment refinement can estimate a central finite-difference
  direction over loop-free segment time increments.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_time_increment_gradient_backtracks_to_improving_step`
  verifies that time-increment finite-difference refinement can backtrack to a smaller improving step
  when the full normalized step is worse.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_finite_difference_refinement_rejects_worse_candidates`
  verifies that worse action and duration finite-difference candidates are rejected.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_refinement_does_not_reduce_objective`
  verifies that bounded local segment-duration refinement can be searched without reducing the
  teacher objective.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_switch_supports_depth_two_conjunction`
  verifies that the Cartpole switch representation supports depth-2 Boolean-tree conjunctions over
  observation inequalities.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_switch_supports_depth_two_disjunction`
  verifies that the Cartpole switch representation also supports depth-2 disjunctions from the
  paper's decision-tree view.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_candidates_include_depth_two` verifies
  that Cartpole switch candidate generation includes depth-2 Boolean-tree candidates, not only linear
  thresholds or single predicates.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_candidates_include_disjunction` verifies
  that the bounded greedy switch search can expand the no-switch leaf into an OR-style candidate.
- `tests/test_cartpole_paper.py::test_cartpole_greedy_boolean_tree_expansion_improves_stump`
  verifies that the greedy Boolean-tree expansion can choose a depth-2 switch when it improves over
  the best depth-1 stump.
- `tests/test_cartpole_paper.py::test_cartpole_switch_prefilter_caps_tied_candidates_deterministically`
  verifies that tied switch-structure candidates are capped to a deterministic top-32 subset before
  bounded distribution rescoring.
- `tests/test_cartpole_paper.py::test_cartpole_switch_structure_rescore_candidates_caps_distribution_scoring`
  verifies that final switch-structure rescoring caps expensive distribution-objective fits to the
  bounded top-32 subset.
- `tests/test_cartpole_paper.py::test_cartpole_best_switch_reuses_rescore_cache` verifies that
  final switch selection reuses the structure-objective cache populated during bounded rescoring.
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_switch_has_gaussian_parameter_per_predicate`
  verifies that a depth-2 Boolean-tree switch gets one Gaussian threshold distribution per predicate.
- `tests/test_cartpole_paper.py::test_cartpole_sampled_switch_uses_gaussian_thresholds` verifies
  that Boolean-tree switch thresholds can be sampled from Gaussian parameter distributions.
- `tests/test_cartpole_paper.py::test_cartpole_probabilistic_student_samples_policy_parameters`
  verifies that a probabilistic Cartpole student can sample a deterministic policy with sampled
  action and switch parameters.
- `tests/test_cartpole_psm_cli.py::test_summarize_student_reports_responsibility_confidence`
  verifies that PSM metrics expose hard latent-mode counts, ambiguous segment count, max
  responsibility, and responsibility entropy for the fitted probabilistic student.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_writes_results_and_manifest`
  verifies that the reproduction runner writes raw results, grouped summary statistics, and a manifest
  with the exact quick-run command settings, PSM teacher overrides, fixed PSM synthesis constants, and
  a per-seed PSM metrics JSON artifact whose final per-iteration evaluation matches the top-level PSM
  row. It also verifies the PSM full-trace sidecar path, per-iteration trace-history serialization,
  trace-count consistency, and the manifest-level paper-protocol status block.
- `tests/test_cartpole_reproduction_runner.py::test_reproduction_protocol_status_keeps_fixed_config_runs_non_paper_scale`
  verifies that a five-seed, full-horizon, 1000-rollout fixed-config runner setup is still not tagged
  as paper-scale because it lacks PPO hyperparameter search, full probabilistic adaptive teaching, and
  the full Direct-Opt protocol.
- `tests/test_cartpole_reproduction_runner.py::test_reproduction_protocol_status_rejects_duplicate_seed_coverage`
  verifies that repeated seed entries are not treated as the paper's five distinct seeds.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_can_include_direct_opt_diagnostic`
  verifies that `--include-direct-opt` adds the bounded Direct-Opt diagnostic row and links its
  metrics artifact from the manifest.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_with_ppo_writes_checkpoints_and_metrics`
  verifies that the reproduction runner writes PPO/PPO-LSTM checkpoints and metrics JSON, that the
  configured PPO evaluation interval produces `eval_history` entries, that PPO update diagnostics are
  persisted, and that each PPO manifest row mirrors the metrics JSON paper-protocol status.
- `tests/test_cartpole_reproduction_runner.py::test_summary_rows_report_mean_std_and_best_train_seed`
  verifies the runner's per-policy mean/std summary and deterministic best-training-seed selection.
- `tests/test_make_paper_figures.py` verifies that figure/table generation reads grouped summary rows
  when present, falls back to raw result rows otherwise, and writes generated abstract-result and
  LaTeX result-table fragments. It also verifies PSM policy-fragment generation,
  switch-boundary parsing/plotting from synthetic metrics, fallback/skip behavior for non-linear
  switches, Figure 19 reference-fragment generation only from manual-reference metrics provenance,
  PPO metrics-file discovery, training-curve PNG generation, local-diagnostic limitation notes,
  checked-in summary/manifest provenance, rejection of explicit non-paper test horizons, and rejection
  of paper-scale result rows that do not use the paper's 1000 evaluation rollouts.

## Completion Criteria Still Required For Full Paper Claim

- Run PPO and PPO-LSTM for `10^7` timesteps.
- Evaluate reported success rates over `1000` rollouts.
- Tune/fix pure PPO-LSTM until it achieves strong Cartpole training performance without supervised
  warm-starting.
- Run 5 random seeds and choose best training performer.
- Run hyperparameter search over the paper's specified ranges.
- Replace the bounded Direct-Opt diagnostic with the paper's full direct optimization protocol:
  keep the combined-reward-over-selected-initial-states objective, but replace the bounded local search with
  the full batch optimization, random restarts when stalled, full continuous one-hot
  switching-condition encoding, and reported two-hour/parallel budget.
- Complete the probabilistic adaptive-teaching implementation: continuous optimization of switch
  Gaussian parameters and the paper's full teacher optimization procedure. The current Cartpole switch
  learner performs a depth-2 greedy Boolean-tree expansion, stores Gaussian threshold distributions
  for each selected switch predicate, can sample deterministic policies from those distributions, and
  scores timing with a discrete approximation to Eq. (12), including transition-at-duration and
  no-transition-before-duration terms. It now performs bounded local mean/std grid, coordinate
  refinement, and finite-difference gradient polishing with backtracking line search, but does not yet
  solve the full continuous Eq. (12) optimization for switch-condition means and standard deviations.
  For depth-2 Boolean trees,
  switch-enable probability now uses an
  exact union of axis-aligned threshold rectangles for conjunctions and disjunctions under
  independent predicate-threshold Gaussians, with threshold samples shared across the segment. The
  current Cartpole teacher samples the first iteration from a Gaussian PSM prior and later iterations
  from the current probabilistic student with fixed initial mode `0`, resampling action/switch parameters on mode entry, refines
  top loop-free candidates with bounded coordinate search over teacher gains when available, integer
  segment durations, per-segment time increments, one-segment local continuous constant-action
  steps, and one teacher-gain plus one action plus one integer-duration plus one time-increment
  finite-difference candidate per refinement iteration with short backtracking line search, evaluates one deterministic top-rho centroid recombination
  candidate plus configurable bounded rounds that refit a Gaussian schedule distribution over teacher
  gains and per-segment action, duration, and time-increment parameters from the refreshed top-rho set
  before drawing the next mean/sample candidates, and scores traces with reward plus Gaussian action likelihood and discrete switch timing
  likelihood under the
  previous student, with the elite-distance kernel including teacher gains plus normalized segment action,
  duration, and time-increment schedules, but it does not yet perform the paper's full CEM procedure or continuous
  gradient-based optimization over loop-free action functions and durations.
