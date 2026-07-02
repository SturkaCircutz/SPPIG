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

## Not Verified From Extracted Text

- Exact numerical Cartpole train/test success rates from Figure 4. The PDF text extraction exposes the
  graphical comparison but not the Cartpole bar values.
- Exact synthesized Cartpole state-machine formula from Figure 19. The extracted text names Figure 19
  but does not expose its switch predicates or constants.

## Implementation Mapping

- `src/cartpole_env.py`: continuous-force Cartpole with the train/test pole length and horizon split.
- `src/ppo_cartpole.py`: local PyTorch PPO implementation with MLP and LSTM policy classes.
- `src/train_cartpole_ppo.py`: CLI for PPO and PPO-LSTM experiments; with `--eval-interval`, it can
  persist per-evaluation train/test metrics to JSON for checkpoint provenance.
- `src/evaluate_cartpole_psm.py`: two-mode constant-action/depth-2-switch programmatic policy evaluator.
- `src/cartpole_direct_opt.py` and `src/train_cartpole_direct_opt.py`: bounded diagnostic Direct-Opt
  baseline over a two-mode constant-action/depth-2 linear-switch Cartpole PSM. This records exact
  search grids and selected program provenance, but is not the paper's full two-hour parallel direct
  optimization protocol.
- `src/train_cartpole_psm.py`: CLI for synthesizing and evaluating the Cartpole programmatic state
  machine; it exposes the current teacher gain, teacher/student iteration, reward-scale,
  regularization, top-rho, and local-refinement settings, and can persist config, policy description,
  fixed local synthesis constants, probabilistic-student parameters, trace count, and train/test
  metrics to JSON. It also persists teacher candidate-source counts, loop-free segment action and
  duration schedules, sampled-trace log-probability provenance, and switch-fit diagnostics comparing
  the selected switch objective tuple to a fixed local reference switch; this is failure-analysis
  provenance, not a controller selection rule.
- `src/cartpole_synthesis.py`: trace-based synthesis of a two-mode constant-action policy, plus a
  partial probabilistic Cartpole student with Gaussian action-parameter distributions and Boolean-tree
  switch candidates.
- `scripts/run_cartpole_reproduction.py`: orchestrated Cartpole runner that writes
  `cartpole_results.csv`, `cartpole_summary.csv`, and `cartpole_manifest.json` for selected seeds
  and settings. Its manifest records the PSM teacher overrides and fixed local synthesis constants,
  and each PSM row links to a per-seed metrics JSON with the fitted probabilistic student and
  teacher-trace provenance. When PPO is included, it also writes per-row PPO checkpoints and metrics
  JSON under the requested output directory; `--ppo-eval-interval` controls whether those metrics
  contain intermediate train/test `eval_history` entries or only the selected final result.
  PPO metrics also contain compact `update_history` rows with rollout reward means and
  train-horizon termination counts.
- `scripts/run_cartpole_ppo_sweep.py`: PPO/PPO-LSTM hyperparameter sweep runner that enumerates the
  paper-reported search ranges, writes a plan/manifest, and can execute jobs with per-config
  checkpoints and metrics JSON. It also writes a per-policy summary selecting the best completed
  config by train success and train reward. This is search infrastructure; the full paper-scale
  sweep has not been run.
- `scripts/make_paper_figures.py`: figure/table generator that prefers grouped summary rows when
  available and falls back to raw per-seed result rows for older artifacts. It also writes the
  generated abstract-result, LaTeX table, and PSM policy fragments consumed by `essay/project.tex`,
  plots the PSM switch-boundary figure from a linear-switch PSM metrics artifact when available, and
  plots PPO training curves when metrics JSON artifacts with `eval_history` are present.

## Current Status

- Implemented and tested: Cartpole dynamics, train/test split, PPO MLP, PPO-LSTM, a bounded
  Direct-Opt diagnostic baseline, and Cartpole programmatic policy synthesis.
- Partially complete against the paper: the Cartpole programmatic policy is synthesized from
  model-based teacher traces into a two-mode constant-action/depth-2-switch policy. The student now
  fits Gaussian distributions over constant action parameters and latent mode responsibilities. Those
  responsibilities now use one bounded forward-backward refinement over the learned switch-timing
  likelihood, but the learner still approximates switch timing and does not implement the full
  probabilistic adaptive-teaching objective from the paper. The switch grammar now includes decision
  stumps plus depth-2 conjunction and disjunction candidates over observation inequalities via a
  bounded greedy leaf-expansion step. Switch threshold Gaussian means and standard deviations are locally refined
  against the Eq. (12)-style timing likelihood with a bounded grid initializer plus coordinate
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
  train success `1.000`, test success over the full 15000-step/300-second horizon `0.200`,
  train reward mean `250.0`, test reward mean `6275.4`.
- Current synthesizer diagnostic command used for the regenerated artifact:
  `python src/train_cartpole_psm.py --num-initial-states 4 --candidate-rollouts 8 --teacher-top-rho 2 --teacher-refinement-steps 1 --eval-rollouts 20 --test-max-steps 15000 --metrics-output artifacts/results/metrics/psm_seed0_full_horizon.json`
- Current synthesizer diagnostic output:
  train success `0.000`, test success over the full 15000-step/300-second horizon `0.000`,
  train reward mean `39.55`, test reward mean `64.8`. The tracked artifact uses the current
  CartPole PSM default loop-free teacher profile (`segment_steps = 1`, `segments_per_trace = 250`)
  so the teacher can span the full 250-step training horizon with one-step segments. Its metadata
  records `rollout_parameter_resampling = on_mode_entry`,
  `bootstrap_source = probabilistic_student_prior`, first-iteration source counts
  `{"bootstrap_student_sample": 3, "bootstrap_student_sample_refined": 1}`, final-iteration source
  counts `{"student_sample": 2, "student_sample_refined": 2}`, and
  `student_sample_segment_budget = chunk_sampled_actions_by_max_segment_duration_then_reroll_loop_free_trace`.
  This remains a local synthesis diagnostic and still demonstrates a full-horizon programmatic-policy
  gap, not a paper-level reproduction result.
- Direct-Opt diagnostic command:
  `python src/train_cartpole_direct_opt.py --seed 0 --num-train-states 10 --random-candidates 256 --eval-rollouts 20 --test-max-steps 15000 --metrics-output artifacts/results/metrics/direct_opt_seed0_full_horizon.json`
- Direct-Opt diagnostic output:
  train success `1.000`, test success over the full 15000-step/300-second horizon `0.100`,
  train reward mean `250.0`, test reward mean `4220.1`. The selected bounded two-mode policy is
  `m0 action=-10.000; m1 action=10.000; mode=1 if 1.000*theta + 0.250*omega >= 0.000, else mode=0`.
  This is an executable local baseline artifact, not the paper's full Direct-Opt protocol.
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
  success `1.000`, but full-horizon test success remains `0.000`.

The PPO diagnostics now verify that the feed-forward PPO baseline can solve the paper's training
split locally. They still do not reproduce the paper-scale PPO/PPO-LSTM protocol.

## Bugs Fixed During Audit

- PPO rollout collection originally reset only on pole/cart failure. It now truncates and resets at
  `env.cfg.max_steps = 250` for the paper's 5-second training horizon.
- PPO now stores raw sampled continuous actions for log-probability calculations and clips only the
  action sent to the environment.
- Vectorized rollouts were added so short local runs get more PPO updates with stable batch shapes.
- LSTM PPO now preserves recurrent state across rollout chunks and replays the same initial state
  during the update.
- Test evaluation defaults now use `15000` steps, matching the paper's 300-second test horizon.
- Programmatic-state-machine synthesis can now write metrics JSON containing the synthesis config,
  policy description, fitted Gaussian action/switch distributions, latent responsibility summary,
  compact teacher-trace examples with segment-duration schedules, per-teacher/student-iteration
  `synthesis_history`, number of teacher traces, evaluation settings, switch-fit diagnostics, and
  train/test metrics.
- The Cartpole switch learner now performs bounded local grid plus coordinate refinement of selected
  switch-threshold Gaussian means and standard deviations against a discrete Eq. (12)-style
  likelihood, while rejecting candidate means that increase hard segment-label mistakes. This moves
  the switch-parameter M-step closer to the paper's numerical optimization, but remains a diagnostic
  approximation: switch structure is prefiltered by a cheaper hard-label/timing objective before
  bounded distribution rescoring, depth-2 Boolean-tree probabilities use a shared-threshold
  rectangle-union calculation, and this is not the paper's full continuous switch-parameter optimizer.
- The first Cartpole teacher iteration now samples from an explicit probabilistic student prior, and
  later teacher candidate pools are sampled from the current probabilistic student before top-rho
  selection. These sampled rollouts now resample action and switch parameters whenever execution
  enters a mode segment, matching the paper's probabilistic PSM execution model more closely. The
  teacher locally refines top sampled loop-free traces by duration/action coordinate search under a
  top-rho elite-distance kernel approximation and records selected trace sources plus sampled-trace
  log probabilities in metrics JSON. This moves toward the sampled-teacher and local-optimization
  phases in Section 4.2, but it is not the paper's full CEM plus gradient-based trajectory optimizer.
- The Cartpole teacher regularizer now scores candidate traces with both Gaussian action likelihood
  and the student's discrete Eq. (12)-style switch timing likelihood, marginalizing over the latent
  mode sequence with a two-state forward pass. For scalar-threshold switches, that timing likelihood
  uses the learned Gaussian switch-parameter distribution with one sampled threshold shared across a
  segment, matching the paper's probabilistic-state-machine sampling model.
- The Cartpole student now refines action-only latent segment responsibilities with a bounded
  forward-backward pass over adjacent segment switch-timing probabilities before refitting the action
  distributions and switch. This moves Eq. (10) closer to the paper by using both `H` and `G` evidence,
  but it remains a local bounded approximation rather than the paper's full EM/M-step optimizer.
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
  EM iterations, Gaussian floors, switch-timing scale, switch-search grids, and teacher-search
  refinement schedule. These values document the current partial implementation; they are not claimed
  as paper-specified constants.
- PPO training runs can now write metrics JSON containing the full evaluation history, compact
  per-update rollout diagnostics, selected result, config, and checkpoint-selection rule.
- The orchestrated reproduction runner now persists PPO/PPO-LSTM checkpoints and metrics JSON for
  `--include-ppo` rows, tying those local diagnostic results to concrete artifacts.
- The orchestrated reproduction runner can include a bounded Direct-Opt diagnostic row through
  `--include-direct-opt`, writing a per-seed metrics JSON with the selected program, searched
  candidate count, exact search grids, and limitation note.
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
  boundary.
- PPO hyperparameter search can now be planned or executed through
  `scripts/run_cartpole_ppo_sweep.py`; the runner records the paper search ranges and the chosen
  learning-rate samples in a manifest, and writes a best-config summary for completed jobs.

## Verified PPO Invariants

These checks are unit-level correctness guards for the local PPO implementation; they do not replace
paper-scale PPO2 runs.

- `tests/test_cartpole_paper.py::test_ppo_rollout_truncates_at_paper_training_horizon` verifies that
  rollout collection treats the 5-second/250-step training horizon as terminal and resets the vector
  environment counter.
- `tests/test_cartpole_paper.py::test_ppo_stores_raw_continuous_actions_for_log_probs` verifies that
  PPO stores the raw sampled Gaussian action for log-probability replay while clipping only the force
  applied to the continuous Cartpole environment.
- `tests/test_cartpole_paper.py::test_lstm_update_replays_rollout_initial_state` verifies that the
  PPO-LSTM update replays the rollout's stored initial recurrent state instead of silently starting
  updates from zeros.
- `tests/test_cartpole_paper.py::test_ppo_writes_eval_history_metrics_json` verifies that PPO
  interval evaluations and per-update rollout diagnostics are persisted to JSON instead of existing
  only in stdout.
- `tests/test_cartpole_psm_cli.py::test_cli_writes_metrics_json` verifies that synthesized
  programmatic-policy metrics are persisted to JSON and that the file records the full paper test
  horizon even when a quick test cap is supplied. It also verifies that exposed teacher
  hyperparameters, fixed local synthesis constants, the fitted probabilistic student summary, and
  per-iteration bounded teacher-trace provenance are persisted.
- `tests/test_cartpole_ppo_sweep.py::test_build_jobs_uses_paper_minibatch_rule_for_lstm` verifies
  that the sweep includes the paper's feed-forward minibatch grid while forcing PPO-LSTM to
  `nminibatches = 1`.
- `tests/test_cartpole_ppo_sweep.py::test_dry_run_writes_plan_and_manifest` verifies that dry-run
  sweep planning writes a CSV plan and manifest with paper-space metadata.
- `tests/test_cartpole_ppo_sweep.py::test_summarize_results_selects_best_train_per_policy` verifies
  the sweep summary selection rule.
- `tests/test_cartpole_ppo_sweep.py::test_quick_execution_writes_results_summary_and_manifest`
  verifies that quick sweep execution writes results, summary, and manifest artifacts.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_returns_policy_and_provenance` verifies that
  the bounded Direct-Opt diagnostic baseline selects a Cartpole PSM and records explicit
  non-paper-scale provenance.
- `tests/test_cartpole_direct_opt.py::test_direct_opt_cli_writes_metrics_json` verifies that the
  Direct-Opt CLI writes config, selected candidate, train/test metrics, and provenance JSON.

## Verified Programmatic-Student Invariants

These checks cover the partial probabilistic Cartpole student, not the complete paper algorithm.

- `tests/test_cartpole_paper.py::test_cartpole_probabilistic_student_uses_gaussian_modes` verifies
  that Cartpole student fitting produces two Gaussian constant-action distributions, positive standard
  deviations, Gaussian switch-parameter distributions, and normalized latent mode responsibilities
  over loop-free teacher segments.
- `tests/test_cartpole_paper.py::test_cartpole_responsibility_refinement_uses_switch_timing` verifies
  that switch-timing likelihood can shift ambiguous latent segment responsibilities away from the
  action-only posterior while preserving normalization.
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
- `tests/test_cartpole_paper.py::test_cartpole_eq12_likelihood_penalizes_early_transition_when_staying`
  verifies the no-transition-before-duration term in the discrete Eq. (12)-style switch likelihood.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_improves_timing_likelihood`
  verifies that bounded switch-threshold refinement can improve the current timing likelihood
  without increasing hard segment-label mistakes.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_can_improve_probabilistic_std`
  verifies that bounded Gaussian standard-deviation refinement can improve the current
  probabilistic timing objective.
- `tests/test_cartpole_paper.py::test_cartpole_switch_coordinate_refinement_polishes_grid_solution`
  verifies that the bounded coordinate pass can improve beyond the discrete std-candidate grid.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_refinement_keeps_std_finite`
  verifies that refined switch Gaussian standard deviations remain finite and above the local
  Gaussian floor.
- `tests/test_cartpole_paper.py::test_cartpole_switch_std_refinement_uses_boundary_variance_candidate`
  verifies that the bounded std grid includes transition-boundary variance evidence.
- `tests/test_cartpole_paper.py::test_cartpole_switch_parameter_refinement_rejects_more_label_mistakes`
  verifies that probabilistic timing refinement does not accept a switch mean that increases hard
  segment-label mistakes.
- `tests/test_cartpole_paper.py::test_cartpole_switch_distribution_timing_loss_rejects_responsibility_mismatch`
  verifies that malformed segment/responsibility inputs do not silently produce a zero timing loss.
- `tests/test_cartpole_paper.py::test_cartpole_switch_probability_uses_gaussian_threshold_distribution`
  verifies that switch-enable probabilities are computed from Gaussian threshold distributions.
- `tests/test_cartpole_paper.py::test_cartpole_switch_transition_probability_uses_shared_threshold_sample`
  verifies that scalar switch timing probability treats the same sampled threshold as shared over the
  segment rather than resampling independently at each simulator step.
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
- `tests/test_cartpole_paper.py::test_cartpole_trace_log_probability_marginalizes_latent_modes`
  verifies that the teacher regularizer marginalizes over latent segment modes instead of treating
  posterior responsibilities as extra priors.
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
- `tests/test_cartpole_paper.py::test_cartpole_teacher_can_refine_student_sampled_trace` verifies
  that a sampled-student loop-free trace can be locally refined without reducing the current
  elite-kernel refinement objective.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_refinement_does_not_reduce_objective`
  verifies that local refinement of Cartpole loop-free teacher gains does not reduce the teacher
  objective after top-candidate sampling. This is a bounded coordinate refinement over the diagnostic
  teacher gains, not the paper's continuous gradient-based trajectory optimizer.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_rollout_records_segment_durations` verifies
  that loop-free teacher traces persist the segment-action and segment-duration schedules used to
  generate them.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_duration_refinement_preserves_action_sequence`
  verifies that duration-only refinement preserves the loop-free teacher's constant-action sequence.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_refinement_changes_one_action_at_a_time`
  verifies that bounded action refinement mutates one loop-free action-function segment at a time.
- `tests/test_cartpole_paper.py::test_cartpole_teacher_action_refinement_uses_continuous_local_steps`
  verifies that bounded action refinement proposes local continuous force steps rather than only
  bang-bang action flips.
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
- `tests/test_cartpole_paper.py::test_cartpole_boolean_tree_switch_has_gaussian_parameter_per_predicate`
  verifies that a depth-2 Boolean-tree switch gets one Gaussian threshold distribution per predicate.
- `tests/test_cartpole_paper.py::test_cartpole_sampled_switch_uses_gaussian_thresholds` verifies
  that Boolean-tree switch thresholds can be sampled from Gaussian parameter distributions.
- `tests/test_cartpole_paper.py::test_cartpole_probabilistic_student_samples_policy_parameters`
  verifies that a probabilistic Cartpole student can sample a deterministic policy with sampled
  action and switch parameters.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_writes_results_and_manifest`
  verifies that the reproduction runner writes raw results, grouped summary statistics, and a manifest
  with the exact quick-run command settings, PSM teacher overrides, fixed PSM synthesis constants, and
  a per-seed PSM metrics JSON artifact whose final per-iteration evaluation matches the top-level PSM
  row.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_can_include_direct_opt_diagnostic`
  verifies that `--include-direct-opt` adds the bounded Direct-Opt diagnostic row and links its
  metrics artifact from the manifest.
- `tests/test_cartpole_reproduction_runner.py::test_quick_runner_with_ppo_writes_checkpoints_and_metrics`
  verifies that the reproduction runner writes PPO/PPO-LSTM checkpoints and metrics JSON, that the
  configured PPO evaluation interval produces `eval_history` entries, and that PPO update diagnostics
  are persisted.
- `tests/test_cartpole_reproduction_runner.py::test_summary_rows_report_mean_std_and_best_train_seed`
  verifies the runner's per-policy mean/std summary and deterministic best-training-seed selection.
- `tests/test_make_paper_figures.py` verifies that figure/table generation reads grouped summary rows
  when present, falls back to raw result rows otherwise, and writes generated abstract-result and
  LaTeX result-table fragments. It also verifies PSM policy-fragment generation,
  switch-boundary parsing/plotting from synthetic metrics, fallback/skip behavior for non-linear
  switches, PPO metrics-file discovery, and training-curve PNG generation.

## Completion Criteria Still Required For Full Paper Claim

- Run PPO and PPO-LSTM for `10^7` timesteps.
- Tune/fix pure PPO-LSTM until it achieves strong Cartpole training performance without supervised
  warm-starting.
- Run 5 random seeds and choose best training performer.
- Run hyperparameter search over the paper's specified ranges.
- Replace the bounded Direct-Opt diagnostic with the paper's full direct optimization protocol:
  optimize the combined reward over all initial states using batch optimization, random restarts when
  stalled, and the reported two-hour/parallel budget.
- Complete the probabilistic adaptive-teaching implementation: continuous optimization of switch
  Gaussian parameters and the paper's full teacher optimization procedure. The current Cartpole switch
  learner performs a depth-2 greedy Boolean-tree expansion, stores Gaussian threshold distributions
  for each selected switch predicate, can sample deterministic policies from those distributions, and
  scores timing with a discrete approximation to Eq. (12), including transition-at-duration and
  no-transition-before-duration terms. It now performs bounded local mean/std grid plus coordinate
  refinement, but does not yet solve the full continuous Eq. (12) optimization for switch-condition
  means and standard deviations. For depth-2 Boolean trees, switch-enable probability now uses an
  exact union of axis-aligned threshold rectangles for conjunctions and disjunctions under
  independent predicate-threshold Gaussians, with threshold samples shared across the segment. The current Cartpole teacher samples the
  first iteration from a Gaussian PSM prior and later iterations from the current probabilistic
  student, resampling action/switch parameters on mode entry, refines top loop-free candidates with
  bounded coordinate search over teacher gains when available, integer segment durations, and
  one-segment local continuous constant-action steps, and scores traces with reward plus Gaussian
  action likelihood and discrete switch timing likelihood under the
  previous student, but it does not yet perform the paper's full CEM procedure or continuous
  gradient-based optimization over loop-free action functions and durations.
- Recover or manually inspect the Figure 19 Cartpole policy if exact state-machine comparison is required.
