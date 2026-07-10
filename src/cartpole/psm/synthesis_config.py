from __future__ import annotations

import bisect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple

from cartpole_env import (
    CARTPOLE_PSM_MODE_UPDATE_ORDER,
    PAPER_EVAL_ROLLOUTS,
    CartpoleConfig,
    CartpoleEnv,
    Observation,
    cartpole_done,
    cartpole_next_state,
    cartpole_reward_spec,
    cartpole_space_spec,
)


MIN_GAUSSIAN_STD = 1e-3
DEFAULT_CARTPOLE_TIME_INCREMENT = 0.02
INITIAL_CARTPOLE_PSM_MODE = 0
PROBABILISTIC_STUDENT_EM_ITERS = 4
PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES = 1
SWITCH_TIMING_STD_STEPS = 2.0
LOG_PROBABILITY_FLOOR = 1e-12
TEACHER_STUDENT_ITERS = 2
TEACHER_STUDENT_REGULARIZER = 1.0
TEACHER_REWARD_LAMBDA = 100.0
TEACHER_TOP_RHO = 10
PAPER_TEACHER_TOP_RHO = 10
PAPER_TEACHER_PARALLEL_THREADS = 10
PAPER_STUDENT_PARALLEL_THREADS = 10
TEACHER_REFINEMENT_STEPS = 2
TEACHER_GAIN_SAMPLE_STD_FRACTION = 0.10
TEACHER_GAIN_SAMPLE_MIN_STD = 1e-6
TEACHER_GAIN_REFINEMENT_DELTA_FRACTION = 0.05
TEACHER_THETA_REFINEMENT_MIN_DELTA = 0.1
TEACHER_OMEGA_REFINEMENT_MIN_DELTA = 0.05
TEACHER_REFINEMENT_DELTA_DECAY = 0.5
TEACHER_GAIN_GRADIENT_STEP_FRACTION = 0.05
TEACHER_GAIN_GRADIENT_EPS_FRACTION = 0.025
TEACHER_DURATION_REFINEMENT_DELTAS = (-1, 1)
TEACHER_ACTION_REFINEMENT_CANDIDATES_PER_SEGMENT = 2
TEACHER_ACTION_REFINEMENT_STEP_FRACTION = 0.25
TEACHER_ACTION_GRADIENT_STEP_FRACTION = 0.10
TEACHER_ACTION_GRADIENT_EPS_FRACTION = 0.05
TEACHER_DURATION_GRADIENT_STEP = 1
TEACHER_DURATION_GRADIENT_EPS = 1
TEACHER_TIME_INCREMENT_REFINEMENT_FRACTION = 0.25
TEACHER_TIME_INCREMENT_GRADIENT_STEP_FRACTION = 0.10
TEACHER_TIME_INCREMENT_GRADIENT_EPS_FRACTION = 0.05
TEACHER_GRADIENT_BACKTRACK_FACTORS = (1.0, 0.5, 0.25, 0.125)
TEACHER_STUDENT_SAMPLE_FRACTION = 1.0
TEACHER_ELITE_DISTRIBUTION_RESAMPLES = PAPER_TEACHER_TOP_RHO
TEACHER_ELITE_DISTRIBUTION_ROUNDS = 1
TEACHER_ELITE_RESAMPLE_MIN_ACTION_STD = 1e-3
TEACHER_ELITE_DISTANCE_DURATION_SCALE_FLOOR = 1.0
TEACHER_BOOTSTRAP_ACTION_STD = 10.0
TEACHER_BOOTSTRAP_SWITCH_THETA_WEIGHT = 1.0
TEACHER_BOOTSTRAP_SWITCH_OMEGA_WEIGHT = 0.25
TEACHER_BOOTSTRAP_SWITCH_THRESHOLD = 0.0
TEACHER_BOOTSTRAP_SWITCH_STD = 1.0
SWITCH_OBLIQUE_THETA_WEIGHTS = (-50.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SWITCH_OBLIQUE_OMEGA_WEIGHTS = (-10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
MAX_SWITCH_THRESHOLD_CANDIDATES = 64
DEFAULT_SWITCH_THRESHOLD_CANDIDATE = 0.0
SWITCH_STD_REFINEMENT_MULTIPLIERS = (0.5, 1.0, 2.0)
SWITCH_PARAMETER_COORDINATE_REFINEMENT_STEPS = 3
SWITCH_PARAMETER_COORDINATE_MEAN_STEP_FRACTION = 0.25
SWITCH_PARAMETER_COORDINATE_LOG_STD_STEP = 0.6931471805599453
SWITCH_PARAMETER_COORDINATE_STEP_DECAY = 0.5
SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS = 2
SWITCH_PARAMETER_GRADIENT_MEAN_STEP_FRACTION = 0.50
SWITCH_PARAMETER_GRADIENT_LOG_STD_STEP = 0.25
SWITCH_PARAMETER_GRADIENT_EPS_FRACTION = 0.25
SWITCH_PARAMETER_GRADIENT_MIN_RELATIVE_IMPROVEMENT = 1e-6
SWITCH_PARAMETER_GRADIENT_BACKTRACK_FACTORS = (1.0, 0.5, 0.25, 0.125)
SWITCH_PARAMETER_LABEL_LOSS_WEIGHT = 1.0
SWITCH_PARAMETER_TIMING_LOSS_WEIGHT = 1.0
SWITCH_SELECTION_OBJECTIVE_ORDER = (
    "responsibility_weighted_label_loss",
    "bounded_eq12_style_distribution_loss",
    "program_complexity",
    "description",
)
SWITCH_PREFILTER_OBJECTIVE_ORDER = (
    "hard_label_mistakes",
    "eq12_style_timing_loss",
    "program_complexity",
    "description",
)
SWITCH_STRUCTURE_RESCORING_TOP_K = 32


@dataclass
class CartpoleSynthesisConfig:
    num_initial_states: int = 32
    candidate_rollouts: int = 128
    segment_steps: int = 1
    segments_per_trace: int = 250
    force_values: Tuple[float, ...] = (-10.0, 10.0)
    seed: int = 0
    teacher_theta_gain: float = 20.0
    teacher_omega_gain: float = 2.0
    teacher_student_iters: int = TEACHER_STUDENT_ITERS
    student_em_iters: int = PROBABILISTIC_STUDENT_EM_ITERS
    student_switch_responsibility_passes: int = PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES
    teacher_student_regularizer: float = TEACHER_STUDENT_REGULARIZER
    teacher_reward_lambda: float = TEACHER_REWARD_LAMBDA
    teacher_top_rho: int = TEACHER_TOP_RHO
    teacher_refinement_steps: int = TEACHER_REFINEMENT_STEPS
    teacher_elite_distribution_resamples: int = TEACHER_ELITE_DISTRIBUTION_RESAMPLES
    teacher_elite_distribution_rounds: int = TEACHER_ELITE_DISTRIBUTION_ROUNDS
    parallel_trace_workers: int = 1
    parallel_switch_workers: int = 1


@dataclass
class CartpoleTrace:
    observations: List[Observation]
    actions: List[float]
    mode_labels: List[int]
    reward: float
    theta_gain: float = 0.0
    omega_gain: float = 0.0
    segment_actions: Tuple[float, ...] = ()
    segment_durations: Tuple[int, ...] = ()
    segment_time_increments: Tuple[float, ...] = ()
    teacher_source: str = "gain_sample"
    student_log_probability: float | None = None
    teacher_objective: float | None = None
    teacher_refinement_objective: float | None = None
    elite_distribution_fit: Dict[str, object] | None = None
    teacher_refinement_elite_summary: Dict[str, object] | None = None
    teacher_candidate_pool_diagnostics: Dict[str, object] | None = None


def cartpole_synthesis_algorithm_provenance() -> Dict[str, object]:
    return {
        "probabilistic_student": {
            "default_em_iters": PROBABILISTIC_STUDENT_EM_ITERS,
            "default_switch_responsibility_passes": PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES,
            "responsibility_evidence": (
                "action_likelihood_initialization_then_directed_switch_forward_backward_action_refits"
            ),
            "switch_responsibility_passes_are_per_em_iteration": True,
            "switch_condition_m_step_schedule": "once_per_student_em_iteration_after_configured_eq10_eq11_passes",
            "initial_switch_before_first_timing_e_step": "fixed_bootstrap_not_data_fit",
            "directed_switch_e_step_schedule": "uses_latest_transition_specific_switches_after_first_bounded_m_step",
            "mode_update_order": CARTPOLE_PSM_MODE_UPDATE_ORDER,
            "rollout_parameter_resampling": "on_mode_entry",
            "transition_specific_switches": "separate_fitted_conditions_for_0_to_1_and_1_to_0",
            "paper_parallel_switch_threads": PAPER_STUDENT_PARALLEL_THREADS,
            "local_parallel_switch_workers": "configurable_via_parallel_switch_workers",
            "parallel_switch_unit": "bounded_switch_candidate_rescoring",
            "paper_parallel_switch_thread_gap": (
                "bounded candidate rescoring can use the configured worker limit, but the current "
                "student M-step is still a bounded depth-2/transition fit rather than the paper's "
                "full continuous switch optimizer"
            ),
            "initial_mode": INITIAL_CARTPOLE_PSM_MODE,
            "initial_mode_prior": "fixed_mode_0",
            "min_gaussian_std": MIN_GAUSSIAN_STD,
            "log_probability_floor": LOG_PROBABILITY_FLOOR,
        },
        "switch_timing": {
            "std_steps": SWITCH_TIMING_STD_STEPS,
            "duration_units": "segment_elapsed_time_normalized_to_default_cartpole_dt",
            "final_segment_stay_evidence": True,
            "scalar_threshold_uses_shared_sample": True,
            "depth2_boolean_probability": "shared_threshold_rectangle_union",
            "std_refinement_multipliers": list(SWITCH_STD_REFINEMENT_MULTIPLIERS),
            "coordinate_refinement_steps": SWITCH_PARAMETER_COORDINATE_REFINEMENT_STEPS,
            "coordinate_mean_step_fraction": SWITCH_PARAMETER_COORDINATE_MEAN_STEP_FRACTION,
            "coordinate_log_std_initial_step": SWITCH_PARAMETER_COORDINATE_LOG_STD_STEP,
            "coordinate_step_decay": SWITCH_PARAMETER_COORDINATE_STEP_DECAY,
            "finite_difference_gradient_refinement_steps": SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS,
            "finite_difference_gradient_mean_step_fraction": SWITCH_PARAMETER_GRADIENT_MEAN_STEP_FRACTION,
            "finite_difference_gradient_log_std_step": SWITCH_PARAMETER_GRADIENT_LOG_STD_STEP,
            "finite_difference_gradient_epsilon_fraction": SWITCH_PARAMETER_GRADIENT_EPS_FRACTION,
            "finite_difference_gradient_min_relative_improvement": (
                SWITCH_PARAMETER_GRADIENT_MIN_RELATIVE_IMPROVEMENT
            ),
            "finite_difference_gradient_convergence": (
                "stop_after_bounded_backtracked_steps_or_tiny_combined_loss_improvement"
            ),
            "finite_difference_gradient_backtracking_factors": list(SWITCH_PARAMETER_GRADIENT_BACKTRACK_FACTORS),
            "finite_difference_gradient_objective": (
                "weighted_sum_of_responsibility_label_loss_and_eq12_timing_loss"
            ),
            "finite_difference_gradient_label_loss_weight": SWITCH_PARAMETER_LABEL_LOSS_WEIGHT,
            "finite_difference_gradient_timing_loss_weight": SWITCH_PARAMETER_TIMING_LOSS_WEIGHT,
            "structure_rescore_uses_pair_posteriors": True,
            "transition_specific_m_step": "bounded_separate_0_to_1_and_1_to_0_switch_fits",
        },
        "switch_search": {
            "boolean_tree_depth": 2,
            "constant_leaf_baselines": "always_switch_and_never_switch",
            "constant_leaf_baselines_preserved_after_prefilter": True,
            "greedy_second_predicate_expands_switch_and_no_switch_leaves": True,
            "greedy_second_predicate_prefilter_top_k": SWITCH_STRUCTURE_RESCORING_TOP_K,
            "structure_label_objective": (
                "responsibility_weighted_expected_label_loss_when_available_else_hard_label_mistakes"
            ),
            "structure_label_observations": "nonboundary_segment_observations_boundary_observations_scored_by_timing_loss",
            "oblique_theta_weights": list(SWITCH_OBLIQUE_THETA_WEIGHTS),
            "oblique_omega_weights": list(SWITCH_OBLIQUE_OMEGA_WEIGHTS),
            "max_threshold_candidates": MAX_SWITCH_THRESHOLD_CANDIDATES,
            "default_threshold_candidate": DEFAULT_SWITCH_THRESHOLD_CANDIDATE,
            "distribution_rescore_top_k": SWITCH_STRUCTURE_RESCORING_TOP_K,
            "prefilter_objective_order": list(SWITCH_PREFILTER_OBJECTIVE_ORDER),
            "selection_objective_order": list(SWITCH_SELECTION_OBJECTIVE_ORDER),
        },
        "teacher_search": {
            "paper_parallel_threads": PAPER_TEACHER_PARALLEL_THREADS,
            "local_parallel_trace_workers": "configurable_via_parallel_trace_workers",
            "parallel_unit": "independent_loop_free_teacher_optimization_per_initial_state",
            "gain_sample_std_fraction": TEACHER_GAIN_SAMPLE_STD_FRACTION,
            "gain_sample_min_std": TEACHER_GAIN_SAMPLE_MIN_STD,
            "gain_refinement_delta_fraction": TEACHER_GAIN_REFINEMENT_DELTA_FRACTION,
            "theta_refinement_min_delta": TEACHER_THETA_REFINEMENT_MIN_DELTA,
            "omega_refinement_min_delta": TEACHER_OMEGA_REFINEMENT_MIN_DELTA,
            "refinement_delta_decay": TEACHER_REFINEMENT_DELTA_DECAY,
            "gain_gradient_step_fraction": TEACHER_GAIN_GRADIENT_STEP_FRACTION,
            "gain_gradient_epsilon_fraction": TEACHER_GAIN_GRADIENT_EPS_FRACTION,
            "duration_refinement_deltas": list(TEACHER_DURATION_REFINEMENT_DELTAS),
            "action_refinement_max_candidates_per_segment": TEACHER_ACTION_REFINEMENT_CANDIDATES_PER_SEGMENT,
            "action_refinement_step_fraction": TEACHER_ACTION_REFINEMENT_STEP_FRACTION,
            "action_gradient_step_fraction": TEACHER_ACTION_GRADIENT_STEP_FRACTION,
            "action_gradient_epsilon_fraction": TEACHER_ACTION_GRADIENT_EPS_FRACTION,
            "duration_gradient_step": TEACHER_DURATION_GRADIENT_STEP,
            "duration_gradient_epsilon": TEACHER_DURATION_GRADIENT_EPS,
            "time_increment_parameterization": "per_segment_delta_i_with_default_environment_dt",
            "time_increment_reward_accounting": "elapsed_time_normalized_to_environment_dt",
            "time_increment_refinement_fraction": TEACHER_TIME_INCREMENT_REFINEMENT_FRACTION,
            "time_increment_gradient_step_fraction": TEACHER_TIME_INCREMENT_GRADIENT_STEP_FRACTION,
            "time_increment_gradient_epsilon_fraction": TEACHER_TIME_INCREMENT_GRADIENT_EPS_FRACTION,
            "finite_difference_gradient_backtracking_factors": list(TEACHER_GRADIENT_BACKTRACK_FACTORS),
            "finite_difference_candidates_per_refinement_iteration": {
                "teacher_gain_schedule": 1,
                "action_schedule": 1,
                "duration_schedule": 1,
                "time_increment_schedule": 1,
                "joint_gain_action_duration_time_increment_schedule": 1,
            },
            "student_sample_fraction_after_first_iteration": TEACHER_STUDENT_SAMPLE_FRACTION,
            "student_sample_probability": "forward_marginalized_action_and_switch_timing_likelihood",
            "student_sample_switch_timing": "uses_transition_specific_switches_when_available",
            "student_sample_segment_budget": (
                "preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood"
            ),
            "student_sample_local_refinement": (
                "mode_preserving_duration_time_increment_continuous_action_gain_and_finite_difference_schedule_search"
            ),
            "candidate_rollout_count": "configurable_via_candidate_rollouts",
            "paper_top_rho": PAPER_TEACHER_TOP_RHO,
            "top_rho_selection": "sort_by_teacher_objective_and_keep_teacher_top_rho_elites",
            "phase_one_objective": (
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_log_p_trace_under_student"
            ),
            "teacher_rollout_horizon": "min_environment_max_steps_and_configured_loop_free_horizon",
            "elite_recombination": "top_rho_segment_mode_action_duration_time_increment_centroid",
            "elite_recombination_candidate_count": "at_most_one_when_elites_have_loop_free_schedules",
            "default_elite_distribution_resamples": TEACHER_ELITE_DISTRIBUTION_RESAMPLES,
            "default_elite_distribution_rounds": TEACHER_ELITE_DISTRIBUTION_ROUNDS,
            "elite_distribution_samples_teacher_gains": True,
            "elite_distribution_mean_candidate_per_round": 1,
            "elite_distribution_default_batch_target": "paper_top_rho",
            "elite_distribution_default_batch_matches_paper_top_rho": (
                TEACHER_ELITE_DISTRIBUTION_RESAMPLES == PAPER_TEACHER_TOP_RHO
            ),
            "elite_distribution_min_action_std": TEACHER_ELITE_RESAMPLE_MIN_ACTION_STD,
            "elite_distribution_phase": "bounded_cem_style_distribution_refit_top_rho_refresh",
            "elite_distribution_update": (
                "fit_objective_weighted_gaussian_schedule_distribution_from_current_top_rho_each_round"
            ),
            "elite_distribution_weighting": "softmax_teacher_objective_when_student_available_else_uniform",
            "elite_distribution_parameters": [
                "teacher_gain_schedule",
                "segment_action_schedule",
                "integer_segment_duration_schedule",
                "segment_time_increment_schedule",
                "majority_segment_mode_schedule",
            ],
            "elite_distribution_selection_objective": (
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_student_log_probability"
            ),
            "elite_distribution_fit_diagnostics": (
                "serialized_on_distribution_mean_and_sample_traces_with_source_weights_objectives_and_gaussian_parameters"
            ),
            "elite_refinement_elite_set": "refreshed_top_rho_after_distribution_rounds",
            "elite_refinement_objective": "reward_plus_top_rho_log_probability_distance_kernel",
            "elite_refinement_selected_trace_diagnostics": (
                "serialized_on_selected_teacher_traces_with_refreshed_elite_count_sources_objectives_distances_and_kernel_terms"
            ),
            "elite_refinement_kernel_weighting": (
                "normalized_student_probability_weights_times_exp_negative_loop_free_distance"
            ),
            "selected_trace_objective_metrics": [
                "teacher_objective",
                "teacher_refinement_objective",
            ],
            "selected_trace_candidate_pool_diagnostics": (
                "serialized_on_selected_teacher_traces_with_counts_for_sampled_candidates_top_rho_elites_"
                "elite_recombination_distribution_candidates_refinement_seeds_refined_candidates_selection_source_"
                "and_selected_candidate_objective_rank_membership"
            ),
            "student_log_probability_cache_policy": (
                "recompute_from_trace_actions_for_current_student_else_use_cached_segment_only_value"
            ),
            "elite_distance_metric": (
                "normalized_l2_over_teacher_gains_segment_modes_actions_durations_and_time_increments"
            ),
            "elite_distance_action_scale": "max_abs_segment_action_floor_1",
            "elite_distance_duration_scale_floor": TEACHER_ELITE_DISTANCE_DURATION_SCALE_FLOOR,
            "bootstrap_source": "probabilistic_student_prior",
            "bootstrap_action_means": "min_and_max_configured_force_values",
            "bootstrap_action_std": TEACHER_BOOTSTRAP_ACTION_STD,
            "bootstrap_switch_mean": {
                "theta_weight": TEACHER_BOOTSTRAP_SWITCH_THETA_WEIGHT,
                "omega_weight": TEACHER_BOOTSTRAP_SWITCH_OMEGA_WEIGHT,
                "threshold": TEACHER_BOOTSTRAP_SWITCH_THRESHOLD,
            },
            "bootstrap_switch_std": TEACHER_BOOTSTRAP_SWITCH_STD,
        },
    }


def cartpole_teacher_cem_protocol_status(cfg: CartpoleSynthesisConfig) -> Dict[str, object]:
    effective_candidate_rollouts = max(1, int(cfg.candidate_rollouts))
    effective_top_rho = max(1, int(cfg.teacher_top_rho))
    effective_parallel_trace_workers = max(1, int(cfg.parallel_trace_workers))
    effective_parallel_trace_initial_states = max(0, int(cfg.num_initial_states))
    effective_parallel_trace_slots = min(
        effective_parallel_trace_workers,
        effective_parallel_trace_initial_states,
    )
    effective_parallel_switch_workers = max(1, int(cfg.parallel_switch_workers))
    transition_switch_fit_count = 2
    effective_parallel_switch_slots = min(
        effective_parallel_switch_workers,
        transition_switch_fit_count,
    )
    effective_student_parallel_switch_candidate_workers_per_fit = max(
        1,
        effective_parallel_switch_workers // max(1, effective_parallel_switch_slots),
    )
    student_switch_candidate_parallel_work_units = SWITCH_STRUCTURE_RESCORING_TOP_K
    effective_student_parallel_switch_candidate_slots = min(
        effective_student_parallel_switch_candidate_workers_per_fit,
        student_switch_candidate_parallel_work_units,
    )
    effective_total_student_switch_worker_slots = min(
        effective_parallel_switch_workers,
        effective_parallel_switch_slots * effective_student_parallel_switch_candidate_slots,
    )
    effective_teacher_elite_distribution_resamples = max(
        0,
        int(cfg.teacher_elite_distribution_resamples),
    )
    effective_teacher_elite_distribution_rounds = max(
        0,
        int(cfg.teacher_elite_distribution_rounds),
    )
    return {
        "teacher_candidate_rollouts": cfg.candidate_rollouts,
        "effective_teacher_candidate_rollouts": effective_candidate_rollouts,
        "selected_teacher_top_rho": cfg.teacher_top_rho,
        "effective_teacher_top_rho": effective_top_rho,
        "paper_teacher_top_rho": PAPER_TEACHER_TOP_RHO,
        "uses_paper_teacher_top_rho": effective_top_rho == PAPER_TEACHER_TOP_RHO,
        "selected_teacher_parallel_trace_workers": cfg.parallel_trace_workers,
        "effective_teacher_parallel_trace_workers": effective_parallel_trace_workers,
        "effective_teacher_parallel_trace_initial_states": effective_parallel_trace_initial_states,
        "effective_teacher_parallel_trace_slots": effective_parallel_trace_slots,
        "paper_teacher_parallel_threads": PAPER_TEACHER_PARALLEL_THREADS,
        "uses_parallel_teacher_trace_optimization": effective_parallel_trace_slots > 1,
        "uses_paper_teacher_parallel_worker_limit": (
            effective_parallel_trace_workers == PAPER_TEACHER_PARALLEL_THREADS
        ),
        "uses_paper_teacher_parallel_threads": (
            effective_parallel_trace_workers == PAPER_TEACHER_PARALLEL_THREADS
            and effective_parallel_trace_slots == PAPER_TEACHER_PARALLEL_THREADS
        ),
        "selected_student_parallel_switch_workers": cfg.parallel_switch_workers,
        "effective_student_parallel_switch_workers": effective_parallel_switch_workers,
        "student_transition_switch_fit_count": transition_switch_fit_count,
        "effective_student_parallel_switch_slots": effective_parallel_switch_slots,
        "effective_student_parallel_switch_candidate_workers_per_fit": (
            effective_student_parallel_switch_candidate_workers_per_fit
        ),
        "student_switch_candidate_rescoring_top_k": SWITCH_STRUCTURE_RESCORING_TOP_K,
        "student_switch_candidate_parallel_work_units": student_switch_candidate_parallel_work_units,
        "effective_student_parallel_switch_candidate_slots": effective_student_parallel_switch_candidate_slots,
        "effective_total_student_switch_worker_slots": effective_total_student_switch_worker_slots,
        "paper_student_parallel_threads": PAPER_STUDENT_PARALLEL_THREADS,
        "uses_parallel_student_switch_optimization": (
            effective_parallel_switch_slots > 1
            or effective_student_parallel_switch_candidate_slots > 1
        ),
        "uses_parallel_student_transition_switch_optimization": effective_parallel_switch_slots > 1,
        "uses_parallel_student_switch_candidate_optimization": (
            effective_student_parallel_switch_candidate_slots > 1
        ),
        "uses_paper_student_parallel_worker_limit": (
            effective_parallel_switch_workers == PAPER_STUDENT_PARALLEL_THREADS
        ),
        "uses_paper_student_parallel_threads": (
            effective_parallel_switch_workers == PAPER_STUDENT_PARALLEL_THREADS
            and effective_parallel_switch_slots == PAPER_STUDENT_PARALLEL_THREADS
        ),
        "student_switch_candidate_parallelism_matches_paper_threads": (
            effective_student_parallel_switch_candidate_slots >= PAPER_STUDENT_PARALLEL_THREADS
        ),
        "teacher_candidate_rollouts_cover_selected_top_rho": effective_candidate_rollouts >= effective_top_rho,
        "teacher_candidate_rollouts_cover_paper_top_rho": effective_candidate_rollouts >= PAPER_TEACHER_TOP_RHO,
        "teacher_cem_phase_matches_paper_rho": (
            effective_top_rho == PAPER_TEACHER_TOP_RHO
            and effective_candidate_rollouts >= PAPER_TEACHER_TOP_RHO
        ),
        "teacher_elite_distribution_resamples": cfg.teacher_elite_distribution_resamples,
        "effective_teacher_elite_distribution_resamples": effective_teacher_elite_distribution_resamples,
        "teacher_elite_distribution_resamples_cover_top_rho": (
            effective_teacher_elite_distribution_resamples >= effective_top_rho
        ),
        "teacher_elite_distribution_rounds": cfg.teacher_elite_distribution_rounds,
        "effective_teacher_elite_distribution_rounds": effective_teacher_elite_distribution_rounds,
        "teacher_elite_distribution_refit_round_enabled": (
            effective_teacher_elite_distribution_rounds > 0
        ),
    }


def cartpole_synthesis_protocol_status(
    cfg: CartpoleSynthesisConfig,
    eval_rollouts: int | None = None,
    test_max_steps: int | None = None,
    quick: bool = False,
    five_seed_selection: bool = False,
) -> Dict[str, object]:
    paper_train_env = CartpoleEnv.train_env()
    paper_test_env = CartpoleEnv.test_env()
    loop_free_training_horizon = cfg.segment_steps * cfg.segments_per_trace
    paper_test_horizon = test_max_steps == paper_test_env.cfg.max_steps if test_max_steps is not None else False
    paper_eval_rollouts = eval_rollouts == PAPER_EVAL_ROLLOUTS if eval_rollouts is not None else False
    cem_status = cartpole_teacher_cem_protocol_status(cfg)
    uses_paper_reward_scale = cfg.teacher_reward_lambda == TEACHER_REWARD_LAMBDA
    two_mode_constant_action_psm = len(cfg.force_values) == 2
    gaussian_action_parameter_distributions = True
    gaussian_switch_parameter_distributions = True
    transition_specific_switch_conditions = True
    resamples_parameters_on_mode_entry = True
    full_continuous_switch_m_step = False
    full_cem_teacher_optimizer = False
    probabilistic_adaptive_teaching_requirements = {
        "cartpole_environment": True,
        "loop_free_teacher_spans_training_horizon": loop_free_training_horizon >= paper_train_env.cfg.max_steps,
        "uses_paper_reward_scale": uses_paper_reward_scale,
        "two_mode_constant_action_psm": two_mode_constant_action_psm,
        "gaussian_action_parameter_distributions": gaussian_action_parameter_distributions,
        "gaussian_switch_parameter_distributions": gaussian_switch_parameter_distributions,
        "transition_specific_switch_conditions": transition_specific_switch_conditions,
        "resamples_parameters_on_mode_entry": resamples_parameters_on_mode_entry,
        "teacher_cem_phase_matches_paper_rho": cem_status["teacher_cem_phase_matches_paper_rho"],
        "teacher_elite_distribution_resamples_cover_top_rho": cem_status[
            "teacher_elite_distribution_resamples_cover_top_rho"
        ],
        "teacher_elite_distribution_refit_round_enabled": cem_status[
            "teacher_elite_distribution_refit_round_enabled"
        ],
        "uses_paper_teacher_parallel_threads": cem_status["uses_paper_teacher_parallel_threads"],
        "uses_paper_student_parallel_threads": cem_status["uses_paper_student_parallel_threads"],
        "student_switch_candidate_parallelism_matches_paper_threads": cem_status[
            "student_switch_candidate_parallelism_matches_paper_threads"
        ],
        "full_continuous_switch_m_step": full_continuous_switch_m_step,
        "full_cem_teacher_optimizer": full_cem_teacher_optimizer,
    }
    missing_probabilistic_adaptive_teaching_requirements = [
        requirement
        for requirement, satisfied in probabilistic_adaptive_teaching_requirements.items()
        if not satisfied
    ]
    full_probabilistic_adaptive_teaching = (
        not missing_probabilistic_adaptive_teaching_requirements
    )
    adaptive_teaching_protocol_requirements = {
        **probabilistic_adaptive_teaching_requirements,
        "five_seed_selection": five_seed_selection,
        "full_test_horizon": paper_test_horizon,
        "paper_eval_rollouts": paper_eval_rollouts,
    }
    missing_adaptive_teaching_protocol_requirements = [
        requirement
        for requirement, satisfied in adaptive_teaching_protocol_requirements.items()
        if not satisfied
    ]
    paper_scale_result = not missing_adaptive_teaching_protocol_requirements
    status = {
        "cartpole_environment": True,
        "train_horizon_seconds": paper_train_env.cfg.horizon_seconds,
        "train_pole_length": paper_train_env.cfg.pole_length,
        "test_horizon_seconds": paper_test_env.cfg.horizon_seconds,
        "test_pole_length": paper_test_env.cfg.pole_length,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(paper_train_env.cfg),
        "training_horizon_steps": paper_train_env.cfg.max_steps,
        "loop_free_teacher_horizon_steps": loop_free_training_horizon,
        "loop_free_teacher_spans_training_horizon": loop_free_training_horizon >= paper_train_env.cfg.max_steps,
        "paper_test_horizon_steps": paper_test_env.cfg.max_steps,
        "uses_full_test_horizon": paper_test_horizon,
        "eval_rollouts": eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": paper_eval_rollouts,
        "five_seed_selection": five_seed_selection,
        "quick_diagnostic": bool(quick),
        "uses_paper_reward_scale": uses_paper_reward_scale,
        "two_mode_constant_action_psm": two_mode_constant_action_psm,
        "boolean_tree_depth": 2,
        "gaussian_action_parameter_distributions": gaussian_action_parameter_distributions,
        "gaussian_switch_parameter_distributions": gaussian_switch_parameter_distributions,
        "transition_specific_switch_conditions": transition_specific_switch_conditions,
        "resamples_parameters_on_mode_entry": resamples_parameters_on_mode_entry,
        "student_em_iters": cfg.student_em_iters,
        "student_switch_responsibility_passes": cfg.student_switch_responsibility_passes,
        "teacher_elite_distribution_resamples": cfg.teacher_elite_distribution_resamples,
        "teacher_elite_distribution_rounds": cfg.teacher_elite_distribution_rounds,
        "student_switch_candidate_parallelism_matches_paper_threads": cem_status[
            "student_switch_candidate_parallelism_matches_paper_threads"
        ],
        "synthesized_by_current_algorithm": True,
        "probabilistic_adaptive_teaching_requirements": probabilistic_adaptive_teaching_requirements,
        "missing_probabilistic_adaptive_teaching_requirements": missing_probabilistic_adaptive_teaching_requirements,
        "full_probabilistic_adaptive_teaching": full_probabilistic_adaptive_teaching,
        "full_continuous_switch_m_step": full_continuous_switch_m_step,
        "full_cem_teacher_optimizer": full_cem_teacher_optimizer,
        "adaptive_teaching_protocol_requirements": adaptive_teaching_protocol_requirements,
        "missing_adaptive_teaching_protocol_requirements": missing_adaptive_teaching_protocol_requirements,
        "paper_scale_result": paper_scale_result,
        "limitation": (
            "Local bounded Cartpole PSM diagnostic: implements Gaussian action/switch distributions "
            "and sampled teacher traces, but not the paper's full probabilistic adaptive-teaching "
            "optimizer or paper-scale result reproduction."
        ),
    }
    status.update(cem_status)
    return status
