from __future__ import annotations

import bisect
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
TEACHER_ELITE_DISTRIBUTION_RESAMPLES = 1
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
SWITCH_PARAMETER_GRADIENT_BACKTRACK_FACTORS = (1.0, 0.5, 0.25, 0.125)
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


def cartpole_synthesis_algorithm_provenance() -> Dict[str, object]:
    return {
        "probabilistic_student": {
            "default_em_iters": PROBABILISTIC_STUDENT_EM_ITERS,
            "default_switch_responsibility_passes": PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES,
            "responsibility_evidence": "action_likelihood_initialization_then_alternating_switch_timing_forward_backward",
            "switch_responsibility_passes_are_per_em_iteration": True,
            "mode_update_order": CARTPOLE_PSM_MODE_UPDATE_ORDER,
            "rollout_parameter_resampling": "on_mode_entry",
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
            "finite_difference_gradient_backtracking_factors": list(SWITCH_PARAMETER_GRADIENT_BACKTRACK_FACTORS),
            "structure_rescore_uses_pair_posteriors": True,
        },
        "switch_search": {
            "boolean_tree_depth": 2,
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
            "student_sample_segment_budget": (
                "preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood"
            ),
            "student_sample_local_refinement": (
                "mode_preserving_duration_time_increment_continuous_action_gain_and_finite_difference_schedule_search"
            ),
            "teacher_rollout_horizon": "min_environment_max_steps_and_configured_loop_free_horizon",
            "elite_recombination": "top_rho_segment_mode_action_duration_time_increment_centroid",
            "elite_recombination_candidate_count": "at_most_one_when_elites_have_loop_free_schedules",
            "default_elite_distribution_resamples": TEACHER_ELITE_DISTRIBUTION_RESAMPLES,
            "default_elite_distribution_rounds": TEACHER_ELITE_DISTRIBUTION_ROUNDS,
            "elite_distribution_samples_teacher_gains": True,
            "elite_distribution_mean_candidate_per_round": 1,
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
            "elite_refinement_elite_set": "refreshed_top_rho_after_distribution_rounds",
            "elite_refinement_objective": "reward_plus_top_rho_log_probability_distance_kernel",
            "selected_trace_objective_metrics": [
                "teacher_objective",
                "teacher_refinement_objective",
            ],
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


def cartpole_synthesis_protocol_status(
    cfg: CartpoleSynthesisConfig,
    eval_rollouts: int | None = None,
    test_max_steps: int | None = None,
    quick: bool = False,
) -> Dict[str, object]:
    paper_train_env = CartpoleEnv.train_env()
    paper_test_env = CartpoleEnv.test_env()
    loop_free_training_horizon = cfg.segment_steps * cfg.segments_per_trace
    paper_test_horizon = test_max_steps == paper_test_env.cfg.max_steps if test_max_steps is not None else False
    paper_eval_rollouts = eval_rollouts == PAPER_EVAL_ROLLOUTS if eval_rollouts is not None else False
    return {
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
        "quick_diagnostic": bool(quick),
        "uses_paper_reward_scale": cfg.teacher_reward_lambda == TEACHER_REWARD_LAMBDA,
        "two_mode_constant_action_psm": len(cfg.force_values) == 2,
        "boolean_tree_depth": 2,
        "gaussian_action_parameter_distributions": True,
        "gaussian_switch_parameter_distributions": True,
        "resamples_parameters_on_mode_entry": True,
        "student_em_iters": cfg.student_em_iters,
        "student_switch_responsibility_passes": cfg.student_switch_responsibility_passes,
        "teacher_elite_distribution_resamples": cfg.teacher_elite_distribution_resamples,
        "teacher_elite_distribution_rounds": cfg.teacher_elite_distribution_rounds,
        "synthesized_by_current_algorithm": True,
        "full_probabilistic_adaptive_teaching": False,
        "full_continuous_switch_m_step": False,
        "full_cem_teacher_optimizer": False,
        "paper_scale_result": False,
        "limitation": (
            "Local bounded Cartpole PSM diagnostic: implements Gaussian action/switch distributions "
            "and sampled teacher traces, but not the paper's full probabilistic adaptive-teaching "
            "optimizer or paper-scale result reproduction."
        ),
    }


@dataclass
class Depth2Switch:
    theta_weight: float
    omega_weight: float
    threshold: float

    def decide(self, observation: Observation) -> int:
        _, _, theta, omega = observation
        return 1 if self.theta_weight * theta + self.omega_weight * omega >= self.threshold else 0

    def describe(self) -> str:
        return (
            f"mode=1 if {self.theta_weight:.3f}*theta + "
            f"{self.omega_weight:.3f}*omega >= {self.threshold:.3f}, else mode=0"
        )


@dataclass(frozen=True)
class ObservationPredicate:
    feature_index: int
    relation: str
    threshold: float

    def evaluate(self, observation: Observation) -> bool:
        value = observation[self.feature_index]
        if self.relation == ">=":
            return value >= self.threshold
        if self.relation == "<=":
            return value <= self.threshold
        raise ValueError(f"unknown relation: {self.relation}")

    def describe(self) -> str:
        return f"o[{self.feature_index}] {self.relation} {self.threshold:.3f}"

    def with_threshold(self, threshold: float) -> "ObservationPredicate":
        return ObservationPredicate(self.feature_index, self.relation, float(threshold))


@dataclass(frozen=True)
class BooleanTreeSwitch:
    first: ObservationPredicate
    second: ObservationPredicate | None = None
    operator: str = "and"

    def decide(self, observation: Observation) -> int:
        first_enabled = self.first.evaluate(observation)
        if self.second is None:
            return 1 if first_enabled else 0
        second_enabled = self.second.evaluate(observation)
        if self.operator == "and":
            return 1 if first_enabled and second_enabled else 0
        if self.operator == "or":
            return 1 if first_enabled or second_enabled else 0
        raise ValueError(f"unknown BooleanTreeSwitch operator: {self.operator}")

    def describe(self) -> str:
        if self.second is None:
            return f"mode=1 if {self.first.describe()}, else mode=0"
        if self.operator not in {"and", "or"}:
            raise ValueError(f"unknown BooleanTreeSwitch operator: {self.operator}")
        return (
            f"mode=1 if {self.first.describe()} {self.operator} "
            f"{self.second.describe()}, else mode=0"
        )

    @property
    def node_count(self) -> int:
        return 1 if self.second is None else 2


SwitchProgram = Depth2Switch | BooleanTreeSwitch


@dataclass
class GaussianScalar:
    mean: float
    std: float

    def log_pdf(self, value: float) -> float:
        std = max(float(self.std), MIN_GAUSSIAN_STD)
        z = (float(value) - float(self.mean)) / std
        return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


@dataclass
class CartpoleSegment:
    observations: List[Observation]
    action_parameter: float
    duration: int
    hard_mode: int
    timing_duration: float | None = None
    timing_step_scale: float = 1.0

    @property
    def end_observation(self) -> Observation:
        return self.observations[-1]

    @property
    def switch_timing_duration(self) -> float:
        return float(self.duration if self.timing_duration is None else self.timing_duration)


@dataclass(frozen=True)
class _SwitchExampleCache:
    labels: Tuple[int, ...]
    columns: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class _SwitchTimingPair:
    observations: Tuple[Observation, ...]
    columns: Tuple[Tuple[float, ...], ...]
    duration: int
    timing_duration: float
    timing_step_scale: float
    off_to_on_weight: float
    on_to_off_weight: float
    stay_off_weight: float
    stay_on_weight: float


@dataclass(frozen=True)
class _ScalarSwitchTimingPair:
    relation: str
    current_value: float | None
    previous_enable_extreme: float | None
    previous_disable_extreme: float | None
    off_to_on_weight: float
    on_to_off_weight: float
    stay_off_weight: float
    stay_on_weight: float


@dataclass
class ProbabilisticCartpoleStudent:
    action_distributions: Dict[int, GaussianScalar]
    switch: SwitchProgram
    switch_threshold_distribution: GaussianScalar
    switch_parameter_distributions: List[GaussianScalar]
    responsibilities: List[Tuple[float, float]]

    def to_deterministic_policy(self) -> "SynthesizedCartpolePSM":
        return SynthesizedCartpolePSM(
            self.action_distributions[0].mean,
            self.action_distributions[1].mean,
            _switch_with_distribution_means(self.switch, self.switch_parameter_distributions),
        )

    def sample_policy(self, rng: random.Random) -> "SynthesizedCartpolePSM":
        return SynthesizedCartpolePSM(
            rng.gauss(self.action_distributions[0].mean, self.action_distributions[0].std),
            rng.gauss(self.action_distributions[1].mean, self.action_distributions[1].std),
            _sample_switch(self.switch, self.switch_parameter_distributions, rng),
        )

    def sample_segment_resampling_policy(self, rng: random.Random) -> "SampledCartpolePSM":
        return SampledCartpolePSM(self, rng)

    def describe(self) -> str:
        left = self.action_distributions[0]
        right = self.action_distributions[1]
        threshold = self.switch_threshold_distribution
        switch_params = ", ".join(
            f"N({param.mean:.3f}, {param.std:.3f})"
            for param in self.switch_parameter_distributions
        )
        return (
            f"H0=N({left.mean:.3f}, {left.std:.3f}); "
            f"H1=N({right.mean:.3f}, {right.std:.3f}); "
            f"threshold=N({threshold.mean:.3f}, {threshold.std:.3f}); "
            f"G=[{switch_params}]"
        )


def cartpole_switch_fit_diagnostics(
    traces: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent,
) -> Dict[str, object]:
    """Summarize trace-fit terms for metrics provenance, not policy selection."""

    segments_by_trace = _segments_from_traces(traces)
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    responsibilities = student.responsibilities
    responsibility_segment_count_match = len(responsibilities) == len(flat_segments)
    if len(responsibilities) != len(flat_segments):
        responsibilities = [
            _mode_responsibilities(segment.action_parameter, student.action_distributions)
            for segment in flat_segments
        ]
    examples = [
        (observation, label)
        for trace in traces
        for observation, label in zip(trace.observations, trace.mode_labels)
    ]
    selected_switch = _switch_with_distribution_means(
        student.switch,
        student.switch_parameter_distributions,
    )
    fixed_reference_switch = Depth2Switch(10.0, 1.0, 0.0)
    num_boundaries = _trace_boundary_count(segments_by_trace)
    return {
        "diagnostic_scope": "local_teacher_trace_fit",
        "not_paper_reproduction": True,
        "note": (
            "Trace-fit diagnostics for the current local synthesizer. These costs "
            "are not paper-scale reproduction results or closed-loop evaluations."
        ),
        "selection_objective_order": list(SWITCH_SELECTION_OBJECTIVE_ORDER),
        "distribution_rescore_top_k": SWITCH_STRUCTURE_RESCORING_TOP_K,
        "prefilter_objective_order": list(SWITCH_PREFILTER_OBJECTIVE_ORDER),
        "example_count": len(examples),
        "num_trace_steps": len(examples),
        "segment_count": len(flat_segments),
        "num_segments": len(flat_segments),
        "num_boundaries": num_boundaries,
        "responsibility_segment_count_match": responsibility_segment_count_match,
        "candidates": {
            "selected_student_switch": _switch_fit_summary(
                selected_switch,
                examples,
                segments_by_trace,
                responsibilities,
            ),
            "fixed_local_reference_switch": _switch_fit_summary(
                fixed_reference_switch,
                examples,
                segments_by_trace,
                responsibilities,
            ),
        },
    }


def _switch_fit_summary(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Dict[str, object]:
    mistakes, timing_loss, complexity, description = _switch_cost(
        switch,
        examples,
        segments_by_trace,
        responsibilities,
    )
    (
        refined_switch,
        structure_label_loss,
        distribution_loss,
        structure_complexity,
        structure_description,
    ) = _fit_switch_structure_objective(
        switch,
        examples,
        segments_by_trace,
        responsibilities,
    )
    example_count = len(examples)
    num_boundaries = _trace_boundary_count(segments_by_trace)
    structure_mistakes = _switch_label_mistakes(refined_switch, examples)
    label_error_rate = structure_mistakes / example_count if example_count else 0.0
    deterministic_label_error_rate = mistakes / example_count if example_count else 0.0
    return {
        "description": description,
        "objective_description": structure_description,
        "label_mistakes": structure_mistakes,
        "label_error_rate": label_error_rate,
        "hard_label_mistakes": structure_mistakes,
        "hard_label_mistake_rate": label_error_rate,
        "responsibility_weighted_label_loss": structure_label_loss,
        "timing_loss_total": distribution_loss,
        "timing_loss_per_boundary": distribution_loss / num_boundaries if num_boundaries else 0.0,
        "bounded_eq12_style_distribution_loss": distribution_loss,
        "eq12_style_timing_loss": timing_loss,
        "program_complexity": structure_complexity,
        "deterministic_hard_label_mistakes": mistakes,
        "deterministic_label_error_rate": deterministic_label_error_rate,
        "deterministic_eq12_style_timing_loss": timing_loss,
        "deterministic_objective_tuple": [mistakes, timing_loss, complexity, description],
        "boundary_alignment": _switch_boundary_alignment(switch, segments_by_trace),
        "objective_boundary_alignment": _switch_boundary_alignment(refined_switch, segments_by_trace),
        "objective_tuple": [structure_label_loss, distribution_loss, structure_complexity, structure_description],
    }


def _trace_boundary_count(segments_by_trace: List[List[CartpoleSegment]]) -> int:
    return sum(max(len(trace_segments) - 1, 0) for trace_segments in segments_by_trace)


def _switch_boundary_alignment(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]],
) -> Dict[str, object]:
    early = 0
    at_boundary = 0
    late = 0
    never = 0
    elapsed_early = 0
    elapsed_at_boundary = 0
    elapsed_late = 0
    deltas: List[int] = []
    timing_deltas: List[float] = []
    for trace_segments in segments_by_trace:
        for segment in trace_segments[:-1]:
            first_enabled = _first_enabled_step(switch, segment.observations)
            if first_enabled > len(segment.observations):
                never += 1
                continue
            delta = first_enabled - segment.duration
            timing_delta = _enabled_step_elapsed_time(first_enabled, segment.timing_step_scale) - segment.switch_timing_duration
            deltas.append(delta)
            timing_deltas.append(timing_delta)
            if first_enabled < segment.duration:
                early += 1
            elif first_enabled == segment.duration:
                at_boundary += 1
            else:
                late += 1
            if timing_delta < -MIN_GAUSSIAN_STD:
                elapsed_early += 1
            elif timing_delta > MIN_GAUSSIAN_STD:
                elapsed_late += 1
            else:
                elapsed_at_boundary += 1
    return {
        "num_boundaries": _trace_boundary_count(segments_by_trace),
        "enabled_boundary_count": len(deltas),
        "early_switch_count": early,
        "at_boundary_count": at_boundary,
        "late_switch_count": late,
        "never_enabled_count": never,
        "elapsed_early_switch_count": elapsed_early,
        "elapsed_at_boundary_count": elapsed_at_boundary,
        "elapsed_late_switch_count": elapsed_late,
        "first_enabled_minus_duration_mean": sum(deltas) / len(deltas) if deltas else None,
        "first_enabled_minus_duration_min": min(deltas) if deltas else None,
        "first_enabled_minus_duration_max": max(deltas) if deltas else None,
        "first_enabled_elapsed_minus_duration_mean": sum(timing_deltas) / len(timing_deltas) if timing_deltas else None,
        "first_enabled_elapsed_minus_duration_min": min(timing_deltas) if timing_deltas else None,
        "first_enabled_elapsed_minus_duration_max": max(timing_deltas) if timing_deltas else None,
    }


class SynthesizedCartpolePSM:
    """Two-mode constant-action Cartpole policy synthesized from traces."""

    def __init__(self, left_force: float, right_force: float, switch: SwitchProgram) -> None:
        self.left_force = left_force
        self.right_force = right_force
        self.switch = switch
        self.mode = 0

    def reset(self) -> None:
        self.mode = 0

    def act(self, observation: Observation) -> float:
        current_mode = self.mode
        action = self.right_force if current_mode == 1 else self.left_force
        self.mode = self.switch.decide(observation)
        return action

    def describe(self) -> str:
        return (
            f"m0 action={self.left_force:.3f}; m1 action={self.right_force:.3f}; "
            f"{self.switch.describe()}"
        )


class SampledCartpolePSM:
    """Probabilistic PSM execution that resamples parameters on mode changes."""

    def __init__(self, student: ProbabilisticCartpoleStudent, rng: random.Random) -> None:
        self.student = student
        self.rng = rng
        self.mode = 0
        self.left_force = 0.0
        self.right_force = 0.0
        self.switch: SwitchProgram = student.switch

    def reset(self) -> None:
        self.mode = 0
        self._resample_segment_parameters(self.mode)

    def act(self, observation: Observation) -> float:
        current_mode = self.mode
        action = self.right_force if current_mode == 1 else self.left_force
        next_mode = self.switch.decide(observation)
        if next_mode != self.mode:
            self.mode = next_mode
            self._resample_segment_parameters(self.mode)
        return action

    def _resample_segment_parameters(self, mode: int) -> None:
        if mode == 0:
            self.left_force = self._sample_action(0)
        else:
            self.right_force = self._sample_action(1)
        self.switch = _sample_switch(
            self.student.switch,
            self.student.switch_parameter_distributions,
            self.rng,
        )

    def _sample_action(self, mode: int) -> float:
        distribution = self.student.action_distributions[mode]
        return self.rng.gauss(distribution.mean, distribution.std)


@dataclass
class CartpoleSynthesisIteration:
    iteration: int
    traces: List[CartpoleTrace]
    student: ProbabilisticCartpoleStudent
    student_fit_history: List["CartpoleStudentFitStep"]


@dataclass
class CartpoleStudentFitStep:
    em_iteration: int
    responsibility_pass: int
    phase: str
    responsibilities: List[Tuple[float, float]]
    switch_pair_responsibilities: List[Tuple[float, float, float, float]]
    action_distributions: Dict[int, GaussianScalar]
    switch: SwitchProgram
    switch_parameter_distributions: List[GaussianScalar]


def synthesize_cartpole_policy(cfg: CartpoleSynthesisConfig) -> tuple[SynthesizedCartpolePSM, List[CartpoleTrace]]:
    student, traces = synthesize_cartpole_student(cfg)
    return student.to_deterministic_policy(), traces


def synthesize_cartpole_student(cfg: CartpoleSynthesisConfig) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleTrace]]:
    student, traces, _ = synthesize_cartpole_student_with_history(cfg)
    return student, traces


def synthesize_cartpole_student_with_history(
    cfg: CartpoleSynthesisConfig,
) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleTrace], List[CartpoleSynthesisIteration]]:
    rng = random.Random(cfg.seed)
    env = CartpoleEnv.train_env(seed=cfg.seed)
    initial_states = [env.reset() for _ in range(cfg.num_initial_states)]
    student: ProbabilisticCartpoleStudent | None = None
    traces: List[CartpoleTrace] = []
    history: List[CartpoleSynthesisIteration] = []
    # Alternate between a teacher that searches for high-reward traces and a
    # student fit that makes later teacher traces easier to explain with the PSM.
    for iteration in range(max(1, cfg.teacher_student_iters)):
        traces = [
            _optimize_loop_free_trace(initial_state, env.cfg, cfg, rng, student)
            for initial_state in initial_states
        ]
        student, student_fit_history = fit_probabilistic_cartpole_student_with_history(traces, cfg)
        history.append(CartpoleSynthesisIteration(iteration + 1, traces, student, student_fit_history))
    if student is None:
        raise RuntimeError("Cartpole synthesis did not produce a student policy")
    return student, traces, history


def fit_probabilistic_cartpole_student(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> ProbabilisticCartpoleStudent:
    student, _ = fit_probabilistic_cartpole_student_with_history(traces, cfg)
    return student


def fit_probabilistic_cartpole_student_with_history(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleStudentFitStep]]:
    """Fit the Cartpole student using Gaussian action-parameter distributions.

    This implements the action-distribution part of the paper's EM-style
    student step for Cartpole's constant-action grammar. The latent segment
    responsibilities are initialized from action likelihoods, then each bounded
    EM iteration alternates switch-timing forward-backward responsibilities with
    refits of the action distributions and switch parameters.
    Switch timing still uses local Gaussian mean/std refinement rather than the
    paper's full continuous M-step.
    """

    segments_by_trace = _segments_from_traces(traces)
    segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    left_default = min(cfg.force_values)
    right_default = max(cfg.force_values)
    action_distributions = {
        0: GaussianScalar(left_default, 1.0),
        1: GaussianScalar(right_default, 1.0),
    }
    responsibilities: List[Tuple[float, float]] = []
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] = []
    switch: SwitchProgram | None = None
    switch_parameter_distributions: List[GaussianScalar] = []
    fit_history: List[CartpoleStudentFitStep] = []

    for iteration in range(max(1, cfg.student_em_iters)):
        if iteration == 0 or cfg.student_switch_responsibility_passes <= 0:
            responsibilities = _action_likelihood_responsibilities(segments, action_distributions)
            responsibilities = _condition_initial_mode_responsibilities(segments_by_trace, responsibilities)
            action_distributions = _fit_action_distributions(
                segments,
                responsibilities,
                left_default,
                right_default,
            )
            switch, switch_parameter_distributions = _fit_student_switch(
                traces,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities or None,
            )
            fit_history.append(
                _student_fit_step(
                    iteration + 1,
                    0,
                    "action_likelihood_initialization" if iteration == 0 else "action_likelihood_refit",
                    responsibilities,
                    switch_pair_responsibilities,
                    action_distributions,
                    switch,
                    switch_parameter_distributions,
                )
            )

        if cfg.student_switch_responsibility_passes <= 0:
            continue
        if switch is None:
            raise RuntimeError("Cartpole student EM requires an initialized switch")
        for pass_index in range(cfg.student_switch_responsibility_passes):
            responsibilities, switch_pair_responsibilities = _refine_responsibilities_and_switch_pairs_with_timing(
                segments_by_trace,
                action_distributions,
                switch,
                switch_parameter_distributions,
            )
            action_distributions = _fit_action_distributions(
                segments,
                responsibilities,
                left_default,
                right_default,
            )
            switch, switch_parameter_distributions = _fit_student_switch(
                traces,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities or None,
            )
            fit_history.append(
                _student_fit_step(
                    iteration + 1,
                    pass_index + 1,
                    "switch_timing_refinement",
                    responsibilities,
                    switch_pair_responsibilities,
                    action_distributions,
                    switch,
                    switch_parameter_distributions,
                )
            )

    if switch is None:
        switch, switch_parameter_distributions = _fit_student_switch(
            traces,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities or None,
        )

    threshold_distribution = (
        switch_parameter_distributions[0]
        if switch_parameter_distributions
        else GaussianScalar(_switch_default_threshold(switch), 1.0)
    )
    student = ProbabilisticCartpoleStudent(
        action_distributions,
        switch,
        threshold_distribution,
        switch_parameter_distributions,
        responsibilities,
    )
    return student, fit_history


def _student_fit_step(
    em_iteration: int,
    responsibility_pass: int,
    phase: str,
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
) -> CartpoleStudentFitStep:
    return CartpoleStudentFitStep(
        em_iteration=em_iteration,
        responsibility_pass=responsibility_pass,
        phase=phase,
        responsibilities=list(responsibilities),
        switch_pair_responsibilities=list(switch_pair_responsibilities),
        action_distributions=dict(action_distributions),
        switch=switch,
        switch_parameter_distributions=list(switch_parameter_distributions),
    )


def _action_likelihood_responsibilities(
    segments: List[CartpoleSegment],
    action_distributions: Dict[int, GaussianScalar],
) -> List[Tuple[float, float]]:
    # The actions are observed, but their latent mode assignments are softened
    # so ambiguous segments can influence both constant-action primitives.
    return [
        _mode_responsibilities(segment.action_parameter, action_distributions)
        for segment in segments
    ]


def _condition_initial_mode_responsibilities(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    conditioned = list(responsibilities)
    offset = 0
    for trace_segments in segments_by_trace:
        if trace_segments and offset < len(conditioned):
            conditioned[offset] = (1.0, 0.0)
        offset += len(trace_segments)
    return conditioned


def _bootstrap_probabilistic_student(cfg: CartpoleSynthesisConfig) -> ProbabilisticCartpoleStudent:
    threshold = GaussianScalar(TEACHER_BOOTSTRAP_SWITCH_THRESHOLD, TEACHER_BOOTSTRAP_SWITCH_STD)
    return ProbabilisticCartpoleStudent(
        action_distributions={
            0: GaussianScalar(min(cfg.force_values), TEACHER_BOOTSTRAP_ACTION_STD),
            1: GaussianScalar(max(cfg.force_values), TEACHER_BOOTSTRAP_ACTION_STD),
        },
        switch=Depth2Switch(
            TEACHER_BOOTSTRAP_SWITCH_THETA_WEIGHT,
            TEACHER_BOOTSTRAP_SWITCH_OMEGA_WEIGHT,
            TEACHER_BOOTSTRAP_SWITCH_THRESHOLD,
        ),
        switch_threshold_distribution=threshold,
        switch_parameter_distributions=[threshold],
        responsibilities=[(0.5, 0.5)],
    )


def _optimize_loop_free_trace(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace:
    # The "teacher" is restricted to loop-free bang-bang traces; ranking by the
    # student likelihood is the local adaptive-teaching approximation.
    scoring_student = student or _bootstrap_probabilistic_student(cfg)
    candidates = _teacher_candidate_traces(initial_state, env_cfg, cfg, rng, student)
    # Refine only the top candidates to keep synthesis cheap while still
    # optimizing around promising sampled loop-free traces.
    elites = _top_teacher_elites(candidates, scoring_student, cfg)
    elite_recombinations, refinement_elites = _elite_recombination_candidates_and_elites(
        elites,
        initial_state,
        env_cfg,
        cfg,
        rng,
        scoring_student,
    )
    refinement_seeds = elites + elite_recombinations
    refined = [
        _refine_loop_free_trace(candidate, initial_state, env_cfg, cfg, scoring_student, refinement_elites)
        for candidate in refinement_seeds
        if candidate.segment_actions and candidate.segment_durations
    ]
    selected = max(
        refinement_seeds + refined,
        key=lambda trace: _teacher_refinement_objective(trace, scoring_student, cfg, refinement_elites),
    )
    return _record_selected_teacher_objectives(selected, scoring_student, cfg, refinement_elites)


def _record_selected_teacher_objectives(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
    refinement_elites: List[CartpoleTrace],
) -> CartpoleTrace:
    trace.teacher_objective = _teacher_objective(trace, student, cfg)
    trace.teacher_refinement_objective = _teacher_refinement_objective(
        trace,
        student,
        cfg,
        refinement_elites,
    )
    return trace


def _teacher_candidate_traces(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None,
) -> List[CartpoleTrace]:
    candidate_count = max(1, cfg.candidate_rollouts)
    if student is None:
        bootstrap = _bootstrap_probabilistic_student(cfg)
        candidates = [
            _rollout_student_sampled_trace(initial_state, env_cfg, cfg, bootstrap, rng)
            for _ in range(candidate_count)
        ]
        for trace in candidates:
            trace.teacher_source = "bootstrap_student_sample"
        return candidates

    # Paper Section 4.2 samples teacher candidates from the current student
    # before keeping the top-rho elite set for local optimization.
    return [
        _rollout_student_sampled_trace(initial_state, env_cfg, cfg, student, rng)
        for _ in range(candidate_count)
    ]


def _elite_recombination_candidates(
    elites: List[CartpoleTrace],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None,
) -> List[CartpoleTrace]:
    candidates, _ = _elite_recombination_candidates_and_elites(
        elites,
        initial_state,
        env_cfg,
        cfg,
        rng,
        student,
    )
    return candidates


def _elite_recombination_candidates_and_elites(
    elites: List[CartpoleTrace],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None,
) -> Tuple[List[CartpoleTrace], List[CartpoleTrace]]:
    current_elites = _top_teacher_elites(elites, student, cfg)
    candidates: List[CartpoleTrace] = []
    centroid = _elite_centroid_trace(current_elites, initial_state, env_cfg, cfg, student)
    if centroid is not None:
        candidates.append(centroid)
    refreshed_elites, distribution_candidates = _refresh_teacher_elites_with_distribution(
        current_elites,
        initial_state,
        env_cfg,
        cfg,
        rng,
        student,
    )
    candidates.extend(distribution_candidates)
    return candidates, refreshed_elites


def _elite_distribution_sample_traces(
    elites: List[CartpoleTrace],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> List[CartpoleTrace]:
    _, samples = _refresh_teacher_elites_with_distribution(
        elites,
        initial_state,
        env_cfg,
        cfg,
        rng,
        student,
    )
    return samples


def _refresh_teacher_elites_with_distribution(
    elites: List[CartpoleTrace],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> Tuple[List[CartpoleTrace], List[CartpoleTrace]]:
    samples: List[CartpoleTrace] = []
    current_elites = _top_teacher_elites(elites, student, cfg)
    rounds = max(0, cfg.teacher_elite_distribution_rounds)
    for _ in range(rounds):
        schedules = _elite_loop_free_schedules(current_elites, env_cfg.dt)
        distribution = _fit_elite_schedule_distribution(schedules, env_cfg, cfg, student)
        if distribution is None:
            break
        round_samples: List[CartpoleTrace] = []
        mean_trace = _elite_distribution_mean_trace_from_distribution(
            distribution,
            initial_state,
            env_cfg,
            cfg,
            student,
        )
        if mean_trace is not None:
            round_samples.append(mean_trace)
        for _ in range(max(0, cfg.teacher_elite_distribution_resamples)):
            sample = _elite_distribution_sample_trace_from_distribution(
                distribution,
                initial_state,
                env_cfg,
                cfg,
                rng,
                student,
            )
            if sample is not None:
                round_samples.append(sample)
        if not round_samples:
            break
        samples.extend(round_samples)
        current_elites = _top_teacher_elites(current_elites + round_samples, student, cfg)
    return current_elites, samples


def _top_teacher_elites(
    traces: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> List[CartpoleTrace]:
    if not traces:
        return []
    top_count = max(1, cfg.teacher_top_rho)
    return sorted(
        traces,
        key=lambda trace: _teacher_objective(trace, student, cfg),
        reverse=True,
    )[:top_count]


EliteSchedule = Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[float, ...], Tuple[int, ...], CartpoleTrace]


@dataclass(frozen=True)
class _EliteSegmentDistribution:
    action: GaussianScalar
    duration: GaussianScalar
    time_increment: GaussianScalar
    mode: int


@dataclass(frozen=True)
class _EliteScheduleDistribution:
    segments: Tuple[_EliteSegmentDistribution, ...]
    theta_gain: GaussianScalar
    omega_gain: GaussianScalar
    source_elites: Tuple[CartpoleTrace, ...]


def _elite_distribution_mean_trace(
    schedules: List[EliteSchedule],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace | None:
    distribution = _fit_elite_schedule_distribution(schedules, env_cfg, cfg, student)
    if distribution is None:
        return None
    return _elite_distribution_mean_trace_from_distribution(
        distribution,
        initial_state,
        env_cfg,
        cfg,
        student,
    )


def _fit_elite_schedule_distribution(
    schedules: List[EliteSchedule],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None = None,
) -> _EliteScheduleDistribution | None:
    if not schedules:
        return None
    max_segments = min(
        max(1, cfg.segments_per_trace),
        max(len(actions) for actions, _, _, _, _ in schedules),
    )
    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    schedule_weights = _elite_schedule_weights(schedules, student, cfg)
    segment_distributions: List[_EliteSegmentDistribution] = []
    for index in range(max_segments):
        pairs = [
            (actions[index], durations[index], increments[index], modes[index], schedule_weights[schedule_index])
            for schedule_index, (actions, durations, increments, modes, _) in enumerate(schedules)
            if (
                index < len(actions)
                and index < len(durations)
                and index < len(increments)
                and index < len(modes)
            )
        ]
        if not pairs:
            break
        weights = [pair[4] for pair in pairs]
        action_mean, action_std = _weighted_mean_and_std(
            [pair[0] for pair in pairs],
            weights,
            TEACHER_ELITE_RESAMPLE_MIN_ACTION_STD,
        )
        action_distribution = GaussianScalar(
            max(lower, min(upper, action_mean)),
            action_std,
        )
        duration_mean, duration_std = _weighted_mean_and_std([float(pair[1]) for pair in pairs], weights, 1.0)
        increment_mean, increment_std = _weighted_mean_and_std([pair[2] for pair in pairs], weights, MIN_GAUSSIAN_STD)
        segment_distributions.append(
            _EliteSegmentDistribution(
                action=action_distribution,
                duration=GaussianScalar(duration_mean, duration_std),
                time_increment=GaussianScalar(increment_mean, increment_std),
                mode=_weighted_majority_mode([pair[3] for pair in pairs], weights),
            )
        )
    if not segment_distributions:
        return None

    theta_gain_mean, theta_gain_std = _weighted_mean_and_std(
        [trace.theta_gain for _, _, _, _, trace in schedules],
        schedule_weights,
        TEACHER_GAIN_SAMPLE_MIN_STD,
    )
    omega_gain_mean, omega_gain_std = _weighted_mean_and_std(
        [trace.omega_gain for _, _, _, _, trace in schedules],
        schedule_weights,
        TEACHER_GAIN_SAMPLE_MIN_STD,
    )
    return _EliteScheduleDistribution(
        segments=tuple(segment_distributions),
        theta_gain=GaussianScalar(theta_gain_mean, theta_gain_std),
        omega_gain=GaussianScalar(omega_gain_mean, omega_gain_std),
        source_elites=tuple(trace for _, _, _, _, trace in schedules),
    )


def _elite_distribution_mean_trace_from_distribution(
    distribution: _EliteScheduleDistribution,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace | None:
    if not distribution.segments:
        return None
    max_duration = max(1, cfg.segment_steps)
    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    mean_actions = [
        max(lower, min(upper, segment.action.mean))
        for segment in distribution.segments
    ]
    mean_durations = [
        min(max_duration, max(1, int(math.floor(segment.duration.mean + 0.5))))
        for segment in distribution.segments
    ]
    mean_increments = [
        _clamp_time_increment(env_cfg, segment.time_increment.mean)
        for segment in distribution.segments
    ]
    mean_modes = [segment.mode for segment in distribution.segments]
    mean_trace = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        distribution.theta_gain.mean,
        distribution.omega_gain.mean,
        tuple(mean_durations),
        tuple(mean_actions),
        tuple(mean_increments),
        tuple(mean_modes),
    )
    mean_trace.teacher_source = _elite_distribution_mean_source(list(distribution.source_elites))
    mean_trace.student_log_probability = (
        _trace_log_probability(mean_trace, student)
        if student is not None
        else None
    )
    return mean_trace


def _elite_centroid_trace(
    elites: List[CartpoleTrace],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace | None:
    schedules = _elite_loop_free_schedules(elites, env_cfg.dt)
    if not schedules:
        return None

    max_segments = min(
        max(1, cfg.segments_per_trace),
        max(len(actions) for actions, _, _, _, _ in schedules),
    )
    max_duration = max(1, cfg.segment_steps)
    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    centroid_actions: List[float] = []
    centroid_durations: List[int] = []
    centroid_increments: List[float] = []
    centroid_modes: List[int] = []
    for index in range(max_segments):
        pairs = [
            (actions[index], durations[index], increments[index], modes[index])
            for actions, durations, increments, modes, _ in schedules
            if (
                index < len(actions)
                and index < len(durations)
                and index < len(increments)
                and index < len(modes)
            )
        ]
        if not pairs:
            break
        action = sum(pair[0] for pair in pairs) / len(pairs)
        duration = sum(pair[1] for pair in pairs) / len(pairs)
        increment = sum(pair[2] for pair in pairs) / len(pairs)
        centroid_actions.append(max(lower, min(upper, action)))
        centroid_durations.append(min(max_duration, max(1, int(math.floor(duration + 0.5)))))
        centroid_increments.append(_clamp_time_increment(env_cfg, increment))
        centroid_modes.append(_majority_mode([pair[3] for pair in pairs]))
    if not centroid_actions or not centroid_durations or not centroid_increments:
        return None

    theta_gain = sum(trace.theta_gain for _, _, _, _, trace in schedules) / len(schedules)
    omega_gain = sum(trace.omega_gain for _, _, _, _, trace in schedules) / len(schedules)
    centroid = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        theta_gain,
        omega_gain,
        tuple(centroid_durations),
        tuple(centroid_actions),
        tuple(centroid_increments),
        tuple(centroid_modes),
    )
    centroid.teacher_source = _elite_centroid_source(elites)
    centroid.student_log_probability = (
        _trace_log_probability(centroid, student)
        if student is not None
        else None
    )
    return centroid


def _elite_distribution_sample_trace(
    schedules: List[EliteSchedule],
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace | None:
    distribution = _fit_elite_schedule_distribution(schedules, env_cfg, cfg)
    if distribution is None:
        return None
    return _elite_distribution_sample_trace_from_distribution(
        distribution,
        initial_state,
        env_cfg,
        cfg,
        rng,
        student,
    )


def _elite_distribution_sample_trace_from_distribution(
    distribution: _EliteScheduleDistribution,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace | None:
    if not distribution.segments:
        return None
    max_duration = max(1, cfg.segment_steps)
    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    sampled_actions: List[float] = []
    sampled_durations: List[int] = []
    sampled_increments: List[float] = []
    sampled_modes: List[int] = []
    for segment in distribution.segments:
        sampled_actions.append(max(lower, min(upper, rng.gauss(segment.action.mean, segment.action.std))))
        sampled_durations.append(
            min(
                max_duration,
                max(1, int(math.floor(rng.gauss(segment.duration.mean, segment.duration.std) + 0.5))),
            )
        )
        sampled_increments.append(
            _clamp_time_increment(env_cfg, rng.gauss(segment.time_increment.mean, segment.time_increment.std))
        )
        sampled_modes.append(segment.mode)
    sample = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        rng.gauss(distribution.theta_gain.mean, distribution.theta_gain.std),
        rng.gauss(distribution.omega_gain.mean, distribution.omega_gain.std),
        tuple(sampled_durations),
        tuple(sampled_actions),
        tuple(sampled_increments),
        tuple(sampled_modes),
    )
    sample.teacher_source = _elite_distribution_source(list(distribution.source_elites))
    sample.student_log_probability = (
        _trace_log_probability(sample, student)
        if student is not None
        else None
    )
    return sample


def _elite_loop_free_schedules(
    elites: List[CartpoleTrace],
    default_time_increment: float,
) -> List[EliteSchedule]:
    schedules: List[EliteSchedule] = []
    for trace in elites:
        actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
        durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
        increments = trace.segment_time_increments or tuple(default_time_increment for _ in durations)
        modes = _segment_modes_from_trace(trace, actions, durations)
        if len(increments) < len(durations):
            increments = tuple(increments) + tuple(
                default_time_increment
                for _ in range(len(durations) - len(increments))
            )
        increments = tuple(increments[: len(durations)])
        if actions and durations and modes:
            schedules.append((actions, durations, increments, modes, trace))
    return schedules


def _segment_modes_from_trace(
    trace: CartpoleTrace,
    actions: Tuple[float, ...],
    durations: Tuple[int, ...],
) -> Tuple[int, ...]:
    modes: List[int] = []
    start = 0
    source_durations = trace.segment_durations or durations
    for index, duration in enumerate(source_durations):
        if trace.mode_labels and start < len(trace.mode_labels):
            modes.append(int(trace.mode_labels[start]))
        elif index < len(actions):
            modes.append(1 if actions[index] > 0.0 else 0)
        start += max(1, int(duration))
        if len(modes) >= len(durations):
            break
    return tuple(modes)


def _majority_mode(modes: List[int]) -> int:
    return 1 if sum(1 for mode in modes if mode == 1) > len(modes) / 2.0 else 0


def _weighted_majority_mode(modes: List[int], weights: List[float]) -> int:
    if not modes or len(modes) != len(weights):
        return _majority_mode(modes)
    positive_weight = sum(weight for mode, weight in zip(modes, weights) if mode == 1)
    total_weight = sum(weights)
    return 1 if positive_weight > total_weight / 2.0 else 0


def _elite_schedule_weights(
    schedules: List[EliteSchedule],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> List[float]:
    if not schedules:
        return []
    if student is None:
        return [1.0 / len(schedules) for _ in schedules]
    objective_values = [
        _teacher_objective(trace, student, cfg)
        for _, _, _, _, trace in schedules
    ]
    max_objective = max(objective_values)
    raw_weights = [math.exp(value - max_objective) for value in objective_values]
    total = sum(raw_weights)
    if total <= 0.0:
        return [1.0 / len(schedules) for _ in schedules]
    return [weight / total for weight in raw_weights]


def _mean_and_std(values: List[float], std_floor: float) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, max(std_floor, math.sqrt(max(variance, 0.0)))


def _weighted_mean_and_std(values: List[float], weights: List[float], std_floor: float) -> Tuple[float, float]:
    if not values:
        return 0.0, std_floor
    if len(values) != len(weights):
        return _mean_and_std(values, std_floor)
    total = sum(weights)
    if total <= 0.0:
        return _mean_and_std(values, std_floor)
    mean = sum(value * weight for value, weight in zip(values, weights)) / total
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total
    return mean, max(std_floor, math.sqrt(max(variance, 0.0)))


def _elite_centroid_source(elites: List[CartpoleTrace]) -> str:
    sources = [trace.teacher_source for trace in elites]
    if sources and all(source.startswith("bootstrap_") for source in sources):
        return "bootstrap_elite_centroid"
    if sources and all(source.startswith("student_") for source in sources):
        return "student_elite_centroid"
    return "elite_centroid"


def _elite_distribution_source(elites: List[CartpoleTrace]) -> str:
    sources = [trace.teacher_source for trace in elites]
    if sources and all(source.startswith("bootstrap_") for source in sources):
        return "bootstrap_elite_distribution_sample"
    if sources and all(source.startswith("student_") for source in sources):
        return "student_elite_distribution_sample"
    return "elite_distribution_sample"


def _elite_distribution_mean_source(elites: List[CartpoleTrace]) -> str:
    sources = [trace.teacher_source for trace in elites]
    if sources and all(source.startswith("bootstrap_") for source in sources):
        return "bootstrap_elite_distribution_mean"
    if sources and all(source.startswith("student_") for source in sources):
        return "student_elite_distribution_mean"
    return "elite_distribution_mean"


def _rollout_loop_free_candidate(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
) -> CartpoleTrace:
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    max_steps = cfg.segment_steps * cfg.segments_per_trace
    theta_gain = rng.gauss(
        cfg.teacher_theta_gain,
        max(TEACHER_GAIN_SAMPLE_MIN_STD, abs(cfg.teacher_theta_gain) * TEACHER_GAIN_SAMPLE_STD_FRACTION),
    )
    omega_gain = rng.gauss(
        cfg.teacher_omega_gain,
        max(TEACHER_GAIN_SAMPLE_MIN_STD, abs(cfg.teacher_omega_gain) * TEACHER_GAIN_SAMPLE_STD_FRACTION),
    )
    return _rollout_with_teacher_gains(initial_state, env_cfg, cfg, theta_gain, omega_gain)


def _rollout_with_teacher_gains(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    theta_gain: float,
    omega_gain: float,
    segment_durations: Tuple[int, ...] | None = None,
    segment_actions: Tuple[float, ...] | None = None,
    segment_time_increments: Tuple[float, ...] | None = None,
    segment_modes: Tuple[int, ...] | None = None,
) -> CartpoleTrace:
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    normalized_reward = 0.0
    max_segment_steps = max(1, cfg.segment_steps)
    max_segments = max(1, cfg.segments_per_trace)
    max_steps = min(env_cfg.max_steps, max_segment_steps * max_segments)
    durations = segment_durations or tuple(max_segment_steps for _ in range(max_segments))
    durations = tuple(durations[:max_segments])
    increments = segment_time_increments or tuple(env_cfg.dt for _ in range(len(durations)))
    increments = tuple(max(MIN_GAUSSIAN_STD, min(env_cfg.dt, float(value))) for value in increments[: len(durations)])
    chosen_actions: List[float] = []
    started_durations: List[int] = []
    started_increments: List[float] = []
    for segment_index, duration in enumerate(durations):
        if cartpole_done(state, env_cfg) or alive >= max_steps:
            break
        duration_steps = min(max(1, duration), max_segment_steps, max_steps - alive)
        if duration_steps <= 0:
            break
        if segment_actions is not None and segment_index < len(segment_actions):
            action = segment_actions[segment_index]
        else:
            _, _, theta, omega = state
            # Random gains choose the next loop-free action function; the final
            # policy is learned from the trace rather than using these gains.
            raw_force = theta_gain * theta + omega_gain * omega
            action = max(cfg.force_values) if raw_force >= 0.0 else min(cfg.force_values)
        segment_dt = increments[segment_index] if segment_index < len(increments) else env_cfg.dt
        label = (
            int(segment_modes[segment_index])
            if segment_modes is not None and segment_index < len(segment_modes)
            else 1 if action > 0.0 else 0
        )
        executed_steps = 0
        for _ in range(duration_steps):
            if cartpole_done(state, env_cfg):
                break
            observations.append(list(state))
            actions.append(action)
            mode_labels.append(label)
            state = cartpole_next_state(state, action, env_cfg, segment_dt)
            alive += 1
            normalized_reward += segment_dt / env_cfg.dt
            executed_steps += 1
            if alive >= max_steps:
                break
        if executed_steps:
            chosen_actions.append(action)
            started_durations.append(executed_steps)
            started_increments.append(segment_dt)
    return CartpoleTrace(
        observations=observations,
        actions=actions,
        mode_labels=mode_labels,
        reward=normalized_reward,
        theta_gain=theta_gain,
        omega_gain=omega_gain,
        segment_actions=tuple(chosen_actions),
        segment_durations=tuple(started_durations),
        segment_time_increments=tuple(started_increments),
        teacher_source="gain_sample",
    )


def _rollout_student_sampled_trace(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent,
    rng: random.Random,
) -> CartpoleTrace:
    # The paper's probabilistic PSM resamples action and switch parameters
    # whenever execution enters a mode segment.
    policy = student.sample_segment_resampling_policy(rng)
    policy.reset()
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    max_steps = min(env_cfg.max_steps, cfg.segment_steps * cfg.segments_per_trace)
    for _ in range(max_steps):
        if cartpole_done(state, env_cfg):
            break
        observation = list(state)
        current_mode = policy.mode
        action = policy.act(observation)
        observations.append(observation)
        actions.append(action)
        mode_labels.append(current_mode)
        state = cartpole_next_state(state, action, env_cfg)
        alive += 1
    segment_actions = _mode_run_actions(actions, mode_labels)
    segment_durations = _mode_run_lengths(mode_labels)
    trace = CartpoleTrace(
        observations=observations,
        actions=actions,
        mode_labels=mode_labels,
        reward=float(alive),
        segment_actions=segment_actions,
        segment_durations=segment_durations,
        segment_time_increments=tuple(env_cfg.dt for _ in segment_durations),
        teacher_source="student_sample",
    )
    trace = _limit_loop_free_trace_segment_budget(trace, initial_state, env_cfg, cfg, student)
    if trace.student_log_probability is None:
        trace.student_log_probability = _trace_log_probability(trace, student)
    return trace


def _limit_loop_free_trace_segment_budget(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    if len(actions) != len(durations):
        raise ValueError("loop-free action count must match duration count")
    if not trace.segment_time_increments and durations:
        trace.segment_time_increments = tuple(env_cfg.dt for _ in durations)
    max_segments = max(1, cfg.segments_per_trace)
    max_segment_steps = max(1, cfg.segment_steps)
    if len(actions) <= max_segments and all(duration <= max_segment_steps for duration in durations):
        return trace

    # Student samples are closed-loop PSMs, but the paper's teacher candidates
    # are loop-free programs with both a segment-count and segment-time budget.
    projected_actions, projected_durations, projected_modes = _chunk_actions_to_loop_free_segments(
        tuple(trace.actions),
        tuple(trace.mode_labels),
        max_segment_steps,
        max_segments,
    )
    limited = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain,
        trace.omega_gain,
        projected_durations,
        projected_actions,
        tuple(env_cfg.dt for _ in projected_durations),
        segment_modes=projected_modes,
    )
    limited.teacher_source = trace.teacher_source
    limited.student_log_probability = (
        _trace_log_probability(limited, student)
        if student is not None
        else trace.student_log_probability
    )
    return limited


def _chunk_actions_to_loop_free_segments(
    actions: Tuple[float, ...],
    mode_labels: Tuple[int, ...],
    max_segment_steps: int,
    max_segments: int,
) -> Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[int, ...]]:
    if max_segment_steps < 1:
        raise ValueError("max_segment_steps must be positive")
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    if mode_labels and len(mode_labels) != len(actions):
        raise ValueError("action count must match mode label count")

    projected_actions: List[float] = []
    projected_durations: List[int] = []
    projected_modes: List[int] = []
    current_action: float | None = None
    current_mode: int | None = None
    current_duration = 0
    for index, action in enumerate(actions):
        mode = int(mode_labels[index]) if mode_labels else 1 if action > 0.0 else 0
        if current_action is None:
            current_action = action
            current_mode = mode
            current_duration = 1
        elif action == current_action and mode == current_mode and current_duration < max_segment_steps:
            current_duration += 1
        else:
            projected_actions.append(float(current_action))
            projected_durations.append(current_duration)
            projected_modes.append(_projected_segment_mode(current_action, current_mode))
            if len(projected_actions) >= max_segments:
                break
            current_action = action
            current_mode = mode
            current_duration = 1
    if current_action is not None and len(projected_actions) < max_segments:
        projected_actions.append(float(current_action))
        projected_durations.append(current_duration)
        projected_modes.append(_projected_segment_mode(current_action, current_mode))
    return tuple(projected_actions), tuple(projected_durations), tuple(projected_modes)


def _projected_segment_mode(action: float, mode: int | None) -> int:
    if mode is not None:
        return int(mode)
    return 1 if action > 0.0 else 0


def _mode_run_lengths(mode_labels: List[int]) -> Tuple[int, ...]:
    if not mode_labels:
        return ()
    durations: List[int] = []
    current = mode_labels[0]
    count = 0
    for label in mode_labels:
        if label != current:
            durations.append(count)
            current = label
            count = 0
        count += 1
    durations.append(count)
    return tuple(durations)


def _mode_run_actions(actions: List[float], mode_labels: List[int]) -> Tuple[float, ...]:
    if not actions or not mode_labels:
        return ()
    if len(actions) != len(mode_labels):
        raise ValueError("action count must match mode label count")
    run_actions = [actions[0]]
    current = mode_labels[0]
    for action, label in zip(actions[1:], mode_labels[1:]):
        if label != current:
            run_actions.append(action)
            current = label
    return tuple(run_actions)


def _refine_loop_free_trace(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None,
    elites: List[CartpoleTrace] | None = None,
) -> CartpoleTrace:
    best = trace
    objective_elites = elites or [trace]
    objective_cache: Dict[
        Tuple[float, float, Tuple[float, ...], Tuple[int, ...], Tuple[float, ...], Tuple[int, ...], float],
        float,
    ] = {}
    elite_log_normalizer = _elite_kernel_log_normalizer(student, objective_elites)

    def objective(candidate: CartpoleTrace) -> float:
        actions = tuple(candidate.segment_actions or _mode_run_actions(candidate.actions, candidate.mode_labels))
        durations = tuple(candidate.segment_durations or _mode_run_lengths(candidate.mode_labels))
        key = (
            candidate.theta_gain,
            candidate.omega_gain,
            actions,
            durations,
            tuple(candidate.segment_time_increments),
            _segment_modes_from_trace(candidate, actions, durations),
            candidate.reward,
        )
        if key not in objective_cache:
            objective_cache[key] = _teacher_refinement_objective(
                candidate,
                student,
                cfg,
                objective_elites,
                elite_log_normalizer,
            )
        return objective_cache[key]

    refine_gains = trace.teacher_source in {"gain_sample", "gain_refined"}
    theta_delta = max(
        abs(trace.theta_gain) * TEACHER_GAIN_REFINEMENT_DELTA_FRACTION,
        TEACHER_THETA_REFINEMENT_MIN_DELTA,
    )
    omega_delta = max(
        abs(trace.omega_gain) * TEACHER_GAIN_REFINEMENT_DELTA_FRACTION,
        TEACHER_OMEGA_REFINEMENT_MIN_DELTA,
    )
    for _ in range(max(0, cfg.teacher_refinement_steps)):
        improved = False
        # Coordinate-search the teacher gains because each rollout is cheap and
        # the objective includes a non-smooth student-likelihood term.
        if refine_gains:
            for theta_step, omega_step in (
                (theta_delta, 0.0),
                (-theta_delta, 0.0),
                (0.0, omega_delta),
                (0.0, -omega_delta),
            ):
                candidate = _rollout_with_teacher_gains(
                    initial_state,
                    env_cfg,
                    cfg,
                    best.theta_gain + theta_step,
                    best.omega_gain + omega_step,
                    best.segment_durations,
                    segment_time_increments=best.segment_time_increments or None,
                )
                if objective(candidate) > objective(best):
                    best = candidate
                    improved = True
            candidate = _gain_gradient_refinement_candidate(best, initial_state, env_cfg, cfg, objective)
            if candidate is not None and objective(candidate) > objective(best):
                best = candidate
                improved = True
        for candidate in _duration_refinement_candidates(best, initial_state, env_cfg, cfg):
            if objective(candidate) > objective(best):
                best = candidate
                improved = True
        for candidate in _time_increment_refinement_candidates(best, initial_state, env_cfg, cfg):
            if objective(candidate) > objective(best):
                best = candidate
                improved = True
        for candidate in _action_refinement_candidates(best, initial_state, env_cfg, cfg):
            if objective(candidate) > objective(best):
                best = candidate
                improved = True
        candidate = _action_gradient_refinement_candidate(best, initial_state, env_cfg, cfg, objective)
        if candidate is not None and objective(candidate) > objective(best):
            best = candidate
            improved = True
        candidate = _duration_gradient_refinement_candidate(best, initial_state, env_cfg, cfg, objective)
        if candidate is not None and objective(candidate) > objective(best):
            best = candidate
            improved = True
        candidate = _time_increment_gradient_refinement_candidate(best, initial_state, env_cfg, cfg, objective)
        if candidate is not None and objective(candidate) > objective(best):
            best = candidate
            improved = True
        candidate = _schedule_gradient_refinement_candidate(best, initial_state, env_cfg, cfg, objective)
        if candidate is not None and objective(candidate) > objective(best):
            best = candidate
            improved = True
        if not improved:
            theta_delta *= TEACHER_REFINEMENT_DELTA_DECAY
            omega_delta *= TEACHER_REFINEMENT_DELTA_DECAY
    if best is not trace:
        if trace.teacher_source.startswith("student_sample"):
            best.teacher_source = "student_sample_refined"
        elif trace.teacher_source.startswith("bootstrap_student_sample"):
            best.teacher_source = "bootstrap_student_sample_refined"
        elif trace.teacher_source in {
            "bootstrap_elite_centroid",
            "student_elite_centroid",
            "elite_centroid",
            "bootstrap_elite_distribution_mean",
            "student_elite_distribution_mean",
            "elite_distribution_mean",
            "bootstrap_elite_distribution_sample",
            "student_elite_distribution_sample",
            "elite_distribution_sample",
        }:
            best.teacher_source = f"{trace.teacher_source}_refined"
        else:
            best.teacher_source = "gain_refined"
        best.student_log_probability = (
            _trace_log_probability(best, student)
            if student is not None
            else best.student_log_probability
        )
    else:
        best.teacher_source = trace.teacher_source
    return best


def _duration_refinement_candidates(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
) -> List[CartpoleTrace]:
    durations = trace.segment_durations or tuple(cfg.segment_steps for _ in range(cfg.segments_per_trace))
    modes = _segment_modes_from_trace(trace, trace.segment_actions, durations)
    candidates: List[CartpoleTrace] = []
    for index, duration in enumerate(durations):
        for delta in TEACHER_DURATION_REFINEMENT_DELTAS:
            updated = list(durations)
            updated[index] = max(1, duration + delta)
            candidates.append(
                _rollout_with_teacher_gains(
                    initial_state,
                    env_cfg,
                    cfg,
                    trace.theta_gain,
                    trace.omega_gain,
                    tuple(updated),
                    trace.segment_actions or None,
                    trace.segment_time_increments or None,
                    modes,
                )
            )
    return candidates


def _time_increment_refinement_candidates(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
) -> List[CartpoleTrace]:
    durations = trace.segment_durations or tuple(cfg.segment_steps for _ in range(cfg.segments_per_trace))
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, trace.segment_actions, durations)
    candidates: List[CartpoleTrace] = []
    for index, increment in enumerate(increments):
        for scale in (
            1.0 - TEACHER_TIME_INCREMENT_REFINEMENT_FRACTION,
            1.0 + TEACHER_TIME_INCREMENT_REFINEMENT_FRACTION,
        ):
            updated_increments = list(increments)
            updated_increments[index] = _clamp_time_increment(env_cfg, increment * scale)
            if abs(updated_increments[index] - increment) < MIN_GAUSSIAN_STD:
                continue
            candidates.append(
                _rollout_with_teacher_gains(
                    initial_state,
                    env_cfg,
                    cfg,
                    trace.theta_gain,
                    trace.omega_gain,
                    durations,
                    trace.segment_actions or None,
                    tuple(updated_increments),
                    modes,
                )
            )
    return candidates


def _action_refinement_candidates(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
) -> List[CartpoleTrace]:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, actions, durations)
    if not actions or not durations:
        return []

    candidates: List[CartpoleTrace] = []
    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    action_step = max(MIN_GAUSSIAN_STD, (upper - lower) * TEACHER_ACTION_REFINEMENT_STEP_FRACTION)
    for index, current_action in enumerate(actions):
        action_candidates = {
            max(lower, min(upper, current_action - action_step)),
            max(lower, min(upper, current_action + action_step)),
        }
        for action in sorted(action_candidates):
            if abs(action - current_action) < MIN_GAUSSIAN_STD:
                continue
            updated = list(actions)
            updated[index] = action
            candidates.append(
                _rollout_with_teacher_gains(
                    initial_state,
                    env_cfg,
                    cfg,
                    trace.theta_gain,
                    trace.omega_gain,
                    durations,
                    tuple(updated),
                    increments,
                    modes,
                )
            )
    return candidates


def _gain_gradient_refinement_candidate(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    objective,
) -> CartpoleTrace | None:
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    theta_scale = max(1.0, abs(trace.theta_gain))
    omega_scale = max(1.0, abs(trace.omega_gain))
    theta_epsilon = max(MIN_GAUSSIAN_STD, theta_scale * TEACHER_GAIN_GRADIENT_EPS_FRACTION)
    omega_epsilon = max(MIN_GAUSSIAN_STD, omega_scale * TEACHER_GAIN_GRADIENT_EPS_FRACTION)

    theta_minus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain - theta_epsilon,
        trace.omega_gain,
        durations,
        None,
        increments or None,
    )
    theta_plus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain + theta_epsilon,
        trace.omega_gain,
        durations,
        None,
        increments or None,
    )
    omega_minus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain,
        trace.omega_gain - omega_epsilon,
        durations,
        None,
        increments or None,
    )
    omega_plus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain,
        trace.omega_gain + omega_epsilon,
        durations,
        None,
        increments or None,
    )
    theta_gradient = (objective(theta_plus) - objective(theta_minus)) / (2.0 * theta_epsilon)
    omega_gradient = (objective(omega_plus) - objective(omega_minus)) / (2.0 * omega_epsilon)
    norm = math.sqrt(theta_gradient * theta_gradient + omega_gradient * omega_gradient)
    if norm < MIN_GAUSSIAN_STD:
        return None

    current_objective = objective(trace)
    for backtrack in TEACHER_GRADIENT_BACKTRACK_FACTORS:
        candidate = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain
            + backtrack * TEACHER_GAIN_GRADIENT_STEP_FRACTION * theta_scale * theta_gradient / norm,
            trace.omega_gain
            + backtrack * TEACHER_GAIN_GRADIENT_STEP_FRACTION * omega_scale * omega_gradient / norm,
            durations,
            None,
            increments or None,
        )
        if objective(candidate) > current_objective:
            return candidate
    return None


def _action_gradient_refinement_candidate(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    objective,
) -> CartpoleTrace | None:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, actions, durations)
    if not actions or not durations:
        return None

    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    action_span = max(MIN_GAUSSIAN_STD, upper - lower)
    epsilon = max(MIN_GAUSSIAN_STD, action_span * TEACHER_ACTION_GRADIENT_EPS_FRACTION)
    step_size = action_span * TEACHER_ACTION_GRADIENT_STEP_FRACTION
    gradients: List[float] = []
    for index, current_action in enumerate(actions):
        minus_action = max(lower, min(upper, current_action - epsilon))
        plus_action = max(lower, min(upper, current_action + epsilon))
        if abs(plus_action - minus_action) < MIN_GAUSSIAN_STD:
            gradients.append(0.0)
            continue
        minus_actions = list(actions)
        plus_actions = list(actions)
        minus_actions[index] = minus_action
        plus_actions[index] = plus_action
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            tuple(minus_actions),
            increments,
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            tuple(plus_actions),
            increments,
            modes,
        )
        gradients.append((objective(plus) - objective(minus)) / (plus_action - minus_action))
    norm = math.sqrt(sum(gradient * gradient for gradient in gradients))
    if norm < MIN_GAUSSIAN_STD:
        return None
    current_objective = objective(trace)
    for backtrack in TEACHER_GRADIENT_BACKTRACK_FACTORS:
        updated_actions = tuple(
            max(lower, min(upper, action + backtrack * step_size * gradient / norm))
            for action, gradient in zip(actions, gradients)
        )
        if all(abs(left - right) < MIN_GAUSSIAN_STD for left, right in zip(updated_actions, actions)):
            continue
        candidate = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            updated_actions,
            increments,
            modes,
        )
        if objective(candidate) > current_objective:
            return candidate
    return None


def _duration_gradient_refinement_candidate(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    objective,
) -> CartpoleTrace | None:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, actions, durations)
    if not actions or not durations:
        return None

    max_duration = max(1, cfg.segment_steps)
    epsilon = max(1, TEACHER_DURATION_GRADIENT_EPS)
    gradients: List[float] = []
    for index, current_duration in enumerate(durations):
        minus_duration = max(1, current_duration - epsilon)
        plus_duration = min(max_duration, current_duration + epsilon)
        if plus_duration == minus_duration:
            gradients.append(0.0)
            continue
        minus_durations = list(durations)
        plus_durations = list(durations)
        minus_durations[index] = minus_duration
        plus_durations[index] = plus_duration
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            tuple(minus_durations),
            actions,
            increments,
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            tuple(plus_durations),
            actions,
            increments,
            modes,
        )
        gradients.append((objective(plus) - objective(minus)) / float(plus_duration - minus_duration))
    norm = math.sqrt(sum(gradient * gradient for gradient in gradients))
    if norm < MIN_GAUSSIAN_STD:
        return None
    current_objective = objective(trace)
    for backtrack in TEACHER_GRADIENT_BACKTRACK_FACTORS:
        updated_durations = tuple(
            min(
                max_duration,
                max(1, int(math.floor(duration + backtrack * TEACHER_DURATION_GRADIENT_STEP * gradient / norm + 0.5))),
            )
            for duration, gradient in zip(durations, gradients)
        )
        if updated_durations == durations:
            continue
        candidate = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            updated_durations,
            actions,
            increments,
            modes,
        )
        if objective(candidate) > current_objective:
            return candidate
    return None


def _time_increment_gradient_refinement_candidate(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    objective,
) -> CartpoleTrace | None:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, actions, durations)
    if not actions or not durations or not increments:
        return None

    epsilon = max(MIN_GAUSSIAN_STD, env_cfg.dt * TEACHER_TIME_INCREMENT_GRADIENT_EPS_FRACTION)
    step_size = env_cfg.dt * TEACHER_TIME_INCREMENT_GRADIENT_STEP_FRACTION
    gradients: List[float] = []
    for index, current_increment in enumerate(increments):
        minus_increment = _clamp_time_increment(env_cfg, current_increment - epsilon)
        plus_increment = _clamp_time_increment(env_cfg, current_increment + epsilon)
        if abs(plus_increment - minus_increment) < MIN_GAUSSIAN_STD:
            gradients.append(0.0)
            continue
        minus_increments = list(increments)
        plus_increments = list(increments)
        minus_increments[index] = minus_increment
        plus_increments[index] = plus_increment
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            actions,
            tuple(minus_increments),
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            actions,
            tuple(plus_increments),
            modes,
        )
        gradients.append((objective(plus) - objective(minus)) / (plus_increment - minus_increment))
    norm = math.sqrt(sum(gradient * gradient for gradient in gradients))
    if norm < MIN_GAUSSIAN_STD:
        return None
    current_objective = objective(trace)
    for backtrack in TEACHER_GRADIENT_BACKTRACK_FACTORS:
        updated_increments = tuple(
            _clamp_time_increment(env_cfg, increment + backtrack * step_size * gradient / norm)
            for increment, gradient in zip(increments, gradients)
        )
        if all(abs(left - right) < MIN_GAUSSIAN_STD for left, right in zip(updated_increments, increments)):
            continue
        candidate = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            actions,
            updated_increments,
            modes,
        )
        if objective(candidate) > current_objective:
            return candidate
    return None


def _schedule_gradient_refinement_candidate(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    objective,
) -> CartpoleTrace | None:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    increments = trace.segment_time_increments or tuple(env_cfg.dt for _ in durations)
    modes = _segment_modes_from_trace(trace, actions, durations)
    length = min(len(actions), len(durations), len(increments), len(modes))
    if length <= 0:
        return None
    actions = tuple(actions[:length])
    durations = tuple(durations[:length])
    increments = tuple(increments[:length])
    modes = tuple(modes[:length])

    lower = max(min(cfg.force_values), -env_cfg.force_limit)
    upper = min(max(cfg.force_values), env_cfg.force_limit)
    action_span = max(MIN_GAUSSIAN_STD, upper - lower)
    action_epsilon = max(MIN_GAUSSIAN_STD, action_span * TEACHER_ACTION_GRADIENT_EPS_FRACTION)
    action_step = action_span * TEACHER_ACTION_GRADIENT_STEP_FRACTION
    max_duration = max(1, cfg.segment_steps)
    duration_epsilon = max(1, TEACHER_DURATION_GRADIENT_EPS)
    increment_epsilon = max(MIN_GAUSSIAN_STD, env_cfg.dt * TEACHER_TIME_INCREMENT_GRADIENT_EPS_FRACTION)
    increment_step = env_cfg.dt * TEACHER_TIME_INCREMENT_GRADIENT_STEP_FRACTION
    theta_scale = max(1.0, abs(trace.theta_gain))
    omega_scale = max(1.0, abs(trace.omega_gain))
    theta_epsilon = max(MIN_GAUSSIAN_STD, theta_scale * TEACHER_GAIN_GRADIENT_EPS_FRACTION)
    omega_epsilon = max(MIN_GAUSSIAN_STD, omega_scale * TEACHER_GAIN_GRADIENT_EPS_FRACTION)

    theta_minus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain - theta_epsilon,
        trace.omega_gain,
        durations,
        actions,
        increments,
        modes,
    )
    theta_plus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain + theta_epsilon,
        trace.omega_gain,
        durations,
        actions,
        increments,
        modes,
    )
    omega_minus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain,
        trace.omega_gain - omega_epsilon,
        durations,
        actions,
        increments,
        modes,
    )
    omega_plus = _rollout_with_teacher_gains(
        initial_state,
        env_cfg,
        cfg,
        trace.theta_gain,
        trace.omega_gain + omega_epsilon,
        durations,
        actions,
        increments,
        modes,
    )
    theta_gradient = (objective(theta_plus) - objective(theta_minus)) / (2.0 * theta_epsilon)
    omega_gradient = (objective(omega_plus) - objective(omega_minus)) / (2.0 * omega_epsilon)

    action_gradients: List[float] = []
    for index, current_action in enumerate(actions):
        minus_action = max(lower, min(upper, current_action - action_epsilon))
        plus_action = max(lower, min(upper, current_action + action_epsilon))
        if abs(plus_action - minus_action) < MIN_GAUSSIAN_STD:
            action_gradients.append(0.0)
            continue
        minus_actions = list(actions)
        plus_actions = list(actions)
        minus_actions[index] = minus_action
        plus_actions[index] = plus_action
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            tuple(minus_actions),
            increments,
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            tuple(plus_actions),
            increments,
            modes,
        )
        action_gradients.append((objective(plus) - objective(minus)) / (plus_action - minus_action))

    duration_gradients: List[float] = []
    for index, current_duration in enumerate(durations):
        minus_duration = max(1, current_duration - duration_epsilon)
        plus_duration = min(max_duration, current_duration + duration_epsilon)
        if plus_duration == minus_duration:
            duration_gradients.append(0.0)
            continue
        minus_durations = list(durations)
        plus_durations = list(durations)
        minus_durations[index] = minus_duration
        plus_durations[index] = plus_duration
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            tuple(minus_durations),
            actions,
            increments,
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            tuple(plus_durations),
            actions,
            increments,
            modes,
        )
        duration_gradients.append((objective(plus) - objective(minus)) / float(plus_duration - minus_duration))

    increment_gradients: List[float] = []
    for index, current_increment in enumerate(increments):
        minus_increment = _clamp_time_increment(env_cfg, current_increment - increment_epsilon)
        plus_increment = _clamp_time_increment(env_cfg, current_increment + increment_epsilon)
        if abs(plus_increment - minus_increment) < MIN_GAUSSIAN_STD:
            increment_gradients.append(0.0)
            continue
        minus_increments = list(increments)
        plus_increments = list(increments)
        minus_increments[index] = minus_increment
        plus_increments[index] = plus_increment
        minus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            actions,
            tuple(minus_increments),
            modes,
        )
        plus = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            trace.theta_gain,
            trace.omega_gain,
            durations,
            actions,
            tuple(plus_increments),
            modes,
        )
        increment_gradients.append((objective(plus) - objective(minus)) / (plus_increment - minus_increment))

    scaled_theta_gradient = theta_gradient * TEACHER_GAIN_GRADIENT_STEP_FRACTION * theta_scale
    scaled_omega_gradient = omega_gradient * TEACHER_GAIN_GRADIENT_STEP_FRACTION * omega_scale
    scaled_action_gradients = [gradient * action_step for gradient in action_gradients]
    scaled_duration_gradients = [gradient * TEACHER_DURATION_GRADIENT_STEP for gradient in duration_gradients]
    scaled_increment_gradients = [gradient * increment_step for gradient in increment_gradients]
    norm = math.sqrt(
        scaled_theta_gradient * scaled_theta_gradient
        + scaled_omega_gradient * scaled_omega_gradient
        + sum(gradient * gradient for gradient in scaled_action_gradients)
        + sum(gradient * gradient for gradient in scaled_duration_gradients)
        + sum(gradient * gradient for gradient in scaled_increment_gradients)
    )
    if norm < MIN_GAUSSIAN_STD:
        return None

    current_objective = objective(trace)
    for backtrack in TEACHER_GRADIENT_BACKTRACK_FACTORS:
        updated_theta_gain = (
            trace.theta_gain
            + backtrack * TEACHER_GAIN_GRADIENT_STEP_FRACTION * theta_scale * scaled_theta_gradient / norm
        )
        updated_omega_gain = (
            trace.omega_gain
            + backtrack * TEACHER_GAIN_GRADIENT_STEP_FRACTION * omega_scale * scaled_omega_gradient / norm
        )
        updated_actions = tuple(
            max(lower, min(upper, action + backtrack * action_step * gradient / norm))
            for action, gradient in zip(actions, scaled_action_gradients)
        )
        updated_durations = tuple(
            min(
                max_duration,
                max(1, int(math.floor(duration + backtrack * TEACHER_DURATION_GRADIENT_STEP * gradient / norm + 0.5))),
            )
            for duration, gradient in zip(durations, scaled_duration_gradients)
        )
        updated_increments = tuple(
            _clamp_time_increment(env_cfg, increment + backtrack * increment_step * gradient / norm)
            for increment, gradient in zip(increments, scaled_increment_gradients)
        )
        if (
            abs(updated_theta_gain - trace.theta_gain) < MIN_GAUSSIAN_STD
            and abs(updated_omega_gain - trace.omega_gain) < MIN_GAUSSIAN_STD
            and all(abs(left - right) < MIN_GAUSSIAN_STD for left, right in zip(updated_actions, actions))
            and updated_durations == durations
            and all(abs(left - right) < MIN_GAUSSIAN_STD for left, right in zip(updated_increments, increments))
        ):
            continue
        candidate = _rollout_with_teacher_gains(
            initial_state,
            env_cfg,
            cfg,
            updated_theta_gain,
            updated_omega_gain,
            updated_durations,
            updated_actions,
            updated_increments,
            modes,
        )
        if objective(candidate) > current_objective:
            return candidate
    return None


def _clamp_time_increment(env_cfg: CartpoleConfig, value: float) -> float:
    return max(MIN_GAUSSIAN_STD, min(env_cfg.dt, float(value)))


def _teacher_objective(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> float:
    if student is None:
        return cfg.teacher_reward_lambda * trace.reward
    # The regularizer rewards traces that the current student can already
    # encode, which is the adaptive-teaching pressure in this local diagnostic.
    log_probability = _current_student_log_probability(trace, student)
    return cfg.teacher_reward_lambda * trace.reward + cfg.teacher_student_regularizer * log_probability


def _teacher_refinement_objective(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
    elites: List[CartpoleTrace],
    elite_log_normalizer: float | None = None,
) -> float:
    if student is None or not elites:
        return _teacher_objective(trace, student, cfg)
    log_probability = _elite_kernel_log_probability(trace, student, elites, elite_log_normalizer)
    return cfg.teacher_reward_lambda * trace.reward + cfg.teacher_student_regularizer * log_probability


def _elite_kernel_log_normalizer(
    student: ProbabilisticCartpoleStudent | None,
    elites: List[CartpoleTrace],
) -> float | None:
    if student is None or not elites:
        return None
    normalizer_terms = [
        _current_student_log_probability(elite, student)
        for elite in elites
    ]
    return _logsumexp(normalizer_terms) if normalizer_terms else None


def _elite_kernel_log_probability(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent,
    elites: List[CartpoleTrace],
    elite_log_normalizer: float | None = None,
) -> float:
    terms: List[float] = []
    for elite in elites:
        elite_log_probability = _current_student_log_probability(elite, student)
        terms.append(elite_log_probability - _loop_free_trace_distance(trace, elite))
    if not terms:
        return _trace_log_probability(trace, student)
    normalizer = elite_log_normalizer
    if normalizer is None:
        normalizer = _elite_kernel_log_normalizer(student, elites)
    return _logsumexp(terms) - (normalizer if normalizer is not None else 0.0)


def _current_student_log_probability(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent,
) -> float:
    if trace.actions:
        return _trace_log_probability(trace, student)
    if trace.student_log_probability is not None:
        return trace.student_log_probability
    return 0.0


def _loop_free_trace_distance(left: CartpoleTrace, right: CartpoleTrace) -> float:
    left_actions = left.segment_actions or _mode_run_actions(left.actions, left.mode_labels)
    right_actions = right.segment_actions or _mode_run_actions(right.actions, right.mode_labels)
    left_durations = left.segment_durations or _mode_run_lengths(left.mode_labels)
    right_durations = right.segment_durations or _mode_run_lengths(right.mode_labels)
    left_increments = _distance_time_increments(left.segment_time_increments, left_durations)
    right_increments = _distance_time_increments(right.segment_time_increments, right_durations)
    left_modes = _segment_modes_from_trace(left, left_actions, left_durations)
    right_modes = _segment_modes_from_trace(right, right_actions, right_durations)
    length = max(
        len(left_actions),
        len(right_actions),
        len(left_durations),
        len(right_durations),
        len(left_increments),
        len(right_increments),
        len(left_modes),
        len(right_modes),
    )
    if length == 0:
        return 0.0

    duration_scale = max(
        TEACHER_ELITE_DISTANCE_DURATION_SCALE_FLOOR,
        max(left_durations or (0,)),
        max(right_durations or (0,)),
    )
    increment_scale = max(
        MIN_GAUSSIAN_STD,
        max(left_increments or (0.0,)),
        max(right_increments or (0.0,)),
    )
    action_scale = max(
        1.0,
        max((abs(action) for action in left_actions), default=0.0),
        max((abs(action) for action in right_actions), default=0.0),
    )
    theta_gain_scale = max(1.0, abs(left.theta_gain), abs(right.theta_gain))
    omega_gain_scale = max(1.0, abs(left.omega_gain), abs(right.omega_gain))
    total = 0.0
    total += ((left.theta_gain - right.theta_gain) / theta_gain_scale) ** 2
    total += ((left.omega_gain - right.omega_gain) / omega_gain_scale) ** 2
    for index in range(length):
        left_action = left_actions[index] if index < len(left_actions) else 0.0
        right_action = right_actions[index] if index < len(right_actions) else 0.0
        left_duration = left_durations[index] if index < len(left_durations) else 0
        right_duration = right_durations[index] if index < len(right_durations) else 0
        left_increment = left_increments[index] if index < len(left_increments) else 0.0
        right_increment = right_increments[index] if index < len(right_increments) else 0.0
        left_mode = left_modes[index] if index < len(left_modes) else 0
        right_mode = right_modes[index] if index < len(right_modes) else 0
        total += float(left_mode != right_mode)
        total += ((left_action - right_action) / action_scale) ** 2
        total += ((left_duration - right_duration) / duration_scale) ** 2
        total += ((left_increment - right_increment) / increment_scale) ** 2
    return math.sqrt(total)


def _distance_time_increments(
    increments: Tuple[float, ...],
    durations: Tuple[int, ...],
) -> Tuple[float, ...]:
    if increments:
        return increments
    return tuple(DEFAULT_CARTPOLE_TIME_INCREMENT for _ in durations)


def _trace_log_probability(trace: CartpoleTrace, student: ProbabilisticCartpoleStudent) -> float:
    trace_segments = _segments_from_traces([trace])[0]
    if not trace_segments:
        return 0.0
    emissions = [
        [
            student.action_distributions[mode].log_pdf(segment.action_parameter)
            for mode in (0, 1)
        ]
        for segment in trace_segments
    ]
    forward: List[List[float]] = [[emissions[0][0], -math.inf]]
    for index in range(1, len(trace_segments)):
        previous = forward[-1]
        pair = _switch_responsibility_pair_log_potentials(
            student.switch,
            student.switch_parameter_distributions,
            trace_segments[index - 1],
        )
        forward.append(
            [
                emissions[index][mode]
                + _logsumexp(
                    [
                        previous[previous_mode] + pair[previous_mode][mode]
                        for previous_mode in (0, 1)
                    ]
                )
                for mode in (0, 1)
            ]
        )
    terminal_stay = _switch_terminal_stay_log_potentials(
        student.switch,
        student.switch_parameter_distributions,
        trace_segments[-1],
    )
    return _logsumexp([
        forward[-1][mode] + terminal_stay[mode]
        for mode in (0, 1)
    ])


def _logsumexp(values: List[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _segments_from_traces(traces: List[CartpoleTrace]) -> List[List[CartpoleSegment]]:
    segments_by_trace: List[List[CartpoleSegment]] = []
    for trace in traces:
        trace_segments: List[CartpoleSegment] = []
        if not trace.actions:
            segments_by_trace.append(trace_segments)
            continue

        if trace.segment_actions and trace.segment_durations:
            segments_by_trace.append(_teacher_schedule_segments(trace))
            continue

        start = 0
        mode = trace.mode_labels[0]
        for index, label in enumerate(trace.mode_labels[1:], start=1):
            if label == mode:
                continue
            # A segment is one maximal same-mode run, matching the student's
            # constant-action primitive rather than individual simulator steps.
            trace_segments.append(_make_segment(trace, start, index, mode))
            start = index
            mode = label
        trace_segments.append(_make_segment(trace, start, len(trace.actions), mode))
        segments_by_trace.append(trace_segments)
    return segments_by_trace


def _teacher_schedule_segments(trace: CartpoleTrace) -> List[CartpoleSegment]:
    if len(trace.segment_actions) != len(trace.segment_durations):
        raise ValueError("loop-free action count must match duration count")

    segments: List[CartpoleSegment] = []
    start = 0
    increments = _distance_time_increments(trace.segment_time_increments, trace.segment_durations)
    for index, (action, duration) in enumerate(zip(trace.segment_actions, trace.segment_durations)):
        end = min(start + max(1, int(duration)), len(trace.actions))
        if start >= end:
            break
        increment = increments[index] if index < len(increments) else DEFAULT_CARTPOLE_TIME_INCREMENT
        timing_step_scale = increment / DEFAULT_CARTPOLE_TIME_INCREMENT
        hard_mode = (
            int(trace.mode_labels[start])
            if trace.mode_labels and start < len(trace.mode_labels)
            else 1 if action > 0.0 else 0
        )
        segments.append(
            CartpoleSegment(
                observations=trace.observations[start:end],
                action_parameter=float(action),
                duration=end - start,
                timing_duration=(end - start) * timing_step_scale,
                timing_step_scale=timing_step_scale,
                hard_mode=hard_mode,
            )
        )
        start = end
    return segments


def _make_segment(trace: CartpoleTrace, start: int, end: int, hard_mode: int) -> CartpoleSegment:
    actions = trace.actions[start:end]
    action_parameter = sum(actions) / len(actions)
    return CartpoleSegment(
        observations=trace.observations[start:end],
        action_parameter=action_parameter,
        duration=end - start,
        hard_mode=hard_mode,
    )


def _mode_responsibilities(
    action_parameter: float,
    action_distributions: Dict[int, GaussianScalar],
) -> Tuple[float, float]:
    log_weights = [
        action_distributions[mode].log_pdf(action_parameter)
        for mode in (0, 1)
    ]
    max_log = max(log_weights)
    weights = [math.exp(value - max_log) for value in log_weights]
    total = sum(weights)
    return weights[0] / total, weights[1] / total


def _fit_action_distributions(
    segments: List[CartpoleSegment],
    responsibilities: List[Tuple[float, float]],
    left_default: float,
    right_default: float,
) -> Dict[int, GaussianScalar]:
    return {
        mode: _weighted_gaussian(
            [segment.action_parameter for segment in segments],
            [resp[mode] for resp in responsibilities],
            left_default if mode == 0 else right_default,
        )
        for mode in (0, 1)
    }


def _fit_student_switch(
    traces: List[CartpoleTrace],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    # Refit both switch structure and Gaussian threshold parameters after each
    # responsibility update so action and timing evidence stay in sync.
    switch = _learn_depth2_switch(
        traces,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    return _switch_with_distribution_means(switch, distributions), distributions


def _refine_responsibilities_with_switch_timing(
    segments_by_trace: List[List[CartpoleSegment]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
) -> List[Tuple[float, float]]:
    responsibilities, _ = _refine_responsibilities_and_switch_pairs_with_timing(
        segments_by_trace,
        action_distributions,
        switch,
        switch_parameter_distributions,
    )
    return responsibilities


def _refine_responsibilities_and_switch_pairs_with_timing(
    segments_by_trace: List[List[CartpoleSegment]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    responsibilities: List[Tuple[float, float]] = []
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] = []
    for trace_segments in segments_by_trace:
        if not trace_segments:
            continue
        # Emissions are the action-function likelihood term from Eq. (10).
        emissions = [
            [
                action_distributions[mode].log_pdf(segment.action_parameter)
                for mode in (0, 1)
            ]
            for segment in trace_segments
        ]
        # Pair potentials encode whether the learned switch prefers a mode
        # transition or a same-mode continuation at each observed boundary.
        pair_potentials = [
            _switch_responsibility_pair_log_potentials(
                switch,
                switch_parameter_distributions,
                segment,
            )
            for segment in trace_segments[:-1]
        ]

        # Forward scores accumulate prefix evidence from the fixed initial
        # memory state used by the executable CartPole PSM.
        forward: List[List[float]] = [[emissions[0][0], -math.inf]]
        for index in range(1, len(trace_segments)):
            previous = forward[-1]
            pair = pair_potentials[index - 1]
            forward.append(
                [
                    emissions[index][mode]
                    + _logsumexp(
                        [
                            previous[previous_mode] + pair[previous_mode][mode]
                            for previous_mode in (0, 1)
                        ]
                    )
                    for mode in (0, 1)
                ]
            )

        terminal_stay = _switch_terminal_stay_log_potentials(
            switch,
            switch_parameter_distributions,
            trace_segments[-1],
        )
        forward_with_terminal_stay = [
            forward[-1][mode] + terminal_stay[mode]
            for mode in (0, 1)
        ]

        # Backward scores accumulate suffix evidence without changing ordering.
        backward: List[List[float]] = [[0.0, 0.0] for _ in trace_segments]
        backward[-1] = list(terminal_stay)
        for index in range(len(trace_segments) - 2, -1, -1):
            pair = pair_potentials[index]
            backward[index] = [
                _logsumexp(
                    [
                        pair[mode][next_mode]
                        + emissions[index + 1][next_mode]
                        + backward[index + 1][next_mode]
                        for next_mode in (0, 1)
                    ]
                )
                for mode in (0, 1)
            ]

        # Posterior marginals are flattened in segment order for later M-steps.
        norm = _logsumexp(forward_with_terminal_stay)
        for index in range(len(trace_segments)):
            posterior_logs = [
                forward[index][mode] + backward[index][mode] - norm
                for mode in (0, 1)
            ]
            weights = [math.exp(value) for value in posterior_logs]
            total = sum(weights)
            responsibilities.append((weights[0] / total, weights[1] / total))
        for index, pair in enumerate(pair_potentials):
            logs = [
                forward[index][0] + pair[0][0] + emissions[index + 1][0] + backward[index + 1][0] - norm,
                forward[index][0] + pair[0][1] + emissions[index + 1][1] + backward[index + 1][1] - norm,
                forward[index][1] + pair[1][0] + emissions[index + 1][0] + backward[index + 1][0] - norm,
                forward[index][1] + pair[1][1] + emissions[index + 1][1] + backward[index + 1][1] - norm,
            ]
            weights = [math.exp(value) for value in logs]
            total = sum(weights)
            switch_pair_responsibilities.append(
                (
                    weights[0] / total,
                    weights[1] / total,
                    weights[2] / total,
                    weights[3] / total,
                )
            )
    return responsibilities, switch_pair_responsibilities


def _switch_responsibility_pair_log_potentials(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segment: CartpoleSegment,
) -> List[List[float]]:
    off_to_on, on_to_off, stay_off, stay_on = _switch_selector_transition_probabilities(
        switch,
        distributions,
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
    # The local two-mode CartPole student uses one selector switch: off means
    # mode 0 and on means mode 1. Eq. (12) is directed by the next mode, so
    # 0->1 consumes switch-on mass while 1->0 consumes switch-off mass.
    return [
        [
            math.log(max(stay_off, LOG_PROBABILITY_FLOOR)),
            math.log(max(off_to_on, LOG_PROBABILITY_FLOOR)),
        ],
        [
            math.log(max(on_to_off, LOG_PROBABILITY_FLOOR)),
            math.log(max(stay_on, LOG_PROBABILITY_FLOOR)),
        ],
    ]


def _switch_terminal_stay_log_potentials(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segment: CartpoleSegment,
) -> List[float]:
    _, _, stay_off, stay_on = _switch_selector_transition_probabilities(
        switch,
        distributions,
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
    return [
        math.log(max(stay_off, LOG_PROBABILITY_FLOOR)),
        math.log(max(stay_on, LOG_PROBABILITY_FLOOR)),
    ]


def _weighted_gaussian(
    values: List[float],
    weights: List[float],
    default_mean: float,
) -> GaussianScalar:
    total = sum(weights)
    if total <= 0.0 or not values:
        return GaussianScalar(default_mean, 1.0)
    mean = sum(weight * value for value, weight in zip(values, weights)) / total
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total
    return GaussianScalar(mean, max(math.sqrt(max(variance, 0.0)), MIN_GAUSSIAN_STD))


def _fit_switch_parameter_distributions(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> List[GaussianScalar]:
    predicates = _switch_predicates(switch)
    if not predicates:
        distribution = _legacy_switch_threshold_distribution(
            switch,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities,
        )
        _, refined = _refine_switch_parameter_distributions(
            switch,
            [distribution],
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities,
        )
        return refined
    distributions = [
        _predicate_threshold_distribution(
            predicate,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities,
        )
        for predicate in predicates
    ]
    _, refined = _refine_switch_parameter_distributions(
        switch,
        distributions,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    return refined


def _switch_with_distribution_means(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
) -> SwitchProgram:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        if len(distributions) < len(predicates):
            return switch
        fitted = [
            predicate.with_threshold(distribution.mean)
            for predicate, distribution in zip(predicates, distributions)
        ]
        if len(fitted) == 1:
            return BooleanTreeSwitch(fitted[0])
        return BooleanTreeSwitch(fitted[0], fitted[1], switch.operator)
    if not distributions:
        return switch
    return Depth2Switch(
        switch.theta_weight,
        switch.omega_weight,
        distributions[0].mean,
    )


def _refine_switch_distribution_means(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    return _refine_switch_parameter_distributions(
        switch,
        distributions,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )


def _refine_switch_parameter_distributions(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    if not distributions or not segments_by_trace:
        return switch, distributions

    examples = [
        (observation, 1 if segment.hard_mode == 1 else 0)
        for trace_segments in segments_by_trace
        for segment in trace_segments
        for observation in segment.observations
    ]
    example_cache = _switch_example_cache(examples)
    timing_pairs = _switch_timing_pairs(
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    scalar_timing_pairs = _scalar_switch_timing_pairs(switch, timing_pairs)
    label_loss_cache: Dict[str, float] = {}

    def label_loss(candidate: SwitchProgram) -> float:
        key = _switch_cache_key(candidate)
        if key not in label_loss_cache:
            label_loss_cache[key] = _switch_structure_label_loss(
                candidate,
                examples,
                segments_by_trace,
                responsibilities,
                example_cache,
            )
        return label_loss_cache[key]

    best_distributions = _distributions_with_switch_means(switch, distributions)
    best_switch = _switch_with_distribution_means(switch, best_distributions)
    best_label_loss = label_loss(best_switch)
    best_loss = _switch_distribution_timing_loss(
        best_switch,
        best_distributions,
        segments_by_trace,
        responsibilities,
        timing_pairs,
        scalar_timing_pairs,
    )

    for param_index, distribution in enumerate(distributions):
        mean_candidates = [
            distribution.mean,
            *_switch_distribution_mean_candidates(switch, param_index, segments_by_trace),
        ]
        std_candidates = _switch_distribution_std_candidates(distribution, switch, param_index, segments_by_trace)
        for candidate_mean in mean_candidates:
            for candidate_std in std_candidates:
                candidate_distributions = list(best_distributions)
                candidate_distributions[param_index] = GaussianScalar(candidate_mean, candidate_std)
                candidate_switch = _switch_with_distribution_means(switch, candidate_distributions)
                candidate_label_loss = label_loss(candidate_switch)
                candidate_loss = _switch_distribution_timing_loss(
                    candidate_switch,
                    candidate_distributions,
                    segments_by_trace,
                    responsibilities,
                    timing_pairs,
                    scalar_timing_pairs,
                )
                if _switch_refinement_improves(candidate_label_loss, candidate_loss, best_label_loss, best_loss):
                    best_distributions = candidate_distributions
                    best_switch = candidate_switch
                    best_label_loss = candidate_label_loss
                    best_loss = candidate_loss
    return _gradient_refine_switch_parameter_distributions(
        switch,
        *_coordinate_refine_switch_parameter_distributions(
            switch,
            best_distributions,
            best_switch,
            best_label_loss,
            best_loss,
            examples,
            example_cache,
            label_loss_cache,
            timing_pairs,
            scalar_timing_pairs,
            segments_by_trace,
            responsibilities,
        ),
        examples,
        example_cache,
        label_loss_cache,
        timing_pairs,
        scalar_timing_pairs,
        segments_by_trace,
        responsibilities,
    )


def _evaluate_switch_parameter_candidate(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    label_loss_cache: Dict[str, float],
    timing_pairs: List[_SwitchTimingPair],
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Tuple[SwitchProgram, float, float]:
    candidate_switch = _switch_with_distribution_means(switch, distributions)
    key = _switch_cache_key(candidate_switch)
    if key not in label_loss_cache:
        label_loss_cache[key] = _switch_structure_label_loss(
            candidate_switch,
            examples,
            segments_by_trace,
            responsibilities,
            example_cache,
        )
    candidate_label_loss = label_loss_cache[key]
    candidate_loss = _switch_distribution_timing_loss(
        candidate_switch,
        distributions,
        segments_by_trace,
        responsibilities,
        timing_pairs,
        scalar_timing_pairs,
    )
    return candidate_switch, candidate_label_loss, candidate_loss


def _switch_cache_key(switch: SwitchProgram) -> str:
    if isinstance(switch, Depth2Switch):
        return f"depth2:{switch.theta_weight!r}:{switch.omega_weight!r}:{switch.threshold!r}"
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        predicate_key = tuple(
            (predicate.feature_index, predicate.relation, predicate.threshold)
            for predicate in predicates
        )
        return f"bool:{switch.operator}:{predicate_key!r}"
    return repr(switch)


def _coordinate_refine_switch_parameter_distributions(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    current_switch: SwitchProgram,
    current_label_loss: float,
    current_loss: float,
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    label_loss_cache: Dict[str, float],
    timing_pairs: List[_SwitchTimingPair],
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    # Start from the grid-refined solution; this is a bounded local polish, not
    # a replacement for the discrete grammar search.
    best_distributions = list(distributions)
    best_switch = current_switch
    best_label_loss = current_label_loss
    best_loss = current_loss
    mean_steps = _switch_distribution_coordinate_mean_steps(switch, segments_by_trace, best_distributions)
    log_std_step = SWITCH_PARAMETER_COORDINATE_LOG_STD_STEP

    for _ in range(SWITCH_PARAMETER_COORDINATE_REFINEMENT_STEPS):
        improved = False
        for param_index, mean_step in enumerate(mean_steps):
            # Try one mean coordinate and one log-std coordinate at a time so
            # accepted moves remain easy to audit against Eq. (12)-style loss.
            for delta_mean, delta_log_std in (
                (mean_step, 0.0),
                (-mean_step, 0.0),
                (0.0, log_std_step),
                (0.0, -log_std_step),
            ):
                candidate_distributions = list(best_distributions)
                current = candidate_distributions[param_index]
                candidate_std = max(MIN_GAUSSIAN_STD, current.std * math.exp(delta_log_std))
                candidate_distributions[param_index] = GaussianScalar(current.mean + delta_mean, candidate_std)
                candidate_switch, candidate_label_loss, candidate_loss = _evaluate_switch_parameter_candidate(
                    switch,
                    candidate_distributions,
                    examples,
                    example_cache,
                    label_loss_cache,
                    timing_pairs,
                    scalar_timing_pairs,
                    segments_by_trace,
                    responsibilities,
                )
                if _switch_refinement_improves(candidate_label_loss, candidate_loss, best_label_loss, best_loss):
                    best_distributions = candidate_distributions
                    best_switch = candidate_switch
                    best_label_loss = candidate_label_loss
                    best_loss = candidate_loss
                    improved = True
        if not improved:
            # When no coordinate helps, shrink the local neighborhood instead
            # of widening the search beyond the fitted switch structure.
            mean_steps = [step * SWITCH_PARAMETER_COORDINATE_STEP_DECAY for step in mean_steps]
            log_std_step *= SWITCH_PARAMETER_COORDINATE_STEP_DECAY
    return best_switch, best_distributions


def _gradient_refine_switch_parameter_distributions(
    switch: SwitchProgram,
    current_switch: SwitchProgram,
    distributions: List[GaussianScalar],
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    label_loss_cache: Dict[str, float],
    timing_pairs: List[_SwitchTimingPair],
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    best_distributions = list(distributions)
    best_switch = current_switch
    _, best_label_loss, best_loss = _evaluate_switch_parameter_candidate(
        switch,
        best_distributions,
        examples,
        example_cache,
        label_loss_cache,
        timing_pairs,
        scalar_timing_pairs,
        segments_by_trace,
        responsibilities,
    )
    mean_steps = _switch_distribution_coordinate_mean_steps(switch, segments_by_trace, best_distributions)

    for _ in range(SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS):
        gradients: List[Tuple[float, float]] = []
        for param_index, distribution in enumerate(best_distributions):
            mean_epsilon = max(
                MIN_GAUSSIAN_STD,
                mean_steps[param_index] * SWITCH_PARAMETER_GRADIENT_EPS_FRACTION,
            )
            log_std_epsilon = max(MIN_GAUSSIAN_STD, SWITCH_PARAMETER_GRADIENT_LOG_STD_STEP)
            mean_gradient = _switch_parameter_loss_gradient(
                switch,
                best_distributions,
                param_index,
                mean_epsilon,
                0.0,
                examples,
                example_cache,
                label_loss_cache,
                timing_pairs,
                scalar_timing_pairs,
                segments_by_trace,
                responsibilities,
            )
            log_std_gradient = _switch_parameter_loss_gradient(
                switch,
                best_distributions,
                param_index,
                0.0,
                log_std_epsilon,
                examples,
                example_cache,
                label_loss_cache,
                timing_pairs,
                scalar_timing_pairs,
                segments_by_trace,
                responsibilities,
            )
            gradients.append((mean_gradient, log_std_gradient))

        norm = math.sqrt(
            sum(
                mean_gradient * mean_gradient + log_std_gradient * log_std_gradient
                for mean_gradient, log_std_gradient in gradients
            )
        )
        if norm < MIN_GAUSSIAN_STD:
            break
        accepted = False
        for backtrack in SWITCH_PARAMETER_GRADIENT_BACKTRACK_FACTORS:
            candidate_distributions = _gradient_switch_parameter_candidate_distributions(
                best_distributions,
                mean_steps,
                gradients,
                norm,
                backtrack,
            )
            candidate_switch, candidate_label_loss, candidate_loss = _evaluate_switch_parameter_candidate(
                switch,
                candidate_distributions,
                examples,
                example_cache,
                label_loss_cache,
                timing_pairs,
                scalar_timing_pairs,
                segments_by_trace,
                responsibilities,
            )
            if _switch_refinement_improves(candidate_label_loss, candidate_loss, best_label_loss, best_loss):
                best_distributions = candidate_distributions
                best_switch = candidate_switch
                best_label_loss = candidate_label_loss
                best_loss = candidate_loss
                accepted = True
                break
        if not accepted:
            break
    return best_switch, best_distributions


def _switch_refinement_improves(
    candidate_label_loss: float,
    candidate_timing_loss: float,
    best_label_loss: float,
    best_timing_loss: float,
) -> bool:
    return (
        candidate_label_loss < best_label_loss
        and candidate_timing_loss <= best_timing_loss
    ) or (
        candidate_label_loss <= best_label_loss
        and candidate_timing_loss < best_timing_loss
    )


def _gradient_switch_parameter_candidate_distributions(
    distributions: List[GaussianScalar],
    mean_steps: List[float],
    gradients: List[Tuple[float, float]],
    norm: float,
    backtrack: float,
) -> List[GaussianScalar]:
    return [
        GaussianScalar(
            distribution.mean
            - backtrack * SWITCH_PARAMETER_GRADIENT_MEAN_STEP_FRACTION * mean_step * mean_gradient / norm,
            max(
                MIN_GAUSSIAN_STD,
                distribution.std
                * math.exp(
                    -backtrack * SWITCH_PARAMETER_GRADIENT_LOG_STD_STEP * log_std_gradient / norm
                ),
            ),
        )
        for distribution, mean_step, (mean_gradient, log_std_gradient) in zip(
            distributions,
            mean_steps,
            gradients,
        )
    ]


def _switch_parameter_loss_gradient(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    param_index: int,
    delta_mean: float,
    delta_log_std: float,
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    label_loss_cache: Dict[str, float],
    timing_pairs: List[_SwitchTimingPair],
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> float:
    current = distributions[param_index]
    minus = list(distributions)
    plus = list(distributions)
    minus[param_index] = GaussianScalar(
        current.mean - delta_mean,
        max(MIN_GAUSSIAN_STD, current.std * math.exp(-delta_log_std)),
    )
    plus[param_index] = GaussianScalar(
        current.mean + delta_mean,
        max(MIN_GAUSSIAN_STD, current.std * math.exp(delta_log_std)),
    )
    _, _, minus_loss = _evaluate_switch_parameter_candidate(
        switch,
        minus,
        examples,
        example_cache,
        label_loss_cache,
        timing_pairs,
        scalar_timing_pairs,
        segments_by_trace,
        responsibilities,
    )
    _, _, plus_loss = _evaluate_switch_parameter_candidate(
        switch,
        plus,
        examples,
        example_cache,
        label_loss_cache,
        timing_pairs,
        scalar_timing_pairs,
        segments_by_trace,
        responsibilities,
    )
    denominator = 2.0 * (delta_mean if delta_mean else delta_log_std)
    return (plus_loss - minus_loss) / denominator


def _switch_distribution_timing_loss(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    timing_pairs: List[_SwitchTimingPair] | None = None,
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None = None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> float:
    if scalar_timing_pairs is not None and len(distributions) == 1:
        return _scalar_switch_distribution_timing_loss(distributions[0], scalar_timing_pairs)
    pairs = (
        timing_pairs
        if timing_pairs is not None
        else _switch_timing_pairs(segments_by_trace, responsibilities, switch_pair_responsibilities)
    )
    loss = 0.0
    for pair in pairs:
        off_to_on, on_to_off, stay_off, stay_on = _switch_selector_transition_probabilities_for_pair(
            switch,
            distributions,
            pair,
        )
        loss -= (
            pair.off_to_on_weight * math.log(max(off_to_on, LOG_PROBABILITY_FLOOR))
            + pair.on_to_off_weight * math.log(max(on_to_off, LOG_PROBABILITY_FLOOR))
            + pair.stay_off_weight * math.log(max(stay_off, LOG_PROBABILITY_FLOOR))
            + pair.stay_on_weight * math.log(max(stay_on, LOG_PROBABILITY_FLOOR))
        )
    return loss


def _switch_timing_pairs(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> List[_SwitchTimingPair]:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match switch timing segments")
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    if switch_pair_responsibilities is not None and len(switch_pair_responsibilities) != pair_count:
        raise ValueError("switch pair responsibility count must match adjacent segment pairs")
    pairs: List[_SwitchTimingPair] = []
    pair_index = 0
    for trace_segments in segments_by_trace:
        for index, current_segment in enumerate(trace_segments):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = (
                responsibility_by_id.get(id(trace_segments[index + 1]), (0.5, 0.5))
                if index + 1 < len(trace_segments)
                else None
            )
            if next_resp is not None and switch_pair_responsibilities is not None:
                stay_off_weight, off_to_on_weight, on_to_off_weight, stay_on_weight = switch_pair_responsibilities[
                    pair_index
                ]
                pair_index += 1
            else:
                stay_off_weight = current_resp[0] * next_resp[0] if next_resp is not None else current_resp[0]
                off_to_on_weight = current_resp[0] * next_resp[1] if next_resp is not None else 0.0
                on_to_off_weight = current_resp[1] * next_resp[0] if next_resp is not None else 0.0
                stay_on_weight = current_resp[1] * next_resp[1] if next_resp is not None else current_resp[1]
            observations = tuple(current_segment.observations)
            pairs.append(
                _SwitchTimingPair(
                    observations=observations,
                    columns=_observation_columns(observations),
                    duration=current_segment.duration,
                    timing_duration=current_segment.switch_timing_duration,
                    timing_step_scale=current_segment.timing_step_scale,
                    off_to_on_weight=off_to_on_weight,
                    on_to_off_weight=on_to_off_weight,
                    stay_off_weight=stay_off_weight,
                    stay_on_weight=stay_on_weight,
                )
            )
    return pairs


def _observation_columns(observations: Tuple[Observation, ...]) -> Tuple[Tuple[float, ...], ...]:
    if not observations:
        return ()
    return tuple(
        tuple(float(observation[index]) for observation in observations)
        for index in range(len(observations[0]))
    )


def _scalar_switch_timing_pairs(
    switch: SwitchProgram,
    timing_pairs: List[_SwitchTimingPair],
) -> List[_ScalarSwitchTimingPair] | None:
    scalar_pairs: List[_ScalarSwitchTimingPair] = []
    for pair in timing_pairs:
        scalar = _scalar_switch_timing_values(switch, pair)
        if scalar is None:
            return None
        values, relation = scalar
        current_index = _timing_duration_step_index(
            pair.timing_duration,
            pair.timing_step_scale,
            len(values),
        )
        previous = values[:current_index]
        previous_enable_extreme = None
        previous_disable_extreme = None
        if previous:
            previous_enable_extreme = max(previous) if relation == ">=" else min(previous)
            previous_disable_extreme = min(previous) if relation == ">=" else max(previous)
        current_value = values[current_index] if current_index < len(values) else None
        scalar_pairs.append(
            _ScalarSwitchTimingPair(
                relation=relation,
                current_value=current_value,
                previous_enable_extreme=previous_enable_extreme,
                previous_disable_extreme=previous_disable_extreme,
                off_to_on_weight=pair.off_to_on_weight,
                on_to_off_weight=pair.on_to_off_weight,
                stay_off_weight=pair.stay_off_weight,
                stay_on_weight=pair.stay_on_weight,
            )
        )
    return scalar_pairs


def _scalar_switch_timing_values(
    switch: SwitchProgram,
    pair: _SwitchTimingPair,
) -> Tuple[Tuple[float, ...], str] | None:
    if isinstance(switch, Depth2Switch):
        theta_values = pair.columns[2]
        omega_values = pair.columns[3]
        return (
            tuple(
                switch.theta_weight * theta + switch.omega_weight * omega
                for theta, omega in zip(theta_values, omega_values)
            ),
            ">=",
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is None:
        predicate = switch.first
        return pair.columns[predicate.feature_index], predicate.relation
    return None


def _scalar_switch_distribution_timing_loss(
    distribution: GaussianScalar,
    scalar_pairs: List[_ScalarSwitchTimingPair],
) -> float:
    loss = 0.0
    for pair in scalar_pairs:
        off_to_on, on_to_off, stay_off, stay_on = _scalar_timing_pair_probabilities(distribution, pair)
        loss -= (
            pair.off_to_on_weight * math.log(max(off_to_on, LOG_PROBABILITY_FLOOR))
            + pair.on_to_off_weight * math.log(max(on_to_off, LOG_PROBABILITY_FLOOR))
            + pair.stay_off_weight * math.log(max(stay_off, LOG_PROBABILITY_FLOOR))
            + pair.stay_on_weight * math.log(max(stay_on, LOG_PROBABILITY_FLOOR))
        )
    return loss


def _scalar_timing_pair_probabilities(
    distribution: GaussianScalar,
    pair: _ScalarSwitchTimingPair,
) -> Tuple[float, float, float, float]:
    if pair.previous_enable_extreme is None:
        previous_enable_probability = 0.0 if pair.relation == ">=" else 1.0
    else:
        previous_enable_probability = _gaussian_cdf(pair.previous_enable_extreme, distribution)
    if pair.previous_disable_extreme is None:
        previous_disable_probability = 1.0 if pair.relation == ">=" else 0.0
    else:
        previous_disable_probability = _gaussian_cdf(pair.previous_disable_extreme, distribution)
    if pair.current_value is None:
        stay_off = _single_threshold_stay_probability(previous_enable_probability, pair.relation)
        stay_on = _single_threshold_disable_stay_probability(previous_disable_probability, pair.relation)
        return 0.0, 0.0, stay_off, stay_on

    current_probability = _gaussian_cdf(pair.current_value, distribution)
    if pair.relation == ">=":
        off_to_on = max(current_probability - previous_enable_probability, 0.0)
        on_to_off = max(previous_disable_probability - current_probability, 0.0)
        stay_off = max(1.0 - previous_enable_probability, 0.0)
        stay_on = max(previous_disable_probability, 0.0)
    elif pair.relation == "<=":
        off_to_on = max(previous_enable_probability - current_probability, 0.0)
        on_to_off = max(current_probability - previous_disable_probability, 0.0)
        stay_off = max(previous_enable_probability, 0.0)
        stay_on = max(1.0 - previous_disable_probability, 0.0)
    else:
        raise ValueError(f"unknown relation: {pair.relation}")
    return off_to_on, on_to_off, stay_off, stay_on


def _switch_distribution_std_candidates(
    distribution: GaussianScalar,
    switch: SwitchProgram,
    param_index: int,
    segments_by_trace: List[List[CartpoleSegment]],
) -> List[float]:
    candidates = {
        max(MIN_GAUSSIAN_STD, distribution.std * multiplier)
        for multiplier in SWITCH_STD_REFINEMENT_MULTIPLIERS
    }
    boundary_values = _switch_distribution_boundary_values(switch, param_index, segments_by_trace)
    if len(boundary_values) > 1:
        mean = sum(boundary_values) / len(boundary_values)
        variance = sum((value - mean) ** 2 for value in boundary_values) / len(boundary_values)
        candidates.add(max(MIN_GAUSSIAN_STD, math.sqrt(max(variance, 0.0))))
    return sorted(candidates)


def _switch_distribution_coordinate_mean_steps(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]],
    distributions: List[GaussianScalar],
) -> List[float]:
    steps: List[float] = []
    for param_index, distribution in enumerate(distributions):
        boundary_values = _switch_distribution_boundary_values(switch, param_index, segments_by_trace)
        if len(boundary_values) > 1:
            span = max(boundary_values) - min(boundary_values)
        else:
            span = abs(distribution.mean) + max(distribution.std, MIN_GAUSSIAN_STD)
        step = max(MIN_GAUSSIAN_STD, span * SWITCH_PARAMETER_COORDINATE_MEAN_STEP_FRACTION)
        steps.append(step)
    return steps


def _switch_distribution_boundary_values(
    switch: SwitchProgram,
    param_index: int,
    segments_by_trace: List[List[CartpoleSegment]],
) -> List[float]:
    values: List[float] = []
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        if param_index >= len(predicates):
            return values
        feature_index = predicates[param_index].feature_index
        for trace_segments in segments_by_trace:
            for current_segment, _ in zip(trace_segments, trace_segments[1:]):
                values.append(current_segment.end_observation[feature_index])
        return values
    for trace_segments in segments_by_trace:
        for current_segment, _ in zip(trace_segments, trace_segments[1:]):
            values.append(_switch_margin(switch, current_segment.end_observation))
    return values


def _distributions_with_switch_means(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
) -> List[GaussianScalar]:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        return [
            GaussianScalar(predicate.threshold, max(MIN_GAUSSIAN_STD, distribution.std))
            for predicate, distribution in zip(predicates, distributions)
        ]
    if not distributions:
        return []
    return [GaussianScalar(_switch_default_threshold(switch), max(MIN_GAUSSIAN_STD, distributions[0].std))]


def _switch_distribution_mean_candidates(
    switch: SwitchProgram,
    param_index: int,
    segments_by_trace: List[List[CartpoleSegment]],
) -> List[float]:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        if param_index >= len(predicates):
            return []
        feature_index = predicates[param_index].feature_index
        values = [
            observation[feature_index]
            for trace_segments in segments_by_trace
            for segment in trace_segments
            for observation in segment.observations
        ]
        return _candidate_thresholds(values)
    values = [
        _switch_margin(switch, observation)
        for trace_segments in segments_by_trace
        for segment in trace_segments
        for observation in segment.observations
    ]
    return _candidate_thresholds(values)


def _switch_label_mistakes(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache | None = None,
) -> int:
    cache = example_cache or _switch_example_cache(examples)
    if not cache.labels:
        return 0
    if isinstance(switch, Depth2Switch):
        theta_weight = switch.theta_weight
        omega_weight = switch.omega_weight
        threshold = switch.threshold
        theta_values = cache.columns[2]
        omega_values = cache.columns[3]
        return sum(
            int(int(theta_weight * theta + omega_weight * omega >= threshold) != label)
            for theta, omega, label in zip(theta_values, omega_values, cache.labels)
        )
    if isinstance(switch, BooleanTreeSwitch):
        return sum(
            int(switch.decide(observation) != label)
            for observation, label in examples
        )
    return sum(
        int(switch.decide(observation) != label)
        for observation, label in examples
    )


def _switch_structure_label_loss(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    example_cache: _SwitchExampleCache | None = None,
) -> float:
    weighted_loss = _responsibility_weighted_switch_label_loss(
        switch,
        segments_by_trace,
        responsibilities,
    )
    if weighted_loss is not None:
        return weighted_loss
    return float(_switch_label_mistakes(switch, examples, example_cache))


def _responsibility_weighted_switch_label_loss(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> float | None:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if not flat_segments or len(flat_segments) != len(responsibilities):
        return None

    loss = 0.0
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    for trace_segments in segments_by_trace:
        for segment_index, segment in enumerate(trace_segments):
            off_weight, on_weight = responsibility_by_id[id(segment)]
            observations = segment.observations
            if segment_index < len(trace_segments) - 1:
                observations = observations[:-1]
            for observation in observations:
                loss += off_weight if switch.decide(observation) == 1 else on_weight
    return loss


def _switch_example_cache(examples: List[Tuple[Observation, int]]) -> _SwitchExampleCache:
    if not examples:
        return _SwitchExampleCache((), ())
    observation_dim = len(examples[0][0])
    labels = tuple(int(label) for _, label in examples)
    columns = tuple(
        tuple(float(observation[index]) for observation, _ in examples)
        for index in range(observation_dim)
    )
    return _SwitchExampleCache(labels, columns)


def _predicate_value_enabled(predicate: ObservationPredicate, value: float) -> bool:
    if predicate.relation == ">=":
        return value >= predicate.threshold
    if predicate.relation == "<=":
        return value <= predicate.threshold
    raise ValueError(f"unknown relation: {predicate.relation}")


def _legacy_switch_threshold_distribution(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> GaussianScalar:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if not flat_segments:
        return GaussianScalar(_switch_default_threshold(switch), 1.0)

    pair_transition_weights = _switch_pair_transition_weights(
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    threshold_samples: List[float] = []
    threshold_weights: List[float] = []
    pair_index = 0
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            # Boundary samples matter most when neighboring segments are likely
            # to belong to different latent modes.
            transition_weight = pair_transition_weights[pair_index]
            pair_index += 1
            threshold_samples.append(_switch_margin(switch, current_segment.end_observation))
            threshold_weights.append(transition_weight)

    if not threshold_samples:
        return GaussianScalar(_switch_default_threshold(switch), 1.0)
    return _weighted_gaussian(threshold_samples, threshold_weights, _switch_default_threshold(switch))


def _predicate_threshold_distribution(
    predicate: ObservationPredicate,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> GaussianScalar:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if not flat_segments:
        return GaussianScalar(predicate.threshold, 1.0)
    pair_transition_weights = _switch_pair_transition_weights(
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    values: List[float] = []
    weights: List[float] = []
    pair_index = 0
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            # Predicate thresholds are estimated from segment endpoints, where
            # the trace actually crosses from one inferred primitive to another.
            transition_weight = pair_transition_weights[pair_index]
            pair_index += 1
            values.append(current_segment.end_observation[predicate.feature_index])
            weights.append(transition_weight)
    if not values:
        return GaussianScalar(predicate.threshold, 1.0)
    return _weighted_gaussian(values, weights, predicate.threshold)


def _switch_pair_transition_weights(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> List[float]:
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    if switch_pair_responsibilities is not None:
        if len(switch_pair_responsibilities) != pair_count:
            raise ValueError("switch pair responsibility count must match adjacent segment pairs")
        return [
            off_to_on + on_to_off
            for _, off_to_on, on_to_off, _ in switch_pair_responsibilities
        ]
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match switch timing segments")
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    transition_weights: List[float] = []
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            transition_weights.append(current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0])
    return transition_weights


def _switch_predicates(switch: SwitchProgram) -> List[ObservationPredicate]:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = [switch.first]
        if switch.second is not None:
            predicates.append(switch.second)
        return predicates
    return []


def _switch_default_threshold(switch: SwitchProgram) -> float:
    if isinstance(switch, BooleanTreeSwitch):
        return switch.first.threshold
    return switch.threshold


def _sample_switch(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    rng: random.Random,
) -> SwitchProgram:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        # Sampling changes only learned thresholds; the selected switch structure
        # stays fixed so sampled students remain in the fitted grammar.
        sampled = [
            predicate.with_threshold(rng.gauss(distribution.mean, distribution.std))
            for predicate, distribution in zip(predicates, distributions)
        ]
        if len(sampled) == 1:
            return BooleanTreeSwitch(sampled[0])
        return BooleanTreeSwitch(sampled[0], sampled[1], switch.operator)
    distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, 1.0)
    return Depth2Switch(
        switch.theta_weight,
        switch.omega_weight,
        rng.gauss(distribution.mean, distribution.std),
    )


def _switch_margin(switch: SwitchProgram, observation: Observation) -> float:
    if isinstance(switch, BooleanTreeSwitch):
        return _predicate_margin(switch.first, observation)
    _, _, theta, omega = observation
    return switch.theta_weight * theta + switch.omega_weight * omega


def _predicate_margin(predicate: ObservationPredicate, observation: Observation) -> float:
    value = observation[predicate.feature_index]
    if predicate.relation == ">=":
        return value - predicate.threshold
    if predicate.relation == "<=":
        return predicate.threshold - value
    raise ValueError(f"unknown relation: {predicate.relation}")


def _switch_enabled_probability(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observation: Observation,
) -> float:
    if isinstance(switch, BooleanTreeSwitch):
        if switch.second is None:
            if not distributions:
                return 1.0 if switch.decide(observation) == 1 else 0.0
            return _predicate_enabled_probability(switch.first, distributions[0], observation)
        if len(distributions) < 2:
            return 1.0 if switch.decide(observation) == 1 else 0.0
        first_probability = _predicate_enabled_probability(switch.first, distributions[0], observation)
        second_probability = _predicate_enabled_probability(switch.second, distributions[1], observation)
        if switch.operator == "and":
            probability = first_probability * second_probability
        elif switch.operator == "or":
            probability = first_probability + second_probability - first_probability * second_probability
        else:
            raise ValueError(f"unknown BooleanTreeSwitch operator: {switch.operator}")
        return min(max(probability, 0.0), 1.0)
    distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, MIN_GAUSSIAN_STD)
    _, _, theta, omega = observation
    value = switch.theta_weight * theta + switch.omega_weight * omega
    return _gaussian_threshold_pass_probability(value, distribution, ">=")


def _predicate_enabled_probability(
    predicate: ObservationPredicate,
    distribution: GaussianScalar,
    observation: Observation,
) -> float:
    return _gaussian_threshold_pass_probability(
        observation[predicate.feature_index],
        distribution,
        predicate.relation,
    )


def _gaussian_threshold_pass_probability(value: float, distribution: GaussianScalar, relation: str) -> float:
    cdf = _gaussian_cdf(value, distribution)
    if relation == ">=":
        return cdf
    if relation == "<=":
        return 1.0 - cdf
    raise ValueError(f"unknown relation: {relation}")


def _gaussian_cdf(value: float, distribution: GaussianScalar) -> float:
    std = max(float(distribution.std), MIN_GAUSSIAN_STD)
    z = (float(value) - float(distribution.mean)) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _learn_depth2_switch(
    traces: List[CartpoleTrace],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
) -> SwitchProgram:
    examples: List[Tuple[Observation, int]] = []
    for trace in traces:
        examples.extend(zip(trace.observations, trace.mode_labels))
    if not examples:
        return Depth2Switch(1.0, 0.0, 0.0)

    example_cache = _switch_example_cache(examples)
    objective_cache: Dict[str, Tuple[float, float, int, str]] = {}
    boolean_switches = _greedy_boolean_tree_candidates(
        examples,
        segments_by_trace=segments_by_trace,
        responsibilities=responsibilities,
        switch_pair_responsibilities=switch_pair_responsibilities,
        cache=objective_cache,
        example_cache=example_cache,
    )
    candidates_with_mistakes: List[Tuple[SwitchProgram, int]] = [
        *_depth2_switch_candidates_with_mistakes(example_cache),
        *[
            (switch, _switch_label_mistakes(switch, examples, example_cache))
            for switch in boolean_switches
        ],
    ]
    candidate_switches = _prefilter_switches_by_label_mistakes(candidates_with_mistakes)

    candidates = []
    for switch in _switch_structure_rescore_candidates(
        candidate_switches,
        examples,
        segments_by_trace=segments_by_trace,
        responsibilities=responsibilities,
        switch_pair_responsibilities=switch_pair_responsibilities,
        cache=objective_cache,
        example_cache=example_cache,
    ):
        candidates.append(
            (
                *_switch_structure_cost(
                    switch,
                    examples,
                    segments_by_trace=segments_by_trace,
                    responsibilities=responsibilities,
                    switch_pair_responsibilities=switch_pair_responsibilities,
                    cache=objective_cache,
                    example_cache=example_cache,
                ),
                switch,
            )
        )
    return min(candidates, key=lambda item: item[:-1])[-1]


def _depth2_switch_candidates_with_mistakes(
    example_cache: _SwitchExampleCache,
) -> List[Tuple[Depth2Switch, int]]:
    theta_values = example_cache.columns[2]
    omega_values = example_cache.columns[3]
    candidates: List[Tuple[Depth2Switch, int]] = []
    for theta_weight in SWITCH_OBLIQUE_THETA_WEIGHTS:
        for omega_weight in SWITCH_OBLIQUE_OMEGA_WEIGHTS:
            scores = [
                theta_weight * theta + omega_weight * omega
                for theta, omega in zip(theta_values, omega_values)
            ]
            thresholds = _candidate_thresholds(scores)
            for threshold, mistakes in _threshold_label_mistakes(scores, example_cache.labels, thresholds):
                candidates.append((Depth2Switch(theta_weight, omega_weight, threshold), mistakes))
    return candidates


def _threshold_label_mistakes(
    scores: List[float],
    labels: Tuple[int, ...],
    thresholds: List[float],
) -> List[Tuple[float, int]]:
    sorted_pairs = sorted(zip(scores, labels))
    sorted_scores = [score for score, _ in sorted_pairs]
    prefix_ones = [0]
    for _, label in sorted_pairs:
        prefix_ones.append(prefix_ones[-1] + int(label))
    total_ones = prefix_ones[-1]
    total_zeros = len(labels) - total_ones

    mistakes: List[Tuple[float, int]] = []
    for threshold in thresholds:
        below = bisect.bisect_left(sorted_scores, threshold)
        ones_below = prefix_ones[below]
        zeros_below = below - ones_below
        mistakes.append((threshold, ones_below + (total_zeros - zeros_below)))
    return mistakes


def _prefilter_switches_by_label_mistakes(
    candidates: List[Tuple[SwitchProgram, int]],
) -> List[SwitchProgram]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item[1],
            item[0].node_count if isinstance(item[0], BooleanTreeSwitch) else 1,
            item[0].describe(),
        ),
    )
    return [switch for switch, _ in ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]]


def _greedy_boolean_tree_candidates(
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    cache: Dict[str, Tuple[float, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> List[BooleanTreeSwitch]:
    stumps = [BooleanTreeSwitch(predicate) for predicate in _predicate_candidates(examples)]
    if not stumps:
        return []
    switch_examples = example_cache or _switch_example_cache(examples)
    best = _best_switch(
        stumps,
        examples,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        cache=cache,
        example_cache=switch_examples,
    )
    best_mistakes = _switch_label_mistakes(best, examples, switch_examples)
    seed_stumps = [
        stump
        for stump in stumps
        if _switch_label_mistakes(stump, examples, switch_examples) == best_mistakes
    ]
    # A second predicate refines either leaf of the stump, yielding bounded
    # depth-2 conjunction/disjunction candidates from the paper's tree view.
    conjunctions: List[BooleanTreeSwitch] = []
    disjunctions: List[BooleanTreeSwitch] = []
    for stump in seed_stumps:
        conjunction_examples = [
            (observation, label)
            for observation, label in examples
            if stump.decide(observation) == 1
        ]
        disjunction_examples = [
            (observation, label)
            for observation, label in examples
            if stump.decide(observation) == 0
        ]
        conjunctions.extend(
            BooleanTreeSwitch(stump.first, predicate)
            for predicate in _predicate_candidates(conjunction_examples)
        )
        disjunctions.extend(
            BooleanTreeSwitch(stump.first, predicate, "or")
            for predicate in _predicate_candidates(disjunction_examples)
        )
    expansions = conjunctions + disjunctions
    if not expansions:
        return [best]
    result = [best]
    if conjunctions:
        result.append(
            _best_switch(
                _prefilter_switches_by_label_mistakes(
                    [
                        (switch, _switch_label_mistakes(switch, examples, switch_examples))
                        for switch in conjunctions
                    ]
                ),
                examples,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities,
                cache=cache,
                example_cache=switch_examples,
            )
        )
    if disjunctions:
        result.append(
            _best_switch(
                _prefilter_switches_by_label_mistakes(
                    [
                        (switch, _switch_label_mistakes(switch, examples, switch_examples))
                        for switch in disjunctions
                    ]
                ),
                examples,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities,
                cache=cache,
                example_cache=switch_examples,
            )
        )
    return result


def _best_switch(
    switches: List[BooleanTreeSwitch],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    cache: Dict[str, Tuple[float, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> BooleanTreeSwitch:
    objective_cache: Dict[str, Tuple[float, float, int, str]] = cache if cache is not None else {}
    switch_examples = example_cache or _switch_example_cache(examples)
    return min(
        _switch_structure_rescore_candidates(
            switches,
            examples,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities=switch_pair_responsibilities,
            cache=objective_cache,
            example_cache=switch_examples,
        ),
        key=lambda switch: _switch_structure_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities=switch_pair_responsibilities,
            cache=objective_cache,
            example_cache=switch_examples,
        ),
    )


def _switch_structure_rescore_candidates(
    switches: List[SwitchProgram],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    cache: Dict[str, Tuple[float, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> List[SwitchProgram]:
    if segments_by_trace is None or responsibilities is None or len(switches) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        return switches

    switch_examples = example_cache or _switch_example_cache(examples)
    prefiltered = _prefilter_switches_by_label_mistakes(
        [
            (switch, _switch_label_mistakes(switch, examples, switch_examples))
            for switch in switches
        ]
    )
    objective_cache: Dict[str, Tuple[float, float, int, str]] = cache if cache is not None else {}
    ranked = sorted(
        prefiltered,
        key=lambda switch: _switch_structure_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities=switch_pair_responsibilities,
            cache=objective_cache,
            example_cache=switch_examples,
        ),
    )
    return ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]


def _switch_structure_cost(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    cache: Dict[str, Tuple[float, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> Tuple[float, float, int, str]:
    if segments_by_trace is None or responsibilities is None:
        return _switch_cost(switch, examples, segments_by_trace, responsibilities, example_cache)
    cache_key = _switch_structure_objective_cache_key(switch, switch_pair_responsibilities)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Score a candidate structure after bounded Gaussian threshold refinement,
    # matching the objective reported in metrics provenance.
    _, label_loss, timing_loss, complexity, description = _fit_switch_structure_objective(
        switch,
        examples,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        example_cache=example_cache,
    )
    result = (label_loss, timing_loss, complexity, description)
    if cache is not None:
        cache[cache_key] = result
    return result


def _switch_structure_objective_cache_key(
    switch: SwitchProgram,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
) -> str:
    key = _switch_cache_key(switch)
    if switch_pair_responsibilities is None:
        return key
    return f"{key}|pair_posteriors={tuple(switch_pair_responsibilities)!r}"


def _fit_switch_structure_objective(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> Tuple[SwitchProgram, float, float, int, str]:
    distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    refined_switch = _switch_with_distribution_means(switch, distributions)
    label_loss = _switch_structure_label_loss(
        refined_switch,
        examples,
        segments_by_trace,
        responsibilities,
        example_cache,
    )
    timing_loss = _switch_distribution_timing_loss(
        refined_switch,
        distributions,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities=switch_pair_responsibilities,
    )
    complexity = refined_switch.node_count if isinstance(refined_switch, BooleanTreeSwitch) else 1
    return refined_switch, label_loss, timing_loss, complexity, refined_switch.describe()


def _switch_cost(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> Tuple[int, float, int, str]:
    mistakes = _switch_label_mistakes(switch, examples, example_cache)
    timing_loss = _switch_timing_loss(switch, segments_by_trace, responsibilities)
    # Lexicographic ordering favors label fidelity, then transition timing, then
    # a smaller/readable program; the description makes ties deterministic.
    complexity = switch.node_count if isinstance(switch, BooleanTreeSwitch) else 1
    return mistakes, timing_loss, complexity, switch.describe()


def _boolean_tree_candidates(examples: List[Tuple[Observation, int]]) -> List[BooleanTreeSwitch]:
    return _greedy_boolean_tree_candidates(examples)


def _predicate_candidates(examples: List[Tuple[Observation, int]]) -> List[ObservationPredicate]:
    if not examples:
        return []
    predicates: List[ObservationPredicate] = []
    observation_dim = len(examples[0][0])
    for feature_index in range(observation_dim):
        values = [observation[feature_index] for observation, _ in examples]
        for threshold in _candidate_thresholds(values):
            predicates.append(ObservationPredicate(feature_index, ">=", threshold))
            predicates.append(ObservationPredicate(feature_index, "<=", threshold))
    return predicates


def _switch_timing_loss(
    switch: SwitchProgram,
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
) -> float:
    if segments_by_trace is None or responsibilities is None:
        return 0.0

    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        return 0.0
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }

    loss = 0.0
    for trace_segments in segments_by_trace:
        for index, current_segment in enumerate(trace_segments):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            if index + 1 < len(trace_segments):
                next_resp = responsibility_by_id.get(id(trace_segments[index + 1]), (0.5, 0.5))
                # This timing term is a local approximation to the paper's
                # switch likelihood: prefer enabling the switch at observed
                # boundaries.
                loss -= _eq12_switch_log_likelihood(
                    switch,
                    current_segment,
                    current_resp,
                    next_resp,
                )
            else:
                loss -= _final_segment_stay_log_likelihood(switch, current_segment, current_resp)
    return loss


def _final_segment_stay_log_likelihood(
    switch: SwitchProgram,
    segment: CartpoleSegment,
    current_resp: Tuple[float, float],
) -> float:
    first_enabled_time = _enabled_step_elapsed_time(
        _first_enabled_step(switch, segment.observations),
        segment.timing_step_scale,
    )
    first_disabled_time = _enabled_step_elapsed_time(
        _first_disabled_step(switch, segment.observations),
        segment.timing_step_scale,
    )
    duration = segment.switch_timing_duration
    return (
        current_resp[0] * _log_no_transition_before_duration(first_enabled_time, duration)
        + current_resp[1] * _log_no_transition_before_duration(first_disabled_time, duration)
    )


def _eq12_switch_log_likelihood(
    switch: SwitchProgram,
    segment: CartpoleSegment,
    current_resp: Tuple[float, float],
    next_resp: Tuple[float, float],
) -> float:
    off_to_on_weight = current_resp[0] * next_resp[1]
    on_to_off_weight = current_resp[1] * next_resp[0]
    stay_off_weight = current_resp[0] * next_resp[0]
    stay_on_weight = current_resp[1] * next_resp[1]
    first_enabled_time = _enabled_step_elapsed_time(
        _first_enabled_step(switch, segment.observations),
        segment.timing_step_scale,
    )
    first_disabled_time = _enabled_step_elapsed_time(
        _first_disabled_step(switch, segment.observations),
        segment.timing_step_scale,
    )
    duration = segment.switch_timing_duration
    return (
        off_to_on_weight * _log_transition_at_duration(first_enabled_time, duration)
        + on_to_off_weight * _log_transition_at_duration(first_disabled_time, duration)
        + stay_off_weight * _log_no_transition_before_duration(first_enabled_time, duration)
        + stay_on_weight * _log_no_transition_before_duration(first_disabled_time, duration)
    )


def _switch_transition_and_stay_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float]:
    scalar = _single_threshold_view(switch, distributions, observations)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_transition_and_stay_probability(
            tuple(values),
            distribution,
            relation,
            duration,
            timing_step_scale,
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        enabled_by_step = _boolean_tree_enabled_cumulative_probabilities(
            switch,
            distributions,
            observations,
        )
        return _cumulative_transition_and_stay_probability(enabled_by_step, duration, timing_step_scale)
    enabled_by_step = _switch_enabled_cumulative_probabilities(switch, distributions, observations)
    return _cumulative_transition_and_stay_probability(enabled_by_step, duration, timing_step_scale)


def _switch_selector_transition_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float, float, float]:
    first_step = _first_step_selector_transition_probabilities(
        switch,
        distributions,
        observations,
        duration,
        timing_step_scale,
    )
    if first_step is not None:
        return first_step
    scalar = _single_threshold_view(switch, distributions, observations)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_selector_transition_probabilities(
            tuple(values),
            distribution,
            relation,
            duration,
            timing_step_scale,
        )
    on_by_step = _switch_enabled_cumulative_probabilities(
        switch,
        distributions,
        observations,
    )
    off_by_step = _switch_disabled_cumulative_probabilities(
        switch,
        distributions,
        observations,
    )
    off_to_on, stay_off = _cumulative_transition_and_stay_probability(
        on_by_step,
        duration,
        timing_step_scale,
    )
    on_to_off, stay_on = _cumulative_transition_and_stay_probability(
        off_by_step,
        duration,
        timing_step_scale,
    )
    return off_to_on, on_to_off, stay_off, stay_on


def _switch_selector_transition_probabilities_for_pair(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    pair: _SwitchTimingPair,
) -> Tuple[float, float, float, float]:
    first_step = _first_step_selector_transition_probabilities(
        switch,
        distributions,
        pair.observations,
        pair.timing_duration,
        pair.timing_step_scale,
    )
    if first_step is not None:
        return first_step
    scalar = _single_threshold_pair_view(switch, distributions, pair)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_selector_transition_probabilities(
            values,
            distribution,
            relation,
            pair.timing_duration,
            pair.timing_step_scale,
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        on_by_step = _boolean_tree_pair_enabled_cumulative_probabilities(switch, distributions, pair)
        off_by_step = _boolean_tree_pair_disabled_cumulative_probabilities(switch, distributions, pair)
    else:
        on_by_step = _switch_enabled_cumulative_probabilities(switch, distributions, pair.observations)
        off_by_step = _switch_disabled_cumulative_probabilities(switch, distributions, pair.observations)
    off_to_on, stay_off = _cumulative_transition_and_stay_probability(
        on_by_step,
        pair.timing_duration,
        pair.timing_step_scale,
    )
    on_to_off, stay_on = _cumulative_transition_and_stay_probability(
        off_by_step,
        pair.timing_duration,
        pair.timing_step_scale,
    )
    return off_to_on, on_to_off, stay_off, stay_on


def _first_step_selector_transition_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: Sequence[Observation],
    duration: float,
    timing_step_scale: float,
) -> Tuple[float, float, float, float] | None:
    if duration <= 0:
        return 0.0, 0.0, 1.0, 1.0
    if not observations:
        return 0.0, 0.0, 1.0, 1.0
    current_index = _timing_duration_step_index(
        duration,
        timing_step_scale,
        len(observations),
    )
    if current_index != 0:
        return None
    enabled = min(max(_switch_enabled_probability(switch, distributions, observations[0]), 0.0), 1.0)
    return enabled, 1.0 - enabled, 1.0, 1.0


def _single_threshold_selector_transition_probabilities(
    values: Tuple[float, ...],
    distribution: GaussianScalar,
    relation: str,
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float, float, float]:
    off_to_on, stay_off = _single_threshold_transition_and_stay_probability(
        values,
        distribution,
        relation,
        duration,
        timing_step_scale,
    )
    on_to_off, stay_on = _single_threshold_disable_transition_and_stay_probability(
        values,
        distribution,
        relation,
        duration,
        timing_step_scale,
    )
    return off_to_on, on_to_off, stay_off, stay_on


def _single_threshold_disable_transition_and_stay_probability(
    values: Tuple[float, ...],
    distribution: GaussianScalar,
    relation: str,
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 1.0
    current_index = _timing_duration_step_index(duration, timing_step_scale, len(values))
    previous = values[:current_index]
    if relation == ">=":
        stay_probability = _gaussian_cdf(min(previous), distribution) if previous else 1.0
        if current_index >= len(values):
            return 0.0, stay_probability
        current_cdf = _gaussian_cdf(values[current_index], distribution)
        transition_probability = max(stay_probability - current_cdf, 0.0)
        return transition_probability, stay_probability
    if relation == "<=":
        previous_cdf = _gaussian_cdf(max(previous), distribution) if previous else 0.0
        stay_probability = max(1.0 - previous_cdf, 0.0)
        if current_index >= len(values):
            return 0.0, stay_probability
        current_cdf = _gaussian_cdf(values[current_index], distribution)
        transition_probability = max(current_cdf - previous_cdf, 0.0)
        return transition_probability, stay_probability
    raise ValueError(f"unknown relation: {relation}")


def _switch_disabled_cumulative_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
) -> List[float]:
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        return _boolean_tree_disabled_cumulative_probabilities(switch, distributions, observations)
    no_disable_probability = 1.0
    disabled_by_step: List[float] = []
    for observation in observations:
        disable_probability = 1.0 - _switch_enabled_probability(switch, distributions, observation)
        no_disable_probability *= 1.0 - min(max(disable_probability, 0.0), 1.0)
        disabled_by_step.append(1.0 - no_disable_probability)
    return disabled_by_step


def _switch_transition_and_stay_probabilities_for_pair(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    pair: _SwitchTimingPair,
) -> Tuple[float, float]:
    scalar = _single_threshold_pair_view(switch, distributions, pair)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_transition_and_stay_probability(
            values,
            distribution,
            relation,
            pair.timing_duration,
            pair.timing_step_scale,
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        enabled_by_step = _boolean_tree_pair_enabled_cumulative_probabilities(switch, distributions, pair)
        return _cumulative_transition_and_stay_probability(
            enabled_by_step,
            pair.timing_duration,
            pair.timing_step_scale,
        )
    enabled_by_step = _switch_enabled_cumulative_probabilities(
        switch,
        distributions,
        pair.observations,
    )
    return _cumulative_transition_and_stay_probability(
        enabled_by_step,
        pair.timing_duration,
        pair.timing_step_scale,
    )


def _switch_transition_probability_at_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: float,
    timing_step_scale: float = 1.0,
) -> float:
    transition_probability, _ = _switch_transition_and_stay_probabilities(
        switch,
        distributions,
        observations,
        duration,
        timing_step_scale,
    )
    return transition_probability


def _switch_no_transition_probability_before_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: float,
    timing_step_scale: float = 1.0,
) -> float:
    _, stay_probability = _switch_transition_and_stay_probabilities(
        switch,
        distributions,
        observations,
        duration,
        timing_step_scale,
    )
    return stay_probability


def _single_threshold_view(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
) -> Tuple[List[float], GaussianScalar, str] | None:
    if isinstance(switch, Depth2Switch):
        distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, MIN_GAUSSIAN_STD)
        values = [
            switch.theta_weight * observation[2] + switch.omega_weight * observation[3]
            for observation in observations
        ]
        return values, distribution, ">="
    if isinstance(switch, BooleanTreeSwitch) and switch.second is None and distributions:
        predicate = switch.first
        values = [observation[predicate.feature_index] for observation in observations]
        return values, distributions[0], predicate.relation
    return None


def _single_threshold_pair_view(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    pair: _SwitchTimingPair,
) -> Tuple[Tuple[float, ...], GaussianScalar, str] | None:
    if isinstance(switch, Depth2Switch):
        distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, MIN_GAUSSIAN_STD)
        theta_values = pair.columns[2]
        omega_values = pair.columns[3]
        return (
            tuple(
                switch.theta_weight * theta + switch.omega_weight * omega
                for theta, omega in zip(theta_values, omega_values)
            ),
            distribution,
            ">=",
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is None and distributions:
        predicate = switch.first
        return pair.columns[predicate.feature_index], distributions[0], predicate.relation
    return None


def _boolean_tree_enabled_cumulative_probabilities(
    switch: BooleanTreeSwitch,
    distributions: List[GaussianScalar],
    observations: List[Observation],
) -> List[float]:
    if switch.second is None or len(distributions) < 2:
        return _switch_enabled_cumulative_probabilities(switch, distributions, observations)
    first = switch.first
    second = switch.second
    first_values = tuple(observation[first.feature_index] for observation in observations)
    second_values = tuple(observation[second.feature_index] for observation in observations)
    return _predicate_pair_enabled_cumulative_probabilities(
        first,
        second,
        distributions[0],
        distributions[1],
        first_values,
        second_values,
        switch.operator,
    )


def _boolean_tree_disabled_cumulative_probabilities(
    switch: BooleanTreeSwitch,
    distributions: List[GaussianScalar],
    observations: List[Observation],
) -> List[float]:
    if switch.second is None or len(distributions) < 2:
        return _switch_disabled_cumulative_probabilities(switch, distributions, observations)
    first = switch.first
    second = switch.second
    first_values = tuple(observation[first.feature_index] for observation in observations)
    second_values = tuple(observation[second.feature_index] for observation in observations)
    return _predicate_pair_disabled_cumulative_probabilities(
        first,
        second,
        distributions[0],
        distributions[1],
        first_values,
        second_values,
        switch.operator,
    )


def _boolean_tree_pair_enabled_cumulative_probabilities(
    switch: BooleanTreeSwitch,
    distributions: List[GaussianScalar],
    pair: _SwitchTimingPair,
) -> List[float]:
    if switch.second is None:
        return _switch_enabled_cumulative_probabilities(switch, distributions, pair.observations)
    first = switch.first
    second = switch.second
    first_values = pair.columns[first.feature_index]
    second_values = pair.columns[second.feature_index]
    return _predicate_pair_enabled_cumulative_probabilities(
        first,
        second,
        distributions[0],
        distributions[1],
        first_values,
        second_values,
        switch.operator,
    )


def _boolean_tree_pair_disabled_cumulative_probabilities(
    switch: BooleanTreeSwitch,
    distributions: List[GaussianScalar],
    pair: _SwitchTimingPair,
) -> List[float]:
    if switch.second is None:
        return _switch_disabled_cumulative_probabilities(switch, distributions, pair.observations)
    first = switch.first
    second = switch.second
    first_values = pair.columns[first.feature_index]
    second_values = pair.columns[second.feature_index]
    return _predicate_pair_disabled_cumulative_probabilities(
        first,
        second,
        distributions[0],
        distributions[1],
        first_values,
        second_values,
        switch.operator,
    )


def _predicate_pair_enabled_cumulative_probabilities(
    first: ObservationPredicate,
    second: ObservationPredicate,
    first_distribution: GaussianScalar,
    second_distribution: GaussianScalar,
    first_values: Tuple[float, ...],
    second_values: Tuple[float, ...],
    operator: str = "and",
) -> List[float]:
    first_enabled = _predicate_enabled_cumulative_probabilities(first, first_distribution, first_values)
    second_enabled = _predicate_enabled_cumulative_probabilities(second, second_distribution, second_values)
    if operator == "or":
        return [
            left + right - left * right
            for left, right in zip(first_enabled, second_enabled)
        ]
    if operator == "and":
        return _predicate_pair_rectangle_cumulative_probabilities(
            first,
            second,
            first_distribution,
            second_distribution,
            first_values,
            second_values,
            operator,
            enabled=True,
        )
    raise ValueError(f"unknown BooleanTreeSwitch operator: {operator}")


def _predicate_pair_disabled_cumulative_probabilities(
    first: ObservationPredicate,
    second: ObservationPredicate,
    first_distribution: GaussianScalar,
    second_distribution: GaussianScalar,
    first_values: Tuple[float, ...],
    second_values: Tuple[float, ...],
    operator: str = "and",
) -> List[float]:
    if operator == "and":
        first_disabled = _predicate_disabled_cumulative_probabilities(first, first_distribution, first_values)
        second_disabled = _predicate_disabled_cumulative_probabilities(second, second_distribution, second_values)
        return [
            left + right - left * right
            for left, right in zip(first_disabled, second_disabled)
        ]
    if operator == "or":
        return _predicate_pair_rectangle_cumulative_probabilities(
            first,
            second,
            first_distribution,
            second_distribution,
            first_values,
            second_values,
            operator,
            enabled=False,
        )
    raise ValueError(f"unknown BooleanTreeSwitch operator: {operator}")


def _predicate_pair_rectangle_cumulative_probabilities(
    first: ObservationPredicate,
    second: ObservationPredicate,
    first_distribution: GaussianScalar,
    second_distribution: GaussianScalar,
    first_values: Tuple[float, ...],
    second_values: Tuple[float, ...],
    operator: str,
    enabled: bool,
) -> List[float]:
    frontier: List[Tuple[float, float]] = []
    probabilities: List[float] = []
    rectangle_fn = _predicate_pair_enabled_rectangles if enabled else _predicate_pair_disabled_rectangles
    for first_value, second_value in zip(first_values, second_values):
        first_probability = _predicate_enabled_probability_from_value(first, first_distribution, first_value)
        second_probability = _predicate_enabled_probability_from_value(second, second_distribution, second_value)
        for rectangle in rectangle_fn(first_probability, second_probability, operator):
            _add_anchored_rectangle_to_frontier(frontier, rectangle)
        probabilities.append(_anchored_rectangle_frontier_area(frontier))
    return probabilities


def _predicate_enabled_cumulative_probabilities(
    predicate: ObservationPredicate,
    distribution: GaussianScalar,
    values: Tuple[float, ...],
) -> List[float]:
    return _relation_cumulative_probabilities(values, distribution, predicate.relation)


def _predicate_disabled_cumulative_probabilities(
    predicate: ObservationPredicate,
    distribution: GaussianScalar,
    values: Tuple[float, ...],
) -> List[float]:
    opposite_relation = "<=" if predicate.relation == ">=" else ">="
    return _relation_cumulative_probabilities(values, distribution, opposite_relation)


def _relation_cumulative_probabilities(
    values: Tuple[float, ...],
    distribution: GaussianScalar,
    relation: str,
) -> List[float]:
    cumulative: List[float] = []
    if relation == ">=":
        best_value: float | None = None
        for value in values:
            best_value = value if best_value is None else max(best_value, value)
            cumulative.append(_gaussian_cdf(best_value, distribution))
        return cumulative
    if relation == "<=":
        best_value = None
        for value in values:
            best_value = value if best_value is None else min(best_value, value)
            cumulative.append(1.0 - _gaussian_cdf(best_value, distribution))
        return cumulative
    raise ValueError(f"unknown relation: {relation}")


def _predicate_pair_enabled_rectangles(
    first_probability: float,
    second_probability: float,
    operator: str,
) -> List[Tuple[float, float]]:
    if operator == "and":
        return [(first_probability, second_probability)]
    if operator == "or":
        return [(first_probability, 1.0), (1.0, second_probability)]
    raise ValueError(f"unknown BooleanTreeSwitch operator: {operator}")


def _predicate_pair_disabled_rectangles(
    first_probability: float,
    second_probability: float,
    operator: str,
) -> List[Tuple[float, float]]:
    first_disabled = max(1.0 - first_probability, 0.0)
    second_disabled = max(1.0 - second_probability, 0.0)
    if operator == "and":
        return [(first_disabled, 1.0), (1.0, second_disabled)]
    if operator == "or":
        return [(first_disabled, second_disabled)]
    raise ValueError(f"unknown BooleanTreeSwitch operator: {operator}")


def _predicate_enabled_probability_from_value(
    predicate: ObservationPredicate,
    distribution: GaussianScalar,
    value: float,
) -> float:
    return _gaussian_threshold_pass_probability(value, distribution, predicate.relation)


def _anchored_rectangle_union_probability(rectangles: List[Tuple[float, float]]) -> float:
    frontier: List[Tuple[float, float]] = []
    for rectangle in rectangles:
        _add_anchored_rectangle_to_frontier(frontier, rectangle)
    return _anchored_rectangle_frontier_area(frontier)


def _add_anchored_rectangle_to_frontier(
    frontier: List[Tuple[float, float]],
    rectangle: Tuple[float, float],
) -> None:
    x_bound = min(max(rectangle[0], 0.0), 1.0)
    y_bound = min(max(rectangle[1], 0.0), 1.0)
    if x_bound <= 0.0 or y_bound <= 0.0:
        return
    if any(x >= x_bound and y >= y_bound for x, y in frontier):
        return
    frontier[:] = [
        (x, y)
        for x, y in frontier
        if not (x <= x_bound and y <= y_bound)
    ]
    frontier.append((x_bound, y_bound))
    frontier.sort(key=lambda item: item[0])


def _anchored_rectangle_frontier_area(frontier: List[Tuple[float, float]]) -> float:
    if not frontier:
        return 0.0
    area = 0.0
    previous_x = 0.0
    for x_bound, y_bound in sorted(frontier):
        if x_bound <= previous_x:
            continue
        area += (x_bound - previous_x) * y_bound
        previous_x = x_bound
    return min(max(area, 0.0), 1.0)


def _single_threshold_transition_and_stay_probability(
    values: Tuple[float, ...],
    distribution: GaussianScalar,
    relation: str,
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 1.0
    current_index = _timing_duration_step_index(duration, timing_step_scale, len(values))
    previous = values[:current_index]
    if not previous:
        previous_probability = 0.0 if relation == ">=" else 1.0
    elif relation == ">=":
        previous_probability = _gaussian_cdf(max(previous), distribution)
    elif relation == "<=":
        previous_probability = _gaussian_cdf(min(previous), distribution)
    else:
        raise ValueError(f"unknown relation: {relation}")
    if current_index >= len(values):
        return 0.0, _single_threshold_stay_probability(previous_probability, relation)

    current_cdf = _gaussian_cdf(values[current_index], distribution)
    if relation == ">=":
        transition_probability = max(current_cdf - previous_probability, 0.0)
    elif relation == "<=":
        transition_probability = max(previous_probability - current_cdf, 0.0)
    else:
        raise ValueError(f"unknown relation: {relation}")
    return transition_probability, _single_threshold_stay_probability(previous_probability, relation)


def _single_threshold_stay_probability(previous_probability: float, relation: str) -> float:
    if relation == ">=":
        return max(1.0 - previous_probability, 0.0)
    if relation == "<=":
        return max(previous_probability, 0.0)
    raise ValueError(f"unknown relation: {relation}")


def _single_threshold_disable_stay_probability(previous_probability: float, relation: str) -> float:
    if relation == ">=":
        return max(previous_probability, 0.0)
    if relation == "<=":
        return max(1.0 - previous_probability, 0.0)
    raise ValueError(f"unknown relation: {relation}")


def _cumulative_transition_and_stay_probability(
    enabled_by_step: List[float],
    duration: float,
    timing_step_scale: float = 1.0,
) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 1.0
    current_index = _timing_duration_step_index(duration, timing_step_scale, len(enabled_by_step))
    previous_index = current_index - 1
    previous_probability = enabled_by_step[previous_index] if previous_index >= 0 else 0.0
    if current_index >= len(enabled_by_step):
        return 0.0, max(1.0 - previous_probability, 0.0)
    transition_probability = max(enabled_by_step[current_index] - previous_probability, 0.0)
    stay_probability = max(1.0 - previous_probability, 0.0)
    return transition_probability, stay_probability


def _timing_duration_step_index(duration: float, timing_step_scale: float, available_steps: int) -> int:
    if available_steps <= 0:
        return 0
    step_scale = max(MIN_GAUSSIAN_STD, float(timing_step_scale))
    return max(0, int(math.ceil(float(duration) / step_scale)) - 1)


def _enabled_step_elapsed_time(first_enabled: int | float, timing_step_scale: float) -> float:
    return float(first_enabled) * max(MIN_GAUSSIAN_STD, float(timing_step_scale))


def _single_threshold_transition_probability(
    values: List[float],
    distribution: GaussianScalar,
    relation: str,
    duration: int,
) -> float:
    if duration <= 0 or duration > len(values):
        return 0.0
    current = values[duration - 1]
    previous = values[: duration - 1]
    if relation == ">=":
        previous_cdf = _gaussian_cdf(max(previous), distribution) if previous else 0.0
        return max(_gaussian_cdf(current, distribution) - previous_cdf, 0.0)
    if relation == "<=":
        previous_cdf = _gaussian_cdf(min(previous), distribution) if previous else 1.0
        return max(previous_cdf - _gaussian_cdf(current, distribution), 0.0)
    raise ValueError(f"unknown relation: {relation}")


def _single_threshold_no_transition_probability(
    values: List[float],
    distribution: GaussianScalar,
    relation: str,
    duration: int,
) -> float:
    previous = values[: max(duration - 1, 0)]
    if not previous:
        return 1.0
    if relation == ">=":
        return max(1.0 - _gaussian_cdf(max(previous), distribution), 0.0)
    if relation == "<=":
        return max(_gaussian_cdf(min(previous), distribution), 0.0)
    raise ValueError(f"unknown relation: {relation}")


def _switch_enabled_cumulative_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
) -> List[float]:
    no_enable_probability = 1.0
    enabled_by_step: List[float] = []
    for observation in observations:
        step_probability = _switch_enabled_probability(switch, distributions, observation)
        no_enable_probability *= 1.0 - min(max(step_probability, 0.0), 1.0)
        enabled_by_step.append(1.0 - no_enable_probability)
    return enabled_by_step

def _log_transition_at_duration(first_enabled: int, duration: float) -> float:
    z = (first_enabled - duration) / SWITCH_TIMING_STD_STEPS
    return -0.5 * z * z


def _log_no_transition_before_duration(first_enabled: int, duration: float) -> float:
    if first_enabled >= duration:
        return 0.0
    z = (duration - first_enabled) / SWITCH_TIMING_STD_STEPS
    return -0.5 * z * z


def _first_enabled_step(switch: SwitchProgram, observations: List[Observation]) -> int:
    for index, observation in enumerate(observations, start=1):
        if switch.decide(observation) == 1:
            return index
    return len(observations) + 1


def _first_disabled_step(switch: SwitchProgram, observations: List[Observation]) -> int:
    for index, observation in enumerate(observations, start=1):
        if switch.decide(observation) == 0:
            return index
    return len(observations) + 1


def _candidate_thresholds(values: List[float]) -> List[float]:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return unique or [0.0]
    if len(unique) > MAX_SWITCH_THRESHOLD_CANDIDATES:
        step = max(1, len(unique) // MAX_SWITCH_THRESHOLD_CANDIDATES)
        unique = unique[::step]
    candidates = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    candidates.append(DEFAULT_SWITCH_THRESHOLD_CANDIDATE)
    return candidates
