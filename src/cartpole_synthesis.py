from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple

from cartpole_env import (
    CartpoleConfig,
    CartpoleEnv,
    Observation,
    cartpole_done,
    cartpole_next_state,
)


MIN_GAUSSIAN_STD = 1e-3
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
TEACHER_DURATION_REFINEMENT_DELTAS = (-1, 1)
TEACHER_ACTION_REFINEMENT_CANDIDATES_PER_SEGMENT = 1
TEACHER_STUDENT_SAMPLE_FRACTION = 1.0
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
SWITCH_SELECTION_OBJECTIVE_ORDER = (
    "hard_label_mistakes",
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
    teacher_student_regularizer: float = TEACHER_STUDENT_REGULARIZER
    teacher_reward_lambda: float = TEACHER_REWARD_LAMBDA
    teacher_top_rho: int = TEACHER_TOP_RHO
    teacher_refinement_steps: int = TEACHER_REFINEMENT_STEPS


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
    teacher_source: str = "gain_sample"
    student_log_probability: float | None = None


def cartpole_synthesis_algorithm_provenance() -> Dict[str, object]:
    return {
        "probabilistic_student": {
            "em_iters": PROBABILISTIC_STUDENT_EM_ITERS,
            "switch_responsibility_passes": PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES,
            "responsibility_evidence": "action_likelihood_then_switch_timing_forward_backward",
            "rollout_parameter_resampling": "on_mode_entry",
            "min_gaussian_std": MIN_GAUSSIAN_STD,
            "log_probability_floor": LOG_PROBABILITY_FLOOR,
        },
        "switch_timing": {
            "std_steps": SWITCH_TIMING_STD_STEPS,
            "scalar_threshold_uses_shared_sample": True,
            "depth2_conjunction_probability": "shared_threshold_rectangle_union",
            "std_refinement_multipliers": list(SWITCH_STD_REFINEMENT_MULTIPLIERS),
            "coordinate_refinement_steps": SWITCH_PARAMETER_COORDINATE_REFINEMENT_STEPS,
            "coordinate_mean_step_fraction": SWITCH_PARAMETER_COORDINATE_MEAN_STEP_FRACTION,
            "coordinate_log_std_initial_step": SWITCH_PARAMETER_COORDINATE_LOG_STD_STEP,
            "coordinate_step_decay": SWITCH_PARAMETER_COORDINATE_STEP_DECAY,
        },
        "switch_search": {
            "boolean_tree_depth": 2,
            "greedy_second_predicate_only_refines_mode1": True,
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
            "duration_refinement_deltas": list(TEACHER_DURATION_REFINEMENT_DELTAS),
            "action_refinement_candidates_per_segment": TEACHER_ACTION_REFINEMENT_CANDIDATES_PER_SEGMENT,
            "student_sample_fraction_after_first_iteration": TEACHER_STUDENT_SAMPLE_FRACTION,
            "student_sample_probability": "trace_log_probability_approximation",
            "student_sample_segment_budget": "chunk_sampled_actions_by_max_segment_duration_then_reroll_loop_free_trace",
            "student_sample_local_refinement": "duration_and_action_coordinate_search",
            "teacher_rollout_horizon": "min_environment_max_steps_and_configured_loop_free_horizon",
            "elite_refinement_objective": "reward_plus_top_rho_log_probability_distance_kernel",
            "elite_distance_metric": "l2_over_segment_actions_and_durations",
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

    def decide(self, observation: Observation) -> int:
        if not self.first.evaluate(observation):
            return 0
        if self.second is not None and not self.second.evaluate(observation):
            return 0
        return 1

    def describe(self) -> str:
        if self.second is None:
            return f"mode=1 if {self.first.describe()}, else mode=0"
        return (
            f"mode=1 if {self.first.describe()} and "
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

    @property
    def end_observation(self) -> Observation:
        return self.observations[-1]


@dataclass(frozen=True)
class _SwitchExampleCache:
    labels: Tuple[int, ...]
    columns: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class _SwitchTimingPair:
    observations: Tuple[Observation, ...]
    columns: Tuple[Tuple[float, ...], ...]
    duration: int
    transition_weight: float
    stay_weight: float


@dataclass(frozen=True)
class _ScalarSwitchTimingPair:
    relation: str
    current_value: float | None
    previous_extreme: float | None
    transition_weight: float
    stay_weight: float


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
        structure_mistakes,
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
    label_error_rate = structure_mistakes / example_count if example_count else 0.0
    deterministic_label_error_rate = mistakes / example_count if example_count else 0.0
    return {
        "description": description,
        "objective_description": structure_description,
        "label_mistakes": structure_mistakes,
        "label_error_rate": label_error_rate,
        "hard_label_mistakes": structure_mistakes,
        "hard_label_mistake_rate": label_error_rate,
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
        "objective_tuple": [structure_mistakes, distribution_loss, structure_complexity, structure_description],
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
    deltas: List[int] = []
    for trace_segments in segments_by_trace:
        for segment in trace_segments[:-1]:
            first_enabled = _first_enabled_step(switch, segment.observations)
            if first_enabled > len(segment.observations):
                never += 1
                continue
            delta = first_enabled - segment.duration
            deltas.append(delta)
            if first_enabled < segment.duration:
                early += 1
            elif first_enabled == segment.duration:
                at_boundary += 1
            else:
                late += 1
    return {
        "num_boundaries": _trace_boundary_count(segments_by_trace),
        "enabled_boundary_count": len(deltas),
        "early_switch_count": early,
        "at_boundary_count": at_boundary,
        "late_switch_count": late,
        "never_enabled_count": never,
        "first_enabled_minus_duration_mean": sum(deltas) / len(deltas) if deltas else None,
        "first_enabled_minus_duration_min": min(deltas) if deltas else None,
        "first_enabled_minus_duration_max": max(deltas) if deltas else None,
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
        self.mode = self.switch.decide(observation)
        return self.right_force if self.mode == 1 else self.left_force

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
        next_mode = self.switch.decide(observation)
        if next_mode != self.mode:
            self.mode = next_mode
            self._resample_segment_parameters(self.mode)
        return self.right_force if self.mode == 1 else self.left_force

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
        student = fit_probabilistic_cartpole_student(traces, cfg)
        history.append(CartpoleSynthesisIteration(iteration + 1, traces, student))
    if student is None:
        raise RuntimeError("Cartpole synthesis did not produce a student policy")
    return student, traces, history


def fit_probabilistic_cartpole_student(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> ProbabilisticCartpoleStudent:
    """Fit the Cartpole student using Gaussian action-parameter distributions.

    This implements the action-distribution part of the paper's EM-style
    student step for Cartpole's constant-action grammar. The latent segment
    responsibilities are first initialized from action likelihoods, then refined
    with a bounded forward-backward pass using switch timing likelihoods.
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

    # The actions are observed, but their latent mode assignments are softened
    # so ambiguous segments can influence both constant-action primitives.
    for _ in range(PROBABILISTIC_STUDENT_EM_ITERS):
        responsibilities = [
            _mode_responsibilities(segment.action_parameter, action_distributions)
            for segment in segments
        ]
        action_distributions = _fit_action_distributions(
            segments,
            responsibilities,
            left_default,
            right_default,
        )

    # Fit the discrete switch after action EM so switch costs can use the same
    # soft transition evidence instead of only the teacher's hard labels.
    switch, switch_parameter_distributions = _fit_student_switch(
        traces,
        segments_by_trace,
        responsibilities,
    )
    for _ in range(PROBABILISTIC_STUDENT_SWITCH_RESPONSIBILITY_PASSES):
        responsibilities = _refine_responsibilities_with_switch_timing(
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
        )
    threshold_distribution = (
        switch_parameter_distributions[0]
        if switch_parameter_distributions
        else GaussianScalar(_switch_default_threshold(switch), 1.0)
    )
    return ProbabilisticCartpoleStudent(
        action_distributions,
        switch,
        threshold_distribution,
        switch_parameter_distributions,
        responsibilities,
    )


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
    ranked = sorted(
        candidates,
        key=lambda trace: _teacher_objective(trace, scoring_student, cfg),
        reverse=True,
    )
    # Refine only the top candidates to keep synthesis cheap while still
    # optimizing around promising sampled loop-free traces.
    elites = ranked[: max(1, cfg.teacher_top_rho)]
    refined = [
        _refine_loop_free_trace(candidate, initial_state, env_cfg, cfg, scoring_student, elites)
        for candidate in elites
        if candidate.segment_actions and candidate.segment_durations
    ]
    return max(
        elites + refined,
        key=lambda trace: _teacher_refinement_objective(trace, scoring_student, cfg, elites),
    )


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
) -> CartpoleTrace:
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    max_segment_steps = max(1, cfg.segment_steps)
    max_segments = max(1, cfg.segments_per_trace)
    max_steps = min(env_cfg.max_steps, max_segment_steps * max_segments)
    durations = segment_durations or tuple(max_segment_steps for _ in range(max_segments))
    durations = tuple(durations[:max_segments])
    chosen_actions: List[float] = []
    started_durations: List[int] = []
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
        chosen_actions.append(action)
        label = 1 if action > 0.0 else 0
        executed_steps = 0
        for _ in range(duration_steps):
            if cartpole_done(state, env_cfg):
                break
            observations.append(list(state))
            actions.append(action)
            mode_labels.append(label)
            state = cartpole_next_state(state, action, env_cfg)
            alive += 1
            executed_steps += 1
            if alive >= max_steps:
                break
        if executed_steps:
            started_durations.append(executed_steps)
    return CartpoleTrace(
        observations=observations,
        actions=actions,
        mode_labels=mode_labels,
        reward=float(alive),
        theta_gain=theta_gain,
        omega_gain=omega_gain,
        segment_actions=tuple(chosen_actions),
        segment_durations=tuple(started_durations),
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
        action = policy.act(observation)
        observations.append(observation)
        actions.append(action)
        mode_labels.append(policy.mode)
        state = cartpole_next_state(state, action, env_cfg)
        alive += 1
    trace = CartpoleTrace(
        observations=observations,
        actions=actions,
        mode_labels=mode_labels,
        reward=float(alive),
        segment_actions=_mode_run_actions(actions, mode_labels),
        segment_durations=_mode_run_lengths(mode_labels),
        teacher_source="student_sample",
    )
    trace = _limit_loop_free_trace_segment_budget(trace, initial_state, env_cfg, cfg)
    trace.student_log_probability = _trace_log_probability(trace, student)
    return trace


def _limit_loop_free_trace_segment_budget(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
) -> CartpoleTrace:
    actions = trace.segment_actions or _mode_run_actions(trace.actions, trace.mode_labels)
    durations = trace.segment_durations or _mode_run_lengths(trace.mode_labels)
    if len(actions) != len(durations):
        raise ValueError("loop-free action count must match duration count")
    max_segments = max(1, cfg.segments_per_trace)
    max_segment_steps = max(1, cfg.segment_steps)
    if len(actions) <= max_segments and all(duration <= max_segment_steps for duration in durations):
        return trace

    # Student samples are closed-loop PSMs, but the paper's teacher candidates
    # are loop-free programs with both a segment-count and segment-time budget.
    projected_actions, projected_durations = _chunk_actions_to_loop_free_segments(
        tuple(trace.actions),
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
    )
    limited.teacher_source = trace.teacher_source
    return limited


def _chunk_actions_to_loop_free_segments(
    actions: Tuple[float, ...],
    max_segment_steps: int,
    max_segments: int,
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    if max_segment_steps < 1:
        raise ValueError("max_segment_steps must be positive")
    if max_segments < 1:
        raise ValueError("max_segments must be positive")

    projected_actions: List[float] = []
    projected_durations: List[int] = []
    limit = min(len(actions), max_segment_steps * max_segments)
    for start in range(0, limit, max_segment_steps):
        chunk = actions[start : start + max_segment_steps]
        if not chunk:
            break
        projected_actions.append(sum(chunk) / len(chunk))
        projected_durations.append(len(chunk))
    return tuple(projected_actions), tuple(projected_durations)


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
    objective_cache: Dict[Tuple[Tuple[float, ...], Tuple[int, ...], float], float] = {}
    elite_log_normalizer = _elite_kernel_log_normalizer(student, objective_elites)

    def objective(candidate: CartpoleTrace) -> float:
        key = (
            tuple(candidate.segment_actions or _mode_run_actions(candidate.actions, candidate.mode_labels)),
            tuple(candidate.segment_durations or _mode_run_lengths(candidate.mode_labels)),
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
                )
                if objective(candidate) > objective(best):
                    best = candidate
                    improved = True
        for candidate in _duration_refinement_candidates(best, initial_state, env_cfg, cfg):
            if objective(candidate) > objective(best):
                best = candidate
                improved = True
        for candidate in _action_refinement_candidates(best, initial_state, env_cfg, cfg):
            if objective(candidate) > objective(best):
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
    if not actions or not durations:
        return []

    candidates: List[CartpoleTrace] = []
    for index, current_action in enumerate(actions):
        for action in cfg.force_values:
            if action == current_action:
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
                )
            )
    return candidates


def _teacher_objective(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> float:
    if student is None:
        return cfg.teacher_reward_lambda * trace.reward
    # The regularizer rewards traces that the current student can already
    # encode, which is the adaptive-teaching pressure in this local diagnostic.
    log_probability = (
        trace.student_log_probability
        if trace.student_log_probability is not None
        else _trace_log_probability(trace, student)
    )
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
        elite.student_log_probability
        if elite.student_log_probability is not None
        else _trace_log_probability(elite, student)
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
        elite_log_probability = (
            elite.student_log_probability
            if elite.student_log_probability is not None
            else _trace_log_probability(elite, student)
        )
        terms.append(elite_log_probability - _loop_free_trace_distance(trace, elite))
    if not terms:
        return _trace_log_probability(trace, student)
    normalizer = elite_log_normalizer
    if normalizer is None:
        normalizer = _elite_kernel_log_normalizer(student, elites)
    return _logsumexp(terms) - (normalizer if normalizer is not None else 0.0)


def _loop_free_trace_distance(left: CartpoleTrace, right: CartpoleTrace) -> float:
    left_actions = left.segment_actions or _mode_run_actions(left.actions, left.mode_labels)
    right_actions = right.segment_actions or _mode_run_actions(right.actions, right.mode_labels)
    left_durations = left.segment_durations or _mode_run_lengths(left.mode_labels)
    right_durations = right.segment_durations or _mode_run_lengths(right.mode_labels)
    length = max(len(left_actions), len(right_actions), len(left_durations), len(right_durations))
    if length == 0:
        return 0.0

    duration_scale = max(
        TEACHER_ELITE_DISTANCE_DURATION_SCALE_FLOOR,
        max(left_durations or (0,)),
        max(right_durations or (0,)),
    )
    total = 0.0
    for index in range(length):
        left_action = left_actions[index] if index < len(left_actions) else 0.0
        right_action = right_actions[index] if index < len(right_actions) else 0.0
        left_duration = left_durations[index] if index < len(left_durations) else 0
        right_duration = right_durations[index] if index < len(right_durations) else 0
        total += (left_action - right_action) ** 2
        total += ((left_duration - right_duration) / duration_scale) ** 2
    return math.sqrt(total)


def _trace_log_probability(trace: CartpoleTrace, student: ProbabilisticCartpoleStudent) -> float:
    total = 0.0
    trace_segments = _segments_from_traces([trace])[0]
    responsibilities = _refine_responsibilities_with_switch_timing(
        [trace_segments],
        student.action_distributions,
        student.switch,
        student.switch_parameter_distributions,
    )
    for segment, resp in zip(trace_segments, responsibilities):
        mode_log_terms = []
        for mode in (0, 1):
            prior = max(resp[mode], LOG_PROBABILITY_FLOOR)
            mode_log_terms.append(
                math.log(prior) + student.action_distributions[mode].log_pdf(segment.action_parameter)
            )
        total += _logsumexp(mode_log_terms)
    for current_index, current_segment in enumerate(trace_segments[:-1]):
        total += _student_switch_log_likelihood(
            student,
            current_segment,
            responsibilities[current_index],
            responsibilities[current_index + 1],
        )
    return total


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
    for action, duration in zip(trace.segment_actions, trace.segment_durations):
        end = min(start + max(1, int(duration)), len(trace.actions))
        if start >= end:
            break
        segments.append(
            CartpoleSegment(
                observations=trace.observations[start:end],
                action_parameter=float(action),
                duration=end - start,
                hard_mode=1 if action > 0.0 else 0,
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
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    # Refit both switch structure and Gaussian threshold parameters after each
    # responsibility update so action and timing evidence stay in sync.
    switch = _learn_depth2_switch(traces, segments_by_trace, responsibilities)
    distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        responsibilities,
    )
    return _switch_with_distribution_means(switch, distributions), distributions


def _refine_responsibilities_with_switch_timing(
    segments_by_trace: List[List[CartpoleSegment]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
) -> List[Tuple[float, float]]:
    responsibilities: List[Tuple[float, float]] = []
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

        # Forward scores accumulate prefix evidence for the two latent modes.
        forward: List[List[float]] = [[emissions[0][0], emissions[0][1]]]
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

        # Backward scores accumulate suffix evidence without changing ordering.
        backward: List[List[float]] = [[0.0, 0.0] for _ in trace_segments]
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
        norm = _logsumexp(forward[-1])
        for index in range(len(trace_segments)):
            posterior_logs = [
                forward[index][mode] + backward[index][mode] - norm
                for mode in (0, 1)
            ]
            weights = [math.exp(value) for value in posterior_logs]
            total = sum(weights)
            responsibilities.append((weights[0] / total, weights[1] / total))
    return responsibilities


def _switch_responsibility_pair_log_potentials(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segment: CartpoleSegment,
) -> List[List[float]]:
    transition_probability = _switch_transition_probability_at_duration(
        switch,
        distributions,
        segment.observations,
        segment.duration,
    )
    stay_probability = _switch_no_transition_probability_before_duration(
        switch,
        distributions,
        segment.observations,
        segment.duration,
    )
    # Different adjacent modes consume transition likelihood; equal modes
    # consume survival likelihood before the observed boundary.
    transition_log = math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
    stay_log = math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
    return [
        [stay_log, transition_log],
        [transition_log, stay_log],
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
) -> List[GaussianScalar]:
    predicates = _switch_predicates(switch)
    if not predicates:
        distribution = _legacy_switch_threshold_distribution(switch, segments_by_trace, responsibilities)
        _, refined = _refine_switch_parameter_distributions(
            switch,
            [distribution],
            segments_by_trace,
            responsibilities,
        )
        return refined
    distributions = [
        _predicate_threshold_distribution(predicate, segments_by_trace, responsibilities)
        for predicate in predicates
    ]
    _, refined = _refine_switch_parameter_distributions(
        switch,
        distributions,
        segments_by_trace,
        responsibilities,
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
        return BooleanTreeSwitch(fitted[0], fitted[1])
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
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    return _refine_switch_parameter_distributions(
        switch,
        distributions,
        segments_by_trace,
        responsibilities,
    )


def _refine_switch_parameter_distributions(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
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
    timing_pairs = _switch_timing_pairs(segments_by_trace, responsibilities)
    scalar_timing_pairs = _scalar_switch_timing_pairs(switch, timing_pairs)
    mistake_cache: Dict[str, int] = {}

    def label_mistakes(candidate: SwitchProgram) -> int:
        key = candidate.describe()
        if key not in mistake_cache:
            mistake_cache[key] = _switch_label_mistakes(candidate, examples, example_cache)
        return mistake_cache[key]

    best_distributions = _distributions_with_switch_means(switch, distributions)
    best_switch = _switch_with_distribution_means(switch, best_distributions)
    best_mistakes = label_mistakes(best_switch)
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
                candidate_mistakes = label_mistakes(candidate_switch)
                if candidate_mistakes > best_mistakes:
                    continue
                candidate_loss = _switch_distribution_timing_loss(
                    candidate_switch,
                    candidate_distributions,
                    segments_by_trace,
                    responsibilities,
                    timing_pairs,
                    scalar_timing_pairs,
                )
                if candidate_loss < best_loss:
                    best_distributions = candidate_distributions
                    best_switch = candidate_switch
                    best_mistakes = candidate_mistakes
                    best_loss = candidate_loss
    return _coordinate_refine_switch_parameter_distributions(
        switch,
        best_distributions,
        best_switch,
        best_mistakes,
        best_loss,
        examples,
        example_cache,
        mistake_cache,
        timing_pairs,
        scalar_timing_pairs,
        segments_by_trace,
        responsibilities,
    )


def _coordinate_refine_switch_parameter_distributions(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    current_switch: SwitchProgram,
    current_mistakes: int,
    current_loss: float,
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    mistake_cache: Dict[str, int],
    timing_pairs: List[_SwitchTimingPair],
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    # Start from the grid-refined solution; this is a bounded local polish, not
    # a replacement for the discrete grammar search.
    best_distributions = list(distributions)
    best_switch = current_switch
    best_mistakes = current_mistakes
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
                candidate_switch = _switch_with_distribution_means(switch, candidate_distributions)
                key = candidate_switch.describe()
                if key not in mistake_cache:
                    mistake_cache[key] = _switch_label_mistakes(candidate_switch, examples, example_cache)
                candidate_mistakes = mistake_cache[key]
                if candidate_mistakes > best_mistakes:
                    continue
                candidate_loss = _switch_distribution_timing_loss(
                    candidate_switch,
                    candidate_distributions,
                    segments_by_trace,
                    responsibilities,
                    timing_pairs,
                    scalar_timing_pairs,
                )
                if candidate_loss < best_loss:
                    best_distributions = candidate_distributions
                    best_switch = candidate_switch
                    best_mistakes = candidate_mistakes
                    best_loss = candidate_loss
                    improved = True
        if not improved:
            # When no coordinate helps, shrink the local neighborhood instead
            # of widening the search beyond the fitted switch structure.
            mean_steps = [step * SWITCH_PARAMETER_COORDINATE_STEP_DECAY for step in mean_steps]
            log_std_step *= SWITCH_PARAMETER_COORDINATE_STEP_DECAY
    return best_switch, best_distributions


def _switch_distribution_timing_loss(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    timing_pairs: List[_SwitchTimingPair] | None = None,
    scalar_timing_pairs: List[_ScalarSwitchTimingPair] | None = None,
) -> float:
    if scalar_timing_pairs is not None and len(distributions) == 1:
        return _scalar_switch_distribution_timing_loss(distributions[0], scalar_timing_pairs)
    pairs = timing_pairs if timing_pairs is not None else _switch_timing_pairs(segments_by_trace, responsibilities)
    loss = 0.0
    for pair in pairs:
        transition_probability, stay_probability = _switch_transition_and_stay_probabilities_for_pair(
            switch,
            distributions,
            pair,
        )
        loss -= (
            pair.transition_weight * math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
            + pair.stay_weight * math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
        )
    return loss


def _switch_timing_pairs(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> List[_SwitchTimingPair]:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match switch timing segments")
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    pairs: List[_SwitchTimingPair] = []
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            observations = tuple(current_segment.observations)
            pairs.append(
                _SwitchTimingPair(
                    observations=observations,
                    columns=_observation_columns(observations),
                    duration=current_segment.duration,
                    transition_weight=current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0],
                    stay_weight=current_resp[0] * next_resp[0] + current_resp[1] * next_resp[1],
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
        previous = values[: pair.duration - 1]
        previous_extreme = None
        if previous:
            previous_extreme = max(previous) if relation == ">=" else min(previous)
        current_value = values[pair.duration - 1] if 0 < pair.duration <= len(values) else None
        scalar_pairs.append(
            _ScalarSwitchTimingPair(
                relation=relation,
                current_value=current_value,
                previous_extreme=previous_extreme,
                transition_weight=pair.transition_weight,
                stay_weight=pair.stay_weight,
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
        transition_probability, stay_probability = _scalar_timing_pair_probabilities(distribution, pair)
        loss -= (
            pair.transition_weight * math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
            + pair.stay_weight * math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
        )
    return loss


def _scalar_timing_pair_probabilities(
    distribution: GaussianScalar,
    pair: _ScalarSwitchTimingPair,
) -> Tuple[float, float]:
    if pair.previous_extreme is None:
        previous_probability = 0.0 if pair.relation == ">=" else 1.0
    else:
        previous_probability = _gaussian_cdf(pair.previous_extreme, distribution)
    if pair.current_value is None:
        return 0.0, _single_threshold_stay_probability(previous_probability, pair.relation)

    current_probability = _gaussian_cdf(pair.current_value, distribution)
    if pair.relation == ">=":
        transition_probability = max(current_probability - previous_probability, 0.0)
    elif pair.relation == "<=":
        transition_probability = max(previous_probability - current_probability, 0.0)
    else:
        raise ValueError(f"unknown relation: {pair.relation}")
    return transition_probability, _single_threshold_stay_probability(previous_probability, pair.relation)


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
        first = switch.first
        second = switch.second
        first_values = cache.columns[first.feature_index]
        if second is None:
            if first.relation == ">=":
                return sum(
                    int(int(value >= first.threshold) != label)
                    for value, label in zip(first_values, cache.labels)
                )
            if first.relation == "<=":
                return sum(
                    int(int(value <= first.threshold) != label)
                    for value, label in zip(first_values, cache.labels)
                )
            return sum(
                int(_predicate_value_enabled(first, value) != bool(label))
                for value, label in zip(first_values, cache.labels)
            )
        second_values = cache.columns[second.feature_index]
        if first.relation == ">=" and second.relation == ">=":
            return sum(
                int(int(first_value >= first.threshold and second_value >= second.threshold) != label)
                for first_value, second_value, label in zip(first_values, second_values, cache.labels)
            )
        if first.relation == ">=" and second.relation == "<=":
            return sum(
                int(int(first_value >= first.threshold and second_value <= second.threshold) != label)
                for first_value, second_value, label in zip(first_values, second_values, cache.labels)
            )
        if first.relation == "<=" and second.relation == ">=":
            return sum(
                int(int(first_value <= first.threshold and second_value >= second.threshold) != label)
                for first_value, second_value, label in zip(first_values, second_values, cache.labels)
            )
        if first.relation == "<=" and second.relation == "<=":
            return sum(
                int(int(first_value <= first.threshold and second_value <= second.threshold) != label)
                for first_value, second_value, label in zip(first_values, second_values, cache.labels)
            )
        return sum(
            int(
                (
                    _predicate_value_enabled(first, first_value)
                    and _predicate_value_enabled(second, second_value)
                )
                != bool(label)
            )
            for first_value, second_value, label in zip(first_values, second_values, cache.labels)
        )
    return sum(
        int(switch.decide(observation) != label)
        for observation, label in examples
    )


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
) -> GaussianScalar:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if not flat_segments:
        return GaussianScalar(_switch_default_threshold(switch), 1.0)

    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    threshold_samples: List[float] = []
    threshold_weights: List[float] = []
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            # Boundary samples matter most when neighboring segments are likely
            # to belong to different latent modes.
            transition_weight = current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0]
            threshold_samples.append(_switch_margin(switch, current_segment.end_observation))
            threshold_weights.append(transition_weight)

    if not threshold_samples:
        return GaussianScalar(_switch_default_threshold(switch), 1.0)
    return _weighted_gaussian(threshold_samples, threshold_weights, _switch_default_threshold(switch))


def _predicate_threshold_distribution(
    predicate: ObservationPredicate,
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> GaussianScalar:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    values: List[float] = []
    weights: List[float] = []
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            # Predicate thresholds are estimated from segment endpoints, where
            # the trace actually crosses from one inferred primitive to another.
            transition_weight = current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0]
            values.append(current_segment.end_observation[predicate.feature_index])
            weights.append(transition_weight)
    if not values:
        return GaussianScalar(predicate.threshold, 1.0)
    return _weighted_gaussian(values, weights, predicate.threshold)


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
        return BooleanTreeSwitch(sampled[0], sampled[1])
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
        probability = _predicate_enabled_probability(switch.first, distributions[0], observation)
        probability *= _predicate_enabled_probability(switch.second, distributions[1], observation)
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
) -> SwitchProgram:
    examples: List[Tuple[Observation, int]] = []
    for trace in traces:
        examples.extend(zip(trace.observations, trace.mode_labels))
    if not examples:
        return Depth2Switch(1.0, 0.0, 0.0)

    example_cache = _switch_example_cache(examples)
    objective_cache: Dict[str, Tuple[int, float, int, str]] = {}
    boolean_switches = _greedy_boolean_tree_candidates(
        examples,
        segments_by_trace=segments_by_trace,
        responsibilities=responsibilities,
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
        example_cache=example_cache,
    ):
        candidates.append(
            (
                *_switch_structure_cost(
                    switch,
                    examples,
                    segments_by_trace=segments_by_trace,
                    responsibilities=responsibilities,
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
    if len(candidates) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        return [switch for switch, _ in candidates]
    ranked_mistakes = sorted(mistakes for _, mistakes in candidates)
    cutoff = ranked_mistakes[SWITCH_STRUCTURE_RESCORING_TOP_K - 1]
    return [switch for switch, mistakes in candidates if mistakes <= cutoff]


def _greedy_boolean_tree_candidates(
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    cache: Dict[str, Tuple[int, float, int, str]] | None = None,
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
        cache=cache,
        example_cache=switch_examples,
    )
    expanded_examples = [
        (observation, label)
        for observation, label in examples
        if best.decide(observation) == 1
    ]
    # A second predicate only refines the mode-1 region, yielding a small
    # conjunction instead of an unrestricted tree search.
    expansions = [
        BooleanTreeSwitch(best.first, predicate)
        for predicate in _predicate_candidates(expanded_examples)
    ]
    if not expansions:
        return [best]
    return [
        best,
        _best_switch(
            expansions,
            examples,
            segments_by_trace,
            responsibilities,
            cache=cache,
            example_cache=switch_examples,
        ),
    ]


def _best_switch(
    switches: List[BooleanTreeSwitch],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
    cache: Dict[str, Tuple[int, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> BooleanTreeSwitch:
    objective_cache: Dict[str, Tuple[int, float, int, str]] = cache if cache is not None else {}
    switch_examples = example_cache or _switch_example_cache(examples)
    return min(
        _switch_structure_rescore_candidates(
            switches,
            examples,
            segments_by_trace,
            responsibilities,
            example_cache=switch_examples,
        ),
        key=lambda switch: _switch_structure_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            cache=objective_cache,
            example_cache=switch_examples,
        ),
    )


def _switch_structure_rescore_candidates(
    switches: List[SwitchProgram],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
    example_cache: _SwitchExampleCache | None = None,
) -> List[SwitchProgram]:
    if segments_by_trace is None or responsibilities is None or len(switches) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        return switches

    switch_examples = example_cache or _switch_example_cache(examples)
    mistake_ranked = sorted(
        switches,
        key=lambda switch: (
            _switch_label_mistakes(switch, examples, switch_examples),
            switch.node_count if isinstance(switch, BooleanTreeSwitch) else 1,
            switch.describe(),
        ),
    )
    if len(mistake_ranked) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        prefiltered = mistake_ranked
    else:
        cutoff_mistakes = _switch_label_mistakes(
            mistake_ranked[SWITCH_STRUCTURE_RESCORING_TOP_K - 1],
            examples,
            switch_examples,
        )
        prefiltered = [
            switch
            for switch in mistake_ranked
            if _switch_label_mistakes(switch, examples, switch_examples) <= cutoff_mistakes
        ]
    ranked = sorted(
        prefiltered,
        key=lambda switch: _switch_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            switch_examples,
        ),
    )
    return ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]


def _switch_structure_cost(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    cache: Dict[str, Tuple[int, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
) -> Tuple[int, float, int, str]:
    if segments_by_trace is None or responsibilities is None:
        return _switch_cost(switch, examples, segments_by_trace, responsibilities, example_cache)
    cache_key = switch.describe()
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Score a candidate structure after bounded Gaussian threshold refinement,
    # matching the objective reported in metrics provenance.
    _, mistakes, timing_loss, complexity, description = _fit_switch_structure_objective(
        switch,
        examples,
        segments_by_trace,
        responsibilities,
        example_cache=example_cache,
    )
    result = (mistakes, timing_loss, complexity, description)
    if cache is not None:
        cache[cache_key] = result
    return result


def _fit_switch_structure_objective(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    example_cache: _SwitchExampleCache | None = None,
) -> Tuple[SwitchProgram, int, float, int, str]:
    distributions = _fit_switch_parameter_distributions(switch, segments_by_trace, responsibilities)
    refined_switch = _switch_with_distribution_means(switch, distributions)
    mistakes = _switch_label_mistakes(refined_switch, examples, example_cache)
    timing_loss = _switch_distribution_timing_loss(
        refined_switch,
        distributions,
        segments_by_trace,
        responsibilities,
    )
    complexity = refined_switch.node_count if isinstance(refined_switch, BooleanTreeSwitch) else 1
    return refined_switch, mistakes, timing_loss, complexity, refined_switch.describe()


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
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            # This timing term is a local approximation to the paper's switch
            # likelihood: prefer enabling the switch at the observed boundary.
            loss -= _eq12_switch_log_likelihood(
                switch,
                current_segment,
                current_resp,
                next_resp,
            )
    return loss


def _eq12_switch_log_likelihood(
    switch: SwitchProgram,
    segment: CartpoleSegment,
    current_resp: Tuple[float, float],
    next_resp: Tuple[float, float],
) -> float:
    transition_weight = current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0]
    stay_weight = current_resp[0] * next_resp[0] + current_resp[1] * next_resp[1]
    # Soft responsibilities split evidence between "switch now" and "stay in
    # the same latent mode" without committing to a hard segment label.
    first_enabled = _first_enabled_step(switch, segment.observations)
    return (
        transition_weight * _log_transition_at_duration(first_enabled, segment.duration)
        + stay_weight * _log_no_transition_before_duration(first_enabled, segment.duration)
    )


def _student_switch_log_likelihood(
    student: ProbabilisticCartpoleStudent,
    segment: CartpoleSegment,
    current_resp: Tuple[float, float],
    next_resp: Tuple[float, float],
) -> float:
    transition_weight = current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0]
    stay_weight = current_resp[0] * next_resp[0] + current_resp[1] * next_resp[1]
    transition_probability, stay_probability = _switch_transition_and_stay_probabilities(
        student.switch,
        student.switch_parameter_distributions,
        segment.observations,
        segment.duration,
    )
    return (
        transition_weight * math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
        + stay_weight * math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
    )


def _switch_transition_and_stay_probabilities(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: int,
) -> Tuple[float, float]:
    scalar = _single_threshold_view(switch, distributions, observations)
    if scalar is not None:
        values, distribution, relation = scalar
        return (
            _single_threshold_transition_probability(values, distribution, relation, duration),
            _single_threshold_no_transition_probability(values, distribution, relation, duration),
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        enabled_by_step = _boolean_tree_enabled_cumulative_probabilities(
            switch,
            distributions,
            observations,
        )
        return _cumulative_transition_and_stay_probability(enabled_by_step, duration)
    enabled_by_step = _switch_enabled_cumulative_probabilities(switch, distributions, observations)
    return _cumulative_transition_and_stay_probability(enabled_by_step, duration)


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
            pair.duration,
        )
    if isinstance(switch, BooleanTreeSwitch) and switch.second is not None and len(distributions) >= 2:
        enabled_by_step = _boolean_tree_pair_enabled_cumulative_probabilities(switch, distributions, pair)
        return _cumulative_transition_and_stay_probability(enabled_by_step, pair.duration)
    enabled_by_step = _switch_enabled_cumulative_probabilities(
        switch,
        distributions,
        pair.observations,
    )
    return _cumulative_transition_and_stay_probability(enabled_by_step, pair.duration)


def _switch_transition_probability_at_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: int,
) -> float:
    transition_probability, _ = _switch_transition_and_stay_probabilities(
        switch,
        distributions,
        observations,
        duration,
    )
    return transition_probability


def _switch_no_transition_probability_before_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: int,
) -> float:
    _, stay_probability = _switch_transition_and_stay_probabilities(
        switch,
        distributions,
        observations,
        duration,
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
    )


def _predicate_pair_enabled_cumulative_probabilities(
    first: ObservationPredicate,
    second: ObservationPredicate,
    first_distribution: GaussianScalar,
    second_distribution: GaussianScalar,
    first_values: Tuple[float, ...],
    second_values: Tuple[float, ...],
) -> List[float]:
    rectangles: List[Tuple[float, float]] = []
    enabled_by_step: List[float] = []
    for first_value, second_value in zip(first_values, second_values):
        rectangles.append(
            (
                _predicate_enabled_probability_from_value(first, first_distribution, first_value),
                _predicate_enabled_probability_from_value(second, second_distribution, second_value),
            )
        )
        enabled_by_step.append(_anchored_rectangle_union_probability(rectangles))
    return enabled_by_step


def _predicate_enabled_probability_from_value(
    predicate: ObservationPredicate,
    distribution: GaussianScalar,
    value: float,
) -> float:
    return _gaussian_threshold_pass_probability(value, distribution, predicate.relation)


def _anchored_rectangle_union_probability(rectangles: List[Tuple[float, float]]) -> float:
    clamped = [
        (min(max(x_bound, 0.0), 1.0), min(max(y_bound, 0.0), 1.0))
        for x_bound, y_bound in rectangles
        if x_bound > 0.0 and y_bound > 0.0
    ]
    if not clamped:
        return 0.0
    x_edges = sorted({0.0, 1.0, *(x_bound for x_bound, _ in clamped)})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        probe = (left + right) / 2.0
        y_bound = max((y for x_bound, y in clamped if probe <= x_bound), default=0.0)
        area += (right - left) * y_bound
    return min(max(area, 0.0), 1.0)


def _single_threshold_transition_and_stay_probability(
    values: Tuple[float, ...],
    distribution: GaussianScalar,
    relation: str,
    duration: int,
) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 1.0
    previous = values[: duration - 1]
    if not previous:
        previous_probability = 0.0 if relation == ">=" else 1.0
    elif relation == ">=":
        previous_probability = _gaussian_cdf(max(previous), distribution)
    elif relation == "<=":
        previous_probability = _gaussian_cdf(min(previous), distribution)
    else:
        raise ValueError(f"unknown relation: {relation}")
    if duration > len(values):
        return 0.0, _single_threshold_stay_probability(previous_probability, relation)

    current_cdf = _gaussian_cdf(values[duration - 1], distribution)
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


def _cumulative_transition_and_stay_probability(
    enabled_by_step: List[float],
    duration: int,
) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 1.0
    previous_probability = (
        enabled_by_step[duration - 2]
        if duration > 1 and duration - 2 < len(enabled_by_step)
        else 0.0
    )
    if duration > len(enabled_by_step):
        return 0.0, max(1.0 - previous_probability, 0.0)
    transition_probability = max(enabled_by_step[duration - 1] - previous_probability, 0.0)
    stay_probability = max(1.0 - previous_probability, 0.0)
    return transition_probability, stay_probability


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

def _log_transition_at_duration(first_enabled: int, duration: int) -> float:
    z = (first_enabled - duration) / SWITCH_TIMING_STD_STEPS
    return -0.5 * z * z


def _log_no_transition_before_duration(first_enabled: int, duration: int) -> float:
    if first_enabled >= duration:
        return 0.0
    z = (duration - first_enabled) / SWITCH_TIMING_STD_STEPS
    return -0.5 * z * z


def _first_enabled_step(switch: SwitchProgram, observations: List[Observation]) -> int:
    for index, observation in enumerate(observations, start=1):
        if switch.decide(observation) == 1:
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
