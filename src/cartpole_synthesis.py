from __future__ import annotations

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
TEACHER_STUDENT_SAMPLE_FRACTION = 0.5
SWITCH_OBLIQUE_THETA_WEIGHTS = (-50.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SWITCH_OBLIQUE_OMEGA_WEIGHTS = (-10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
MAX_SWITCH_THRESHOLD_CANDIDATES = 64
DEFAULT_SWITCH_THRESHOLD_CANDIDATE = 0.0
SWITCH_STD_REFINEMENT_MULTIPLIERS = (0.5, 1.0, 2.0)
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
SWITCH_STRUCTURE_RESCORING_TOP_K = 128


@dataclass
class CartpoleSynthesisConfig:
    num_initial_states: int = 32
    candidate_rollouts: int = 128
    segment_steps: int = 8
    segments_per_trace: int = 32
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
            "min_gaussian_std": MIN_GAUSSIAN_STD,
            "log_probability_floor": LOG_PROBABILITY_FLOOR,
        },
        "switch_timing": {
            "std_steps": SWITCH_TIMING_STD_STEPS,
            "scalar_threshold_uses_shared_sample": True,
            "depth2_conjunction_probability": "independence_approximation",
            "std_refinement_multipliers": list(SWITCH_STD_REFINEMENT_MULTIPLIERS),
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
            "student_sample_local_refinement": "duration_and_action_coordinate_search",
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
    student step for Cartpole's constant-action grammar. Switch timing uses a
    bounded Gaussian mean/std refinement against the local Eq. (12)-style
    timing likelihood; it is still not the paper's full continuous M-step.
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
        action_distributions = {
            mode: _weighted_gaussian(
                [segment.action_parameter for segment in segments],
                [resp[mode] for resp in responsibilities],
                left_default if mode == 0 else right_default,
            )
            for mode in (0, 1)
        }

    # Fit the discrete switch after action EM so switch costs can use the same
    # soft transition evidence instead of only the teacher's hard labels.
    switch = _learn_depth2_switch(traces, segments_by_trace, responsibilities)
    switch_parameter_distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        responsibilities,
    )
    switch = _switch_with_distribution_means(switch, switch_parameter_distributions)
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


def _optimize_loop_free_trace(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None = None,
) -> CartpoleTrace:
    # The "teacher" is restricted to loop-free bang-bang traces; ranking by the
    # student likelihood is the local adaptive-teaching approximation.
    candidates = _teacher_candidate_traces(initial_state, env_cfg, cfg, rng, student)
    ranked = sorted(
        candidates,
        key=lambda trace: _teacher_objective(trace, student, cfg),
        reverse=True,
    )
    # Refine only the top candidates to keep synthesis cheap while still
    # optimizing around promising sampled loop-free traces.
    refined = [
        _refine_loop_free_trace(candidate, initial_state, env_cfg, cfg, student)
        for candidate in ranked[: max(1, cfg.teacher_top_rho)]
        if candidate.segment_actions and candidate.segment_durations
    ]
    return max(
        ranked + refined,
        key=lambda trace: _teacher_objective(trace, student, cfg),
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
        return [
            _rollout_loop_free_candidate(initial_state, env_cfg, cfg, rng)
            for _ in range(candidate_count)
        ]

    # Paper Section 4.2 samples teacher candidates from the current student
    # before optimizing them. This bounded local version keeps random gain
    # samples too, so a bad early student cannot fully determine exploration.
    student_count = max(1, int(candidate_count * TEACHER_STUDENT_SAMPLE_FRACTION))
    gain_count = max(0, candidate_count - student_count)
    candidates = [
        _rollout_student_sampled_trace(initial_state, env_cfg, cfg, student, rng)
        for _ in range(student_count)
    ]
    candidates.extend(
        _rollout_loop_free_candidate(initial_state, env_cfg, cfg, rng)
        for _ in range(gain_count)
    )
    return candidates


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
    durations = segment_durations or tuple(cfg.segment_steps for _ in range(cfg.segments_per_trace))
    chosen_actions: List[float] = []
    started_durations: List[int] = []
    for segment_index, duration in enumerate(durations):
        if cartpole_done(state, env_cfg):
            break
        duration_steps = max(1, duration)
        if segment_actions is not None and segment_index < len(segment_actions):
            action = segment_actions[segment_index]
        else:
            _, _, theta, omega = state
            # Random gains choose the next loop-free action function; the final
            # policy is learned from the trace rather than using these gains.
            raw_force = theta_gain * theta + omega_gain * omega
            action = max(cfg.force_values) if raw_force >= 0.0 else min(cfg.force_values)
        chosen_actions.append(action)
        started_durations.append(duration_steps)
        label = 1 if action > 0.0 else 0
        for _ in range(duration_steps):
            if cartpole_done(state, env_cfg):
                break
            observations.append(list(state))
            actions.append(action)
            mode_labels.append(label)
            state = cartpole_next_state(state, action, env_cfg)
            alive += 1
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
    # Sample one deterministic PSM from the student's Gaussian parameters and
    # record the induced trace probability for Eq. (8)-style ranking.
    policy = student.sample_policy(rng)
    policy.reset()
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    for _ in range(cfg.segment_steps * cfg.segments_per_trace):
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
    trace.student_log_probability = _trace_log_probability(trace, student)
    return trace


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
) -> CartpoleTrace:
    best = trace
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
                if _teacher_objective(candidate, student, cfg) > _teacher_objective(best, student, cfg):
                    best = candidate
                    improved = True
        for candidate in _duration_refinement_candidates(best, initial_state, env_cfg, cfg):
            if _teacher_objective(candidate, student, cfg) > _teacher_objective(best, student, cfg):
                best = candidate
                improved = True
        for candidate in _action_refinement_candidates(best, initial_state, env_cfg, cfg):
            if _teacher_objective(candidate, student, cfg) > _teacher_objective(best, student, cfg):
                best = candidate
                improved = True
        if not improved:
            theta_delta *= TEACHER_REFINEMENT_DELTA_DECAY
            omega_delta *= TEACHER_REFINEMENT_DELTA_DECAY
    if best is not trace:
        best.teacher_source = (
            "student_sample_refined"
            if trace.teacher_source.startswith("student_sample")
            else "gain_refined"
        )
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


def _trace_log_probability(trace: CartpoleTrace, student: ProbabilisticCartpoleStudent) -> float:
    total = 0.0
    trace_segments = _segments_from_traces([trace])[0]
    responsibilities: List[Tuple[float, float]] = []
    for segment in trace_segments:
        # Recompute responsibilities under the current student so the teacher
        # objective stays aligned with the latest fitted action primitives.
        resp = _mode_responsibilities(segment.action_parameter, student.action_distributions)
        responsibilities.append(resp)
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
    best_distributions = _distributions_with_switch_means(switch, distributions)
    best_switch = _switch_with_distribution_means(switch, best_distributions)
    best_mistakes = _switch_label_mistakes(best_switch, examples)
    best_loss = _switch_distribution_timing_loss(best_switch, best_distributions, segments_by_trace, responsibilities)

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
                candidate_mistakes = _switch_label_mistakes(candidate_switch, examples)
                if candidate_mistakes > best_mistakes:
                    continue
                candidate_loss = _switch_distribution_timing_loss(
                    candidate_switch,
                    candidate_distributions,
                    segments_by_trace,
                    responsibilities,
                )
                if candidate_loss < best_loss:
                    best_distributions = candidate_distributions
                    best_switch = candidate_switch
                    best_mistakes = candidate_mistakes
                    best_loss = candidate_loss
    return best_switch, best_distributions


def _switch_distribution_timing_loss(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> float:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match switch timing segments")
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    loss = 0.0
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
            next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
            transition_weight = current_resp[0] * next_resp[1] + current_resp[1] * next_resp[0]
            stay_weight = current_resp[0] * next_resp[0] + current_resp[1] * next_resp[1]
            transition_probability = _switch_transition_probability_at_duration(
                switch,
                distributions,
                current_segment.observations,
                current_segment.duration,
            )
            stay_probability = _switch_no_transition_probability_before_duration(
                switch,
                distributions,
                current_segment.observations,
                current_segment.duration,
            )
            loss -= (
                transition_weight * math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
                + stay_weight * math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
            )
    return loss


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
) -> int:
    return sum(
        int(switch.decide(observation) != label)
        for observation, label in examples
    )


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
        predicates = _switch_predicates(switch)
        if len(distributions) < len(predicates):
            return 1.0 if switch.decide(observation) == 1 else 0.0
        probability = 1.0
        for predicate, distribution in zip(predicates, distributions):
            probability *= _predicate_enabled_probability(predicate, distribution, observation)
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

    candidates = []
    candidate_switches: List[SwitchProgram] = []
    # Search a compact oblique threshold family over CartPole angle and angular
    # velocity before considering predicate-tree alternatives.
    for theta_weight in SWITCH_OBLIQUE_THETA_WEIGHTS:
        for omega_weight in SWITCH_OBLIQUE_OMEGA_WEIGHTS:
            scores = [theta_weight * obs[2] + omega_weight * obs[3] for obs, _ in examples]
            thresholds = _candidate_thresholds(scores)
            for threshold in thresholds:
                candidate_switches.append(Depth2Switch(theta_weight, omega_weight, threshold))
    candidate_switches.extend(_greedy_boolean_tree_candidates(examples, segments_by_trace, responsibilities))
    for switch in _switch_structure_rescore_candidates(
        candidate_switches,
        examples,
        segments_by_trace,
        responsibilities,
    ):
        candidates.append((*_switch_structure_cost(switch, examples, segments_by_trace, responsibilities), switch))
    return min(candidates, key=lambda item: item[:-1])[-1]


def _greedy_boolean_tree_candidates(
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
) -> List[BooleanTreeSwitch]:
    stumps = [BooleanTreeSwitch(predicate) for predicate in _predicate_candidates(examples)]
    if not stumps:
        return []
    best = _best_switch(stumps, examples, segments_by_trace, responsibilities)
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
    return [best, _best_switch(expansions, examples, segments_by_trace, responsibilities)]


def _best_switch(
    switches: List[BooleanTreeSwitch],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
) -> BooleanTreeSwitch:
    return min(
        _switch_structure_rescore_candidates(switches, examples, segments_by_trace, responsibilities),
        key=lambda switch: _switch_structure_cost(switch, examples, segments_by_trace, responsibilities),
    )


def _switch_structure_rescore_candidates(
    switches: List[SwitchProgram],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None,
    responsibilities: List[Tuple[float, float]] | None,
) -> List[SwitchProgram]:
    if segments_by_trace is None or responsibilities is None or len(switches) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        return switches

    ranked = sorted(
        switches,
        key=lambda switch: _switch_cost(switch, examples, segments_by_trace, responsibilities),
    )
    return ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]


def _switch_structure_cost(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
) -> Tuple[int, float, int, str]:
    if segments_by_trace is None or responsibilities is None:
        return _switch_cost(switch, examples, segments_by_trace, responsibilities)

    # Score a candidate structure after bounded Gaussian threshold refinement,
    # matching the objective reported in metrics provenance.
    _, mistakes, timing_loss, complexity, description = _fit_switch_structure_objective(
        switch,
        examples,
        segments_by_trace,
        responsibilities,
    )
    return mistakes, timing_loss, complexity, description


def _fit_switch_structure_objective(
    switch: SwitchProgram,
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> Tuple[SwitchProgram, int, float, int, str]:
    distributions = _fit_switch_parameter_distributions(switch, segments_by_trace, responsibilities)
    refined_switch = _switch_with_distribution_means(switch, distributions)
    mistakes = _switch_label_mistakes(refined_switch, examples)
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
) -> Tuple[int, float, int, str]:
    mistakes = sum(
        int(switch.decide(observation) != label)
        for observation, label in examples
    )
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
    transition_probability = _switch_transition_probability_at_duration(
        student.switch,
        student.switch_parameter_distributions,
        segment.observations,
        segment.duration,
    )
    stay_probability = _switch_no_transition_probability_before_duration(
        student.switch,
        student.switch_parameter_distributions,
        segment.observations,
        segment.duration,
    )
    return (
        transition_weight * math.log(max(transition_probability, LOG_PROBABILITY_FLOOR))
        + stay_weight * math.log(max(stay_probability, LOG_PROBABILITY_FLOOR))
    )


def _switch_transition_probability_at_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: int,
) -> float:
    scalar = _single_threshold_view(switch, distributions, observations)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_transition_probability(values, distribution, relation, duration)
    enabled_by_step = _switch_enabled_cumulative_probabilities(switch, distributions, observations)
    if duration <= 0 or duration > len(enabled_by_step):
        return 0.0
    previous_probability = enabled_by_step[duration - 2] if duration > 1 else 0.0
    return max(enabled_by_step[duration - 1] - previous_probability, 0.0)


def _switch_no_transition_probability_before_duration(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    observations: List[Observation],
    duration: int,
) -> float:
    scalar = _single_threshold_view(switch, distributions, observations)
    if scalar is not None:
        values, distribution, relation = scalar
        return _single_threshold_no_transition_probability(values, distribution, relation, duration)
    enabled_by_step = _switch_enabled_cumulative_probabilities(switch, distributions, observations)
    previous_probability = enabled_by_step[duration - 2] if duration > 1 and duration - 2 < len(enabled_by_step) else 0.0
    return max(1.0 - previous_probability, 0.0)


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
