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
TEACHER_STUDENT_ITERS = 2
TEACHER_STUDENT_REGULARIZER = 1.0
TEACHER_REWARD_LAMBDA = 100.0
TEACHER_TOP_RHO = 10
TEACHER_REFINEMENT_STEPS = 2


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
    segment_durations: Tuple[int, ...] = ()


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


def synthesize_cartpole_policy(cfg: CartpoleSynthesisConfig) -> tuple[SynthesizedCartpolePSM, List[CartpoleTrace]]:
    student, traces = synthesize_cartpole_student(cfg)
    return student.to_deterministic_policy(), traces


def synthesize_cartpole_student(cfg: CartpoleSynthesisConfig) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleTrace]]:
    rng = random.Random(cfg.seed)
    env = CartpoleEnv.train_env(seed=cfg.seed)
    initial_states = [env.reset() for _ in range(cfg.num_initial_states)]
    student: ProbabilisticCartpoleStudent | None = None
    traces: List[CartpoleTrace] = []
    # Alternate between a teacher that searches for high-reward traces and a
    # student fit that makes later teacher traces easier to explain with the PSM.
    for _ in range(max(1, cfg.teacher_student_iters)):
        traces = [
            _optimize_loop_free_trace(initial_state, env.cfg, cfg, rng, student)
            for initial_state in initial_states
        ]
        student = fit_probabilistic_cartpole_student(traces, cfg)
    if student is None:
        raise RuntimeError("Cartpole synthesis did not produce a student policy")
    return student, traces


def fit_probabilistic_cartpole_student(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> ProbabilisticCartpoleStudent:
    """Fit the Cartpole student using Gaussian action-parameter distributions.

    This implements the action-distribution part of the paper's EM-style
    student step for Cartpole's constant-action grammar.  Switch timing uses a
    bounded threshold-mean refinement against the local Eq. (12)-style timing
    likelihood; it is still not the paper's full continuous M-step.
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
    candidates = [
        _rollout_loop_free_candidate(initial_state, env_cfg, cfg, rng)
        for _ in range(max(1, cfg.candidate_rollouts))
    ]
    ranked = sorted(
        candidates,
        key=lambda trace: _teacher_objective(trace, student, cfg),
        reverse=True,
    )
    # Refine only the top candidates to keep synthesis cheap while still
    # improving the gain parameters that generated promising traces.
    refined = [
        _refine_loop_free_trace(candidate, initial_state, env_cfg, cfg, student)
        for candidate in ranked[: max(1, cfg.teacher_top_rho)]
    ]
    return max(
        ranked + refined,
        key=lambda trace: _teacher_objective(trace, student, cfg),
    )


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
    theta_gain = rng.gauss(cfg.teacher_theta_gain, max(1e-6, abs(cfg.teacher_theta_gain) * 0.10))
    omega_gain = rng.gauss(cfg.teacher_omega_gain, max(1e-6, abs(cfg.teacher_omega_gain) * 0.10))
    return _rollout_with_teacher_gains(initial_state, env_cfg, cfg, theta_gain, omega_gain)


def _rollout_with_teacher_gains(
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    theta_gain: float,
    omega_gain: float,
    segment_durations: Tuple[int, ...] | None = None,
) -> CartpoleTrace:
    observations: List[Observation] = []
    actions: List[float] = []
    mode_labels: List[int] = []
    state = list(initial_state)
    alive = 0
    durations = segment_durations or tuple(cfg.segment_steps for _ in range(cfg.segments_per_trace))
    for duration in durations:
        if cartpole_done(state, env_cfg):
            break
        _, _, theta, omega = state
        # Random gains produce a loop-free switching trace; the final policy is
        # learned from the trace rather than using these gains directly.
        raw_force = theta_gain * theta + omega_gain * omega
        action = max(cfg.force_values) if raw_force >= 0.0 else min(cfg.force_values)
        label = 1 if action > 0.0 else 0
        for _ in range(max(1, duration)):
            if cartpole_done(state, env_cfg):
                break
            observations.append(list(state))
            actions.append(action)
            mode_labels.append(label)
            state = cartpole_next_state(state, action, env_cfg)
            alive += 1
    return CartpoleTrace(observations, actions, mode_labels, float(alive), theta_gain, omega_gain, tuple(durations))


def _refine_loop_free_trace(
    trace: CartpoleTrace,
    initial_state: Sequence[float],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    student: ProbabilisticCartpoleStudent | None,
) -> CartpoleTrace:
    best = trace
    theta_delta = max(abs(trace.theta_gain) * 0.05, 0.1)
    omega_delta = max(abs(trace.omega_gain) * 0.05, 0.05)
    for _ in range(max(0, cfg.teacher_refinement_steps)):
        improved = False
        # Coordinate-search the teacher gains because each rollout is cheap and
        # the objective includes a non-smooth student-likelihood term.
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
        if not improved:
            theta_delta *= 0.5
            omega_delta *= 0.5
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
        for delta in (-1, 1):
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
    return cfg.teacher_reward_lambda * trace.reward + cfg.teacher_student_regularizer * _trace_log_probability(trace, student)


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
            prior = max(resp[mode], 1e-12)
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
        _, refined = _refine_switch_distribution_means(
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
    _, refined = _refine_switch_distribution_means(
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
    best_loss = _switch_timing_loss(best_switch, segments_by_trace, responsibilities)

    for param_index, distribution in enumerate(distributions):
        candidates = [
            distribution.mean,
            *_switch_distribution_mean_candidates(switch, param_index, segments_by_trace),
        ]
        for candidate_mean in candidates:
            candidate_distributions = list(best_distributions)
            candidate_distributions[param_index] = GaussianScalar(candidate_mean, distribution.std)
            candidate_switch = _switch_with_distribution_means(switch, candidate_distributions)
            candidate_mistakes = _switch_label_mistakes(candidate_switch, examples)
            if candidate_mistakes > best_mistakes:
                continue
            candidate_loss = _switch_timing_loss(candidate_switch, segments_by_trace, responsibilities)
            if candidate_loss < best_loss:
                best_distributions = candidate_distributions
                best_switch = candidate_switch
                best_mistakes = candidate_mistakes
                best_loss = candidate_loss
    return best_switch, best_distributions


def _distributions_with_switch_means(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
) -> List[GaussianScalar]:
    if isinstance(switch, BooleanTreeSwitch):
        predicates = _switch_predicates(switch)
        return [
            GaussianScalar(predicate.threshold, distribution.std)
            for predicate, distribution in zip(predicates, distributions)
        ]
    if not distributions:
        return []
    return [GaussianScalar(_switch_default_threshold(switch), distributions[0].std)]


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
    for theta_weight in (-50.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        for omega_weight in (-10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
            scores = [theta_weight * obs[2] + omega_weight * obs[3] for obs, _ in examples]
            thresholds = _candidate_thresholds(scores)
            for threshold in thresholds:
                candidate_switches.append(Depth2Switch(theta_weight, omega_weight, threshold))
    candidate_switches.extend(_greedy_boolean_tree_candidates(examples, segments_by_trace, responsibilities))
    for switch in candidate_switches:
        candidates.append((*_switch_cost(switch, examples, segments_by_trace, responsibilities), switch))
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
        switches,
        key=lambda switch: _switch_cost(switch, examples, segments_by_trace, responsibilities),
    )


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
        transition_weight * math.log(max(transition_probability, 1e-12))
        + stay_weight * math.log(max(stay_probability, 1e-12))
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
    if len(unique) > 64:
        step = max(1, len(unique) // 64)
        unique = unique[::step]
    candidates = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    candidates.append(0.0)
    return candidates
