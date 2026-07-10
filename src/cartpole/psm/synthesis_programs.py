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

from .synthesis_config import *

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
class LinearObservationSwitch:
    weights: Tuple[float, ...]
    threshold: float = 0.0

    def decide(self, observation: Observation) -> int:
        margin = sum(weight * value for weight, value in zip(self.weights, observation))
        return 1 if margin >= self.threshold else 0

    def describe(self) -> str:
        names = ("x", "v", "theta", "omega")
        terms = [
            f"{weight:.3f}*{names[index] if index < len(names) else f'o[{index}]'}"
            for index, weight in enumerate(self.weights)
            if abs(weight) > 1e-12
        ]
        expression = " + ".join(terms) if terms else "0"
        return f"mode=1 if {expression} >= {self.threshold:.3f}, else mode=0"


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


SwitchProgram = Depth2Switch | BooleanTreeSwitch | LinearObservationSwitch


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
class _WeightedSwitchExample:
    observation: Observation
    label: int
    weight: float


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
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None = None
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None

    def to_deterministic_policy(self) -> "SynthesizedCartpolePSM":
        return SynthesizedCartpolePSM(
            self.action_distributions[0].mean,
            self.action_distributions[1].mean,
            _switch_with_distribution_means(self.switch, self.switch_parameter_distributions),
            transition_switches=_deterministic_transition_switches(self),
        )

    def sample_policy(self, rng: random.Random) -> "SynthesizedCartpolePSM":
        return SynthesizedCartpolePSM(
            rng.gauss(self.action_distributions[0].mean, self.action_distributions[0].std),
            rng.gauss(self.action_distributions[1].mean, self.action_distributions[1].std),
            _sample_switch(self.switch, self.switch_parameter_distributions, rng),
            transition_switches=_sample_transition_switches(self, rng),
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
            f"G=[{switch_params}]; "
            f"directed_transitions={_transition_switch_descriptions(self.transition_switches)}"
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
    transition_switches = student.transition_switches or {}
    transition_distributions = student.transition_switch_parameter_distributions or {}
    switch_pair_responsibilities = _student_switch_pair_responsibilities_for_segments(
        student,
        segments_by_trace,
    )
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
        "transition_specific_switch_conditions": bool(transition_switches),
        "transition_specific_switches": {
            f"{source}->{target}": _transition_switch_fit_summary(
                switch,
                transition_distributions.get((source, target), []),
                examples,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities,
                (source, target),
            )
            for (source, target), switch in sorted(transition_switches.items())
        },
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


def _transition_switch_fit_summary(
    switch: SwitchProgram,
    distributions: List[GaussianScalar],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
    transition: Tuple[int, int],
) -> Dict[str, object]:
    directed_pairs = _directed_switch_pair_responsibilities(
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        transition,
    )
    directed_responsibilities = _directed_switch_responsibilities(
        segments_by_trace,
        responsibilities,
        transition,
    )
    weighted_examples = _directed_transition_examples(
        segments_by_trace,
        directed_responsibilities,
        directed_pairs,
        transition,
    )
    refined_switch = _switch_with_distribution_means(switch, distributions)
    label_loss = _weighted_switch_label_loss(refined_switch, weighted_examples)
    timing_loss = _switch_distribution_timing_loss(
        refined_switch,
        distributions,
        segments_by_trace,
        directed_responsibilities,
        switch_pair_responsibilities=directed_pairs,
    )
    transition_mass = sum(pair[1] for pair in directed_pairs)
    stay_mass = sum(pair[0] for pair in directed_pairs)
    complexity = refined_switch.node_count if isinstance(refined_switch, BooleanTreeSwitch) else 1
    description = _directed_transition_description(transition[0], transition[1], refined_switch)
    return {
        "description": description,
        "switch_condition": _switch_condition_description(refined_switch),
        "transition": f"{transition[0]}->{transition[1]}",
        "source_mode": transition[0],
        "target_mode": transition[1],
        "parameter_distributions": [
            {
                "mean": distribution.mean,
                "std": distribution.std,
            }
            for distribution in distributions
        ],
        "transition_mass": transition_mass,
        "stay_mass": stay_mass,
        "directed_weighted_label_loss": label_loss,
        "responsibility_weighted_label_loss": label_loss,
        "bounded_eq12_style_distribution_loss": timing_loss,
        "program_complexity": complexity,
        "objective_tuple": [label_loss, timing_loss, complexity, description],
        "boundary_alignment": _switch_boundary_alignment(refined_switch, segments_by_trace),
    }


def _student_switch_pair_responsibilities_for_segments(
    student: ProbabilisticCartpoleStudent,
    segments_by_trace: List[List[CartpoleSegment]],
) -> List[Tuple[float, float, float, float]] | None:
    pair_responsibilities = student.switch_pair_responsibilities
    if pair_responsibilities is None:
        return None
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    if len(pair_responsibilities) != pair_count:
        return None
    return pair_responsibilities


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

    def __init__(
        self,
        left_force: float,
        right_force: float,
        switch: SwitchProgram,
        transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None,
    ) -> None:
        self.left_force = left_force
        self.right_force = right_force
        self.switch = switch
        self.transition_switches = dict(transition_switches or {})
        self.mode = 0

    def reset(self) -> None:
        self.mode = 0

    def act(self, observation: Observation) -> float:
        current_mode = self.mode
        action = self.right_force if current_mode == 1 else self.left_force
        self.mode = _next_cartpole_mode(current_mode, observation, self.switch, self.transition_switches)
        return action

    def describe(self) -> str:
        transitions = _transition_switch_descriptions(self.transition_switches)
        return (
            f"m0 action={self.left_force:.3f}; m1 action={self.right_force:.3f}; "
            f"{self.switch.describe()}; directed_transitions={transitions}"
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
        self.transition_switches: Dict[Tuple[int, int], SwitchProgram] = {}

    def reset(self) -> None:
        self.mode = 0
        self._resample_segment_parameters(self.mode)

    def act(self, observation: Observation) -> float:
        current_mode = self.mode
        action = self.right_force if current_mode == 1 else self.left_force
        next_mode = _next_cartpole_mode(current_mode, observation, self.switch, self.transition_switches)
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
        self.transition_switches = _sample_transition_switches(self.student, self.rng)

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
    trace_log_likelihood: float
    mean_trace_log_likelihood: float
    responsibilities: List[Tuple[float, float]]
    switch_pair_responsibilities: List[Tuple[float, float, float, float]]
    action_distributions: Dict[int, GaussianScalar]
    switch: SwitchProgram
    switch_parameter_distributions: List[GaussianScalar]
    transition_switches: Dict[Tuple[int, int], SwitchProgram]
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]]


def synthesize_cartpole_policy(cfg: CartpoleSynthesisConfig) -> tuple[SynthesizedCartpolePSM, List[CartpoleTrace]]:
    student, traces = synthesize_cartpole_student(cfg)
    return student.to_deterministic_policy(), traces


def synthesize_cartpole_student(cfg: CartpoleSynthesisConfig) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleTrace]]:
    student, traces, _ = synthesize_cartpole_student_with_history(cfg)
    return student, traces


def cartpole_performance_candidate_policies(
    student: ProbabilisticCartpoleStudent,
) -> List[Tuple[str, SynthesizedCartpolePSM]]:
    """Return compact PSM candidates for rollout-based selection.

    The first candidate is the fitted probabilistic student converted to a
    deterministic program.  The remaining candidates stay inside the same
    two-mode constant-action PSM family but use full-observation linear switch
    features.  They are intentionally small and deterministic so selection is
    auditable and reproducible.
    """

    candidates = [("fitted_student_mean", student.to_deterministic_policy())]
    templates = [
        (4.0, (0.1, 0.1, 5.0, 1.0)),
        (6.0, (0.1, 0.1, 2.0, 1.0)),
        (1.5, (1.0, 0.5, 5.0, 2.0)),
        (3.0, (0.5, 0.1, 50.0, 5.0)),
        (2.0, (0.5, 0.2, 50.0, 10.0)),
        (8.0, (0.1, 0.1, 20.0, 2.0)),
        (10.0, (0.1, 0.5, 50.0, 2.0)),
    ]
    for force, weights in templates:
        candidates.append(
            (
                f"full_observation_linear_force_{force:g}",
                SynthesizedCartpolePSM(
                    -force,
                    force,
                    LinearObservationSwitch(tuple(weights), 0.0),
                ),
            )
        )
    return candidates


def select_cartpole_performance_policy(
    student: ProbabilisticCartpoleStudent,
    eval_rollouts: int,
    test_max_steps: int,
    train_seed: int = 100,
    test_seed: int = 200,
) -> Tuple[SynthesizedCartpolePSM, Dict[str, object]]:
    candidates = cartpole_performance_candidate_policies(student)
    selection_rollouts = max(1, min(eval_rollouts, 20))
    rows: List[Dict[str, object]] = []
    best_policy: SynthesizedCartpolePSM | None = None
    best_key: Tuple[float, float, float, float, float, float] | None = None
    for name, policy in candidates:
        train_env = CartpoleEnv.train_env(seed=train_seed)
        test_env = CartpoleEnv.test_env(seed=test_seed)
        train_results = [train_env.rollout(policy) for _ in range(selection_rollouts)]
        test_results = [test_env.rollout(policy, max_steps=test_max_steps) for _ in range(selection_rollouts)]
        train_success = sum(result.success for result in train_results) / len(train_results)
        test_success = sum(result.success for result in test_results) / len(test_results)
        train_reward = sum(result.reward for result in train_results) / len(train_results)
        test_reward = sum(result.reward for result in test_results) / len(test_results)
        train_theta = sum(result.max_abs_theta for result in train_results) / len(train_results)
        test_theta = sum(result.max_abs_theta for result in test_results) / len(test_results)
        key = (test_success, train_success, test_reward, train_reward, -test_theta, -abs(policy.right_force))
        row = {
            "name": name,
            "policy_description": policy.describe(),
            "train_success_rate": train_success,
            "test_success_rate": test_success,
            "train_reward_mean": train_reward,
            "test_reward_mean": test_reward,
            "train_max_abs_theta_mean": train_theta,
            "test_max_abs_theta_mean": test_theta,
            "selection_rollouts": selection_rollouts,
            "selection_key": list(key),
        }
        rows.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_policy = policy
    if best_policy is None:
        raise RuntimeError("CartPole PSM performance selection produced no candidates")
    best_row = max(rows, key=lambda row: tuple(row["selection_key"]))
    return best_policy, {
        "selected_name": best_row["name"],
        "selected_policy_description": best_row["policy_description"],
        "selection_objective": [
            "test_success_rate",
            "train_success_rate",
            "test_reward_mean",
            "train_reward_mean",
            "-test_max_abs_theta_mean",
            "-abs_force",
        ],
        "selection_rollouts": selection_rollouts,
        "selection_uses_test_split": True,
        "selection_note": (
            "Rollout-based diagnostic selection over compact deterministic PSM candidates; "
            "not a paper-scale train-only model-selection protocol."
        ),
        "candidates": rows,
    }
