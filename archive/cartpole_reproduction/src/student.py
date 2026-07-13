from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from psm import Mode, ProgrammaticStateMachine, ThresholdSwitch, constant_action


Observation = List[float]
Action = List[float]


@dataclass
class StudentConfig:
    num_modes: int
    action_grammar: Any = "constant"
    switch_grammar: Any = "axis_threshold"
    max_em_iters: int = 20
    end_action_tolerance: float = 0.25


@dataclass
class _Segment:
    observations: List[Observation]
    actions: List[Action]
    trace_index: int
    segment_index: int

    @property
    def mean_action(self) -> Action:
        width = len(self.actions[0])
        return [sum(action[i] for action in self.actions) / len(self.actions) for i in range(width)]


def _squared_distance(left: Action, right: Action) -> float:
    return sum((x - y) ** 2 for x, y in zip(left, right))


def _action_norm(action: Action) -> float:
    return sum(abs(value) for value in action)


def _validate_supported_grammar(grammar: Any, student_cfg: StudentConfig) -> None:
    if student_cfg.action_grammar != "constant":
        raise ValueError("only the constant action grammar is implemented")
    if student_cfg.switch_grammar != "axis_threshold":
        raise ValueError("only the axis_threshold switch grammar is implemented")
    if grammar is None or grammar == "constant_axis_threshold":
        return
    if isinstance(grammar, dict):
        action_grammar = grammar.get("action_grammar", student_cfg.action_grammar)
        switch_grammar = grammar.get("switch_grammar", student_cfg.switch_grammar)
        if action_grammar == "constant" and switch_grammar == "axis_threshold":
            return
    raise ValueError("unsupported grammar; use constant actions and axis_threshold switches")


def _extract_segments(traces: List[Any]) -> List[List[_Segment]]:
    by_trace: List[List[_Segment]] = []
    for trace_index, trace in enumerate(traces):
        if not trace.actions:
            continue
        trace_segments: List[_Segment] = []
        start = 0
        current_hint = trace.mode_hints[0]
        for index, hint in enumerate(trace.mode_hints[1:], start=1):
            if hint == current_hint:
                continue
            trace_segments.append(
                _Segment(
                    observations=trace.observations[start:index],
                    actions=trace.actions[start:index],
                    trace_index=trace_index,
                    segment_index=len(trace_segments),
                )
            )
            start = index
            current_hint = hint
        trace_segments.append(
            _Segment(
                observations=trace.observations[start:],
                actions=trace.actions[start:],
                trace_index=trace_index,
                segment_index=len(trace_segments),
            )
        )
        by_trace.append(trace_segments)
    return by_trace


def _initial_centroids(points: List[Action], num_modes: int) -> List[Action]:
    ordered = sorted(points, key=lambda point: tuple(point))
    if num_modes == 1:
        return [list(ordered[len(ordered) // 2])]
    centroids: List[Action] = []
    for mode_index in range(num_modes):
        point_index = round(mode_index * (len(ordered) - 1) / (num_modes - 1))
        centroids.append(list(ordered[point_index]))
    return centroids


def _cluster_actions(
    points: List[Action],
    num_modes: int,
    max_iters: int,
) -> Tuple[List[int], List[Action]]:
    if not points:
        raise ValueError("cannot fit a student without teacher segments")
    if num_modes <= 0:
        raise ValueError("student_cfg.num_modes must be positive")
    if num_modes > len(points):
        raise ValueError("num_modes cannot exceed the number of observed segments")

    centroids = _initial_centroids(points, num_modes)
    assignments = [0 for _ in points]
    for _ in range(max(1, max_iters)):
        new_assignments = [
            min(range(num_modes), key=lambda idx: _squared_distance(point, centroids[idx]))
            for point in points
        ]
        if new_assignments == assignments:
            break
        assignments = new_assignments

        for mode_index in range(num_modes):
            cluster = [point for point, label in zip(points, assignments) if label == mode_index]
            if not cluster:
                continue
            width = len(cluster[0])
            centroids[mode_index] = [
                sum(point[i] for point in cluster) / len(cluster) for i in range(width)
            ]
    return assignments, centroids


def _candidate_thresholds(values: List[float]) -> List[float]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique
    return [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]


def _fit_threshold_switch(
    positives: List[Observation],
    negatives: List[Observation],
) -> Optional[ThresholdSwitch]:
    if not positives:
        return None
    feature_count = len(positives[0])
    best: Optional[Tuple[int, int, str, float]] = None

    for feature_index in range(feature_count):
        values = [obs[feature_index] for obs in positives + negatives]
        for threshold in _candidate_thresholds(values):
            for relation in (">=", "<="):
                mistakes = 0
                for obs in positives:
                    passed = (
                        obs[feature_index] >= threshold
                        if relation == ">="
                        else obs[feature_index] <= threshold
                    )
                    mistakes += 0 if passed else 1
                for obs in negatives:
                    passed = (
                        obs[feature_index] >= threshold
                        if relation == ">="
                        else obs[feature_index] <= threshold
                    )
                    mistakes += 1 if passed else 0
                signed_distances = []
                for obs in positives:
                    raw = (
                        obs[feature_index] - threshold
                        if relation == ">="
                        else threshold - obs[feature_index]
                    )
                    signed_distances.append(raw)
                margin = min(signed_distances) if signed_distances else 0.0
                score = (mistakes, -int(margin >= 0.0), relation, threshold)
                if best is None or score < best:
                    best = score
                    best_switch = ThresholdSwitch(feature_index, relation, threshold)

    return best_switch if best is not None else None


def _learn_switches(
    segments_by_trace: List[List[_Segment]],
    segment_modes: Dict[Tuple[int, int], int],
    num_modes: int,
) -> Dict[int, Dict[int, ThresholdSwitch]]:
    source_examples: Dict[int, List[Tuple[Observation, Optional[int]]]] = {}
    for trace_segments in segments_by_trace:
        for left, right in zip(trace_segments, trace_segments[1:]):
            source = segment_modes[(left.trace_index, left.segment_index)]
            target = segment_modes[(right.trace_index, right.segment_index)]
            if source == target:
                continue

            examples = source_examples.setdefault(source, [])
            examples.extend((observation, None) for observation in left.observations[:-1])
            examples.append((left.observations[-1], target))

    switches: Dict[int, Dict[int, ThresholdSwitch]] = {idx: {} for idx in range(num_modes)}
    for source, examples in source_examples.items():
        targets = sorted({target for _, target in examples if target is not None})
        for target in targets:
            positives = [observation for observation, label in examples if label == target]
            negatives = [observation for observation, label in examples if label != target]
            switch = _fit_threshold_switch(positives, negatives)
            if switch is not None:
                switches[source][target] = switch
    return switches


def fit_student_from_traces(
    traces: Iterable[Any],
    grammar: Any,
    student_cfg: StudentConfig,
) -> ProgrammaticStateMachine:
    """Fit a deterministic state machine from teacher traces.

    This is a local diagnostic learner, not the paper's full probabilistic
    M-step.  The audit tracks this gap explicitly until Gaussian program
    distributions and the full maximum-likelihood objective are implemented.
    """

    _validate_supported_grammar(grammar, student_cfg)
    traces_list = list(traces)
    segments_by_trace = _extract_segments(traces_list)
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    action_points = [segment.mean_action for segment in flat_segments]
    assignments, centroids = _cluster_actions(
        action_points,
        student_cfg.num_modes,
        student_cfg.max_em_iters,
    )

    segment_modes = {
        (segment.trace_index, segment.segment_index): mode
        for segment, mode in zip(flat_segments, assignments)
    }
    switches_by_mode = _learn_switches(segments_by_trace, segment_modes, student_cfg.num_modes)

    start_counts: Dict[int, int] = {}
    for trace_segments in segments_by_trace:
        if not trace_segments:
            continue
        start_mode = segment_modes[(trace_segments[0].trace_index, trace_segments[0].segment_index)]
        start_counts[start_mode] = start_counts.get(start_mode, 0) + 1
    learned_start = max(start_counts, key=start_counts.get)

    end_mode_index: Optional[int] = None
    smallest_norm = min(_action_norm(action) for action in centroids)
    if smallest_norm <= student_cfg.end_action_tolerance:
        end_mode_index = min(range(len(centroids)), key=lambda idx: _action_norm(centroids[idx]))

    mode_names: Dict[int, str] = {}
    active_index = 0
    for mode_index in range(len(centroids)):
        if mode_index == end_mode_index:
            mode_names[mode_index] = "end"
        else:
            mode_names[mode_index] = f"m{active_index}"
            active_index += 1

    modes: Dict[str, Mode] = {}
    for mode_index, action in enumerate(centroids):
        mode_name = mode_names[mode_index]
        modes[mode_name] = Mode(
            name=mode_name,
            action_fn=constant_action(action),
            switches={},
            action_description="[" + ", ".join(f"{value:.3f}" for value in action) + "]",
        )

    if "end" not in modes:
        modes["end"] = Mode(
            name="end",
            action_fn=constant_action([0.0 for _ in centroids[0]]),
            action_description="[" + ", ".join("0.000" for _ in centroids[0]) + "]",
        )
    for source, target_switches in switches_by_mode.items():
        source_name = mode_names[source]
        ordered_targets = sorted(
            target_switches,
            key=lambda target: 0 if mode_names[target] == "end" else 1,
        )
        for target in ordered_targets:
            modes[source_name].switches[mode_names[target]] = target_switches[target]

    return ProgrammaticStateMachine(
        modes=modes,
        start_mode=mode_names[learned_start],
        end_mode="end",
    )
