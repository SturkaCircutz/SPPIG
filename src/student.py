"""Student-side fitting utilities for the simplified supervised prototype."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from psm import ConstantAction, Mode, ProgrammaticStateMachine, ThresholdSwitch
from teacher import TeacherTrace


@dataclass
class StudentConfig:
    """Configuration for fitting a compact state machine from teacher traces."""

    num_modes: int
    action_grammar: Any = None
    switch_grammar: Any = None
    max_em_iters: int = 20
    min_guard_examples: int = 1


def _sorted_mode_ids(traces: List[TeacherTrace], max_modes: int) -> List[int]:
    """Pick the most common mode ids and keep them in a stable sorted order."""

    counts = Counter()
    for trace in traces:
        counts.update(trace.mode_hints)

    most_common = counts.most_common(max_modes)
    return sorted(mode_id for mode_id, _ in most_common)


def _mean_action(actions: List[List[float]]) -> List[float]:
    """Fit a constant action by averaging teacher actions assigned to one mode."""

    if not actions:
        return [0.0]

    dimension = len(actions[0])
    means = [0.0] * dimension
    for action in actions:
        for index, value in enumerate(action):
            means[index] += float(value)
    return [value / len(actions) for value in means]


def _candidate_thresholds(values: List[float]) -> List[float]:
    """Enumerate threshold candidates from observed values."""

    unique_values = sorted(set(values))
    if not unique_values:
        return [0.0]
    if len(unique_values) == 1:
        return [unique_values[0]]

    thresholds = [unique_values[0] - 1e-6, unique_values[-1] + 1e-6]
    for left, right in zip(unique_values, unique_values[1:]):
        thresholds.append((left + right) / 2.0)
    return thresholds


def _threshold_accuracy(
    positives: List[List[float]],
    negatives: List[List[float]],
    index: int,
    direction: str,
    threshold: float,
) -> Tuple[float, float]:
    """Score one threshold rule by accuracy first and margin second."""

    switch_fn = ThresholdSwitch(index=index, threshold=threshold, direction=direction)

    correct = 0
    total = len(positives) + len(negatives)
    margin = 0.0
    for observation in positives:
        score = switch_fn(observation)
        if score >= 0.0:
            correct += 1
        margin += score
    for observation in negatives:
        score = switch_fn(observation)
        if score < 0.0:
            correct += 1
        margin -= score

    accuracy = correct / total if total else 0.0
    return accuracy, margin


def _fit_threshold_switch(
    positives: List[List[float]],
    negatives: List[List[float]],
) -> Optional[ThresholdSwitch]:
    """Fit the best one-dimensional threshold guard for a transition."""

    if not positives:
        return None

    if not negatives:
        # When only positive examples exist, prefer the tightest observation
        # dimension so the learned rule stays as specific as possible.
        ranges = []
        for index in range(len(positives[0])):
            values = [observation[index] for observation in positives]
            ranges.append((max(values) - min(values), index, values))

        _, index, values = min(ranges, key=lambda item: (item[0], item[1]))
        threshold = sum(values) / len(values)
        direction = "<=" if threshold <= 0.5 else ">="
        edge_value = max(values) if direction == "<=" else min(values)
        return ThresholdSwitch(index=index, threshold=edge_value, direction=direction)

    dimension = len(positives[0])
    best_rule: Optional[ThresholdSwitch] = None
    best_score = (-1.0, float("-inf"))
    for index in range(dimension):
        values = [observation[index] for observation in positives]
        values.extend(observation[index] for observation in negatives)
        for threshold in _candidate_thresholds(values):
            for direction in (">=", "<="):
                accuracy, margin = _threshold_accuracy(positives, negatives, index, direction, threshold)
                score = (accuracy, margin)
                if score > best_score:
                    best_score = score
                    best_rule = ThresholdSwitch(index=index, threshold=threshold, direction=direction)
    return best_rule


def _normalize_trace(raw_trace: Any) -> TeacherTrace:
    """Coerce trace-like inputs into the TeacherTrace dataclass."""

    if isinstance(raw_trace, TeacherTrace):
        return raw_trace

    return TeacherTrace(
        initial_state=getattr(raw_trace, "initial_state", None),
        observations=[list(observation) for observation in raw_trace.observations],
        actions=[list(action) for action in raw_trace.actions],
        mode_hints=list(raw_trace.mode_hints),
        reward=float(getattr(raw_trace, "reward", 0.0)),
    )


def fit_student_from_traces(
    traces: Iterable[Any],
    grammar: Any,
    student_cfg: StudentConfig,
) -> ProgrammaticStateMachine:
    """
    Fit a simple state machine from teacher traces with mode labels.

    This prototype skips latent mode inference and instead uses `mode_hints`
    directly. Each mode gets the mean action of the examples assigned to it, and
    each cross-mode transition receives a single threshold guard chosen to best
    separate positive and negative examples from the trace data.
    """

    normalized_traces = [_normalize_trace(trace) for trace in traces]
    if not normalized_traces:
        raise ValueError("Cannot fit a student without teacher traces.")

    mode_ids = _sorted_mode_ids(normalized_traces, student_cfg.num_modes)
    if not mode_ids:
        raise ValueError("Teacher traces did not contain any mode hints.")
    mode_id_set = set(mode_ids)

    action_examples: Dict[int, List[List[float]]] = defaultdict(list)
    transition_examples: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    negative_examples: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    first_modes: List[int] = []
    last_modes: List[int] = []

    outgoing_targets: Dict[int, set[int]] = defaultdict(set)
    for trace in normalized_traces:
        if trace.mode_hints:
            if trace.mode_hints[0] in mode_id_set:
                first_modes.append(trace.mode_hints[0])
            if trace.mode_hints[-1] in mode_id_set:
                last_modes.append(trace.mode_hints[-1])

        for action, mode_id in zip(trace.actions, trace.mode_hints):
            if mode_id in mode_id_set:
                action_examples[mode_id].append([float(value) for value in action])

        for time_index in range(len(trace.mode_hints) - 1):
            source = trace.mode_hints[time_index]
            target = trace.mode_hints[time_index + 1]
            if source in mode_id_set and target in mode_id_set and source != target:
                outgoing_targets[source].add(target)

    for trace in normalized_traces:
        horizon = min(len(trace.observations), len(trace.mode_hints) - 1)
        for time_index in range(horizon):
            source = trace.mode_hints[time_index]
            if source not in outgoing_targets:
                continue

            target = trace.mode_hints[time_index + 1]
            observation = [float(value) for value in trace.observations[time_index]]
            for candidate_target in outgoing_targets[source]:
                key = (source, candidate_target)
                if target == candidate_target and source != target:
                    transition_examples[key].append(observation)
                else:
                    negative_examples[key].append(observation)

    mode_names = {mode_id: f"mode_{mode_id}" for mode_id in mode_ids}
    fitted_modes: Dict[str, Mode] = {}
    for mode_id in mode_ids:
        fitted_modes[mode_names[mode_id]] = Mode(
            name=mode_names[mode_id],
            action_fn=ConstantAction(_mean_action(action_examples[mode_id])),
        )

    for (source, target), positives in transition_examples.items():
        negatives = negative_examples[(source, target)]
        if len(positives) < student_cfg.min_guard_examples:
            continue

        # Each transition gets a single symbolic guard in the current prototype.
        switch_rule = _fit_threshold_switch(positives, negatives)
        if switch_rule is None:
            continue
        fitted_modes[mode_names[source]].switches[mode_names[target]] = switch_rule

    start_mode_id = Counter(first_modes).most_common(1)[0][0] if first_modes else mode_ids[0]
    end_mode_id = Counter(last_modes).most_common(1)[0][0] if last_modes else mode_ids[-1]
    return ProgrammaticStateMachine(
        modes=fitted_modes,
        start_mode=mode_names[start_mode_id],
        end_mode=mode_names[end_mode_id],
    )
