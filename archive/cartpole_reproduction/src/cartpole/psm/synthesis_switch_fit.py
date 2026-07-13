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
from .synthesis_programs import *
from .synthesis_student import *
from .synthesis_teacher import *

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
    cfg: CartpoleSynthesisConfig | None = None,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    # Refit switch structure and Gaussian threshold parameters for the bounded
    # Eq. (12)-style M-step.
    switch = _learn_depth2_switch(
        traces,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        parallel_workers=max(1, int(cfg.parallel_switch_workers)) if cfg is not None else 1,
    )
    distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
    )
    return _switch_with_distribution_means(switch, distributions), distributions


def _fit_transition_switches(
    traces: List[CartpoleTrace],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
    fallback_switch: SwitchProgram,
    fallback_distributions: List[GaussianScalar],
    cfg: CartpoleSynthesisConfig | None = None,
) -> Tuple[Dict[Tuple[int, int], SwitchProgram], Dict[Tuple[int, int], List[GaussianScalar]]]:
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    if pair_count == 0:
        return {}, {}

    transitions: Dict[Tuple[int, int], SwitchProgram] = {}
    distributions: Dict[Tuple[int, int], List[GaussianScalar]] = {}
    for transition, switch, switch_distributions in _fit_directed_transition_switches(
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        fallback_switch,
        cfg,
    ):
        transitions[transition] = switch
        distributions[transition] = switch_distributions
    return transitions, distributions


def _fit_directed_transition_switches(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
    fallback_switch: SwitchProgram,
    cfg: CartpoleSynthesisConfig | None = None,
) -> List[Tuple[Tuple[int, int], SwitchProgram, List[GaussianScalar]]]:
    transitions = [(0, 1), (1, 0)]

    def fit_one(transition: Tuple[int, int]) -> Tuple[Tuple[int, int], SwitchProgram, List[GaussianScalar]]:
        directed_pairs = _directed_switch_pair_responsibilities(
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities,
            transition,
        )
        switch, switch_distributions = _fit_directed_transition_switch(
            segments_by_trace,
            responsibilities,
            directed_pairs,
            transition,
            fallback_switch,
            parallel_workers=directed_fit_parallel_workers,
        )
        return transition, switch, switch_distributions

    parallel_workers = max(1, int(cfg.parallel_switch_workers)) if cfg is not None else 1
    outer_workers = min(parallel_workers, len(transitions))
    directed_fit_parallel_workers = max(1, parallel_workers // max(1, outer_workers))
    if outer_workers <= 1:
        return [fit_one(transition) for transition in transitions]
    with ThreadPoolExecutor(max_workers=outer_workers) as executor:
        return list(executor.map(fit_one, transitions))


def _fit_directed_transition_switch(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    directed_pair_responsibilities: List[Tuple[float, float, float, float]],
    transition: Tuple[int, int],
    fallback_switch: SwitchProgram,
    parallel_workers: int = 1,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    directed_responsibilities = _directed_switch_responsibilities(
        segments_by_trace,
        responsibilities,
        transition,
    )
    weighted_examples = _directed_transition_examples(
        segments_by_trace,
        directed_responsibilities,
        directed_pair_responsibilities,
        transition,
    )
    examples = [
        (example.observation, example.label)
        for example in weighted_examples
        if example.weight > 0.0
    ]
    labels = {example.label for example in weighted_examples if example.weight > 0.0}
    if not weighted_examples or labels == {0}:
        return _constant_directed_switch(weighted_examples, fire=False)
    if labels == {1}:
        return _constant_directed_switch(weighted_examples, fire=True)
    switch = _learn_switch_from_examples(
        weighted_examples,
        parallel_workers=parallel_workers,
    )
    distributions = _fit_switch_parameter_distributions(
        switch,
        segments_by_trace,
        directed_responsibilities,
        directed_pair_responsibilities,
    )
    refined_switch = _switch_with_distribution_means(switch, distributions)
    if _weighted_switch_label_loss(refined_switch, weighted_examples) > _weighted_switch_label_loss(switch, weighted_examples):
        distributions = _distributions_with_switch_means(switch, distributions)
        refined_switch = _switch_with_distribution_means(switch, distributions)
    return refined_switch, distributions


def _constant_directed_switch(
    examples: List[_WeightedSwitchExample],
    fire: bool,
) -> Tuple[SwitchProgram, List[GaussianScalar]]:
    threshold = -1e9 if fire else 1e9
    switch = Depth2Switch(1.0, 0.0, threshold)
    return switch, [GaussianScalar(threshold, MIN_GAUSSIAN_STD)]


def _directed_switch_responsibilities(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    transition: Tuple[int, int],
) -> List[Tuple[float, float]]:
    source_mode = transition[0]
    directed: List[Tuple[float, float]] = []
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match directed switch segments")
    for responsibility in responsibilities:
        directed.append((responsibility[source_mode], 0.0))
    return directed


def _directed_transition_examples(
    segments_by_trace: List[List[CartpoleSegment]],
    directed_responsibilities: List[Tuple[float, float]],
    directed_pair_responsibilities: List[Tuple[float, float, float, float]],
    transition: Tuple[int, int],
) -> List[_WeightedSwitchExample]:
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    final_segment_count = sum(1 for trace_segments in segments_by_trace if trace_segments)
    if len(directed_pair_responsibilities) not in {pair_count, pair_count + final_segment_count}:
        raise ValueError("directed pair responsibility count must match adjacent pairs plus optional final stays")
    has_final_stay_rows = (
        len(directed_pair_responsibilities) == pair_count + final_segment_count
        and final_segment_count > 0
    )
    examples: List[_WeightedSwitchExample] = []
    pair_index = 0
    segment_index = 0
    for trace_segments in segments_by_trace:
        for index, segment in enumerate(trace_segments):
            source_weight = (
                directed_responsibilities[segment_index][0]
                if segment_index < len(directed_responsibilities)
                else 0.0
            )
            segment_index += 1
            interior = segment.observations[:-1] if index + 1 < len(trace_segments) else segment.observations
            if source_weight > 0.0 and interior:
                examples.extend(
                    _WeightedSwitchExample(observation, 0, source_weight)
                    for observation in interior
                )
            if index + 1 >= len(trace_segments):
                continue
            transition_weight = directed_pair_responsibilities[pair_index][1]
            no_transition_weight = max(source_weight - transition_weight, 0.0)
            pair_index += 1
            if transition_weight > 0.0:
                examples.append(_WeightedSwitchExample(segment.end_observation, 1, transition_weight))
            if no_transition_weight > 0.0:
                examples.append(_WeightedSwitchExample(segment.end_observation, 0, no_transition_weight))
        if has_final_stay_rows and trace_segments:
            pair_index += 1
    return examples


def _learn_switch_from_examples(
    weighted_examples: List[_WeightedSwitchExample],
    parallel_workers: int = 1,
) -> SwitchProgram:
    examples = [
        (example.observation, example.label)
        for example in weighted_examples
        if example.weight > 0.0
    ]
    if not examples:
        return Depth2Switch(1.0, 0.0, 0.0)

    example_cache = _switch_example_cache(examples)
    boolean_switches = _directed_boolean_tree_candidates(weighted_examples, example_cache)
    constant_leaf_switches = [
        _constant_directed_switch(weighted_examples, fire=False)[0],
        _constant_directed_switch(weighted_examples, fire=True)[0],
    ]
    candidates_with_mistakes: List[Tuple[SwitchProgram, int]] = [
        *[
            (switch, _switch_label_mistakes(switch, examples, example_cache))
            for switch in constant_leaf_switches
        ],
        *_depth2_switch_candidates_with_mistakes(example_cache),
        *[
            (switch, _switch_label_mistakes(switch, examples, example_cache))
            for switch in boolean_switches
        ],
    ]
    candidate_switches = _prefilter_switches_by_weighted_label_loss(
        [switch for switch, _ in candidates_with_mistakes],
        weighted_examples,
        required_switches=constant_leaf_switches,
    )

    return _best_directed_switch(
        candidate_switches,
        weighted_examples,
        examples,
        example_cache,
        parallel_workers=parallel_workers,
    )


def _prefilter_switches_by_weighted_label_loss(
    switches: List[SwitchProgram],
    weighted_examples: List[_WeightedSwitchExample],
    required_switches: List[SwitchProgram] | None = None,
) -> List[SwitchProgram]:
    ranked = sorted(
        switches,
        key=lambda switch: (
            _weighted_switch_label_loss(switch, weighted_examples),
            switch.node_count if isinstance(switch, BooleanTreeSwitch) else 1,
            switch.describe(),
        ),
    )
    return _with_required_switches(ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K], required_switches)


def _with_required_switches(
    selected: List[SwitchProgram],
    required_switches: List[SwitchProgram] | None,
) -> List[SwitchProgram]:
    if not required_switches:
        return selected
    result = list(selected)
    selected_key_counts: Dict[str, int] = {}
    for switch in result:
        key = _switch_cache_key(switch)
        selected_key_counts[key] = selected_key_counts.get(key, 0) + 1
    required_keys = {_switch_cache_key(switch) for switch in required_switches}
    for switch in required_switches:
        key = _switch_cache_key(switch)
        if selected_key_counts.get(key, 0) > 0:
            continue
        while len(result) >= SWITCH_STRUCTURE_RESCORING_TOP_K:
            for index in range(len(result) - 1, -1, -1):
                candidate_key = _switch_cache_key(result[index])
                if candidate_key not in required_keys:
                    result.pop(index)
                    selected_key_counts[candidate_key] -= 1
                    if selected_key_counts[candidate_key] <= 0:
                        del selected_key_counts[candidate_key]
                    break
            else:
                return result
        result.append(switch)
        selected_key_counts[key] = selected_key_counts.get(key, 0) + 1
    return result


def _directed_boolean_tree_candidates(
    weighted_examples: List[_WeightedSwitchExample],
    example_cache: _SwitchExampleCache,
) -> List[BooleanTreeSwitch]:
    examples = [
        (example.observation, example.label)
        for example in weighted_examples
        if example.weight > 0.0
    ]
    stumps = [BooleanTreeSwitch(predicate) for predicate in _predicate_candidates(examples)]
    if not stumps:
        return []
    best = min(stumps, key=lambda switch: _weighted_switch_label_loss(switch, weighted_examples))
    best_loss = _weighted_switch_label_loss(best, weighted_examples)
    seed_stumps = [
        stump
        for stump in stumps
        if _weighted_switch_label_loss(stump, weighted_examples) == best_loss
    ]
    expansions: List[BooleanTreeSwitch] = []
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
        expansions.extend(
            BooleanTreeSwitch(stump.first, predicate)
            for predicate in _predicate_candidates(conjunction_examples)
        )
        expansions.extend(
            BooleanTreeSwitch(stump.first, predicate, "or")
            for predicate in _predicate_candidates(disjunction_examples)
        )
    if not expansions:
        return [best]
    return [
        best,
        *_prefilter_switches_by_weighted_label_loss(expansions, weighted_examples),
    ]


def _directed_switch_structure_cost(
    switch: SwitchProgram,
    weighted_examples: List[_WeightedSwitchExample],
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
) -> Tuple[float, float, int, str]:
    label_loss = _weighted_switch_label_loss(switch, weighted_examples)
    complexity = switch.node_count if isinstance(switch, BooleanTreeSwitch) else 1
    return label_loss, float(_switch_label_mistakes(switch, examples, example_cache)), complexity, switch.describe()


def _best_directed_switch(
    switches: List[SwitchProgram],
    weighted_examples: List[_WeightedSwitchExample],
    examples: List[Tuple[Observation, int]],
    example_cache: _SwitchExampleCache,
    parallel_workers: int = 1,
) -> SwitchProgram:
    if parallel_workers <= 1 or len(switches) <= 1:
        return min(
            switches,
            key=lambda switch: _directed_switch_structure_cost(
                switch,
                weighted_examples,
                examples,
                example_cache,
            ),
        )
    with ThreadPoolExecutor(max_workers=min(max(1, int(parallel_workers)), len(switches))) as executor:
        costs = list(
            executor.map(
                lambda switch: _directed_switch_structure_cost(
                    switch,
                    weighted_examples,
                    examples,
                    example_cache,
                ),
                switches,
            )
        )
    return min(zip(costs, switches), key=lambda item: item[0])[1]


def _weighted_switch_label_loss(
    switch: SwitchProgram,
    examples: List[_WeightedSwitchExample],
) -> float:
    return sum(
        example.weight
        for example in examples
        if example.weight > 0.0 and switch.decide(example.observation) != example.label
    )


def _directed_switch_pair_responsibilities(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
    transition: Tuple[int, int],
) -> List[Tuple[float, float, float, float]]:
    flat_segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    if len(flat_segments) != len(responsibilities):
        raise ValueError("responsibility count must match directed switch timing segments")
    pair_count = sum(max(0, len(trace_segments) - 1) for trace_segments in segments_by_trace)
    final_segment_count = sum(1 for trace_segments in segments_by_trace if trace_segments)
    if (
        switch_pair_responsibilities is not None
        and len(switch_pair_responsibilities) not in {pair_count, pair_count + final_segment_count}
    ):
        raise ValueError("switch pair responsibility count must match adjacent pairs plus optional final stays")
    has_final_stay_rows = (
        switch_pair_responsibilities is not None
        and len(switch_pair_responsibilities) == pair_count + final_segment_count
        and final_segment_count > 0
    )
    responsibility_by_id = {
        id(segment): resp for segment, resp in zip(flat_segments, responsibilities)
    }
    directed: List[Tuple[float, float, float, float]] = []
    pair_index = 0
    for trace_segments in segments_by_trace:
        for current_segment, next_segment in zip(trace_segments, trace_segments[1:]):
            if switch_pair_responsibilities is not None:
                stay_off_weight, off_to_on_weight, on_to_off_weight, stay_on_weight = switch_pair_responsibilities[
                    pair_index
                ]
            else:
                current_resp = responsibility_by_id.get(id(current_segment), (0.5, 0.5))
                next_resp = responsibility_by_id.get(id(next_segment), (0.5, 0.5))
                stay_off_weight = current_resp[0] * next_resp[0]
                off_to_on_weight = current_resp[0] * next_resp[1]
                on_to_off_weight = current_resp[1] * next_resp[0]
                stay_on_weight = current_resp[1] * next_resp[1]
            pair_index += 1
            if transition == (0, 1):
                directed.append((stay_off_weight, off_to_on_weight, 0.0, 0.0))
            elif transition == (1, 0):
                directed.append((stay_on_weight, on_to_off_weight, 0.0, 0.0))
            else:
                raise ValueError(f"unsupported CartPole transition: {transition}")
        if trace_segments:
            if has_final_stay_rows:
                final_stay_off_weight, _, _, final_stay_on_weight = switch_pair_responsibilities[
                    pair_index
                ]
                pair_index += 1
            else:
                final_segment = trace_segments[-1]
                final_resp = responsibility_by_id.get(id(final_segment), (0.5, 0.5))
                final_stay_off_weight, final_stay_on_weight = final_resp
            if transition == (0, 1):
                directed.append((final_stay_off_weight, 0.0, 0.0, 0.0))
            elif transition == (1, 0):
                directed.append((final_stay_on_weight, 0.0, 0.0, 0.0))
            else:
                raise ValueError(f"unsupported CartPole transition: {transition}")
    return directed


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
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None,
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None = None,
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
            _transition_switch_pair_log_potentials(
                transition_switches,
                transition_switch_parameter_distributions,
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

        terminal_stay = _transition_switch_terminal_stay_log_potentials(
            transition_switches,
            transition_switch_parameter_distributions,
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
    if isinstance(switch, LinearObservationSwitch):
        if not distributions:
            return switch
        return LinearObservationSwitch(switch.weights, distributions[0].mean)
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
    best_combined_loss = _switch_parameter_gradient_loss(best_label_loss, best_loss)
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
            candidate_combined_loss = _switch_parameter_gradient_loss(candidate_label_loss, candidate_loss)
            if candidate_combined_loss < best_combined_loss:
                previous_combined_loss = best_combined_loss
                best_distributions = candidate_distributions
                best_switch = candidate_switch
                best_label_loss = candidate_label_loss
                best_loss = candidate_loss
                best_combined_loss = candidate_combined_loss
                accepted = True
                break
        if not accepted:
            break
        if _switch_parameter_gradient_converged(previous_combined_loss, best_combined_loss):
            break
    return best_switch, best_distributions


def _switch_parameter_gradient_improves(
    candidate_label_loss: float,
    candidate_timing_loss: float,
    best_label_loss: float,
    best_timing_loss: float,
) -> bool:
    return _switch_parameter_gradient_loss(
        candidate_label_loss,
        candidate_timing_loss,
    ) < _switch_parameter_gradient_loss(
        best_label_loss,
        best_timing_loss,
    )


def _switch_parameter_gradient_converged(previous_loss: float, current_loss: float) -> bool:
    improvement = previous_loss - current_loss
    scale = max(1.0, abs(previous_loss))
    return improvement <= SWITCH_PARAMETER_GRADIENT_MIN_RELATIVE_IMPROVEMENT * scale


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
    _, minus_label_loss, minus_timing_loss = _evaluate_switch_parameter_candidate(
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
    _, plus_label_loss, plus_timing_loss = _evaluate_switch_parameter_candidate(
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
    minus_loss = _switch_parameter_gradient_loss(minus_label_loss, minus_timing_loss)
    plus_loss = _switch_parameter_gradient_loss(plus_label_loss, plus_timing_loss)
    denominator = 2.0 * (delta_mean if delta_mean else delta_log_std)
    return (plus_loss - minus_loss) / denominator


def _switch_parameter_gradient_loss(label_loss: float, timing_loss: float) -> float:
    return (
        SWITCH_PARAMETER_LABEL_LOSS_WEIGHT * label_loss
        + SWITCH_PARAMETER_TIMING_LOSS_WEIGHT * timing_loss
    )


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
    final_segment_count = sum(1 for trace_segments in segments_by_trace if trace_segments)
    if (
        switch_pair_responsibilities is not None
        and len(switch_pair_responsibilities) not in {pair_count, pair_count + final_segment_count}
    ):
        raise ValueError("switch pair responsibility count must match adjacent pairs plus optional final stays")
    has_final_stay_rows = (
        switch_pair_responsibilities is not None
        and len(switch_pair_responsibilities) == pair_count + final_segment_count
        and final_segment_count > 0
    )
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
            if next_resp is None and has_final_stay_rows:
                stay_off_weight, off_to_on_weight, on_to_off_weight, stay_on_weight = switch_pair_responsibilities[
                    pair_index
                ]
                pair_index += 1
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
    if isinstance(switch, LinearObservationSwitch):
        return (
            tuple(
                sum(weight * column[index] for weight, column in zip(switch.weights, pair.columns))
                for index in range(len(pair.columns[0]))
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
    if isinstance(switch, LinearObservationSwitch):
        if not distributions:
            return []
        return [GaussianScalar(switch.threshold, max(MIN_GAUSSIAN_STD, distributions[0].std))]
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
    if isinstance(switch, LinearObservationSwitch):
        weights = switch.weights
        threshold = switch.threshold
        return sum(
            int(int(sum(weight * column[row] for weight, column in zip(weights, cache.columns)) >= threshold) != label)
            for row, label in enumerate(cache.labels)
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
    final_segment_count = sum(1 for trace_segments in segments_by_trace if trace_segments)
    if switch_pair_responsibilities is not None:
        if len(switch_pair_responsibilities) not in {pair_count, pair_count + final_segment_count}:
            raise ValueError("switch pair responsibility count must match adjacent pairs plus optional final stays")
        has_final_stay_rows = (
            len(switch_pair_responsibilities) == pair_count + final_segment_count
            and final_segment_count > 0
        )
        adjacent_pair_responsibilities: List[Tuple[float, float, float, float]] = []
        pair_index = 0
        for trace_segments in segments_by_trace:
            adjacent_count = max(0, len(trace_segments) - 1)
            adjacent_pair_responsibilities.extend(
                switch_pair_responsibilities[pair_index : pair_index + adjacent_count]
            )
            pair_index += adjacent_count
            if has_final_stay_rows and trace_segments:
                pair_index += 1
        return [
            off_to_on + on_to_off
            for _, off_to_on, on_to_off, _ in adjacent_pair_responsibilities
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
    if isinstance(switch, LinearObservationSwitch):
        return switch.threshold
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
    if isinstance(switch, LinearObservationSwitch):
        distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, 1.0)
        return LinearObservationSwitch(
            switch.weights,
            rng.gauss(distribution.mean, distribution.std),
        )
    distribution = distributions[0] if distributions else GaussianScalar(switch.threshold, 1.0)
    return Depth2Switch(
        switch.theta_weight,
        switch.omega_weight,
        rng.gauss(distribution.mean, distribution.std),
    )
