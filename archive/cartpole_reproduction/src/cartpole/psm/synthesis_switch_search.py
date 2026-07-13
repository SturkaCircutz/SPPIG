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
from .synthesis_switch_fit import *

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
    parallel_workers: int = 1,
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
        parallel_workers=parallel_workers,
    )
    weighted_examples = [
        _WeightedSwitchExample(observation, label, 1.0)
        for observation, label in examples
    ]
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
    candidate_switches = _prefilter_switches_by_label_mistakes(
        candidates_with_mistakes,
        required_switches=constant_leaf_switches,
    )

    rescored_switches = _switch_structure_rescore_candidates(
        candidate_switches,
        examples,
        segments_by_trace=segments_by_trace,
        responsibilities=responsibilities,
        switch_pair_responsibilities=switch_pair_responsibilities,
        cache=objective_cache,
        example_cache=example_cache,
        parallel_workers=parallel_workers,
    )
    costs = _switch_structure_costs(
        rescored_switches,
        examples,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        objective_cache,
        example_cache,
        parallel_workers=parallel_workers,
    )
    return min(zip(costs, rescored_switches), key=lambda item: item[0])[1]


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
    required_switches: List[SwitchProgram] | None = None,
) -> List[SwitchProgram]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item[1],
            item[0].node_count if isinstance(item[0], BooleanTreeSwitch) else 1,
            item[0].describe(),
        ),
    )
    selected = [switch for switch, _ in ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]]
    return _with_required_switches(selected, required_switches)


def _greedy_boolean_tree_candidates(
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]] | None = None,
    responsibilities: List[Tuple[float, float]] | None = None,
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None = None,
    cache: Dict[str, Tuple[float, float, int, str]] | None = None,
    example_cache: _SwitchExampleCache | None = None,
    parallel_workers: int = 1,
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
        parallel_workers=parallel_workers,
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
                parallel_workers=parallel_workers,
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
                parallel_workers=parallel_workers,
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
    parallel_workers: int = 1,
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
            parallel_workers=parallel_workers,
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
    parallel_workers: int = 1,
) -> List[SwitchProgram]:
    if segments_by_trace is None or responsibilities is None:
        return switches
    if len(switches) <= SWITCH_STRUCTURE_RESCORING_TOP_K and parallel_workers <= 1:
        return switches

    switch_examples = example_cache or _switch_example_cache(examples)
    if len(switches) <= SWITCH_STRUCTURE_RESCORING_TOP_K:
        prefiltered = switches
    else:
        prefiltered = _prefilter_switches_by_label_mistakes(
            [
                (switch, _switch_label_mistakes(switch, examples, switch_examples))
                for switch in switches
            ]
        )
    objective_cache: Dict[str, Tuple[float, float, int, str]] = cache if cache is not None else {}
    costs = _switch_structure_costs(
        prefiltered,
        examples,
        segments_by_trace,
        responsibilities,
        switch_pair_responsibilities,
        objective_cache,
        switch_examples,
        parallel_workers=parallel_workers,
    )
    ranked = [switch for _, switch in sorted(zip(costs, prefiltered), key=lambda item: item[0])]
    return ranked[:SWITCH_STRUCTURE_RESCORING_TOP_K]


def _switch_structure_costs(
    switches: List[SwitchProgram],
    examples: List[Tuple[Observation, int]],
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] | None,
    cache: Dict[str, Tuple[float, float, int, str]],
    example_cache: _SwitchExampleCache,
    parallel_workers: int = 1,
) -> List[Tuple[float, float, int, str]]:
    if parallel_workers <= 1 or len(switches) <= 1:
        return [
            _switch_structure_cost(
                switch,
                examples,
                segments_by_trace,
                responsibilities,
                switch_pair_responsibilities=switch_pair_responsibilities,
                cache=cache,
                example_cache=example_cache,
            )
            for switch in switches
        ]

    missing_switches = [
        switch
        for switch in switches
        if _switch_structure_objective_cache_key(switch, switch_pair_responsibilities) not in cache
    ]
    if missing_switches:
        with ThreadPoolExecutor(max_workers=min(max(1, int(parallel_workers)), len(missing_switches))) as executor:
            missing_costs = list(
                executor.map(
                    lambda switch: _switch_structure_cost(
                        switch,
                        examples,
                        segments_by_trace,
                        responsibilities,
                        switch_pair_responsibilities=switch_pair_responsibilities,
                        cache=None,
                        example_cache=example_cache,
                    ),
                    missing_switches,
                )
            )
        for switch, cost in zip(missing_switches, missing_costs):
            cache[_switch_structure_objective_cache_key(switch, switch_pair_responsibilities)] = cost
    return [
        cache[_switch_structure_objective_cache_key(switch, switch_pair_responsibilities)]
        for switch in switches
    ]


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
