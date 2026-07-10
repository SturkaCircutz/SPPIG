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
    candidate_pool_diagnostics = _teacher_candidate_pool_diagnostics(
        selected,
        candidates,
        elites,
        elite_recombinations,
        refinement_elites,
        refinement_seeds,
        refined,
        scoring_student,
        cfg,
    )
    selected.teacher_candidate_pool_diagnostics = candidate_pool_diagnostics
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
    trace.teacher_refinement_elite_summary = _teacher_refinement_elite_summary(
        trace,
        student,
        cfg,
        refinement_elites,
    )
    return trace


def _teacher_refinement_elite_summary(
    trace: CartpoleTrace,
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
    refinement_elites: List[CartpoleTrace],
) -> Dict[str, object] | None:
    if not refinement_elites:
        return None
    elite_objectives = [_teacher_objective(elite, student, cfg) for elite in refinement_elites]
    elite_refinement_objectives = [
        _teacher_refinement_objective(elite, student, cfg, refinement_elites)
        for elite in refinement_elites
    ]
    source_counts: Dict[str, int] = {}
    for elite in refinement_elites:
        source_counts[elite.teacher_source] = source_counts.get(elite.teacher_source, 0) + 1
    distances = [_loop_free_trace_distance(trace, elite) for elite in refinement_elites]
    nearest_index = min(range(len(distances)), key=lambda index: distances[index])
    summary: Dict[str, object] = {
        "elite_count": len(refinement_elites),
        "top_rho": max(1, cfg.teacher_top_rho),
        "source_counts": source_counts,
        "reward_mean": sum(elite.reward for elite in refinement_elites) / len(refinement_elites),
        "reward_min": min(elite.reward for elite in refinement_elites),
        "reward_max": max(elite.reward for elite in refinement_elites),
        "teacher_objective_mean": sum(elite_objectives) / len(elite_objectives),
        "teacher_objective_min": min(elite_objectives),
        "teacher_objective_max": max(elite_objectives),
        "teacher_refinement_objective_mean": sum(elite_refinement_objectives) / len(elite_refinement_objectives),
        "teacher_refinement_objective_min": min(elite_refinement_objectives),
        "teacher_refinement_objective_max": max(elite_refinement_objectives),
        "selected_distance_to_nearest_elite": distances[nearest_index],
        "selected_distance_to_elite_mean": sum(distances) / len(distances),
        "selected_distance_to_elite_min": min(distances),
        "selected_distance_to_elite_max": max(distances),
        "nearest_elite_source": refinement_elites[nearest_index].teacher_source,
        "nearest_elite_reward": refinement_elites[nearest_index].reward,
        "nearest_elite_teacher_objective": elite_objectives[nearest_index],
    }
    if student is not None:
        elite_log_probabilities = [
            _current_student_log_probability(elite, student)
            for elite in refinement_elites
        ]
        elite_log_normalizer = _logsumexp(elite_log_probabilities)
        kernel_terms = [
            log_probability - distance
            for log_probability, distance in zip(elite_log_probabilities, distances)
        ]
        kernel_log_normalizer = _logsumexp(kernel_terms)
        summary["elite_log_normalizer"] = elite_log_normalizer
        summary["kernel_log_normalizer"] = kernel_log_normalizer
        summary["elite_student_log_probabilities"] = elite_log_probabilities
        summary["elite_probability_weights"] = [
            math.exp(log_probability - elite_log_normalizer)
            for log_probability in elite_log_probabilities
        ]
        summary["selected_kernel_component_weights"] = [
            math.exp(term - kernel_log_normalizer)
            for term in kernel_terms
        ]
        summary["nearest_elite_probability_weight"] = summary["elite_probability_weights"][nearest_index]
        summary["nearest_elite_kernel_component_weight"] = summary["selected_kernel_component_weights"][nearest_index]
        summary["selected_elite_kernel_log_probability"] = kernel_log_normalizer - elite_log_normalizer
    return summary


def _teacher_candidate_pool_diagnostics(
    selected: CartpoleTrace,
    candidates: List[CartpoleTrace],
    elites: List[CartpoleTrace],
    elite_recombinations: List[CartpoleTrace],
    refinement_elites: List[CartpoleTrace],
    refinement_seeds: List[CartpoleTrace],
    refined: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> Dict[str, object]:
    selected_pool = refinement_seeds + refined
    return {
        "diagnostic_scope": "bounded_loop_free_teacher_candidate_pool",
        "not_full_paper_cem": True,
        "candidate_rollouts": cfg.candidate_rollouts,
        "effective_candidate_rollouts": max(1, int(cfg.candidate_rollouts)),
        "sampled_candidate_count": len(candidates),
        "top_rho": max(1, int(cfg.teacher_top_rho)),
        "elite_count": len(elites),
        "elite_recombination_candidate_count": len(elite_recombinations),
        "refinement_elite_count": len(refinement_elites),
        "refinement_seed_count": len(refinement_seeds),
        "refined_candidate_count": len(refined),
        "selection_pool_count": len(selected_pool),
        "selected_source": selected.teacher_source,
        "selected_was_refined_candidate": any(selected is candidate for candidate in refined),
        "selected_candidate": _selected_teacher_candidate_diagnostics(
            selected,
            candidates,
            elites,
            elite_recombinations,
            refinement_elites,
            refined,
            selected_pool,
            student,
            cfg,
        ),
        "source_counts": _teacher_source_counts(selected_pool),
        "sampled_source_counts": _teacher_source_counts(candidates),
        "elite_source_counts": _teacher_source_counts(elites),
        "refinement_elite_source_counts": _teacher_source_counts(refinement_elites),
        "sampled_teacher_objective": _teacher_objective_stats(candidates, student, cfg),
        "selection_pool_refinement_objective": _teacher_refinement_objective_stats(
            selected_pool,
            student,
            cfg,
            refinement_elites,
        ),
    }


def _selected_teacher_candidate_diagnostics(
    selected: CartpoleTrace,
    sampled_candidates: List[CartpoleTrace],
    initial_elites: List[CartpoleTrace],
    elite_recombinations: List[CartpoleTrace],
    refinement_elites: List[CartpoleTrace],
    refined_candidates: List[CartpoleTrace],
    selection_pool: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> Dict[str, object]:
    ranked = sorted(
        enumerate(selection_pool),
        key=lambda item: _teacher_refinement_objective(item[1], student, cfg, refinement_elites),
        reverse=True,
    )
    selected_pool_index = next((index for index, trace in enumerate(selection_pool) if trace is selected), None)
    selected_rank = next((rank for rank, (_, trace) in enumerate(ranked, start=1) if trace is selected), None)
    selected_refinement_objective = _teacher_refinement_objective(
        selected,
        student,
        cfg,
        refinement_elites,
    )
    is_recombination_or_distribution = any(selected is trace for trace in elite_recombinations)
    return {
        "source": selected.teacher_source,
        "reward": selected.reward,
        "student_log_probability": selected.student_log_probability,
        "teacher_objective": _teacher_objective(selected, student, cfg),
        "teacher_refinement_objective": selected_refinement_objective,
        "selection_pool_index": selected_pool_index,
        "selection_rank_by_refinement_objective": selected_rank,
        "selection_pool_count": len(selection_pool),
        "is_sampled_candidate": any(selected is trace for trace in sampled_candidates),
        "is_initial_top_rho_elite": any(selected is trace for trace in initial_elites),
        "is_elite_recombination_candidate": is_recombination_or_distribution
        and "elite_centroid" in selected.teacher_source,
        "is_elite_distribution_candidate": is_recombination_or_distribution
        and "elite_distribution" in selected.teacher_source,
        "is_elite_recombination_or_distribution_candidate": is_recombination_or_distribution,
        "is_refinement_elite": any(selected is trace for trace in refinement_elites),
        "is_refined_candidate": any(selected is trace for trace in refined_candidates),
    }


def _teacher_source_counts(traces: List[CartpoleTrace]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trace in traces:
        counts[trace.teacher_source] = counts.get(trace.teacher_source, 0) + 1
    return counts


def _teacher_objective_stats(
    traces: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> Dict[str, object]:
    values = [_teacher_objective(trace, student, cfg) for trace in traces]
    return _float_stats(values)


def _teacher_refinement_objective_stats(
    traces: List[CartpoleTrace],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
    refinement_elites: List[CartpoleTrace],
) -> Dict[str, object]:
    values = [
        _teacher_refinement_objective(trace, student, cfg, refinement_elites)
        for trace in traces
    ]
    return _float_stats(values)


def _float_stats(values: List[float]) -> Dict[str, object]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


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
    source_weights: Tuple[float, ...]
    source_teacher_objectives: Tuple[float, ...]
    weighting: str


@dataclass(frozen=True)
class _EliteScheduleWeightDetails:
    weights: Tuple[float, ...]
    teacher_objectives: Tuple[float, ...]
    weighting: str


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
    weight_details = _elite_schedule_weight_details(schedules, student, cfg)
    schedule_weights = list(weight_details.weights)
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
        source_weights=weight_details.weights,
        source_teacher_objectives=weight_details.teacher_objectives,
        weighting=weight_details.weighting,
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
    mean_trace.elite_distribution_fit = _summarize_elite_schedule_distribution(distribution)
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
    distribution = _fit_elite_schedule_distribution(schedules, env_cfg, cfg, student)
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
    sample.elite_distribution_fit = _summarize_elite_schedule_distribution(distribution)
    sample.student_log_probability = (
        _trace_log_probability(sample, student)
        if student is not None
        else None
    )
    return sample


def _gaussian_scalar_summary(distribution: GaussianScalar) -> Dict[str, float]:
    return {"mean": distribution.mean, "std": distribution.std}


def _summarize_elite_schedule_distribution(distribution: _EliteScheduleDistribution) -> Dict[str, object]:
    return {
        "weighting": distribution.weighting,
        "source_count": len(distribution.source_elites),
        "source_weights": list(distribution.source_weights),
        "source_teacher_objectives": list(distribution.source_teacher_objectives),
        "theta_gain": _gaussian_scalar_summary(distribution.theta_gain),
        "omega_gain": _gaussian_scalar_summary(distribution.omega_gain),
        "segments": [
            {
                "mode": segment.mode,
                "action": _gaussian_scalar_summary(segment.action),
                "duration": _gaussian_scalar_summary(segment.duration),
                "time_increment": _gaussian_scalar_summary(segment.time_increment),
            }
            for segment in distribution.segments
        ],
    }


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
    return list(_elite_schedule_weight_details(schedules, student, cfg).weights)


def _elite_schedule_weight_details(
    schedules: List[EliteSchedule],
    student: ProbabilisticCartpoleStudent | None,
    cfg: CartpoleSynthesisConfig,
) -> _EliteScheduleWeightDetails:
    if not schedules:
        return _EliteScheduleWeightDetails((), (), "empty")
    if student is None:
        weights = tuple(1.0 / len(schedules) for _ in schedules)
        objectives = tuple(_teacher_objective(trace, None, cfg) for _, _, _, _, trace in schedules)
        return _EliteScheduleWeightDetails(weights, objectives, "uniform_no_student")
    objective_values = [
        _teacher_objective(trace, student, cfg)
        for _, _, _, _, trace in schedules
    ]
    max_objective = max(objective_values)
    raw_weights = [math.exp(value - max_objective) for value in objective_values]
    total = sum(raw_weights)
    if total <= 0.0:
        weights = tuple(1.0 / len(schedules) for _ in schedules)
        return _EliteScheduleWeightDetails(weights, tuple(objective_values), "uniform_degenerate_objective")
    return _EliteScheduleWeightDetails(
        tuple(weight / total for weight in raw_weights),
        tuple(objective_values),
        "softmax_teacher_objective",
    )


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
        best.elite_distribution_fit = trace.elite_distribution_fit
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
        pair = _student_trace_pair_log_potentials(student, trace_segments[index - 1])
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
    terminal_stay = _student_trace_terminal_stay_log_potentials(
        student,
        trace_segments[-1],
    )
    return _logsumexp([
        forward[-1][mode] + terminal_stay[mode]
        for mode in (0, 1)
    ])


def _student_trace_pair_log_potentials(
    student: ProbabilisticCartpoleStudent,
    segment: CartpoleSegment,
) -> List[List[float]]:
    return _transition_switch_pair_log_potentials(
        student.transition_switches,
        student.transition_switch_parameter_distributions,
        student.switch,
        student.switch_parameter_distributions,
        segment,
    )


def _student_trace_terminal_stay_log_potentials(
    student: ProbabilisticCartpoleStudent,
    segment: CartpoleSegment,
) -> List[float]:
    return _transition_switch_terminal_stay_log_potentials(
        student.transition_switches,
        student.transition_switch_parameter_distributions,
        student.switch,
        student.switch_parameter_distributions,
        segment,
    )


def _transition_switch_pair_log_potentials(
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None,
    transition_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None,
    fallback_switch: SwitchProgram,
    fallback_distributions: List[GaussianScalar],
    segment: CartpoleSegment,
) -> List[List[float]]:
    transition_switches = transition_switches or {}
    if (0, 1) not in transition_switches or (1, 0) not in transition_switches:
        return _switch_responsibility_pair_log_potentials(
            fallback_switch,
            fallback_distributions,
            segment,
        )
    transition_distributions = transition_distributions or {}
    off_to_on, stay_off = _switch_transition_and_stay_probabilities(
        transition_switches[(0, 1)],
        transition_distributions.get((0, 1), []),
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
    on_to_off, stay_on = _switch_transition_and_stay_probabilities(
        transition_switches[(1, 0)],
        transition_distributions.get((1, 0), []),
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
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


def _transition_switch_terminal_stay_log_potentials(
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None,
    transition_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None,
    fallback_switch: SwitchProgram,
    fallback_distributions: List[GaussianScalar],
    segment: CartpoleSegment,
) -> List[float]:
    transition_switches = transition_switches or {}
    if (0, 1) not in transition_switches or (1, 0) not in transition_switches:
        return _switch_terminal_stay_log_potentials(
            fallback_switch,
            fallback_distributions,
            segment,
        )
    transition_distributions = transition_distributions or {}
    _, stay_off = _switch_transition_and_stay_probabilities(
        transition_switches[(0, 1)],
        transition_distributions.get((0, 1), []),
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
    _, stay_on = _switch_transition_and_stay_probabilities(
        transition_switches[(1, 0)],
        transition_distributions.get((1, 0), []),
        segment.observations,
        segment.switch_timing_duration,
        segment.timing_step_scale,
    )
    return [
        math.log(max(stay_off, LOG_PROBABILITY_FLOOR)),
        math.log(max(stay_on, LOG_PROBABILITY_FLOOR)),
    ]

def _logsumexp(values: List[float]) -> float:
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))
