from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict

from cartpole_env import (
    PAPER_EVAL_ROLLOUTS,
    CartpoleEnv,
    cartpole_reward_spec,
    cartpole_space_spec,
    summarize_cartpole_results,
)
from cartpole_synthesis import (
    CartpoleSynthesisIteration,
    CartpoleSynthesisConfig,
    CartpoleStudentFitStep,
    CartpoleTrace,
    ProbabilisticCartpoleStudent,
    cartpole_synthesis_algorithm_provenance,
    cartpole_synthesis_protocol_status,
    cartpole_teacher_cem_protocol_status,
    cartpole_switch_fit_diagnostics,
    synthesize_cartpole_student_with_history,
    _directed_transition_description,
)


def summarize_rollouts(results):
    return summarize_cartpole_results(results)


def summarize_policy_evaluation(
    policy,
    eval_rollouts: int,
    test_max_steps: int,
    train_seed: int = 100,
    test_seed: int = 200,
):
    train_env = CartpoleEnv.train_env(seed=train_seed)
    test_env = CartpoleEnv.test_env(seed=test_seed)
    train_results = [train_env.rollout(policy) for _ in range(eval_rollouts)]
    test_results = [test_env.rollout(policy, max_steps=test_max_steps) for _ in range(eval_rollouts)]
    return {
        "train": summarize_rollouts(train_results),
        "test": summarize_rollouts(test_results),
    }


def summarize_student(student: ProbabilisticCartpoleStudent):
    transition_distributions = student.transition_switch_parameter_distributions or {}
    return {
        "description": student.describe(),
        "action_distributions": {
            str(mode): {
                "mean": distribution.mean,
                "std": distribution.std,
            }
            for mode, distribution in sorted(student.action_distributions.items())
        },
        "switch": student.switch.describe(),
        "switch_threshold_distribution": {
            "mean": student.switch_threshold_distribution.mean,
            "std": student.switch_threshold_distribution.std,
        },
        "switch_parameter_distributions": [
            {
                "mean": distribution.mean,
                "std": distribution.std,
            }
            for distribution in student.switch_parameter_distributions
        ],
        "transition_switches": {
            f"{source}->{target}": _directed_transition_description(source, target, switch)
            for (source, target), switch in sorted((student.transition_switches or {}).items())
        },
        "transition_switch_parameter_distributions": {
            f"{source}->{target}": [
                {
                    "mean": distribution.mean,
                    "std": distribution.std,
                }
                for distribution in transition_distributions.get((source, target), [])
            ]
            for source, target in sorted((student.transition_switches or {}).keys())
        },
        "responsibility_summary": summarize_responsibilities(student.responsibilities),
        "switch_pair_responsibility_summary": summarize_switch_pair_responsibilities(
            student.switch_pair_responsibilities or []
        ),
    }


def summarize_responsibilities(responsibilities):
    if responsibilities:
        mean_left = sum(left for left, _ in responsibilities) / len(responsibilities)
        mean_right = sum(right for _, right in responsibilities) / len(responsibilities)
        max_weights = [max(left, right) for left, right in responsibilities]
        entropy_values = [
            -sum(weight * math.log(weight) for weight in (left, right) if weight > 0.0)
            for left, right in responsibilities
        ]
        hard_mode_0 = sum(1 for left, right in responsibilities if left >= right)
        hard_mode_1 = len(responsibilities) - hard_mode_0
        ambiguous_segments = sum(1 for weight in max_weights if weight < 0.75)
        mean_max_weight = sum(max_weights) / len(max_weights)
        min_max_weight = min(max_weights)
        mean_entropy = sum(entropy_values) / len(entropy_values)
        max_entropy = max(entropy_values)
    else:
        mean_left = 0.0
        mean_right = 0.0
        hard_mode_0 = 0
        hard_mode_1 = 0
        ambiguous_segments = 0
        mean_max_weight = 0.0
        min_max_weight = 0.0
        mean_entropy = 0.0
        max_entropy = 0.0
    return {
        "segments": len(responsibilities),
        "mean_mode_0": mean_left,
        "mean_mode_1": mean_right,
        "hard_mode_0_count": hard_mode_0,
        "hard_mode_1_count": hard_mode_1,
        "ambiguous_segment_count": ambiguous_segments,
        "ambiguous_segment_threshold": 0.75,
        "mean_max_responsibility": mean_max_weight,
        "min_max_responsibility": min_max_weight,
        "mean_entropy_nats": mean_entropy,
        "max_entropy_nats": max_entropy,
    }


def summarize_traces(traces: list[CartpoleTrace], max_examples: int = 3):
    rewards = [trace.reward for trace in traces]
    lengths = [len(trace.actions) for trace in traces]
    source_counts: dict[str, int] = {}
    for trace in traces:
        source_counts[trace.teacher_source] = source_counts.get(trace.teacher_source, 0) + 1
    examples = []
    for trace in traces[:max_examples]:
        example = {
            "reward": trace.reward,
            "steps": len(trace.actions),
            "switches": sum(
                int(left != right)
                for left, right in zip(trace.mode_labels, trace.mode_labels[1:])
            ),
            "theta_gain": trace.theta_gain,
            "omega_gain": trace.omega_gain,
            "segment_actions": list(trace.segment_actions),
            "segment_durations": list(trace.segment_durations),
            "segment_time_increments": list(trace.segment_time_increments),
            "teacher_source": trace.teacher_source,
            "student_log_probability": trace.student_log_probability,
            "teacher_objective": trace.teacher_objective,
            "teacher_refinement_objective": trace.teacher_refinement_objective,
            "first_observation": trace.observations[0] if trace.observations else None,
            "last_observation": trace.observations[-1] if trace.observations else None,
            "mode_prefix": trace.mode_labels[: min(8, len(trace.mode_labels))],
        }
        if trace.teacher_refinement_elite_summary is not None:
            example["teacher_refinement_elite_summary"] = trace.teacher_refinement_elite_summary
        if trace.teacher_candidate_pool_diagnostics is not None:
            example["teacher_candidate_pool_diagnostics"] = trace.teacher_candidate_pool_diagnostics
        if trace.elite_distribution_fit is not None:
            example["elite_distribution_fit"] = trace.elite_distribution_fit
        examples.append(example)
    return {
        "count": len(traces),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "teacher_source_counts": source_counts,
        "examples": examples,
    }


def serialize_trace(trace: CartpoleTrace):
    payload = {
        "observations": [list(observation) for observation in trace.observations],
        "actions": list(trace.actions),
        "mode_labels": list(trace.mode_labels),
        "reward": trace.reward,
        "theta_gain": trace.theta_gain,
        "omega_gain": trace.omega_gain,
        "segment_actions": list(trace.segment_actions),
        "segment_durations": list(trace.segment_durations),
        "segment_time_increments": list(trace.segment_time_increments),
        "teacher_source": trace.teacher_source,
        "student_log_probability": trace.student_log_probability,
        "teacher_objective": trace.teacher_objective,
        "teacher_refinement_objective": trace.teacher_refinement_objective,
    }
    if trace.teacher_refinement_elite_summary is not None:
        payload["teacher_refinement_elite_summary"] = trace.teacher_refinement_elite_summary
    if trace.teacher_candidate_pool_diagnostics is not None:
        payload["teacher_candidate_pool_diagnostics"] = trace.teacher_candidate_pool_diagnostics
    if trace.elite_distribution_fit is not None:
        payload["elite_distribution_fit"] = trace.elite_distribution_fit
    return payload


def serialize_traces(traces: list[CartpoleTrace]):
    return [serialize_trace(trace) for trace in traces]


def serialize_trace_history(history: list[CartpoleSynthesisIteration]):
    return [
        {
            "iteration": entry.iteration,
            "num_traces": len(entry.traces),
            "traces": serialize_traces(entry.traces),
        }
        for entry in history
    ]


def summarize_student_fit_step(step: CartpoleStudentFitStep):
    transition_distributions = step.transition_switch_parameter_distributions or {}
    return {
        "em_iteration": step.em_iteration,
        "responsibility_pass": step.responsibility_pass,
        "phase": step.phase,
        "trace_log_likelihood": step.trace_log_likelihood,
        "mean_trace_log_likelihood": step.mean_trace_log_likelihood,
        "action_distributions": {
            str(mode): {
                "mean": distribution.mean,
                "std": distribution.std,
            }
            for mode, distribution in sorted(step.action_distributions.items())
        },
        "switch": step.switch.describe(),
        "switch_parameter_distributions": [
            {
                "mean": distribution.mean,
                "std": distribution.std,
            }
            for distribution in step.switch_parameter_distributions
        ],
        "transition_switches": {
            f"{source}->{target}": _directed_transition_description(source, target, switch)
            for (source, target), switch in sorted((step.transition_switches or {}).items())
        },
        "transition_switch_parameter_distributions": {
            f"{source}->{target}": [
                {
                    "mean": distribution.mean,
                    "std": distribution.std,
                }
                for distribution in transition_distributions.get((source, target), [])
            ]
            for source, target in sorted((step.transition_switches or {}).keys())
        },
        "responsibility_summary": summarize_responsibilities(step.responsibilities),
        "switch_pair_responsibility_summary": summarize_switch_pair_responsibilities(
            step.switch_pair_responsibilities
        ),
    }


def summarize_switch_pair_responsibilities(pair_responsibilities):
    if not pair_responsibilities:
        return {
            "pairs": 0,
            "transition_mass": 0.0,
            "stay_mass": 0.0,
            "off_to_on_mass": 0.0,
            "on_to_off_mass": 0.0,
        }
    stay_off = sum(pair[0] for pair in pair_responsibilities)
    off_to_on = sum(pair[1] for pair in pair_responsibilities)
    on_to_off = sum(pair[2] for pair in pair_responsibilities)
    stay_on = sum(pair[3] for pair in pair_responsibilities)
    return {
        "pairs": len(pair_responsibilities),
        "transition_mass": off_to_on + on_to_off,
        "stay_mass": stay_off + stay_on,
        "off_to_on_mass": off_to_on,
        "on_to_off_mass": on_to_off,
    }


def summarize_student_fit_history(history: list[CartpoleStudentFitStep]):
    return [summarize_student_fit_step(step) for step in history]


def _mean_or_none(values: list[float]):
    return sum(values) / len(values) if values else None


def _summary_stats(values: list[float]):
    return {
        "count": len(values),
        "mean": _mean_or_none(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_adaptive_teacher_iteration(
    entry: CartpoleSynthesisIteration,
    cfg: CartpoleSynthesisConfig,
):
    traces = entry.traces
    cem_status = cartpole_teacher_cem_protocol_status(cfg)
    rewards = [trace.reward for trace in traces]
    reward_terms = [cfg.teacher_reward_lambda * trace.reward for trace in traces]
    log_probabilities = [
        trace.student_log_probability
        for trace in traces
        if trace.student_log_probability is not None
    ]
    regularizer_terms = [
        cfg.teacher_student_regularizer * trace.student_log_probability
        for trace in traces
        if trace.student_log_probability is not None
    ]
    teacher_objectives = [
        trace.teacher_objective
        for trace in traces
        if trace.teacher_objective is not None
    ]
    teacher_refinement_objectives = [
        trace.teacher_refinement_objective
        for trace in traces
        if trace.teacher_refinement_objective is not None
    ]
    recorded_objectives = [
        cfg.teacher_reward_lambda * trace.reward
        + cfg.teacher_student_regularizer * trace.student_log_probability
        for trace in traces
        if trace.student_log_probability is not None
    ]
    direct_objective_residuals = [
        trace.teacher_objective
        - (
            cfg.teacher_reward_lambda * trace.reward
            + cfg.teacher_student_regularizer * trace.student_log_probability
        )
        for trace in traces
        if trace.teacher_objective is not None
        and trace.student_log_probability is not None
    ]
    refinement_objective_deltas = [
        trace.teacher_refinement_objective - trace.teacher_objective
        for trace in traces
        if trace.teacher_refinement_objective is not None
        and trace.teacher_objective is not None
    ]
    refinement_elite_summaries = [
        trace.teacher_refinement_elite_summary
        for trace in traces
        if trace.teacher_refinement_elite_summary is not None
    ]
    refinement_elite_counts = [
        float(summary["elite_count"])
        for summary in refinement_elite_summaries
        if isinstance(summary.get("elite_count"), (int, float))
    ]
    refinement_nearest_distances = [
        summary["selected_distance_to_nearest_elite"]
        for summary in refinement_elite_summaries
        if isinstance(summary.get("selected_distance_to_nearest_elite"), (int, float))
    ]
    refinement_kernel_probabilities = [
        summary["selected_elite_kernel_log_probability"]
        for summary in refinement_elite_summaries
        if isinstance(summary.get("selected_elite_kernel_log_probability"), (int, float))
    ]
    source_counts: dict[str, int] = {}
    for trace in traces:
        source_counts[trace.teacher_source] = source_counts.get(trace.teacher_source, 0) + 1

    return {
        "iteration": entry.iteration,
        "teacher_sampling_model": (
            "bootstrap_probabilistic_prior"
            if entry.iteration == 1
            else "previous_iteration_student"
        ),
        "teacher_objective_formula": (
            "teacher_reward_lambda * reward + "
            "teacher_student_regularizer * recorded_student_log_probability"
        ),
        "teacher_reward_lambda": cfg.teacher_reward_lambda,
        "teacher_student_regularizer": cfg.teacher_student_regularizer,
        "trace_count": len(traces),
        "candidate_rollouts": cfg.candidate_rollouts,
        "effective_candidate_rollouts": cem_status["effective_teacher_candidate_rollouts"],
        "selected_top_rho": cfg.teacher_top_rho,
        "effective_top_rho": cem_status["effective_teacher_top_rho"],
        "paper_top_rho": cem_status["paper_teacher_top_rho"],
        "uses_paper_top_rho": cem_status["uses_paper_teacher_top_rho"],
        "candidate_rollouts_cover_selected_top_rho": cem_status[
            "teacher_candidate_rollouts_cover_selected_top_rho"
        ],
        "candidate_rollouts_cover_paper_top_rho": cem_status[
            "teacher_candidate_rollouts_cover_paper_top_rho"
        ],
        "cem_phase_matches_paper_rho": cem_status["teacher_cem_phase_matches_paper_rho"],
        "teacher_source_counts": source_counts,
        "refinement_elite_summary": {
            "count": len(refinement_elite_summaries),
            "elite_count": _summary_stats(refinement_elite_counts),
            "selected_distance_to_nearest_elite": _summary_stats(refinement_nearest_distances),
            "selected_elite_kernel_log_probability": _summary_stats(refinement_kernel_probabilities),
        },
        "objective_component_summary": {
            "reward_term": _summary_stats(reward_terms),
            "student_log_probability": _summary_stats(log_probabilities),
            "student_regularizer_term": _summary_stats(regularizer_terms),
            "direct_objective": _summary_stats(teacher_objectives),
            "direct_objective_formula_residual": _summary_stats(direct_objective_residuals),
            "refinement_objective": _summary_stats(teacher_refinement_objectives),
            "refinement_minus_direct_objective": _summary_stats(refinement_objective_deltas),
        },
        "reward_mean": _mean_or_none(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "recorded_student_log_probability_count": len(log_probabilities),
        "recorded_student_log_probability_fraction": (
            len(log_probabilities) / len(traces) if traces else 0.0
        ),
        "recorded_student_log_probability_mean": _mean_or_none(log_probabilities),
        "recorded_student_log_probability_min": (
            min(log_probabilities) if log_probabilities else None
        ),
        "recorded_student_log_probability_max": (
            max(log_probabilities) if log_probabilities else None
        ),
        "recorded_teacher_objective_direct_count": len(teacher_objectives),
        "recorded_teacher_objective_direct_fraction": (
            len(teacher_objectives) / len(traces) if traces else 0.0
        ),
        "recorded_teacher_objective_direct_mean": _mean_or_none(teacher_objectives),
        "recorded_teacher_objective_direct_min": (
            min(teacher_objectives) if teacher_objectives else None
        ),
        "recorded_teacher_objective_direct_max": (
            max(teacher_objectives) if teacher_objectives else None
        ),
        "recorded_teacher_refinement_objective_count": len(teacher_refinement_objectives),
        "recorded_teacher_refinement_objective_fraction": (
            len(teacher_refinement_objectives) / len(traces) if traces else 0.0
        ),
        "recorded_teacher_refinement_objective_mean": _mean_or_none(teacher_refinement_objectives),
        "recorded_teacher_refinement_objective_min": (
            min(teacher_refinement_objectives) if teacher_refinement_objectives else None
        ),
        "recorded_teacher_refinement_objective_max": (
            max(teacher_refinement_objectives) if teacher_refinement_objectives else None
        ),
        "recorded_teacher_objective_mean": _mean_or_none(recorded_objectives),
        "recorded_teacher_objective_min": (
            min(recorded_objectives) if recorded_objectives else None
        ),
        "recorded_teacher_objective_max": (
            max(recorded_objectives) if recorded_objectives else None
        ),
        "recorded_teacher_objective_covers_all_traces": (
            len(recorded_objectives) == len(traces)
        ),
    }


def summarize_adaptive_teacher_history(
    history: list[CartpoleSynthesisIteration],
    cfg: CartpoleSynthesisConfig,
):
    return [summarize_adaptive_teacher_iteration(entry, cfg) for entry in history]


def summarize_synthesis_history(
    history: list[CartpoleSynthesisIteration],
    eval_rollouts: int | None = None,
    test_max_steps: int | None = None,
    train_seed: int = 100,
    test_seed: int = 200,
    cfg: CartpoleSynthesisConfig | None = None,
):
    rows = []
    for entry in history:
        row = {
            "iteration": entry.iteration,
            "trace_summary": summarize_traces(entry.traces, max_examples=1),
            "probabilistic_student": summarize_student(entry.student),
            "student_fit_history": summarize_student_fit_history(entry.student_fit_history),
            "switch_fit_diagnostics": cartpole_switch_fit_diagnostics(entry.traces, entry.student),
        }
        if cfg is not None:
            row["adaptive_teacher_summary"] = summarize_adaptive_teacher_iteration(entry, cfg)
        if eval_rollouts is not None and test_max_steps is not None:
            row["evaluation"] = summarize_policy_evaluation(
                entry.student.to_deterministic_policy(),
                eval_rollouts,
                test_max_steps,
                train_seed,
                test_seed,
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a Cartpole programmatic state machine.")
    default_cfg = CartpoleSynthesisConfig()
    parser.add_argument("--num-initial-states", type=int, default=32)
    parser.add_argument("--candidate-rollouts", type=int, default=128)
    parser.add_argument("--segment-steps", type=int, default=default_cfg.segment_steps)
    parser.add_argument("--segments-per-trace", type=int, default=default_cfg.segments_per_trace)
    parser.add_argument("--teacher-theta-gain", type=float, default=default_cfg.teacher_theta_gain)
    parser.add_argument("--teacher-omega-gain", type=float, default=default_cfg.teacher_omega_gain)
    parser.add_argument("--teacher-student-iters", type=int, default=default_cfg.teacher_student_iters)
    parser.add_argument("--student-em-iters", type=int, default=default_cfg.student_em_iters)
    parser.add_argument(
        "--student-switch-responsibility-passes",
        type=int,
        default=default_cfg.student_switch_responsibility_passes,
    )
    parser.add_argument("--teacher-student-regularizer", type=float, default=default_cfg.teacher_student_regularizer)
    parser.add_argument("--teacher-reward-lambda", type=float, default=default_cfg.teacher_reward_lambda)
    parser.add_argument("--teacher-top-rho", type=int, default=default_cfg.teacher_top_rho)
    parser.add_argument("--teacher-refinement-steps", type=int, default=default_cfg.teacher_refinement_steps)
    parser.add_argument(
        "--teacher-elite-distribution-resamples",
        type=int,
        default=default_cfg.teacher_elite_distribution_resamples,
    )
    parser.add_argument(
        "--teacher-elite-distribution-rounds",
        type=int,
        default=default_cfg.teacher_elite_distribution_rounds,
    )
    parser.add_argument("--parallel-trace-workers", type=int, default=default_cfg.parallel_trace_workers)
    parser.add_argument("--parallel-switch-workers", type=int, default=default_cfg.parallel_switch_workers)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15000)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--traces-output", default=None)
    args = parser.parse_args()

    cfg = CartpoleSynthesisConfig(
        num_initial_states=args.num_initial_states,
        candidate_rollouts=args.candidate_rollouts,
        segment_steps=args.segment_steps,
        segments_per_trace=args.segments_per_trace,
        teacher_theta_gain=args.teacher_theta_gain,
        teacher_omega_gain=args.teacher_omega_gain,
        teacher_student_iters=args.teacher_student_iters,
        student_em_iters=args.student_em_iters,
        student_switch_responsibility_passes=args.student_switch_responsibility_passes,
        teacher_student_regularizer=args.teacher_student_regularizer,
        teacher_reward_lambda=args.teacher_reward_lambda,
        teacher_top_rho=args.teacher_top_rho,
        teacher_refinement_steps=args.teacher_refinement_steps,
        teacher_elite_distribution_resamples=args.teacher_elite_distribution_resamples,
        teacher_elite_distribution_rounds=args.teacher_elite_distribution_rounds,
        parallel_trace_workers=args.parallel_trace_workers,
        parallel_switch_workers=args.parallel_switch_workers,
        seed=args.seed,
    )
    student, traces, synthesis_history = synthesize_cartpole_student_with_history(cfg)
    policy = student.to_deterministic_policy()
    evaluation = summarize_policy_evaluation(policy, args.eval_rollouts, args.test_max_steps)
    train = evaluation["train"]
    test = evaluation["test"]
    metrics = {
        "command": " ".join(sys.argv),
        "config": asdict(cfg),
        "algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "paper_protocol_status": cartpole_synthesis_protocol_status(
            cfg,
            args.eval_rollouts,
            args.test_max_steps,
        ),
        "eval_rollouts": args.eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": args.eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "test_max_steps": args.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "num_traces": len(traces),
        "traces_output": args.traces_output,
        "adaptive_teacher_summary": summarize_adaptive_teacher_history(
            synthesis_history,
            cfg,
        ),
        "synthesis_history": summarize_synthesis_history(
            synthesis_history,
            args.eval_rollouts,
            args.test_max_steps,
            cfg=cfg,
        ),
        "trace_summary": summarize_traces(traces),
        "policy_description": policy.describe(),
        "probabilistic_student": summarize_student(student),
        "switch_fit_diagnostics": cartpole_switch_fit_diagnostics(traces, student),
        "train": train,
        "test": test,
    }
    if args.metrics_output is not None:
        metrics_dir = os.path.dirname(args.metrics_output)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    if args.traces_output is not None:
        traces_dir = os.path.dirname(args.traces_output)
        if traces_dir:
            os.makedirs(traces_dir, exist_ok=True)
        with open(args.traces_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "command": " ".join(sys.argv),
                    "config": asdict(cfg),
                    "num_traces": len(traces),
                    "traces": serialize_traces(traces),
                    "trace_history": serialize_trace_history(synthesis_history),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    print("Synthesized Cartpole programmatic state machine")
    print(f"  traces={len(traces)}")
    print(f"  policy={policy.describe()}")
    print(f"  train_success_rate={train['success_rate']:.3f}")
    print(f"  test_success_rate={test['success_rate']:.3f}")
    print(f"  train_reward_mean={train['reward_mean']:.1f}")
    print(f"  test_reward_mean={test['reward_mean']:.1f}")
    if args.metrics_output is not None:
        print(f"  metrics={args.metrics_output}")
    if args.traces_output is not None:
        print(f"  traces={args.traces_output}")


if __name__ == "__main__":
    main()
