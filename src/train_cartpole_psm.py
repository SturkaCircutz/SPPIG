from __future__ import annotations

import argparse
import json
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
    CartpoleTrace,
    ProbabilisticCartpoleStudent,
    cartpole_synthesis_algorithm_provenance,
    cartpole_synthesis_protocol_status,
    cartpole_switch_fit_diagnostics,
    synthesize_cartpole_student_with_history,
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
    responsibilities = student.responsibilities
    if responsibilities:
        mean_left = sum(left for left, _ in responsibilities) / len(responsibilities)
        mean_right = sum(right for _, right in responsibilities) / len(responsibilities)
    else:
        mean_left = 0.0
        mean_right = 0.0
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
        "responsibility_summary": {
            "segments": len(responsibilities),
            "mean_mode_0": mean_left,
            "mean_mode_1": mean_right,
        },
    }


def summarize_traces(traces: list[CartpoleTrace], max_examples: int = 3):
    rewards = [trace.reward for trace in traces]
    lengths = [len(trace.actions) for trace in traces]
    source_counts: dict[str, int] = {}
    for trace in traces:
        source_counts[trace.teacher_source] = source_counts.get(trace.teacher_source, 0) + 1
    return {
        "count": len(traces),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
        "teacher_source_counts": source_counts,
        "examples": [
            {
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
                "first_observation": trace.observations[0] if trace.observations else None,
                "last_observation": trace.observations[-1] if trace.observations else None,
                "mode_prefix": trace.mode_labels[: min(8, len(trace.mode_labels))],
            }
            for trace in traces[:max_examples]
        ],
    }


def _mean_or_none(values: list[float]):
    return sum(values) / len(values) if values else None


def summarize_adaptive_teacher_iteration(
    entry: CartpoleSynthesisIteration,
    cfg: CartpoleSynthesisConfig,
):
    traces = entry.traces
    rewards = [trace.reward for trace in traces]
    log_probabilities = [
        trace.student_log_probability
        for trace in traces
        if trace.student_log_probability is not None
    ]
    recorded_objectives = [
        cfg.teacher_reward_lambda * trace.reward
        + cfg.teacher_student_regularizer * trace.student_log_probability
        for trace in traces
        if trace.student_log_probability is not None
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
        "teacher_source_counts": source_counts,
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15000)
    parser.add_argument("--metrics-output", default=None)
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

    print("Synthesized Cartpole programmatic state machine")
    print(f"  traces={len(traces)}")
    print(f"  policy={policy.describe()}")
    print(f"  train_success_rate={train['success_rate']:.3f}")
    print(f"  test_success_rate={test['success_rate']:.3f}")
    print(f"  train_reward_mean={train['reward_mean']:.1f}")
    print(f"  test_reward_mean={test['reward_mean']:.1f}")
    if args.metrics_output is not None:
        print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
