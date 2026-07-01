from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from cartpole_env import CartpoleEnv
from cartpole_synthesis import CartpoleSynthesisConfig, CartpoleTrace, ProbabilisticCartpoleStudent, synthesize_cartpole_student


def summarize_rollouts(results):
    return {
        "success_rate": sum(result.success for result in results) / len(results),
        "reward_mean": sum(result.reward for result in results) / len(results),
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
    return {
        "count": len(traces),
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
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
                "segment_durations": list(trace.segment_durations),
                "first_observation": trace.observations[0] if trace.observations else None,
                "last_observation": trace.observations[-1] if trace.observations else None,
                "mode_prefix": trace.mode_labels[: min(8, len(trace.mode_labels))],
            }
            for trace in traces[:max_examples]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a Cartpole programmatic state machine.")
    default_cfg = CartpoleSynthesisConfig()
    parser.add_argument("--num-initial-states", type=int, default=32)
    parser.add_argument("--candidate-rollouts", type=int, default=128)
    parser.add_argument("--segment-steps", type=int, default=8)
    parser.add_argument("--segments-per-trace", type=int, default=32)
    parser.add_argument("--teacher-theta-gain", type=float, default=default_cfg.teacher_theta_gain)
    parser.add_argument("--teacher-omega-gain", type=float, default=default_cfg.teacher_omega_gain)
    parser.add_argument("--teacher-student-iters", type=int, default=default_cfg.teacher_student_iters)
    parser.add_argument("--teacher-student-regularizer", type=float, default=default_cfg.teacher_student_regularizer)
    parser.add_argument("--teacher-reward-lambda", type=float, default=default_cfg.teacher_reward_lambda)
    parser.add_argument("--teacher-top-rho", type=int, default=default_cfg.teacher_top_rho)
    parser.add_argument("--teacher-refinement-steps", type=int, default=default_cfg.teacher_refinement_steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-rollouts", type=int, default=20)
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
        teacher_student_regularizer=args.teacher_student_regularizer,
        teacher_reward_lambda=args.teacher_reward_lambda,
        teacher_top_rho=args.teacher_top_rho,
        teacher_refinement_steps=args.teacher_refinement_steps,
        seed=args.seed,
    )
    student, traces = synthesize_cartpole_student(cfg)
    policy = student.to_deterministic_policy()
    train_env = CartpoleEnv.train_env(seed=100)
    test_env = CartpoleEnv.test_env(seed=200)
    train_results = [train_env.rollout(policy) for _ in range(args.eval_rollouts)]
    test_results = [test_env.rollout(policy, max_steps=args.test_max_steps) for _ in range(args.eval_rollouts)]
    train = summarize_rollouts(train_results)
    test = summarize_rollouts(test_results)
    metrics = {
        "command": " ".join(sys.argv),
        "config": asdict(cfg),
        "eval_rollouts": args.eval_rollouts,
        "test_max_steps": args.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "num_traces": len(traces),
        "trace_summary": summarize_traces(traces),
        "policy_description": policy.describe(),
        "probabilistic_student": summarize_student(student),
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
