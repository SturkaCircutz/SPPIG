from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Keep this script runnable from a fresh checkout without requiring package install.
sys.path.insert(0, str(SRC))

from cartpole_env import (  # noqa: E402
    PAPER_EVAL_ROLLOUTS,
    CartpoleEnv,
    PaperFigure19CartpolePSM,
    cartpole_paper_figure19_policy_spec,
    cartpole_reward_spec,
    cartpole_space_spec,
    summarize_cartpole_results,
)
from cartpole_synthesis import Depth2Switch, SynthesizedCartpolePSM  # noqa: E402


def summarize_rollouts(results):
    return summarize_cartpole_results(results)


def fixed_program_protocol_status(
    eval_rollouts: int,
    test_max_steps: int,
    policy_source: str = "fixed_two_mode_program_parameters",
) -> dict[str, object]:
    train_env = CartpoleEnv.train_env()
    test_env = CartpoleEnv.test_env()
    return {
        "artifact_kind": "fixed_cartpole_program_reevaluation",
        "policy_source": policy_source,
        "synthesized_by_current_algorithm": False,
        "full_probabilistic_adaptive_teaching": False,
        "train_horizon_seconds": train_env.cfg.horizon_seconds,
        "train_pole_length": train_env.cfg.pole_length,
        "training_horizon_steps": train_env.cfg.max_steps,
        "test_horizon_seconds": test_env.cfg.horizon_seconds,
        "test_pole_length": test_env.cfg.pole_length,
        "paper_test_horizon_steps": test_env.cfg.max_steps,
        "selected_test_max_steps": test_max_steps,
        "uses_full_test_horizon": test_max_steps == test_env.cfg.max_steps,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": eval_rollouts,
        "uses_paper_eval_rollouts": eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(train_env.cfg),
        "paper_scale_fixed_program_result": False,
        "limitation": (
            "Reevaluates a fixed two-mode CartPole program under the requested horizon and "
            "rollout count; it is not evidence that the current synthesis implementation "
            "reproduced the paper's probabilistic adaptive-teaching result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed two-mode CartPole program.")
    parser.add_argument("--paper-figure19", action="store_true")
    parser.add_argument("--theta-weight", type=float)
    parser.add_argument("--omega-weight", type=float)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--left-force", type=float, default=-10.0)
    parser.add_argument("--right-force", type=float, default=10.0)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    if args.paper_figure19:
        policy = PaperFigure19CartpolePSM()
        policy_source = "paper_figure19_manual_transcription"
        program_parameters = cartpole_paper_figure19_policy_spec()
    else:
        if args.theta_weight is None or args.omega_weight is None:
            parser.error("--theta-weight and --omega-weight are required unless --paper-figure19 is set")
        policy = SynthesizedCartpolePSM(
            args.left_force,
            args.right_force,
            Depth2Switch(args.theta_weight, args.omega_weight, args.threshold),
        )
        policy_source = "fixed_two_mode_program_parameters"
        program_parameters = {
            "theta_weight": args.theta_weight,
            "omega_weight": args.omega_weight,
            "threshold": args.threshold,
            "left_force": args.left_force,
            "right_force": args.right_force,
        }

    train_env = CartpoleEnv.train_env(seed=100)
    test_env = CartpoleEnv.test_env(seed=200)
    train = summarize_rollouts([train_env.rollout(policy) for _ in range(args.eval_rollouts)])
    test = summarize_rollouts([test_env.rollout(policy, max_steps=args.test_max_steps) for _ in range(args.eval_rollouts)])
    metrics = {
        "command": " ".join(sys.argv),
        "policy_description": policy.describe(),
        "eval_rollouts": args.eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": args.eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(train_env.cfg),
        "paper_protocol_status": fixed_program_protocol_status(
            args.eval_rollouts,
            args.test_max_steps,
            policy_source,
        ),
        "test_max_steps": args.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "program_parameters": program_parameters,
        "train": train,
        "test": test,
    }

    metrics_dir = os.path.dirname(args.metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print("CartPole fixed program evaluation")
    print(f"  policy={policy.describe()}")
    print(f"  train_success_rate={train['success_rate']:.3f}")
    print(f"  test_success_rate={test['success_rate']:.3f}")
    print(f"  train_reward_mean={train['reward_mean']:.1f}")
    print(f"  test_reward_mean={test['reward_mean']:.1f}")
    print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
