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

from cartpole_env import CartpoleEnv  # noqa: E402
from cartpole_synthesis import Depth2Switch, SynthesizedCartpolePSM  # noqa: E402


def summarize_rollouts(results):
    return {
        "success_rate": sum(result.success for result in results) / len(results),
        "reward_mean": sum(result.reward for result in results) / len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fixed two-mode CartPole program.")
    parser.add_argument("--theta-weight", type=float, required=True)
    parser.add_argument("--omega-weight", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--left-force", type=float, default=-10.0)
    parser.add_argument("--right-force", type=float, default=10.0)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    policy = SynthesizedCartpolePSM(
        args.left_force,
        args.right_force,
        Depth2Switch(args.theta_weight, args.omega_weight, args.threshold),
    )
    train_env = CartpoleEnv.train_env(seed=100)
    test_env = CartpoleEnv.test_env(seed=200)
    train = summarize_rollouts([train_env.rollout(policy) for _ in range(args.eval_rollouts)])
    test = summarize_rollouts([test_env.rollout(policy, max_steps=args.test_max_steps) for _ in range(args.eval_rollouts)])
    metrics = {
        "command": " ".join(sys.argv),
        "policy_description": policy.describe(),
        "eval_rollouts": args.eval_rollouts,
        "test_max_steps": args.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "program_parameters": {
            "theta_weight": args.theta_weight,
            "omega_weight": args.omega_weight,
            "threshold": args.threshold,
            "left_force": args.left_force,
            "right_force": args.right_force,
        },
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
