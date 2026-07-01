from __future__ import annotations

import argparse
import json
import os
import sys

from cartpole_direct_opt import (
    DirectOptConfig,
    direct_opt_metrics,
    run_cartpole_direct_opt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CartPole Direct-Opt diagnostic baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-states", type=int, default=10)
    parser.add_argument("--random-candidates", type=int, default=256)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    cfg = DirectOptConfig(
        seed=args.seed,
        num_train_states=2 if args.quick else args.num_train_states,
        random_candidates=8 if args.quick else args.random_candidates,
        eval_rollouts=args.eval_rollouts,
        test_max_steps=args.test_max_steps,
        quick=args.quick,
    )
    result = run_cartpole_direct_opt(cfg)
    metrics = {"command": " ".join(sys.argv), **direct_opt_metrics(result)}

    metrics_dir = os.path.dirname(args.metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print("CartPole Direct-Opt diagnostic")
    print(f"  policy={result.policy.describe()}")
    print(f"  searched_candidates={result.searched_candidates}")
    print(f"  train_success_rate={result.train_success_rate:.3f}")
    print(f"  test_success_rate={result.test_success_rate:.3f}")
    print(f"  train_reward_mean={result.train_reward_mean:.1f}")
    print(f"  test_reward_mean={result.test_reward_mean:.1f}")
    print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
