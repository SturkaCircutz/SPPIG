from __future__ import annotations

import argparse
import json
import os
import sys

from cartpole_env import PAPER_EVAL_ROLLOUTS
from cartpole_direct_opt import (
    DirectOptConfig,
    direct_opt_metrics,
    run_cartpole_direct_opt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CartPole Direct-Opt diagnostic baseline.")
    defaults = DirectOptConfig()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-states", type=int, default=None)
    parser.add_argument("--random-candidates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-refinement-rounds", type=int, default=None)
    parser.add_argument("--local-refinement-steps", type=int, default=None)
    parser.add_argument("--restart-candidates-on-stall", type=int, default=None)
    parser.add_argument("--local-step-fraction", type=float, default=defaults.local_step_fraction)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    cfg = DirectOptConfig(
        seed=args.seed,
        num_train_states=(
            args.num_train_states
            if args.num_train_states is not None
            else (2 if args.quick else defaults.num_train_states)
        ),
        random_candidates=(
            args.random_candidates
            if args.random_candidates is not None
            else (8 if args.quick else defaults.random_candidates)
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else (2 if args.quick else defaults.batch_size)
        ),
        batch_refinement_rounds=(
            args.batch_refinement_rounds
            if args.batch_refinement_rounds is not None
            else defaults.batch_refinement_rounds
        ),
        local_refinement_steps=(
            args.local_refinement_steps
            if args.local_refinement_steps is not None
            else (1 if args.quick else defaults.local_refinement_steps)
        ),
        restart_candidates_on_stall=(
            args.restart_candidates_on_stall
            if args.restart_candidates_on_stall is not None
            else defaults.restart_candidates_on_stall
        ),
        local_step_fraction=args.local_step_fraction,
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
    print(f"  batch_count={result.search_diagnostics['batch_count']}")
    print(f"  train_success_rate={result.train_success_rate:.3f}")
    print(f"  test_success_rate={result.test_success_rate:.3f}")
    print(f"  train_reward_mean={result.train_reward_mean:.1f}")
    print(f"  test_reward_mean={result.test_reward_mean:.1f}")
    print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
