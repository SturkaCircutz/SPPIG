from __future__ import annotations

import argparse
import os

from cartpole_env import PAPER_EVAL_ROLLOUTS
from ppo_cartpole import PAPER_PPO_TIMESTEPS, PPOConfig, train_ppo_cartpole


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper-style Cartpole PPO baselines.")
    parser.add_argument("--policy", choices=("mlp", "lstm"), default="mlp")
    parser.add_argument("--timesteps", type=int, default=PAPER_PPO_TIMESTEPS)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--initial-log-std", type=float, default=0.0)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15000)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=0.001)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--no-keep-best", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/cartpole_ppo.pt")
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    cfg = PPOConfig(
        policy_type=args.policy,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatches=1 if args.policy == "lstm" else args.minibatches,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        initial_log_std=args.initial_log_std,
        eval_rollouts=args.eval_rollouts,
        eval_test_max_steps=args.test_max_steps,
        pretrain_steps=args.pretrain_steps,
        pretrain_learning_rate=args.pretrain_learning_rate,
        num_envs=args.num_envs,
        hidden_size=args.hidden_size,
        eval_interval=args.eval_interval,
        keep_best=not args.no_keep_best,
        verbose=args.verbose,
        metrics_output=args.metrics_output,
        seed=args.seed,
    )
    _, result = train_ppo_cartpole(cfg, output=args.output)
    print("Cartpole PPO")
    print(f"  policy={args.policy}")
    print(f"  timesteps={result.timesteps}")
    print(f"  train_success_rate={result.train_success_rate:.3f}")
    print(f"  test_success_rate={result.test_success_rate:.3f}")
    print(f"  train_reward_mean={result.train_reward_mean:.1f}")
    print(f"  test_reward_mean={result.test_reward_mean:.1f}")
    print(f"  saved_model={args.output}")
    metrics_output = args.metrics_output or f"{os.path.splitext(args.output)[0]}_metrics.json"
    print(f"  metrics={metrics_output}")


if __name__ == "__main__":
    main()
