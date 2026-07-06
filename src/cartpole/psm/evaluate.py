from __future__ import annotations

from cartpole_env import BangBangCartpolePSM, evaluate_cartpole_policy


def main() -> None:
    policy = BangBangCartpolePSM(force=10.0)
    metrics = evaluate_cartpole_policy(policy, train_rollouts=20, test_rollouts=20, test_max_steps=5_000)
    print("Cartpole programmatic state-machine policy")
    print(f"  policy={policy.describe()}")
    print(f"  train_success_rate={metrics['train_success_rate']:.3f}")
    print(f"  test_success_rate={metrics['test_success_rate']:.3f}")


if __name__ == "__main__":
    main()
