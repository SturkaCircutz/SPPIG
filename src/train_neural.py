from __future__ import annotations

import argparse
import os

from torch_neural_policy import (
    TorchTrainConfig,
    evaluate_torch_policy,
    save_checkpoint,
    train_torch_behavior_cloning,
)
from shuttle_env import ShuttleLineEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch neural shuttle baseline.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-traces", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", choices=("lstm", "mlp"), default="lstm")
    parser.add_argument("--output", default="artifacts/torch_neural_shuttle.pt")
    args = parser.parse_args()

    env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6, 8])
    cfg = TorchTrainConfig(
        model=args.model,
        epochs=args.epochs,
        num_traces=args.num_traces,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
    )
    model, result = train_torch_behavior_cloning(env, cfg)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_checkpoint(args.output, model, cfg, result)

    print("PyTorch neural behavior-cloning training")
    print(f"  model={args.model}")
    print(f"  examples={result.num_examples}")
    print(f"  initial_loss={result.initial_loss:.4f}")
    print(f"  final_loss={result.final_loss:.4f}")
    print(f"  saved_model={args.output}")

    print("\nEvaluation")
    print("crossings | split | success | steps | reward | final_mode")
    for split, crossings_set in (("train", env.train_crossings), ("test", env.test_crossings)):
        for crossings in crossings_set:
            evaluation = evaluate_torch_policy(env, model, crossings)
            print(
                f"{crossings:9d} | {split:5s} | {str(evaluation.success):7s} | "
                f"{evaluation.steps:5d} | {evaluation.reward:.1f}    | {evaluation.final_mode}"
            )


if __name__ == "__main__":
    main()
