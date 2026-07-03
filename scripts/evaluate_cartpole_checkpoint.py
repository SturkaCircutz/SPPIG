from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Keep this script runnable from a fresh checkout without requiring package install.
sys.path.insert(0, str(SRC))

from cartpole_env import PAPER_EVAL_ROLLOUTS, CartpoleEnv, cartpole_reward_spec, cartpole_space_spec  # noqa: E402
from ppo_cartpole import (  # noqa: E402
    PAPER_PPO_TIMESTEPS,
    LSTMActorCritic,
    MLPActorCritic,
    evaluate_ppo_model,
    result_to_metrics,
)


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if config["policy_type"] == "mlp":
        model = MLPActorCritic(
            config["hidden_size"],
            config.get("initial_log_std", 0.0),
            config.get("action_scale", 10.0),
        )
    elif config["policy_type"] == "lstm":
        model = LSTMActorCritic(
            config["hidden_size"],
            config.get("initial_log_std", 0.0),
            config.get("action_scale", 10.0),
        )
    else:
        raise ValueError("checkpoint config policy_type must be 'mlp' or 'lstm'")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model


def checkpoint_reevaluation_protocol_status(
    checkpoint_config: dict,
    eval_rollouts: int,
    test_max_steps: int,
) -> dict[str, object]:
    paper_test_steps = CartpoleEnv.test_env().cfg.max_steps
    checkpoint_timesteps = int(checkpoint_config.get("total_timesteps", 0))
    checkpoint_eval_steps = int(checkpoint_config.get("eval_test_max_steps", 0))
    return {
        "artifact_kind": "ppo_checkpoint_reevaluation",
        "policy_type": checkpoint_config.get("policy_type"),
        "paper_timestep_budget": PAPER_PPO_TIMESTEPS,
        "checkpoint_total_timesteps": checkpoint_timesteps,
        "checkpoint_uses_paper_timestep_budget": checkpoint_timesteps == PAPER_PPO_TIMESTEPS,
        "checkpoint_eval_test_max_steps": checkpoint_eval_steps,
        "checkpoint_eval_used_full_test_horizon": checkpoint_eval_steps == paper_test_steps,
        "paper_test_horizon_steps": paper_test_steps,
        "selected_test_max_steps": test_max_steps,
        "reevaluation_uses_full_test_horizon": test_max_steps == paper_test_steps,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": eval_rollouts,
        "uses_paper_eval_rollouts": eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "paper_scale_checkpoint_result": False,
        "limitation": (
            "Reevaluates an existing local PPO checkpoint under the requested horizon and rollout "
            "count; it does not turn a short or warm-started checkpoint into the paper's full "
            "10^7-timestep, five-seed PPO/PPO-LSTM baseline protocol."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reevaluate a CartPole PPO checkpoint and write metrics JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint, model = load_model(checkpoint_path)
    prior_result = checkpoint.get("result", {})
    timesteps = int(prior_result.get("timesteps", checkpoint["config"].get("total_timesteps", 0)))
    result = evaluate_ppo_model(
        model,
        timesteps=timesteps,
        rollouts=args.eval_rollouts,
        test_max_steps=args.test_max_steps,
    )
    metrics = {
        "command": " ".join(sys.argv),
        "checkpoint": str(checkpoint_path),
        "checkpoint_config": checkpoint["config"],
        "checkpoint_result": prior_result,
        "eval_rollouts": args.eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": args.eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "paper_protocol_status": checkpoint_reevaluation_protocol_status(
            checkpoint["config"],
            args.eval_rollouts,
            args.test_max_steps,
        ),
        "test_max_steps": args.test_max_steps,
        "paper_test_horizon_steps": 15_000,
        "selected_result": result_to_metrics(result),
    }

    metrics_dir = os.path.dirname(args.metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print("CartPole checkpoint evaluation")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  train_success_rate={result.train_success_rate:.3f}")
    print(f"  test_success_rate={result.test_success_rate:.3f}")
    print(f"  train_reward_mean={result.train_reward_mean:.1f}")
    print(f"  test_reward_mean={result.test_reward_mean:.1f}")
    print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
