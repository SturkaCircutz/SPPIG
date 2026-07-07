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
    CARTPOLE_PSM_MODE_UPDATE_ORDER,
    CARTPOLE_PSM_PRETRAIN_TEACHER_POLICY,
    PAPER_EVAL_ROLLOUTS,
    CartpoleEnv,
    cartpole_reward_spec,
    cartpole_space_spec,
)

PAPER_PPO_TIMESTEPS = 10_000_000


def _load_ppo_runtime():
    try:
        import torch
        from ppo_cartpole import LSTMActorCritic, MLPActorCritic, evaluate_ppo_model, result_to_metrics
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError("PyTorch is required to reevaluate PPO checkpoints") from exc
        raise
    return torch, LSTMActorCritic, MLPActorCritic, evaluate_ppo_model, result_to_metrics


def load_model(checkpoint_path: Path):
    torch, LSTMActorCritic, MLPActorCritic, _, _ = _load_ppo_runtime()
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
    pretrain_steps = int(checkpoint_config.get("pretrain_steps", 0))
    recorded_teacher_policy = checkpoint_config.get("pretrain_teacher_policy")
    recorded_teacher_order = checkpoint_config.get("pretrain_teacher_mode_update_order")
    if pretrain_steps > 0:
        teacher_policy_matches = recorded_teacher_policy == CARTPOLE_PSM_PRETRAIN_TEACHER_POLICY
        teacher_order_matches = recorded_teacher_order == CARTPOLE_PSM_MODE_UPDATE_ORDER
        if not recorded_teacher_policy:
            teacher_policy_status = "missing_from_checkpoint_config"
        elif teacher_policy_matches:
            teacher_policy_status = "recorded_matches_current_implementation"
        else:
            teacher_policy_status = "recorded_mismatch_current_implementation"
        if not recorded_teacher_order:
            teacher_order_status = "missing_from_checkpoint_config"
        elif teacher_order_matches:
            teacher_order_status = "recorded_matches_current_implementation"
        else:
            teacher_order_status = "recorded_mismatch_current_implementation"
    else:
        teacher_policy_matches = True
        teacher_order_matches = True
        teacher_order_status = "not_applicable_no_pretraining"
        teacher_policy_status = "not_applicable_no_pretraining"
    return {
        "artifact_kind": "ppo_checkpoint_reevaluation",
        "policy_type": checkpoint_config.get("policy_type"),
        "checkpoint_training_reused": True,
        "training_launched": False,
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
        "checkpoint_pretrain_steps": pretrain_steps,
        "current_pretrain_teacher_policy": CARTPOLE_PSM_PRETRAIN_TEACHER_POLICY if pretrain_steps > 0 else None,
        "checkpoint_pretrain_teacher_policy": recorded_teacher_policy,
        "checkpoint_pretrain_teacher_policy_status": teacher_policy_status,
        "checkpoint_pretrain_teacher_policy_recorded": pretrain_steps == 0 or bool(recorded_teacher_policy),
        "checkpoint_pretrain_teacher_policy_matches_current_implementation": teacher_policy_matches,
        "current_pretrain_teacher_mode_update_order": CARTPOLE_PSM_MODE_UPDATE_ORDER if pretrain_steps > 0 else None,
        "checkpoint_pretrain_teacher_mode_update_order": recorded_teacher_order,
        "checkpoint_pretrain_teacher_mode_order_status": teacher_order_status,
        "checkpoint_pretrain_teacher_mode_order_recorded": pretrain_steps == 0 or bool(recorded_teacher_order),
        "checkpoint_pretrain_teacher_mode_order_matches_current_implementation": teacher_order_matches,
        "paper_scale_checkpoint_result": False,
        "limitation": (
            "Reevaluates an existing local PPO checkpoint under the requested horizon and rollout "
            "count; it does not turn a short or warm-started checkpoint into the paper's full "
            "10^7-timestep, five-seed PPO/PPO-LSTM baseline protocol. Warm-start checkpoints "
            "created before explicit teacher-policy and teacher-order provenance cannot prove which "
            "pretraining teacher policy or state-machine update order was used."
        ),
    }


def evaluate_checkpoint_metrics(
    checkpoint_path: Path,
    eval_rollouts: int,
    test_max_steps: int,
    command: str,
) -> dict[str, object]:
    checkpoint, model = load_model(checkpoint_path)
    _, _, _, evaluate_ppo_model, result_to_metrics = _load_ppo_runtime()
    prior_result = checkpoint.get("result", {})
    timesteps = int(prior_result.get("timesteps", checkpoint["config"].get("total_timesteps", 0)))
    result = evaluate_ppo_model(
        model,
        timesteps=timesteps,
        rollouts=eval_rollouts,
        test_max_steps=test_max_steps,
    )
    return {
        "command": command,
        "checkpoint": str(checkpoint_path),
        "checkpoint_config": checkpoint["config"],
        "checkpoint_result": prior_result,
        "eval_rollouts": eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "paper_protocol_status": checkpoint_reevaluation_protocol_status(
            checkpoint["config"],
            eval_rollouts,
            test_max_steps,
        ),
        "test_max_steps": test_max_steps,
        "paper_test_horizon_steps": 15_000,
        "selected_result": result_to_metrics(result),
    }


def write_metrics(path: str | os.PathLike[str], metrics: dict[str, object]) -> None:
    metrics_dir = os.path.dirname(path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reevaluate a CartPole PPO checkpoint and write metrics JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--metrics-output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    metrics = evaluate_checkpoint_metrics(
        checkpoint_path,
        args.eval_rollouts,
        args.test_max_steps,
        " ".join(sys.argv),
    )

    write_metrics(args.metrics_output, metrics)

    print("CartPole checkpoint evaluation")
    print(f"  checkpoint={checkpoint_path}")
    selected = metrics["selected_result"]
    print(f"  train_success_rate={selected['train_success_rate']:.3f}")
    print(f"  test_success_rate={selected['test_success_rate']:.3f}")
    print(f"  train_reward_mean={selected['train_reward_mean']:.1f}")
    print(f"  test_reward_mean={selected['test_reward_mean']:.1f}")
    print(f"  metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
