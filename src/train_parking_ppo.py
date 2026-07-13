"""Train a small NumPy PPO baseline on the parking benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_teaching_sim import (
    ACTION_SPEC as PARKING_ACTION_SPEC,
    compact_trace,
    plot_trajectories,
    save_json,
    serialize_task,
    serialize_trajectories,
    summarize_traces,
)
from parking_env import ParkingTask, Trajectory, collision_or_bounds, is_success, make_tasks, observe, step_dynamics


ACTION_LOW = np.array([-1.40, -0.85], dtype=float)
ACTION_HIGH = np.array([1.60, 0.85], dtype=float)
OBS_SCALE = np.array([12.0, 4.0, 1.35, 12.0, 12.0, 12.0, 4.0, 12.0], dtype=float)
ACTION_SPEC = {
    **PARKING_ACTION_SPEC,
    "bounds": {
        "low": ACTION_LOW.astype(float).tolist(),
        "high": ACTION_HIGH.astype(float).tolist(),
    },
}


@dataclass
class PpoConfig:
    train_n: int = 8
    test_n: int = 8
    updates: int = 4
    rollouts_per_update: int = 8
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    policy_lr: float = 0.015
    value_lr: float = 0.040
    entropy_coef: float = 0.002
    seed: int = 0


@dataclass
class PpoBatch:
    features: np.ndarray
    raw_actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class ParkingPpoPolicy:
    """Linear Gaussian PPO policy for the parking observation vector."""

    def __init__(
        self,
        policy_w: np.ndarray,
        policy_b: np.ndarray,
        log_std: np.ndarray,
        value_w: np.ndarray,
        value_b: float,
    ):
        self.policy_w = policy_w.astype(float)
        self.policy_b = policy_b.astype(float)
        self.log_std = log_std.astype(float)
        self.value_w = value_w.astype(float)
        self.value_b = float(value_b)

    @classmethod
    def initialize(cls, rng: np.random.Generator) -> "ParkingPpoPolicy":
        policy_w = rng.normal(0.0, 0.035, size=(8, 2))
        policy_b = np.array([0.0, 0.0], dtype=float)
        # A light goal-seeking initialization shortens smoke-test training
        # without changing the PPO update path.
        policy_w[5, 0] = 0.55
        policy_w[2, 1] = -0.35
        policy_w[6, 1] = 0.20
        log_std = np.log(np.array([0.70, 0.45], dtype=float))
        value_w = np.zeros(8, dtype=float)
        return cls(policy_w, policy_b, log_std, value_w, 0.0)

    def mean(self, features: np.ndarray) -> np.ndarray:
        return features @ self.policy_w + self.policy_b

    def value(self, features: np.ndarray) -> np.ndarray:
        return features @ self.value_w + self.value_b

    def act(
        self,
        obs: Sequence[float],
        rng: np.random.Generator,
        deterministic: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        features = featurize(obs)
        mean = self.mean(features)
        std = np.exp(self.log_std)
        raw_action = mean if deterministic else rng.normal(mean, std)
        action = clip_action(raw_action)
        log_prob = gaussian_log_prob(raw_action.reshape(1, -1), mean.reshape(1, -1), self.log_std)[0]
        value = float(self.value(features))
        return action, raw_action.astype(float), float(log_prob), value

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy_w": self.policy_w.astype(float).tolist(),
            "policy_b": self.policy_b.astype(float).tolist(),
            "log_std": self.log_std.astype(float).tolist(),
            "value_w": self.value_w.astype(float).tolist(),
            "value_b": self.value_b,
        }


def clip_action(action: Sequence[float]) -> np.ndarray:
    array = np.asarray(action, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"parking PPO action must be [velocity, steering], got shape {array.shape}")
    return np.minimum(np.maximum(array, ACTION_LOW), ACTION_HIGH)


def featurize(obs: Sequence[float]) -> np.ndarray:
    return np.tanh(np.asarray(obs, dtype=float) / OBS_SCALE)


def gaussian_log_prob(raw_actions: np.ndarray, means: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    std = np.exp(log_std)
    normalized = (raw_actions - means) / std
    return -0.5 * np.sum(normalized * normalized + 2.0 * log_std + math.log(2.0 * math.pi), axis=1)


def parking_reward(state: np.ndarray, action: np.ndarray, task: ParkingTask, collision: bool, success: bool) -> float:
    goal_error = float(np.linalg.norm(state[:2] - task.goal[:2]))
    heading_error = float(abs(state[2]))
    reward = -0.03 - 0.55 * goal_error - 0.30 * heading_error
    reward -= 0.025 * float(abs(action[0])) + 0.015 * float(abs(action[1]))
    if collision:
        reward -= 55.0
    if success:
        reward += 120.0
    return float(reward)


def rollout_policy(
    policy: ParkingPpoPolicy,
    task: ParkingTask,
    rng: np.random.Generator,
    deterministic: bool,
    collect: bool = False,
) -> Tuple[Trajectory, List[dict], float]:
    state = task.start.copy()
    states = [state.tolist()]
    observations: List[List[float]] = []
    actions: List[List[float]] = []
    modes: List[str] = []
    transitions: List[dict] = []
    total_reward = 0.0
    collision = False
    success = False

    for _ in range(task.max_steps):
        obs = observe(state, task)
        action, raw_action, log_prob, value = policy.act(obs, rng, deterministic)
        next_state = step_dynamics(state, action)
        collision = collision_or_bounds(next_state, task)
        success = is_success(next_state, "done", task)
        reward = parking_reward(next_state, action, task, collision, success)

        observations.append([float(value) for value in obs])
        actions.append(action.astype(float).tolist())
        modes.append("ppo")
        total_reward += reward

        if collect:
            transitions.append(
                {
                    "features": featurize(obs),
                    "raw_action": raw_action,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                    "done": collision or success,
                }
            )

        state = next_state
        states.append(state.tolist())
        if collision or success:
            break

    params = {
        "controller": "numpy_linear_ppo",
        "total_reward": total_reward,
        "action_spec": ACTION_SPEC,
    }
    trajectory = Trajectory(
        task_id=task.task_id,
        states=states,
        observations=observations,
        actions=actions,
        modes=modes,
        success=success and not collision,
        collision=collision,
        score=float(total_reward),
        loop_count=0,
        params=params,
    )
    last_value = 0.0
    if collect and transitions and not transitions[-1]["done"]:
        last_value = float(policy.value(featurize(observe(state, task))))
    return trajectory, transitions, last_value


def compute_returns_and_advantages(transitions: List[dict], last_value: float, cfg: PpoConfig) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.array([item["reward"] for item in transitions], dtype=float)
    values = np.array([item["value"] for item in transitions], dtype=float)
    dones = np.array([item["done"] for item in transitions], dtype=bool)
    advantages = np.zeros_like(rewards)

    gae = 0.0
    next_value = float(last_value)
    for idx in range(len(transitions) - 1, -1, -1):
        next_nonterminal = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + cfg.gamma * next_value * next_nonterminal - values[idx]
        gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * gae
        advantages[idx] = gae
        next_value = values[idx]
    returns = advantages + values
    return returns, advantages


def collect_batch(
    policy: ParkingPpoPolicy,
    tasks: List[ParkingTask],
    rng: np.random.Generator,
    cfg: PpoConfig,
) -> Tuple[PpoBatch, List[Trajectory]]:
    all_features = []
    all_actions = []
    all_log_probs = []
    all_returns = []
    all_advantages = []
    traces = []

    for _ in range(cfg.rollouts_per_update):
        task = tasks[int(rng.integers(0, len(tasks)))]
        trace, transitions, last_value = rollout_policy(policy, task, rng, deterministic=False, collect=True)
        traces.append(trace)
        if not transitions:
            continue
        returns, advantages = compute_returns_and_advantages(transitions, last_value, cfg)
        all_features.extend(item["features"] for item in transitions)
        all_actions.extend(item["raw_action"] for item in transitions)
        all_log_probs.extend(item["log_prob"] for item in transitions)
        all_returns.extend(returns.tolist())
        all_advantages.extend(advantages.tolist())

    if not all_features:
        raise RuntimeError("PPO collected no transitions")

    return (
        PpoBatch(
            features=np.asarray(all_features, dtype=float),
            raw_actions=np.asarray(all_actions, dtype=float),
            old_log_probs=np.asarray(all_log_probs, dtype=float),
            returns=np.asarray(all_returns, dtype=float),
            advantages=np.asarray(all_advantages, dtype=float),
        ),
        traces,
    )


def clip_gradient(array: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    norm = float(np.linalg.norm(array))
    if norm <= max_norm or norm <= 1e-12:
        return array
    return array * (max_norm / norm)


def ppo_update(policy: ParkingPpoPolicy, batch: PpoBatch, cfg: PpoConfig) -> Dict[str, float]:
    features = batch.features
    raw_actions = batch.raw_actions
    old_log_probs = batch.old_log_probs
    returns = batch.returns
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = max(1, len(advantages))

    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    for _ in range(cfg.epochs):
        means = policy.mean(features)
        log_probs = gaussian_log_prob(raw_actions, means, policy.log_std)
        ratios = np.exp(np.clip(log_probs - old_log_probs, -20.0, 20.0))
        clipped = np.clip(ratios, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        active = np.ones_like(advantages)
        active[(advantages >= 0.0) & (ratios > 1.0 + cfg.clip_ratio)] = 0.0
        active[(advantages < 0.0) & (ratios < 1.0 - cfg.clip_ratio)] = 0.0

        std = np.exp(policy.log_std)
        coeff = (active * advantages * ratios / n).reshape(-1, 1)
        dlogp_dmean = (raw_actions - means) / (std * std)
        grad_mean = coeff * dlogp_dmean
        grad_w = clip_gradient(features.T @ grad_mean)
        grad_b = clip_gradient(grad_mean.sum(axis=0))
        grad_log_std = np.sum(coeff * (((raw_actions - means) ** 2 / (std * std)) - 1.0), axis=0)
        grad_log_std = clip_gradient(grad_log_std + cfg.entropy_coef)

        policy.policy_w += cfg.policy_lr * grad_w
        policy.policy_b += cfg.policy_lr * grad_b
        policy.log_std = np.clip(policy.log_std + cfg.policy_lr * grad_log_std, -2.5, 0.5)

        values = policy.value(features)
        value_error = values - returns
        value_grad_w = clip_gradient(features.T @ value_error / n)
        value_grad_b = float(np.mean(value_error))
        policy.value_w -= cfg.value_lr * value_grad_w
        policy.value_b -= cfg.value_lr * value_grad_b

        surrogate = np.minimum(ratios * advantages, clipped * advantages)
        final_policy_loss = float(-np.mean(surrogate))
        final_value_loss = float(np.mean(value_error * value_error))
        final_entropy = float(np.sum(policy.log_std + 0.5 * math.log(2.0 * math.pi * math.e)))

    return {
        "policy_loss": final_policy_loss,
        "value_loss": final_value_loss,
        "entropy": final_entropy,
        "num_transitions": int(n),
    }


def evaluate(policy: ParkingPpoPolicy, tasks: List[ParkingTask], seed: int) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    return [rollout_policy(policy, task, rng, deterministic=True, collect=False)[0] for task in tasks]


def action_lengths(traces: Sequence[Trajectory]) -> List[int]:
    return sorted({len(action) for trace in traces for action in trace.actions})


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(getattr(args, "metrics_output", None) or outdir / "metrics.json").resolve()
    traces_path = Path(getattr(args, "traces_output", None) or outdir / "traces.json").resolve()

    cfg = PpoConfig(
        train_n=args.train_n,
        test_n=args.test_n,
        updates=args.updates,
        rollouts_per_update=args.rollouts_per_update,
        epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
    rng = np.random.default_rng(cfg.seed)
    train_tasks = make_tasks(cfg.train_n, "train", rng)
    test_tasks = make_tasks(cfg.test_n, "test", rng)
    policy = ParkingPpoPolicy.initialize(rng)

    initial_train = evaluate(policy, train_tasks, cfg.seed + 11)
    history = []
    last_collection: List[Trajectory] = []
    for update_idx in range(cfg.updates):
        batch, collected_traces = collect_batch(policy, train_tasks, rng, cfg)
        update_stats = ppo_update(policy, batch, cfg)
        train_eval = evaluate(policy, train_tasks, cfg.seed + 100 + update_idx)
        history.append(
            {
                "update": update_idx,
                "collected": summarize_traces(collected_traces),
                "train_eval": summarize_traces(train_eval),
                "ppo": update_stats,
            }
        )
        last_collection = collected_traces

    train_eval = evaluate(policy, train_tasks, cfg.seed + 1000)
    test_eval = evaluate(policy, test_tasks, cfg.seed + 2000)
    command = " ".join(sys.argv)
    config = asdict(cfg)
    config["outdir"] = str(outdir)
    metrics = {
        "artifact_kind": "parking_ppo_training_metrics",
        "command": command,
        "config": config,
        "action_spec": ACTION_SPEC,
        "algorithm": {
            "name": "numpy_linear_ppo",
            "policy": "linear_gaussian",
            "value_function": "linear",
            "action_clipping": True,
        },
        "metrics_output": str(metrics_path),
        "traces_output": str(traces_path),
        "initial_train": summarize_traces(initial_train),
        "collected_train": summarize_traces(last_collection),
        "ppo_train": summarize_traces(train_eval),
        "ppo_test": summarize_traces(test_eval),
        "history": history,
        "policy_parameters": policy.to_dict(),
        "train_action_lengths": action_lengths(train_eval),
        "test_action_lengths": action_lengths(test_eval),
        "train_examples": [compact_trace(trace) for trace in train_eval[:5]],
        "test_examples": [compact_trace(trace) for trace in test_eval[:5]],
    }
    trace_payload = {
        "artifact_kind": "parking_ppo_training_traces",
        "command": command,
        "config": config,
        "action_spec": ACTION_SPEC,
        "train_tasks": [serialize_task(task) for task in train_tasks],
        "test_tasks": [serialize_task(task) for task in test_tasks],
        "collected_train_traces": serialize_trajectories(last_collection),
        "ppo_train_traces": serialize_trajectories(train_eval),
        "ppo_test_traces": serialize_trajectories(test_eval),
    }
    save_json(metrics_path, metrics)
    save_json(traces_path, trace_payload)
    plot_trajectories(outdir / "ppo_trajectories.png", train_eval, test_eval, train_tasks, test_tasks)
    return metrics


def verify_metrics(metrics: Dict[str, object]) -> None:
    if metrics.get("artifact_kind") != "parking_ppo_training_metrics":
        raise AssertionError("wrong PPO metrics artifact kind")
    action_spec = metrics.get("action_spec", {})
    if action_spec.get("components") != ["velocity", "steering"] or action_spec.get("dimension") != 2:
        raise AssertionError("PPO action spec must be two-action velocity/steering")
    if metrics.get("train_action_lengths") != [2] or metrics.get("test_action_lengths") != [2]:
        raise AssertionError("PPO traces must contain only two-action controls")
    for key in ("initial_train", "ppo_train", "ppo_test", "history", "policy_parameters"):
        if key not in metrics:
            raise AssertionError(f"missing PPO metric {key}")
    if not np.isfinite(float(metrics["ppo_train"]["mean_score"])):
        raise AssertionError("PPO train score is not finite")
    metrics_text = json.dumps(metrics).lower()
    for forbidden in ("lateral_rate", "center_lateral", "arc_lateral", "counter_lateral"):
        if forbidden in metrics_text:
            raise AssertionError(f"PPO metrics contain non-paper action term {forbidden}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO baseline on the parking benchmark.")
    parser.add_argument("--outdir", default=Path("artifacts/parking_ppo"), type=Path)
    parser.add_argument("--train-n", default=8, type=int)
    parser.add_argument("--test-n", default=8, type=int)
    parser.add_argument("--updates", default=4, type=int)
    parser.add_argument("--rollouts-per-update", default=8, type=int)
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--gae-lambda", default=0.95, type=float)
    parser.add_argument("--clip-ratio", default=0.20, type=float)
    parser.add_argument("--policy-lr", default=0.015, type=float)
    parser.add_argument("--value-lr", default=0.040, type=float)
    parser.add_argument("--entropy-coef", default=0.002, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--traces-output", default=None, type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_experiment(args)
    if args.verify:
        verify_metrics(metrics)
    print(
        "metrics: "
        f"initial_train={metrics['initial_train']['success_rate']:.2f}, "
        f"ppo_train={metrics['ppo_train']['success_rate']:.2f}, "
        f"ppo_test={metrics['ppo_test']['success_rate']:.2f}, "
        f"test_mean_score={metrics['ppo_test']['mean_score']:.1f}"
    )
    print(f"metrics written to {metrics['metrics_output']}")
    print(f"traces written to {metrics['traces_output']}")
    print(f"plots written to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
