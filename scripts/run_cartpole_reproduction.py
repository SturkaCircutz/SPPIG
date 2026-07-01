from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Keep this script runnable from a fresh checkout without requiring package install.
sys.path.insert(0, str(SRC))

from cartpole_env import CartpoleEnv  # noqa: E402
from cartpole_synthesis import CartpoleSynthesisConfig, synthesize_cartpole_policy  # noqa: E402

try:
    from ppo_cartpole import PPOConfig, train_ppo_cartpole  # noqa: E402

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


RESULT_FIELDS = [
    "policy",
    "seed",
    "train_success",
    "test_success",
    "train_reward",
    "test_reward",
    "timesteps",
    "checkpoint",
    "metrics_output",
]

SUMMARY_FIELDS = [
    "policy",
    "n",
    "train_success_mean",
    "train_success_std",
    "test_success_mean",
    "test_success_std",
    "train_reward_mean",
    "train_reward_std",
    "test_reward_mean",
    "test_reward_std",
    "best_seed_by_train",
    "best_train_success",
    "best_test_success",
    "best_train_reward",
    "best_test_reward",
    "best_timesteps",
    "best_checkpoint",
    "best_metrics_output",
]


def _summarize_results(results: Iterable[Any]) -> Dict[str, float]:
    result_list = list(results)
    return {
        "success": sum(result.success for result in result_list) / len(result_list),
        "reward": sum(result.reward for result in result_list) / len(result_list),
    }


def run_psm(seed: int, eval_rollouts: int, test_max_steps: int, quick: bool) -> Dict[str, Any]:
    # The quick path is a CI/local smoke test; the non-quick path preserves the
    # larger candidate pool and trace count expected for reproduction runs.
    cfg = CartpoleSynthesisConfig(
        num_initial_states=4 if quick else 64,
        candidate_rollouts=4 if quick else 128,
        segment_steps=2 if quick else 8,
        segments_per_trace=8 if quick else 32,
        teacher_student_iters=1 if quick else 2,
        seed=seed,
    )
    policy, traces = synthesize_cartpole_policy(cfg)
    train_env = CartpoleEnv.train_env(seed=100 + seed)
    test_env = CartpoleEnv.test_env(seed=200 + seed)
    train = _summarize_results(train_env.rollout(policy) for _ in range(eval_rollouts))
    # The paper's test horizon is 300s; test_max_steps is only exposed so tests
    # can cap runtime without changing the environment definition itself.
    test = _summarize_results(test_env.rollout(policy, max_steps=test_max_steps) for _ in range(eval_rollouts))
    return {
        "policy": "Programmatic state machine",
        "seed": seed,
        "train_success": train["success"],
        "test_success": test["success"],
        "train_reward": train["reward"],
        "test_reward": test["reward"],
        "timesteps": 0,
        "config": asdict(cfg),
        "policy_description": policy.describe(),
        "num_traces": len(traces),
    }


def run_ppo(
    policy: str,
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    outdir: Path,
    quick: bool,
) -> Dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run PPO baselines")
    artifact_stem = f"ppo_{policy}_seed{seed}"
    checkpoint_path = outdir / "checkpoints" / f"{artifact_stem}.pt"
    metrics_path = outdir / "metrics" / f"{artifact_stem}.json"
    # Non-quick PPO keeps the paper-scale 10^7 timestep budget. This runner
    # intentionally records one fixed config; it is not the missing five-seed
    # hyperparameter search from the paper.
    cfg = PPOConfig(
        policy_type=policy,
        total_timesteps=64 if quick else 10_000_000,
        rollout_steps=32 if quick else 128,
        update_epochs=1 if quick else 8,
        minibatches=1 if policy == "lstm" else (1 if quick else 8),
        hidden_size=8 if quick else 64,
        num_envs=1 if quick else 8,
        eval_rollouts=eval_rollouts,
        eval_test_max_steps=test_max_steps,
        seed=seed,
        initial_log_std=-1.0,
        metrics_output=str(metrics_path),
    )
    _, result = train_ppo_cartpole(cfg, output=str(checkpoint_path))
    return {
        "policy": "PPO-LSTM" if policy == "lstm" else "PPO MLP",
        "seed": seed,
        "train_success": result.train_success_rate,
        "test_success": result.test_success_rate,
        "train_reward": result.train_reward_mean,
        "test_reward": result.test_reward_mean,
        "timesteps": result.timesteps,
        "checkpoint": str(checkpoint_path),
        "metrics_output": str(metrics_path),
        "config": asdict(cfg),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    policy_order: List[str] = []
    for row in rows:
        policy = row["policy"]
        if policy not in groups:
            groups[policy] = []
            policy_order.append(policy)
        groups[policy].append(row)

    summary: List[Dict[str, Any]] = []
    for policy in policy_order:
        group = groups[policy]
        best = max(
            group,
            key=lambda row: (
                float(row["train_success"]),
                float(row["train_reward"]),
                -int(row["seed"]),
            ),
        )
        summary.append(
            {
                "policy": policy,
                "n": len(group),
                "train_success_mean": _mean([float(row["train_success"]) for row in group]),
                "train_success_std": _sample_std([float(row["train_success"]) for row in group]),
                "test_success_mean": _mean([float(row["test_success"]) for row in group]),
                "test_success_std": _sample_std([float(row["test_success"]) for row in group]),
                "train_reward_mean": _mean([float(row["train_reward"]) for row in group]),
                "train_reward_std": _sample_std([float(row["train_reward"]) for row in group]),
                "test_reward_mean": _mean([float(row["test_reward"]) for row in group]),
                "test_reward_std": _sample_std([float(row["test_reward"]) for row in group]),
                "best_seed_by_train": int(best["seed"]),
                "best_train_success": float(best["train_success"]),
                "best_test_success": float(best["test_success"]),
                "best_train_reward": float(best["train_reward"]),
                "best_test_reward": float(best["test_reward"]),
                "best_timesteps": int(best["timesteps"]),
                "best_checkpoint": best.get("checkpoint", ""),
                "best_metrics_output": best.get("metrics_output", ""),
            }
        )
    return summary


def write_results(rows: List[Dict[str, Any]], outdir: Path, manifest: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "cartpole_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
    summary = summarize_rows(rows)
    summary_path = outdir / "cartpole_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    # Keep full configs and synthesized policy descriptions out of the flat CSV
    # but preserve them in the manifest for experiment provenance.
    (outdir / "cartpole_manifest.json").write_text(
        json.dumps({**manifest, "rows": rows, "summary": summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CartPole reproduction experiments and write artifacts.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "results")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--include-ppo", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a small diagnostic configuration for CI/local checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.append(run_psm(seed, args.eval_rollouts, args.test_max_steps, args.quick))
        if args.include_ppo:
            # Baselines share the same seed list and evaluation budget so their
            # raw rows remain comparable under one reproduction manifest.
            rows.append(run_ppo("mlp", seed, args.eval_rollouts, args.test_max_steps, args.outdir, args.quick))
            rows.append(run_ppo("lstm", seed, args.eval_rollouts, args.test_max_steps, args.outdir, args.quick))

    manifest = {
        "command": " ".join(sys.argv),
        "quick": args.quick,
        "include_ppo": args.include_ppo,
        "seeds": seeds,
        "eval_rollouts": args.eval_rollouts,
        "test_max_steps": args.test_max_steps,
        "paper_scale_note": (
            "Without --quick, PPO uses 10^7 timesteps per seed. "
            "This runner records exact configs but does not perform hyperparameter search."
        ),
        "summary_note": (
            "cartpole_summary.csv reports per-policy means and sample standard deviations over "
            "the requested seeds; with one seed, std is reported as 0. Best seed is selected by "
            "train_success, then train_reward, then lower seed."
        ),
        "ppo_artifact_note": (
            "When --include-ppo is set, PPO rows include checkpoint and metrics_output paths "
            "under the requested output directory."
        ),
    }
    write_results(rows, args.outdir, manifest)
    print(f"wrote {args.outdir / 'cartpole_results.csv'}")
    print(f"wrote {args.outdir / 'cartpole_summary.csv'}")
    print(f"wrote {args.outdir / 'cartpole_manifest.json'}")


if __name__ == "__main__":
    main()
