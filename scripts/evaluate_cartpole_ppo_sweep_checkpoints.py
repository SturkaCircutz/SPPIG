from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Keep this script runnable from a fresh checkout without requiring package install.
sys.path.insert(0, str(SRC))

from evaluate_cartpole_checkpoint import evaluate_checkpoint_metrics, write_metrics  # noqa: E402


DEFAULT_RESULTS_CSV = ROOT / "artifacts" / "ppo_sweep_cuda_medium_core" / "cartpole_ppo_sweep_results.csv"
DEFAULT_OUTDIR = ROOT / "artifacts" / "ppo_sweep_cuda_medium_core" / "checkpoint_reevaluations"
SUMMARY_FILENAME = "cartpole_ppo_sweep_checkpoint_reeval_summary.csv"
MANIFEST_FILENAME = "cartpole_ppo_sweep_checkpoint_reeval_manifest.json"
REQUIRED_RESULTS_FIELDS = {
    "job_id",
    "policy",
    "seed",
    "output",
    "eval_rollouts",
    "test_max_steps",
}
INTEGER_PATTERN = re.compile(r"[+-]?\d+")


def _load_checkpoint_config(checkpoint_path: Path) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to validate PPO sweep checkpoints") from exc
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint lacks config mapping: {checkpoint_path}")
    return config


def read_sweep_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_RESULTS_FIELDS - fieldnames)
        if missing:
            raise ValueError(f"sweep results CSV {path} is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"sweep results CSV {path} has no checkpoint rows")
    return rows


def repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def row_int(row: dict[str, str], field: str) -> int:
    try:
        value = row[field].strip()
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"sweep row {row.get('job_id', '<unknown>')} lacks integer {field}") from exc
    if INTEGER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"sweep row {row.get('job_id', '<unknown>')} lacks integer {field}")
    return int(value)


def positive_int(value: int, field: str, row: dict[str, str]) -> int:
    if value <= 0:
        raise ValueError(f"sweep row {row.get('job_id', '<unknown>')} has nonpositive {field}: {value}")
    return value


def reevaluation_metrics_name(row: dict[str, str], job_id: int, seed: int) -> str:
    policy = row.get("policy", "policy")
    return f"{job_id:05d}_{policy}_seed{seed}_reeval.json"


def validate_checkpoint_row(
    row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Path, int, int, int, int, dict[str, object]]:
    job_id = row_int(row, "job_id")
    seed = row_int(row, "seed")
    checkpoint_path = repo_path(row["output"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint for job {row.get('job_id')}: {checkpoint_path}")
    eval_rollouts = positive_int(
        args.eval_rollouts if args.eval_rollouts is not None else row_int(row, "eval_rollouts"),
        "eval_rollouts",
        row,
    )
    test_max_steps = positive_int(
        args.test_max_steps if args.test_max_steps is not None else row_int(row, "test_max_steps"),
        "test_max_steps",
        row,
    )
    config = _load_checkpoint_config(checkpoint_path)
    checkpoint_policy = config.get("policy_type")
    if checkpoint_policy != row["policy"]:
        raise ValueError(
            f"sweep row {row.get('job_id', '<unknown>')} policy {row['policy']} "
            f"does not match checkpoint policy_type {checkpoint_policy}"
        )
    if "seed" not in config:
        raise ValueError(f"checkpoint lacks seed provenance for job {row.get('job_id', '<unknown>')}")
    try:
        checkpoint_seed = int(config["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint seed is not an integer for job {row.get('job_id', '<unknown>')}") from exc
    if checkpoint_seed != seed:
        raise ValueError(
            f"sweep row {row.get('job_id', '<unknown>')} seed {row['seed']} "
            f"does not match checkpoint seed {checkpoint_seed}"
        )
    return checkpoint_path, eval_rollouts, test_max_steps, job_id, seed, config


def summary_row(row: dict[str, str], metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    selected = metrics["selected_result"]
    status = metrics["paper_protocol_status"]
    if not isinstance(selected, dict) or not isinstance(status, dict):
        raise ValueError(f"checkpoint reevaluation metrics for job {row.get('job_id')} are malformed")
    return {
        "job_id": row["job_id"],
        "policy": row["policy"],
        "seed": row["seed"],
        "checkpoint": row["output"],
        "metrics_output": str(metrics_path),
        "eval_rollouts": metrics["eval_rollouts"],
        "test_max_steps": metrics["test_max_steps"],
        "train_success": selected["train_success_rate"],
        "test_success": selected["test_success_rate"],
        "train_reward": selected["train_reward_mean"],
        "test_reward": selected["test_reward_mean"],
        "train_steps": selected["train_steps_mean"],
        "test_steps": selected["test_steps_mean"],
        "train_survival_seconds": selected["train_survival_seconds_mean"],
        "test_survival_seconds": selected["test_survival_seconds_mean"],
        "checkpoint_total_timesteps": status["checkpoint_total_timesteps"],
        "checkpoint_uses_paper_timestep_budget": status["checkpoint_uses_paper_timestep_budget"],
        "reevaluation_uses_full_test_horizon": status["reevaluation_uses_full_test_horizon"],
        "uses_paper_eval_rollouts": status["uses_paper_eval_rollouts"],
        "paper_scale_checkpoint_result": status["paper_scale_checkpoint_result"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "policy",
        "seed",
        "checkpoint",
        "metrics_output",
        "eval_rollouts",
        "test_max_steps",
        "train_success",
        "test_success",
        "train_reward",
        "test_reward",
        "train_steps",
        "test_steps",
        "train_survival_seconds",
        "test_survival_seconds",
        "checkpoint_total_timesteps",
        "checkpoint_uses_paper_timestep_budget",
        "reevaluation_uses_full_test_horizon",
        "uses_paper_eval_rollouts",
        "paper_scale_checkpoint_result",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    command: str,
    rows: list[dict[str, str]],
    summary_rows: list[dict[str, object]],
) -> None:
    manifest = {
        "artifact_kind": "cartpole_ppo_sweep_checkpoint_reevaluation_manifest",
        "command": command,
        "source_results_csv": str(args.results_csv),
        "metrics_dir": str(args.outdir / "metrics"),
        "summary_csv": str(args.outdir / SUMMARY_FILENAME),
        "jobs_planned": len(rows),
        "jobs_completed": len(summary_rows),
        "training_launched": False,
        "eval_rollouts_override": args.eval_rollouts,
        "test_max_steps_override": args.test_max_steps,
        "paper_protocol_status": {
            "artifact_kind": "ppo_sweep_checkpoint_reevaluation",
            "checkpoint_training_reused": True,
            "training_launched": False,
            "paper_scale_checkpoint_result": False,
            "limitation": (
                "Reevaluates existing PPO/PPO-LSTM sweep checkpoints without PPO training. "
                "The result is checkpoint-reuse evidence only; it does not make the original "
                "partial sweep a paper-scale 10^7-timestep, five-seed baseline protocol."
            ),
        },
        "summary": summary_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def reevaluate_sweep_checkpoints(args: argparse.Namespace, command: str | None = None) -> list[dict[str, object]]:
    command = " ".join(sys.argv) if command is None else command
    rows = read_sweep_rows(args.results_csv)
    validated_rows = [
        (row, *validate_checkpoint_row(row, args))
        for row in rows
    ]
    metrics_dir = args.outdir / "metrics"
    summary_rows: list[dict[str, object]] = []
    for row, checkpoint_path, eval_rollouts, test_max_steps, job_id, seed, _ in validated_rows:
        metrics_path = metrics_dir / reevaluation_metrics_name(row, job_id, seed)
        metrics = evaluate_checkpoint_metrics(
            checkpoint_path,
            eval_rollouts,
            test_max_steps,
            command,
        )
        metrics["sweep_source_results_csv"] = str(args.results_csv)
        metrics["sweep_source_row"] = row
        write_metrics(metrics_path, metrics)
        summary_rows.append(summary_row(row, metrics_path, metrics))

    write_csv(args.outdir / SUMMARY_FILENAME, summary_rows)
    write_manifest(args.outdir / MANIFEST_FILENAME, args, command, rows, summary_rows)
    return summary_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reevaluate PPO/PPO-LSTM checkpoints listed in a CartPole sweep results CSV."
    )
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=None,
        help="Override per-row eval_rollouts from the sweep results CSV.",
    )
    parser.add_argument(
        "--test-max-steps",
        type=int,
        default=None,
        help="Override per-row test_max_steps from the sweep results CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command = " ".join(sys.argv if argv is None else [sys.argv[0], *argv])
    summary_rows = reevaluate_sweep_checkpoints(args, command)
    print("CartPole PPO sweep checkpoint reevaluation")
    print(f"  source_results_csv={args.results_csv}")
    print(f"  jobs_completed={len(summary_rows)}")
    print(f"  summary={args.outdir / SUMMARY_FILENAME}")
    print(f"  manifest={args.outdir / MANIFEST_FILENAME}")


if __name__ == "__main__":
    main()
