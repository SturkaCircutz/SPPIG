from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


PAPER_TIMESTEPS = 10_000_000
PAPER_NMINIBATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
PAPER_ENT_COEFS = [0.0, 0.01, 0.05, 0.1]
PAPER_UPDATE_EPOCHS = list(range(3, 37))
PAPER_CLIP_RANGES = [0.1, 0.2, 0.3]
DEFAULT_LEARNING_RATES = [5e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

PLAN_FIELDS = [
    "job_id",
    "policy",
    "seed",
    "total_timesteps",
    "rollout_steps",
    "num_envs",
    "hidden_size",
    "update_epochs",
    "minibatches",
    "learning_rate",
    "entropy_coef",
    "clip_range",
    "eval_rollouts",
    "test_max_steps",
    "eval_interval",
    "output",
    "metrics_output",
]

RESULT_FIELDS = PLAN_FIELDS + [
    "train_success",
    "test_success",
    "train_reward",
    "test_reward",
    "selected_timesteps",
]

SUMMARY_FIELDS = [
    "policy",
    "jobs_completed",
    "best_job_id",
    "best_seed",
    "best_train_success",
    "best_test_success",
    "best_train_reward",
    "best_test_reward",
    "best_selected_timesteps",
    "best_minibatches",
    "best_learning_rate",
    "best_entropy_coef",
    "best_update_epochs",
    "best_clip_range",
    "best_output",
    "best_metrics_output",
]

FAILURE_FIELDS = PLAN_FIELDS + [
    "error_type",
    "error_message",
]


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_floats(value: str) -> List[float]:
    return [float(item) for item in value.split(",") if item]


def _parse_update_epochs(value: str) -> List[int]:
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return _parse_ints(value)


def _job_name(job_id: int, policy: str, seed: int) -> str:
    return f"{job_id:05d}_{policy}_seed{seed}"


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policies = [policy for policy in args.policies.split(",") if policy]
    seeds = _parse_ints(args.seeds)
    learning_rates = _parse_floats(args.learning_rates)
    nminibatches = _parse_ints(args.nminibatches)
    ent_coefs = _parse_floats(args.ent_coefs)
    update_epochs = _parse_update_epochs(args.update_epochs)
    clip_ranges = _parse_floats(args.clip_ranges)

    jobs: List[Dict[str, Any]] = []
    for policy in policies:
        policy_minibatches = [1] if policy == "lstm" else nminibatches
        for seed in seeds:
            for minibatches in policy_minibatches:
                for entropy_coef in ent_coefs:
                    for epochs in update_epochs:
                        for clip_range in clip_ranges:
                            for learning_rate in learning_rates:
                                job_id = len(jobs)
                                name = _job_name(job_id, policy, seed)
                                jobs.append(
                                    {
                                        "job_id": job_id,
                                        "policy": policy,
                                        "seed": seed,
                                        "total_timesteps": args.timesteps,
                                        "rollout_steps": args.rollout_steps,
                                        "num_envs": args.num_envs,
                                        "hidden_size": args.hidden_size,
                                        "update_epochs": epochs,
                                        "minibatches": minibatches,
                                        "learning_rate": learning_rate,
                                        "entropy_coef": entropy_coef,
                                        "clip_range": clip_range,
                                        "eval_rollouts": args.eval_rollouts,
                                        "test_max_steps": args.test_max_steps,
                                        "eval_interval": args.eval_interval,
                                        "output": str(args.outdir / "checkpoints" / f"{name}.pt"),
                                        "metrics_output": str(args.outdir / "metrics" / f"{name}.json"),
                                    }
                                )
                                if args.max_configs is not None and len(jobs) >= args.max_configs:
                                    return jobs
    return jobs


def count_uncapped_jobs(args: argparse.Namespace) -> int:
    policies = [policy for policy in args.policies.split(",") if policy]
    seeds = _parse_ints(args.seeds)
    nminibatches = _parse_ints(args.nminibatches)
    ent_coefs = _parse_floats(args.ent_coefs)
    update_epochs = _parse_update_epochs(args.update_epochs)
    clip_ranges = _parse_floats(args.clip_ranges)
    learning_rates = _parse_floats(args.learning_rates)

    total = 0
    for policy in policies:
        policy_minibatches = 1 if policy == "lstm" else len(nminibatches)
        total += (
            len(seeds)
            * policy_minibatches
            * len(ent_coefs)
            * len(update_epochs)
            * len(clip_ranges)
            * len(learning_rates)
        )
    return total


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def summarize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    policies: List[str] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        policy = str(row["policy"])
        if policy not in by_policy:
            policies.append(policy)
            by_policy[policy] = []
        by_policy[policy].append(row)

    summary: List[Dict[str, Any]] = []
    for policy in policies:
        group = by_policy[policy]
        best = max(
            group,
            key=lambda row: (
                float(row["train_success"]),
                float(row["train_reward"]),
                -int(row["job_id"]),
            ),
        )
        summary.append(
            {
                "policy": policy,
                "jobs_completed": len(group),
                "best_job_id": int(best["job_id"]),
                "best_seed": int(best["seed"]),
                "best_train_success": float(best["train_success"]),
                "best_test_success": float(best["test_success"]),
                "best_train_reward": float(best["train_reward"]),
                "best_test_reward": float(best["test_reward"]),
                "best_selected_timesteps": int(best["selected_timesteps"]),
                "best_minibatches": int(best["minibatches"]),
                "best_learning_rate": float(best["learning_rate"]),
                "best_entropy_coef": float(best["entropy_coef"]),
                "best_update_epochs": int(best["update_epochs"]),
                "best_clip_range": float(best["clip_range"]),
                "best_output": best["output"],
                "best_metrics_output": best["metrics_output"],
            }
        )
    return summary


def read_existing_results(path: Path | str) -> Dict[int, Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["job_id"]): row for row in rows if row.get("job_id", "").isdigit()}


def resumable_result_for_job(
    job: Dict[str, Any],
    existing_results: Dict[int, Dict[str, str]],
) -> Dict[str, str] | None:
    row = existing_results.get(int(job["job_id"]))
    if row is None:
        return None
    for field in PLAN_FIELDS:
        if str(row.get(field, "")) != str(job[field]):
            return None
    for field in ("output", "metrics_output"):
        if not Path(str(row[field])).exists():
            return None
    return row


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    from ppo_cartpole import PPOConfig, train_ppo_cartpole

    cfg = PPOConfig(
        policy_type=job["policy"],
        total_timesteps=int(job["total_timesteps"]),
        rollout_steps=int(job["rollout_steps"]),
        update_epochs=int(job["update_epochs"]),
        minibatches=int(job["minibatches"]),
        learning_rate=float(job["learning_rate"]),
        entropy_coef=float(job["entropy_coef"]),
        clip_range=float(job["clip_range"]),
        hidden_size=int(job["hidden_size"]),
        num_envs=int(job["num_envs"]),
        eval_rollouts=int(job["eval_rollouts"]),
        eval_test_max_steps=int(job["test_max_steps"]),
        eval_interval=int(job["eval_interval"]),
        metrics_output=job["metrics_output"],
        seed=int(job["seed"]),
        initial_log_std=-1.0,
    )
    _, result = train_ppo_cartpole(cfg, output=job["output"])
    return {
        **job,
        "train_success": result.train_success_rate,
        "test_success": result.test_success_rate,
        "train_reward": result.train_reward_mean,
        "test_reward": result.test_reward_mean,
        "selected_timesteps": result.timesteps,
    }


def failed_job_row(job: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return {
        **job,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def write_manifest(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    completed: int,
    skipped: int,
    failed: int,
) -> None:
    manifest = {
        "command": " ".join(sys.argv),
        "dry_run": args.dry_run,
        "quick": args.quick,
        "resume": args.resume,
        "continue_on_error": args.continue_on_error,
        "policies": [policy for policy in args.policies.split(",") if policy],
        "seeds": _parse_ints(args.seeds),
        "jobs_planned": len(jobs),
        "jobs_uncapped_for_selected_space": count_uncapped_jobs(args),
        "jobs_completed": completed,
        "jobs_failed": failed,
        "jobs_skipped_existing": skipped,
        "jobs_run_this_invocation": completed - skipped,
        "max_configs": args.max_configs,
        "paper_space": {
            "timesteps": PAPER_TIMESTEPS,
            "nminibatches": PAPER_NMINIBATCHES,
            "lstm_nminibatches": [1],
            "ent_coef": PAPER_ENT_COEFS,
            "noptepochs": [3, 36],
            "cliprange": PAPER_CLIP_RANGES,
            "learning_rate_interval": [5e-6, 0.003],
            "learning_rate_values_used": _parse_floats(args.learning_rates),
            "learning_rate_note": (
                "The extracted paper text reports a learning-rate interval, not the exact samples. "
                "This runner uses the explicit --learning-rates values within that interval."
            ),
        },
        "artifacts": {
            "plan": str(args.outdir / "cartpole_ppo_sweep_plan.csv"),
            "results": str(args.outdir / "cartpole_ppo_sweep_results.csv"),
            "summary": str(args.outdir / "cartpole_ppo_sweep_summary.csv"),
            "failures": str(args.outdir / "cartpole_ppo_sweep_failures.csv"),
            "checkpoints": str(args.outdir / "checkpoints"),
            "metrics": str(args.outdir / "metrics"),
        },
        "selection_rule": "per policy: max train_success, then train_reward, then lower job_id",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "cartpole_ppo_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan the CartPole PPO hyperparameter sweep.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "ppo_sweep")
    parser.add_argument("--policies", default="mlp,lstm")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--timesteps", type=int, default=PAPER_TIMESTEPS)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--learning-rates", default=",".join(str(value) for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--nminibatches", default=",".join(str(value) for value in PAPER_NMINIBATCHES))
    parser.add_argument("--ent-coefs", default=",".join(str(value) for value in PAPER_ENT_COEFS))
    parser.add_argument("--update-epochs", default="3-36")
    parser.add_argument("--clip-ranges", default=",".join(str(value) for value in PAPER_CLIP_RANGES))
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed jobs from an existing results CSV when plan fields and artifacts still match.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed jobs to a failures CSV and continue. By default the first job failure stops the sweep.",
    )
    parser.add_argument("--quick", action="store_true", help="Use tiny local settings; pair with --max-configs.")
    args = parser.parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 64)
        args.rollout_steps = min(args.rollout_steps, 32)
        args.num_envs = 1
        args.hidden_size = min(args.hidden_size, 8)
        args.eval_rollouts = min(args.eval_rollouts, 1)
        args.test_max_steps = min(args.test_max_steps, 20)
        args.eval_interval = args.eval_interval or 32
    return args


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_csv(args.outdir / "cartpole_ppo_sweep_plan.csv", PLAN_FIELDS, jobs)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    skipped = 0
    if not args.dry_run:
        existing_results = (
            read_existing_results(args.outdir / "cartpole_ppo_sweep_results.csv")
            if args.resume
            else {}
        )
        for job in jobs:
            existing = resumable_result_for_job(job, existing_results) if args.resume else None
            if existing is not None:
                results.append(existing)
                skipped += 1
            else:
                try:
                    results.append(run_job(job))
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    failures.append(failed_job_row(job, exc))
                    write_csv(args.outdir / "cartpole_ppo_sweep_failures.csv", FAILURE_FIELDS, failures)
                    continue
            write_csv(args.outdir / "cartpole_ppo_sweep_results.csv", RESULT_FIELDS, results)
            if results:
                write_csv(args.outdir / "cartpole_ppo_sweep_summary.csv", SUMMARY_FIELDS, summarize_results(results))
        write_csv(args.outdir / "cartpole_ppo_sweep_results.csv", RESULT_FIELDS, results)
        write_csv(args.outdir / "cartpole_ppo_sweep_summary.csv", SUMMARY_FIELDS, summarize_results(results))
    if failures:
        write_csv(args.outdir / "cartpole_ppo_sweep_failures.csv", FAILURE_FIELDS, failures)
    write_manifest(args, jobs, len(results), skipped, len(failures))
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_plan.csv'}")
    if not args.dry_run:
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_results.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_summary.csv'}")
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_manifest.json'}")


if __name__ == "__main__":
    main()
