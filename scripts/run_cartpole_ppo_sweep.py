from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from cartpole_env import PAPER_EVAL_ROLLOUTS, CartpoleEnv, cartpole_reward_spec, cartpole_space_spec  # noqa: E402


PAPER_TIMESTEPS = 10_000_000
PAPER_NMINIBATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
PAPER_ENT_COEFS = [0.0, 0.01, 0.05, 0.1]
PAPER_UPDATE_EPOCHS = list(range(3, 37))
PAPER_CLIP_RANGES = [0.1, 0.2, 0.3]
PAPER_HYPERPARAMETER_SAMPLES = 10
PAPER_LEARNING_RATE_MIN = 5e-6
PAPER_LEARNING_RATE_MAX = 0.003
PAPER_TEST_MAX_STEPS = 15_000
DEFAULT_LEARNING_RATES = [5e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

PLAN_FIELDS = [
    "job_id",
    "policy",
    "seed",
    "hyperparam_mode",
    "hyperparam_sample",
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
    "train_steps",
    "test_steps",
    "train_survival_seconds",
    "test_survival_seconds",
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
    "best_train_steps",
    "best_test_steps",
    "best_train_survival_seconds",
    "best_test_survival_seconds",
    "best_selected_timesteps",
    "best_minibatches",
    "best_learning_rate",
    "best_entropy_coef",
    "best_update_epochs",
    "best_clip_range",
    "best_output",
    "best_metrics_output",
]

HYPERPARAMETER_SUMMARY_FIELDS = [
    "policy",
    "hyperparam_mode",
    "hyperparam_sample",
    "jobs_completed",
    "seed_count",
    "seeds_completed",
    "train_success_mean",
    "train_success_std",
    "test_success_mean",
    "test_success_std",
    "train_reward_mean",
    "train_reward_std",
    "test_reward_mean",
    "test_reward_std",
    "train_steps_mean",
    "train_steps_std",
    "test_steps_mean",
    "test_steps_std",
    "train_survival_seconds_mean",
    "train_survival_seconds_std",
    "test_survival_seconds_mean",
    "test_survival_seconds_std",
    "minibatches",
    "learning_rate",
    "entropy_coef",
    "update_epochs",
    "clip_range",
    "best_job_id",
    "best_seed",
    "best_train_success",
    "best_test_success",
    "best_train_reward",
    "best_test_reward",
    "best_train_steps",
    "best_test_steps",
    "best_train_survival_seconds",
    "best_test_survival_seconds",
    "best_selected_timesteps",
    "best_output",
    "best_metrics_output",
    "is_best_hyperparam_for_policy",
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


def _grid_hyperparameter_configs(args: argparse.Namespace, policy: str) -> List[Dict[str, Any]]:
    nminibatches = _parse_ints(args.nminibatches)
    ent_coefs = _parse_floats(args.ent_coefs)
    update_epochs = _parse_update_epochs(args.update_epochs)
    clip_ranges = _parse_floats(args.clip_ranges)
    learning_rates = _parse_floats(args.learning_rates)
    policy_minibatches = [1] if policy == "lstm" else nminibatches
    configs: List[Dict[str, Any]] = []
    for minibatches in policy_minibatches:
        for entropy_coef in ent_coefs:
            for epochs in update_epochs:
                for clip_range in clip_ranges:
                    for learning_rate in learning_rates:
                        configs.append(
                            {
                                "minibatches": minibatches,
                                "learning_rate": learning_rate,
                                "entropy_coef": entropy_coef,
                                "update_epochs": epochs,
                                "clip_range": clip_range,
                            }
                        )
    return configs


def _paper_random_hyperparameter_configs(args: argparse.Namespace, policy: str) -> List[Dict[str, Any]]:
    rng = random.Random(args.hyperparam_seed + (0 if policy == "mlp" else 1_000_003))
    configs: List[Dict[str, Any]] = []
    for _ in range(max(0, args.hyperparam_samples)):
        configs.append(
            {
                "minibatches": 1 if policy == "lstm" else rng.choice(PAPER_NMINIBATCHES),
                "learning_rate": rng.uniform(PAPER_LEARNING_RATE_MIN, PAPER_LEARNING_RATE_MAX),
                "entropy_coef": rng.choice(PAPER_ENT_COEFS),
                "update_epochs": rng.choice(PAPER_UPDATE_EPOCHS),
                "clip_range": rng.choice(PAPER_CLIP_RANGES),
            }
        )
    return configs


def hyperparameter_configs(args: argparse.Namespace, policy: str) -> List[Dict[str, Any]]:
    if args.hyperparam_mode == "paper-random":
        return _paper_random_hyperparameter_configs(args, policy)
    return _grid_hyperparameter_configs(args, policy)


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policies = [policy for policy in args.policies.split(",") if policy]
    seeds = _parse_ints(args.seeds)

    jobs: List[Dict[str, Any]] = []
    for policy in policies:
        configs = hyperparameter_configs(args, policy)
        for seed in seeds:
            for sample_index, config in enumerate(configs):
                job_id = len(jobs)
                name = _job_name(job_id, policy, seed)
                jobs.append(
                    {
                        "job_id": job_id,
                        "policy": policy,
                        "seed": seed,
                        "hyperparam_mode": args.hyperparam_mode,
                        "hyperparam_sample": sample_index,
                        "total_timesteps": args.timesteps,
                        "rollout_steps": args.rollout_steps,
                        "num_envs": args.num_envs,
                        "hidden_size": args.hidden_size,
                        "update_epochs": config["update_epochs"],
                        "minibatches": config["minibatches"],
                        "learning_rate": config["learning_rate"],
                        "entropy_coef": config["entropy_coef"],
                        "clip_range": config["clip_range"],
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
    return sum(len(seeds) * len(hyperparameter_configs(args, policy)) for policy in policies)


def sampled_hyperparameter_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for policy in [policy for policy in args.policies.split(",") if policy]:
        for sample_index, config in enumerate(hyperparameter_configs(args, policy)):
            configs.append(
                {
                    "policy": policy,
                    "hyperparam_mode": args.hyperparam_mode,
                    "hyperparam_sample": sample_index,
                    "minibatches": int(config["minibatches"]),
                    "learning_rate": float(config["learning_rate"]),
                    "entropy_coef": float(config["entropy_coef"]),
                    "update_epochs": int(config["update_epochs"]),
                    "clip_range": float(config["clip_range"]),
                }
            )
    return configs


def paper_protocol_status(
    args: argparse.Namespace,
    jobs_planned: int | None = None,
    jobs_completed: int | None = None,
    jobs_failed: int | None = None,
) -> Dict[str, Any]:
    policies = [policy for policy in args.policies.split(",") if policy]
    seeds = _parse_ints(args.seeds)
    nminibatches = _parse_ints(args.nminibatches)
    ent_coefs = _parse_floats(args.ent_coefs)
    update_epochs = _parse_update_epochs(args.update_epochs)
    clip_ranges = _parse_floats(args.clip_ranges)
    learning_rates = _parse_floats(args.learning_rates)
    requested_policy_set = set(policies)
    full_baseline_policy_set = requested_policy_set == {"mlp", "lstm"} and len(policies) == 2
    grid_mode = args.hyperparam_mode == "grid"
    paper_random_mode = args.hyperparam_mode == "paper-random"
    jobs_expected_for_selected_space = count_uncapped_jobs(args)
    has_full_mlp_grid = (
        grid_mode
        and "mlp" in policies
        and nminibatches == PAPER_NMINIBATCHES
        and ent_coefs == PAPER_ENT_COEFS
        and update_epochs == PAPER_UPDATE_EPOCHS
        and clip_ranges == PAPER_CLIP_RANGES
    )
    paper_timestep_budget = int(args.timesteps) == PAPER_TIMESTEPS
    paper_test_horizon = int(args.test_max_steps) == PAPER_TEST_MAX_STEPS
    paper_eval_rollouts = int(args.eval_rollouts) == PAPER_EVAL_ROLLOUTS
    paper_seed_count = len(seeds) == 5 and len(set(seeds)) == 5
    grid_learning_rates_in_interval = (
        grid_mode
        and bool(learning_rates)
        and all(PAPER_LEARNING_RATE_MIN <= value <= PAPER_LEARNING_RATE_MAX for value in learning_rates)
    )
    paper_random_learning_rates_in_interval = paper_random_mode and int(args.hyperparam_samples) > 0
    learning_rates_in_interval = grid_learning_rates_in_interval or paper_random_learning_rates_in_interval
    full_default_learning_rate_grid = (
        grid_mode
        and len(learning_rates) == len(DEFAULT_LEARNING_RATES)
        and set(learning_rates) == set(DEFAULT_LEARNING_RATES)
    )
    paper_random_sample_count = (
        paper_random_mode
        and int(args.hyperparam_samples) == PAPER_HYPERPARAMETER_SAMPLES
    )
    truncated = args.max_configs is not None
    paper_scale_plan = (
        paper_timestep_budget
        and paper_test_horizon
        and paper_eval_rollouts
        and paper_seed_count
        and full_baseline_policy_set
        and paper_random_sample_count
        and not truncated
        and not args.quick
    )
    all_planned_jobs_completed = (
        jobs_planned is not None
        and jobs_completed is not None
        and jobs_failed is not None
        and jobs_planned > 0
        and jobs_completed == jobs_planned
        and jobs_failed == 0
    )
    planned_job_count_matches_selected_space = (
        jobs_planned is not None and jobs_planned == jobs_expected_for_selected_space
    )
    return {
        "paper_timestep_budget": paper_timestep_budget,
        "paper_test_horizon": paper_test_horizon,
        "paper_test_horizon_steps": PAPER_TEST_MAX_STEPS,
        "selected_test_max_steps": int(args.test_max_steps),
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": int(args.eval_rollouts),
        "uses_paper_eval_rollouts": paper_eval_rollouts,
        "paper_seed_count": paper_seed_count,
        "selected_seed_count": len(seeds),
        "includes_ppo_mlp": "mlp" in requested_policy_set,
        "includes_ppo_lstm": "lstm" in requested_policy_set,
        "full_baseline_policy_set": full_baseline_policy_set,
        "hyperparam_mode": args.hyperparam_mode,
        "grid_hyperparameter_search": grid_mode,
        "paper_random_hyperparameter_search": paper_random_mode,
        "paper_random_hyperparameter_samples": int(args.hyperparam_samples),
        "paper_random_sample_count": paper_random_sample_count,
        "full_reported_mlp_grid": has_full_mlp_grid,
        "ppo_lstm_minibatches_fixed_to_one": "lstm" in requested_policy_set,
        "learning_rate_interval_only": True,
        "grid_learning_rate_values_within_reported_interval": grid_learning_rates_in_interval,
        "paper_random_learning_rate_values_within_reported_interval": paper_random_learning_rates_in_interval,
        "learning_rate_values_within_reported_interval": learning_rates_in_interval,
        "full_default_learning_rate_grid": full_default_learning_rate_grid,
        "truncated_by_max_configs": truncated,
        "quick_diagnostic": bool(args.quick),
        "dry_run_only": bool(args.dry_run),
        "jobs_expected_for_selected_space": jobs_expected_for_selected_space,
        "planned_job_count_matches_selected_space": planned_job_count_matches_selected_space,
        "all_planned_jobs_completed": all_planned_jobs_completed,
        "paper_scale_plan": paper_scale_plan,
        "paper_scale_execution": (
            paper_scale_plan
            and not args.dry_run
            and planned_job_count_matches_selected_space
            and all_planned_jobs_completed
        ),
    }


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
                "best_train_steps": float(best["train_steps"]),
                "best_test_steps": float(best["test_steps"]),
                "best_train_survival_seconds": float(best["train_survival_seconds"]),
                "best_test_survival_seconds": float(best["test_survival_seconds"]),
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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _best_completed_job(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["train_success"]),
            float(row["train_reward"]),
            -int(row["job_id"]),
        ),
    )


def _hyperparameter_summary_rank(row: Dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(row["train_success_mean"]),
        float(row["train_reward_mean"]),
        int(row["seed_count"]),
        -int(row["hyperparam_sample"]),
    )


def summarize_hyperparameter_configs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int], List[Dict[str, Any]]] = {}
    order: List[tuple[str, str, int]] = []
    for row in results:
        key = (
            str(row["policy"]),
            str(row["hyperparam_mode"]),
            int(row["hyperparam_sample"]),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    summary: List[Dict[str, Any]] = []
    for policy, hyperparam_mode, hyperparam_sample in order:
        group = grouped[(policy, hyperparam_mode, hyperparam_sample)]
        first = min(group, key=lambda row: int(row["job_id"]))
        best = _best_completed_job(group)
        seeds = sorted({int(row["seed"]) for row in group})
        train_success = [float(row["train_success"]) for row in group]
        test_success = [float(row["test_success"]) for row in group]
        train_reward = [float(row["train_reward"]) for row in group]
        test_reward = [float(row["test_reward"]) for row in group]
        train_steps = [float(row["train_steps"]) for row in group]
        test_steps = [float(row["test_steps"]) for row in group]
        train_survival_seconds = [float(row["train_survival_seconds"]) for row in group]
        test_survival_seconds = [float(row["test_survival_seconds"]) for row in group]
        summary.append(
            {
                "policy": policy,
                "hyperparam_mode": hyperparam_mode,
                "hyperparam_sample": hyperparam_sample,
                "jobs_completed": len(group),
                "seed_count": len(seeds),
                "seeds_completed": ",".join(str(seed) for seed in seeds),
                "train_success_mean": _mean(train_success),
                "train_success_std": _sample_std(train_success),
                "test_success_mean": _mean(test_success),
                "test_success_std": _sample_std(test_success),
                "train_reward_mean": _mean(train_reward),
                "train_reward_std": _sample_std(train_reward),
                "test_reward_mean": _mean(test_reward),
                "test_reward_std": _sample_std(test_reward),
                "train_steps_mean": _mean(train_steps),
                "train_steps_std": _sample_std(train_steps),
                "test_steps_mean": _mean(test_steps),
                "test_steps_std": _sample_std(test_steps),
                "train_survival_seconds_mean": _mean(train_survival_seconds),
                "train_survival_seconds_std": _sample_std(train_survival_seconds),
                "test_survival_seconds_mean": _mean(test_survival_seconds),
                "test_survival_seconds_std": _sample_std(test_survival_seconds),
                "minibatches": int(first["minibatches"]),
                "learning_rate": float(first["learning_rate"]),
                "entropy_coef": float(first["entropy_coef"]),
                "update_epochs": int(first["update_epochs"]),
                "clip_range": float(first["clip_range"]),
                "best_job_id": int(best["job_id"]),
                "best_seed": int(best["seed"]),
                "best_train_success": float(best["train_success"]),
                "best_test_success": float(best["test_success"]),
                "best_train_reward": float(best["train_reward"]),
                "best_test_reward": float(best["test_reward"]),
                "best_train_steps": float(best["train_steps"]),
                "best_test_steps": float(best["test_steps"]),
                "best_train_survival_seconds": float(best["train_survival_seconds"]),
                "best_test_survival_seconds": float(best["test_survival_seconds"]),
                "best_selected_timesteps": int(best["selected_timesteps"]),
                "best_output": best["output"],
                "best_metrics_output": best["metrics_output"],
                "is_best_hyperparam_for_policy": False,
            }
        )

    best_by_policy: Dict[str, Dict[str, Any]] = {}
    for row in summary:
        policy = str(row["policy"])
        current_best = best_by_policy.get(policy)
        if current_best is None or _hyperparameter_summary_rank(row) > _hyperparameter_summary_rank(current_best):
            best_by_policy[policy] = row
    for row in summary:
        row["is_best_hyperparam_for_policy"] = row is best_by_policy[str(row["policy"])]
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
        "train_steps": result.train_steps_mean,
        "test_steps": result.test_steps_mean,
        "train_survival_seconds": result.train_survival_seconds_mean,
        "test_survival_seconds": result.test_survival_seconds_mean,
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
        "hyperparam_mode": args.hyperparam_mode,
        "hyperparam_samples": args.hyperparam_samples,
        "hyperparam_seed": args.hyperparam_seed,
        "sampled_hyperparameters": sampled_hyperparameter_manifest(args),
        "paper_protocol_status": paper_protocol_status(args, len(jobs), completed, failed),
        "paper_space": {
            "timesteps": PAPER_TIMESTEPS,
            "test_max_steps": PAPER_TEST_MAX_STEPS,
            "eval_rollouts": PAPER_EVAL_ROLLOUTS,
            "reward_spec": cartpole_reward_spec(),
            "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
            "hyperparameter_sampling": (
                "10 uniformly sampled configs from the reported space per policy, evaluated for each seed"
            ),
            "hyperparameter_samples": PAPER_HYPERPARAMETER_SAMPLES,
            "nminibatches": PAPER_NMINIBATCHES,
            "lstm_nminibatches": [1],
            "ent_coef": PAPER_ENT_COEFS,
            "noptepochs": [3, 36],
            "cliprange": PAPER_CLIP_RANGES,
            "learning_rate_interval": [PAPER_LEARNING_RATE_MIN, PAPER_LEARNING_RATE_MAX],
            "learning_rate_values_used": _parse_floats(args.learning_rates),
            "learning_rate_note": (
                "The paper reports uniform sampling from this interval, not the exact sampled values. "
                "Use --hyperparam-mode paper-random for reproducible local samples; grid mode is "
                "a local diagnostic extension."
            ),
        },
        "artifacts": {
            "plan": str(args.outdir / "cartpole_ppo_sweep_plan.csv"),
            "results": str(args.outdir / "cartpole_ppo_sweep_results.csv"),
            "summary": str(args.outdir / "cartpole_ppo_sweep_summary.csv"),
            "hyperparameter_summary": str(args.outdir / "cartpole_ppo_sweep_hyperparam_summary.csv"),
            "failures": str(args.outdir / "cartpole_ppo_sweep_failures.csv"),
            "checkpoints": str(args.outdir / "checkpoints"),
            "metrics": str(args.outdir / "metrics"),
        },
        "selection_rule": "single completed job per policy: max train_success, then train_reward, then lower job_id",
        "hyperparameter_selection_rule": (
            "per policy and hyperparameter sample: aggregate completed seeds by mean train_success, "
            "then mean train_reward, then completed seed count, then lower sample id"
        ),
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
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument(
        "--hyperparam-mode",
        choices=("paper-random", "grid"),
        default="paper-random",
        help=(
            "paper-random samples the paper's 10 hyperparameter instances from the reported ranges; "
            "grid enumerates the explicit local grids below."
        ),
    )
    parser.add_argument("--hyperparam-samples", type=int, default=PAPER_HYPERPARAMETER_SAMPLES)
    parser.add_argument("--hyperparam-seed", type=int, default=0)
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
                write_csv(
                    args.outdir / "cartpole_ppo_sweep_hyperparam_summary.csv",
                    HYPERPARAMETER_SUMMARY_FIELDS,
                    summarize_hyperparameter_configs(results),
                )
        write_csv(args.outdir / "cartpole_ppo_sweep_results.csv", RESULT_FIELDS, results)
        write_csv(args.outdir / "cartpole_ppo_sweep_summary.csv", SUMMARY_FIELDS, summarize_results(results))
        write_csv(
            args.outdir / "cartpole_ppo_sweep_hyperparam_summary.csv",
            HYPERPARAMETER_SUMMARY_FIELDS,
            summarize_hyperparameter_configs(results),
        )
    if failures:
        write_csv(args.outdir / "cartpole_ppo_sweep_failures.csv", FAILURE_FIELDS, failures)
    write_manifest(args, jobs, len(results), skipped, len(failures))
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_plan.csv'}")
    if not args.dry_run:
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_results.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_summary.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_hyperparam_summary.csv'}")
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_manifest.json'}")


if __name__ == "__main__":
    main()
