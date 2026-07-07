from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
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
    "device",
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
    "selected_seed_count",
    "selected_seeds",
    "missing_seeds",
    "complete_seed_coverage",
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
                        "device": args.device,
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
                    "minibatches": config["minibatches"],
                    "learning_rate": float(config["learning_rate"]),
                    "entropy_coef": float(config["entropy_coef"]),
                    "update_epochs": config["update_epochs"],
                    "clip_range": float(config["clip_range"]),
                }
            )
    return configs


def _policy_hyperparameter_configs(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    return {
        policy: hyperparameter_configs(args, policy)
        for policy in [policy for policy in args.policies.split(",") if policy]
    }


def _config_int_value_in(value: Any, allowed: Iterable[int]) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not numeric.is_integer():
        return False
    return int(numeric) in set(allowed)


def _all_config_int_values_in(values: Iterable[Any], allowed: Iterable[int]) -> bool:
    allowed_set = set(allowed)
    return all(_config_int_value_in(value, allowed_set) for value in values)


def _all_config_float_values_in(values: Iterable[Any], allowed: Iterable[float]) -> bool:
    allowed_values = tuple(float(value) for value in allowed)
    return all(
        any(math.isclose(float(value), allowed_value, rel_tol=0.0, abs_tol=1e-12) for allowed_value in allowed_values)
        for value in values
    )


def _all_learning_rates_in_paper_interval(configs_by_policy: Dict[str, List[Dict[str, Any]]]) -> bool:
    values = [
        float(config["learning_rate"])
        for configs in configs_by_policy.values()
        for config in configs
    ]
    return bool(values) and all(PAPER_LEARNING_RATE_MIN <= value <= PAPER_LEARNING_RATE_MAX for value in values)


def _configs_follow_paper_ranges(configs_by_policy: Dict[str, List[Dict[str, Any]]]) -> bool:
    all_configs = [config for configs in configs_by_policy.values() for config in configs]
    if not all_configs:
        return False
    return (
        _all_config_float_values_in((config["entropy_coef"] for config in all_configs), PAPER_ENT_COEFS)
        and _all_config_int_values_in((config["update_epochs"] for config in all_configs), PAPER_UPDATE_EPOCHS)
        and _all_config_float_values_in((config["clip_range"] for config in all_configs), PAPER_CLIP_RANGES)
        and _all_learning_rates_in_paper_interval(configs_by_policy)
    )


def _configs_follow_paper_minibatch_rules(configs_by_policy: Dict[str, List[Dict[str, Any]]]) -> bool:
    mlp_configs = configs_by_policy.get("mlp", [])
    lstm_configs = configs_by_policy.get("lstm", [])
    mlp_ok = all(_config_int_value_in(config["minibatches"], PAPER_NMINIBATCHES) for config in mlp_configs)
    lstm_ok = all(_config_int_value_in(config["minibatches"], [1]) for config in lstm_configs)
    return mlp_ok and lstm_ok


def _configs_have_paper_sample_count(configs_by_policy: Dict[str, List[Dict[str, Any]]]) -> bool:
    return all(
        len(configs_by_policy.get(policy, [])) == PAPER_HYPERPARAMETER_SAMPLES
        for policy in ("mlp", "lstm")
    )


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
    distinct_seeds = sorted(set(seeds))
    full_baseline_policy_set = requested_policy_set == {"mlp", "lstm"} and len(policies) == 2
    grid_mode = args.hyperparam_mode == "grid"
    paper_random_mode = args.hyperparam_mode == "paper-random"
    configs_by_policy = _policy_hyperparameter_configs(args)
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
    sampled_learning_rates_in_interval = _all_learning_rates_in_paper_interval(configs_by_policy)
    sampled_configs_follow_paper_ranges = _configs_follow_paper_ranges(configs_by_policy)
    sampled_configs_follow_paper_minibatch_rules = _configs_follow_paper_minibatch_rules(configs_by_policy)
    paper_random_learning_rates_in_interval = paper_random_mode and sampled_learning_rates_in_interval
    learning_rates_in_interval = grid_learning_rates_in_interval or paper_random_learning_rates_in_interval
    full_default_learning_rate_grid = (
        grid_mode
        and len(learning_rates) == len(DEFAULT_LEARNING_RATES)
        and set(learning_rates) == set(DEFAULT_LEARNING_RATES)
    )
    requested_paper_random_sample_count = (
        paper_random_mode
        and int(args.hyperparam_samples) == PAPER_HYPERPARAMETER_SAMPLES
    )
    generated_paper_random_sample_count = paper_random_mode and _configs_have_paper_sample_count(configs_by_policy)
    paper_random_sample_count = requested_paper_random_sample_count and generated_paper_random_sample_count
    truncated = args.max_configs is not None
    paper_scale_plan = (
        paper_timestep_budget
        and paper_test_horizon
        and paper_eval_rollouts
        and paper_seed_count
        and full_baseline_policy_set
        and paper_random_sample_count
        and sampled_configs_follow_paper_ranges
        and sampled_configs_follow_paper_minibatch_rules
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
        "selected_seeds": seeds,
        "distinct_seeds": distinct_seeds,
        "selected_seed_count": len(seeds),
        "distinct_seed_count": len(distinct_seeds),
        "selected_policies": policies,
        "distinct_policies": sorted(requested_policy_set),
        "includes_ppo_mlp": "mlp" in requested_policy_set,
        "includes_ppo_lstm": "lstm" in requested_policy_set,
        "full_baseline_policy_set": full_baseline_policy_set,
        "hyperparam_mode": args.hyperparam_mode,
        "grid_hyperparameter_search": grid_mode,
        "paper_random_hyperparameter_search": paper_random_mode,
        "paper_random_hyperparameter_samples": int(args.hyperparam_samples),
        "requested_paper_random_sample_count": requested_paper_random_sample_count,
        "generated_paper_random_sample_count": generated_paper_random_sample_count,
        "paper_random_sample_count": paper_random_sample_count,
        "full_reported_mlp_grid": has_full_mlp_grid,
        "sampled_hyperparameters_follow_paper_ranges": sampled_configs_follow_paper_ranges,
        "sampled_hyperparameters_follow_paper_minibatch_rules": sampled_configs_follow_paper_minibatch_rules,
        "ppo_lstm_minibatches_fixed_to_one": (
            "lstm" in requested_policy_set
            and all(_config_int_value_in(config["minibatches"], [1]) for config in configs_by_policy.get("lstm", []))
        ),
        "learning_rate_interval_only": True,
        "grid_learning_rate_values_within_reported_interval": grid_learning_rates_in_interval,
        "sampled_learning_rate_values_within_reported_interval": sampled_learning_rates_in_interval,
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


def _hyperparameter_summary_rank(row: Dict[str, Any]) -> tuple[int, float, float, int, int]:
    return (
        int(bool(row["complete_seed_coverage"])),
        float(row["train_success_mean"]),
        float(row["train_reward_mean"]),
        int(row["seed_count"]),
        -int(row["hyperparam_sample"]),
    )


def summarize_hyperparameter_configs(
    results: List[Dict[str, Any]],
    selected_seeds: List[int] | None = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, int], List[Dict[str, Any]]] = {}
    order: List[tuple[str, str, int]] = []
    expected_seeds = sorted(set(selected_seeds)) if selected_seeds is not None else None
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
        seed_target = expected_seeds if expected_seeds is not None else seeds
        missing_seeds = [seed for seed in seed_target if seed not in seeds]
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
                "selected_seed_count": len(seed_target),
                "selected_seeds": ",".join(str(seed) for seed in seed_target),
                "missing_seeds": ",".join(str(seed) for seed in missing_seeds),
                "complete_seed_coverage": bool(seed_target) and not missing_seeds,
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


def _metric_float_matches(row: Dict[str, str], metrics: Dict[str, Any], row_field: str, metric_field: str) -> bool:
    try:
        row_value = float(row[row_field])
        metric_value = float(metrics[metric_field])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isclose(row_value, metric_value, rel_tol=1e-9, abs_tol=1e-9)


def _metric_int_matches(row: Dict[str, str], metrics: Dict[str, Any], row_field: str, metric_field: str) -> bool:
    try:
        row_value = int(float(row[row_field]))
        metric_value = int(metrics[metric_field])
    except (KeyError, TypeError, ValueError):
        return False
    return row_value == metric_value


def _status_bool_matches(status: Dict[str, Any], field: str, expected: bool) -> bool:
    return status.get(field) is expected


def _status_int_matches(status: Dict[str, Any], field: str, expected: int) -> bool:
    try:
        return int(status[field]) == expected
    except (KeyError, TypeError, ValueError):
        return False


def _expected_selected_device(requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested.startswith("cuda"):
        return requested
    return requested


def _torch_device_block_matches_resumable_job(job: Dict[str, Any], torch_device: Dict[str, Any]) -> bool:
    requested = str(job["device"]).strip().lower()
    if torch_device.get("requested") != requested:
        return False
    selected = torch_device.get("selected")
    if not isinstance(selected, str):
        return False
    if requested == "auto":
        return selected in {"cpu", "cuda"} or selected.startswith("cuda:")
    if requested == "cpu":
        return selected == "cpu"
    if requested.startswith("cuda"):
        return selected == _expected_selected_device(requested) and torch_device.get("fallback_reason") is None
    return selected == requested


def _torch_device_matches_resumable_job(job: Dict[str, Any], status: Dict[str, Any]) -> bool:
    torch_device = status.get("torch_device")
    return isinstance(torch_device, dict) and _torch_device_block_matches_resumable_job(job, torch_device)


def _protocol_status_matches_resumable_job(job: Dict[str, Any], status: Dict[str, Any]) -> bool:
    policy = str(job["policy"])
    total_timesteps = int(job["total_timesteps"])
    test_max_steps = int(job["test_max_steps"])
    eval_rollouts = int(job["eval_rollouts"])
    minibatches = int(job["minibatches"])
    paper_timestep_budget = total_timesteps == PAPER_TIMESTEPS
    paper_test_horizon = test_max_steps == PAPER_TEST_MAX_STEPS
    paper_eval_rollouts = eval_rollouts == PAPER_EVAL_ROLLOUTS
    lstm_minibatches_ok = policy != "lstm" or minibatches == 1
    single_run_matches_paper_budget = (
        paper_timestep_budget
        and paper_test_horizon
        and paper_eval_rollouts
        and lstm_minibatches_ok
    )

    return (
        status.get("policy_type") == policy
        and _torch_device_matches_resumable_job(job, status)
        and _status_int_matches(status, "selected_test_max_steps", test_max_steps)
        and _status_int_matches(status, "paper_test_horizon_steps", PAPER_TEST_MAX_STEPS)
        and _status_int_matches(status, "selected_eval_rollouts", eval_rollouts)
        and _status_int_matches(status, "paper_eval_rollouts", PAPER_EVAL_ROLLOUTS)
        and _status_bool_matches(status, "paper_timestep_budget", paper_timestep_budget)
        and _status_bool_matches(status, "paper_test_horizon", paper_test_horizon)
        and _status_bool_matches(status, "uses_paper_eval_rollouts", paper_eval_rollouts)
        and _status_bool_matches(status, "ppo_lstm_minibatches_fixed_to_one", lstm_minibatches_ok)
        and _status_bool_matches(status, "local_supervised_warm_start", False)
        and _status_bool_matches(status, "no_local_supervised_warm_start", True)
        and _status_bool_matches(status, "single_run_matches_paper_budget", single_run_matches_paper_budget)
        and _status_bool_matches(status, "five_seed_hyperparameter_search", False)
        and _status_bool_matches(status, "paper_scale_baseline_protocol", False)
    )


def _metrics_match_resumable_row(job: Dict[str, Any], row: Dict[str, str]) -> bool:
    try:
        metrics = json.loads(Path(str(row["metrics_output"])).read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    if not isinstance(metrics, dict):
        return False
    command = metrics.get("command")
    config = metrics.get("config")
    selected_result = metrics.get("selected_result")
    paper_status = metrics.get("paper_protocol_status")
    torch_device = metrics.get("torch_device")
    if not isinstance(command, str) or not command.strip():
        return False
    if not isinstance(config, dict) or not isinstance(selected_result, dict) or not isinstance(paper_status, dict):
        return False
    if not isinstance(torch_device, dict) or not _torch_device_block_matches_resumable_job(job, torch_device):
        return False

    config_checks = {
        "policy_type": str(job["policy"]),
        "total_timesteps": int(job["total_timesteps"]),
        "rollout_steps": int(job["rollout_steps"]),
        "num_envs": int(job["num_envs"]),
        "hidden_size": int(job["hidden_size"]),
        "update_epochs": int(job["update_epochs"]),
        "minibatches": int(job["minibatches"]),
        "eval_rollouts": int(job["eval_rollouts"]),
        "eval_test_max_steps": int(job["test_max_steps"]),
        "eval_interval": int(job["eval_interval"]),
        "seed": int(job["seed"]),
        "metrics_output": str(job["metrics_output"]),
        "device": str(job["device"]),
    }
    for key, expected in config_checks.items():
        if config.get(key) != expected:
            return False
    for key in ("learning_rate", "entropy_coef", "clip_range"):
        try:
            if not math.isclose(float(config[key]), float(job[key]), rel_tol=1e-12, abs_tol=1e-12):
                return False
        except (KeyError, TypeError, ValueError):
            return False

    if not _protocol_status_matches_resumable_job(job, paper_status):
        return False

    if not _metric_int_matches(row, selected_result, "selected_timesteps", "timesteps"):
        return False
    metric_pairs = (
        ("train_success", "train_success_rate"),
        ("test_success", "test_success_rate"),
        ("train_reward", "train_reward_mean"),
        ("test_reward", "test_reward_mean"),
        ("train_steps", "train_steps_mean"),
        ("test_steps", "test_steps_mean"),
        ("train_survival_seconds", "train_survival_seconds_mean"),
        ("test_survival_seconds", "test_survival_seconds_mean"),
    )
    return all(_metric_float_matches(row, selected_result, row_field, metric_field) for row_field, metric_field in metric_pairs)


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
    if not _metrics_match_resumable_row(job, row):
        return None
    return row


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job["policy"] not in {"mlp", "lstm"}:
        raise ValueError(f"unknown policy_type: {job['policy']}")

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
        device=str(job["device"]),
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


def _log_progress(message: str) -> None:
    print(message, flush=True)


def _job_progress_prefix(job: Dict[str, Any], index: int, total: int) -> str:
    return (
        f"job {index}/{total} id={job['job_id']} policy={job['policy']} "
        f"seed={job['seed']} sample={job['hyperparam_sample']} "
        f"timesteps={job['total_timesteps']} device={job['device']}"
    )


def failed_job_row(job: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    return {
        **job,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _torch_runtime_status() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {
            "torch_importable": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "torch_import_error": f"{type(exc).__name__}: {exc}",
            "cuda_probe_error": None,
        }

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return {
            "torch_importable": True,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "torch_import_error": None,
            "cuda_probe_error": f"{type(exc).__name__}: {exc}",
        }
    devices = []
    cuda_probe_error = None
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
            for index in range(device_count):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_bytes": int(properties.total_memory),
                    }
                )
        except Exception as exc:
            cuda_available = False
            devices = []
            cuda_probe_error = f"{type(exc).__name__}: {exc}"
    return {
        "torch_importable": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "torch_import_error": None,
        "cuda_probe_error": cuda_probe_error,
    }


def _nvidia_smi_status() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    gpus = []
    for index, line in enumerate(completed.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_total_mib = int(float(parts[1]))
        except ValueError:
            memory_total_mib = None
        gpus.append(
            {
                "index": index,
                "name": parts[0],
                "memory_total_mib": memory_total_mib,
                "driver_version": parts[2],
            }
        )
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "error": None,
    }


def runtime_preflight(args: argparse.Namespace, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    planned_training_timesteps = sum(int(job["total_timesteps"]) for job in jobs)
    planned_eval_rollouts = sum(int(job["eval_rollouts"]) for job in jobs)
    selected_space_reference_job_count = count_uncapped_jobs(args)
    paper_scale_job_count = 2 * 5 * PAPER_HYPERPARAMETER_SAMPLES
    return {
        "torch": _torch_runtime_status(),
        "nvidia_smi": _nvidia_smi_status(),
        "jobs_planned": len(jobs),
        "jobs_uncapped_for_selected_space": selected_space_reference_job_count,
        "selected_space_reference_job_count": selected_space_reference_job_count,
        "paper_scale_reference_job_count": paper_scale_job_count,
        "planned_training_timesteps": planned_training_timesteps,
        "paper_scale_reference_training_timesteps": paper_scale_job_count * PAPER_TIMESTEPS,
        "planned_eval_rollouts": planned_eval_rollouts,
        "paper_scale_reference_eval_rollouts": paper_scale_job_count * PAPER_EVAL_ROLLOUTS,
        "note": (
            "This is a launch preflight for local capacity/provenance only. It does not prove "
            "paper-scale execution; paper_scale_execution still requires completed matching jobs."
        ),
    }


def write_manifest(
    args: argparse.Namespace,
    jobs: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    completed: int,
    skipped: int,
    jobs_run_this_invocation: int | None = None,
) -> None:
    summary_rows = summarize_results(results) if results else []
    hyperparameter_summary_rows = summarize_hyperparameter_configs(results, _parse_ints(args.seeds)) if results else []
    if jobs_run_this_invocation is None:
        jobs_run_this_invocation = completed - skipped
    manifest = {
        "artifact_kind": "cartpole_ppo_sweep_manifest",
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
        "jobs_failed": len(failures),
        "jobs_skipped_existing": skipped,
        "jobs_run_this_invocation": jobs_run_this_invocation,
        "max_configs": args.max_configs,
        "hyperparam_mode": args.hyperparam_mode,
        "hyperparam_samples": args.hyperparam_samples,
        "hyperparam_seed": args.hyperparam_seed,
        "device": args.device,
        "sampled_hyperparameters": sampled_hyperparameter_manifest(args),
        "paper_protocol_status": paper_protocol_status(args, len(jobs), completed, len(failures)),
        "runtime_preflight": runtime_preflight(args, jobs),
        "summary": summary_rows,
        "hyperparameter_summary": hyperparameter_summary_rows,
        "failure_summary": failures,
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
            "per policy and hyperparameter sample: prefer configs with complete selected-seed coverage, "
            "then aggregate completed seeds by mean train_success, then mean train_reward, "
            "then completed seed count, then lower sample id"
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
    parser.add_argument("--device", default="auto", help="Torch device for PPO jobs: auto, cpu, cuda, or cuda:N.")
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
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild plan, summaries, and manifest from existing result/failure CSVs without running jobs.",
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


def read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validated_existing_results_for_jobs(
    jobs: List[Dict[str, Any]],
    existing_results: Dict[int, Dict[str, str]],
) -> tuple[List[Dict[str, str]], int]:
    results: List[Dict[str, str]] = []
    rejected = 0
    for job in jobs:
        if int(job["job_id"]) not in existing_results:
            continue
        result = resumable_result_for_job(job, existing_results)
        if result is None:
            rejected += 1
            continue
        results.append(result)
    return results, rejected


def failure_row_matches_job(job: Dict[str, Any], row: Dict[str, str]) -> bool:
    return all(str(row.get(field, "")) == str(job[field]) for field in PLAN_FIELDS)


def validated_failure_rows_for_jobs(
    jobs: List[Dict[str, Any]],
    failure_rows: List[Dict[str, str]],
    completed_job_ids: set[int] | None = None,
) -> tuple[List[Dict[str, str]], int]:
    completed_job_ids = completed_job_ids or set()
    jobs_by_id = {int(job["job_id"]): job for job in jobs}
    failures: List[Dict[str, str]] = []
    rejected = 0
    for row in failure_rows:
        job_id = row.get("job_id", "")
        if not str(job_id).isdigit():
            rejected += 1
            continue
        numeric_job_id = int(job_id)
        if numeric_job_id in completed_job_ids:
            rejected += 1
            continue
        job = jobs_by_id.get(numeric_job_id)
        if job is None or not failure_row_matches_job(job, row):
            rejected += 1
            continue
        failures.append(row)
    return failures, rejected


def write_result_sidecars(args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    write_csv(args.outdir / "cartpole_ppo_sweep_results.csv", RESULT_FIELDS, results)
    if results:
        write_csv(args.outdir / "cartpole_ppo_sweep_summary.csv", SUMMARY_FIELDS, summarize_results(results))
        write_csv(
            args.outdir / "cartpole_ppo_sweep_hyperparam_summary.csv",
            HYPERPARAMETER_SUMMARY_FIELDS,
            summarize_hyperparameter_configs(results, _parse_ints(args.seeds)),
        )
    else:
        write_csv(args.outdir / "cartpole_ppo_sweep_summary.csv", SUMMARY_FIELDS, [])
        write_csv(args.outdir / "cartpole_ppo_sweep_hyperparam_summary.csv", HYPERPARAMETER_SUMMARY_FIELDS, [])


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_csv(args.outdir / "cartpole_ppo_sweep_plan.csv", PLAN_FIELDS, jobs)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    skipped = 0
    if args.refresh_manifest:
        results, rejected_results = validated_existing_results_for_jobs(
            jobs,
            read_existing_results(args.outdir / "cartpole_ppo_sweep_results.csv"),
        )
        failures, rejected_failures = validated_failure_rows_for_jobs(
            jobs,
            read_csv_rows(args.outdir / "cartpole_ppo_sweep_failures.csv"),
            {int(row["job_id"]) for row in results},
        )
        write_result_sidecars(args, results)
        write_manifest(
            args,
            jobs,
            results,
            failures,
            len(results),
            skipped,
            jobs_run_this_invocation=0,
        )
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_plan.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_manifest.json'}")
        if rejected_results:
            print(f"ignored {rejected_results} result rows without matching checkpoint/metrics provenance")
        if rejected_failures:
            print(f"ignored {rejected_failures} failure rows that do not match the current plan")
        return
    if not args.dry_run:
        _log_progress(
            f"starting CartPole PPO sweep: jobs_planned={len(jobs)} "
            f"resume={args.resume} outdir={args.outdir}"
        )
        existing_results = (
            read_existing_results(args.outdir / "cartpole_ppo_sweep_results.csv")
            if args.resume
            else {}
        )
        for index, job in enumerate(jobs, start=1):
            prefix = _job_progress_prefix(job, index, len(jobs))
            existing = resumable_result_for_job(job, existing_results) if args.resume else None
            if existing is not None:
                _log_progress(f"skipping completed {prefix}")
                results.append(existing)
                skipped += 1
            else:
                try:
                    _log_progress(f"running {prefix}")
                    result = run_job(job)
                    results.append(result)
                    _log_progress(
                        f"finished {prefix} train_success={result['train_success']:.3f} "
                        f"test_success={result['test_success']:.3f} "
                        f"selected_timesteps={result['selected_timesteps']}"
                    )
                except Exception as exc:
                    failure = failed_job_row(job, exc)
                    failures.append(failure)
                    write_csv(args.outdir / "cartpole_ppo_sweep_failures.csv", FAILURE_FIELDS, failures)
                    write_result_sidecars(args, results)
                    write_manifest(args, jobs, results, failures, len(results), skipped)
                    if not args.continue_on_error:
                        raise
                    _log_progress(f"failed {prefix} error={type(exc).__name__}: {exc}")
                    continue
            write_result_sidecars(args, results)
            write_manifest(args, jobs, results, failures, len(results), skipped)
        write_result_sidecars(args, results)
    if failures:
        write_csv(args.outdir / "cartpole_ppo_sweep_failures.csv", FAILURE_FIELDS, failures)
    write_manifest(args, jobs, results, failures, len(results), skipped)
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_plan.csv'}")
    if not args.dry_run:
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_results.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_summary.csv'}")
        print(f"wrote {args.outdir / 'cartpole_ppo_sweep_hyperparam_summary.csv'}")
    print(f"wrote {args.outdir / 'cartpole_ppo_sweep_manifest.json'}")


if __name__ == "__main__":
    main()
