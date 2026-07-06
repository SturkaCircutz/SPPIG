from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Keep this script runnable from a fresh checkout without requiring package install.
sys.path.insert(0, str(SRC))

from cartpole_env import PAPER_EVAL_ROLLOUTS, CartpoleEnv, cartpole_reward_spec, cartpole_space_spec  # noqa: E402
from cartpole_direct_opt import (  # noqa: E402
    DirectOptConfig,
    cartpole_direct_opt_protocol_status,
    direct_opt_metrics,
    run_cartpole_direct_opt,
)
from cartpole_synthesis import (  # noqa: E402
    CartpoleSynthesisConfig,
    cartpole_synthesis_algorithm_provenance,
    cartpole_synthesis_protocol_status,
    cartpole_switch_fit_diagnostics,
    synthesize_cartpole_student_with_history,
)
from train_cartpole_psm import (  # noqa: E402
    serialize_trace_history,
    serialize_traces,
    summarize_adaptive_teacher_history,
    summarize_policy_evaluation,
    summarize_student,
    summarize_synthesis_history,
    summarize_traces,
)

try:
    from ppo_cartpole import PPOConfig, ppo_paper_protocol_status, train_ppo_cartpole  # noqa: E402

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
    "train_steps",
    "test_steps",
    "train_survival_seconds",
    "test_survival_seconds",
    "eval_rollouts",
    "test_horizon_steps",
    "timesteps",
    "checkpoint",
    "metrics_output",
    "traces_output",
    "command",
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
    "train_steps_mean",
    "train_steps_std",
    "test_steps_mean",
    "test_steps_std",
    "train_survival_seconds_mean",
    "train_survival_seconds_std",
    "test_survival_seconds_mean",
    "test_survival_seconds_std",
    "best_seed_by_train",
    "best_train_success",
    "best_test_success",
    "best_train_reward",
    "best_test_reward",
    "best_train_steps",
    "best_test_steps",
    "best_train_survival_seconds",
    "best_test_survival_seconds",
    "eval_rollouts",
    "test_horizon_steps",
    "best_timesteps",
    "best_checkpoint",
    "best_metrics_output",
    "best_traces_output",
    "best_command",
]

DIRECT_OPT_PROTOCOL_REQUIREMENT_KEYS = {
    "paper_batch_size_and_batch_refinement",
    "paper_parallel_threads",
    "paper_time_limit",
    "full_continuous_one_hot_switch_grammar",
    "full_initial_state_distribution",
    "full_test_horizon",
    "paper_eval_rollouts",
}


def run_psm(
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    quick: bool,
    outdir: Path,
    teacher_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = psm_config(seed, quick, teacher_overrides)
    student, traces, synthesis_history = synthesize_cartpole_student_with_history(cfg)
    policy = student.to_deterministic_policy()
    # The paper's test horizon is 300s; test_max_steps is only exposed so tests
    # can cap runtime without changing the environment definition itself.
    evaluation = summarize_policy_evaluation(
        policy,
        eval_rollouts,
        test_max_steps,
        train_seed=100 + seed,
        test_seed=200 + seed,
    )
    train = evaluation["train"]
    test = evaluation["test"]
    metrics_path = outdir / "metrics" / f"psm_seed{seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path = outdir / "traces" / f"psm_seed{seed}_teacher_traces.json"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    metrics = {
        "command": command,
        "config": asdict(cfg),
        "algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "paper_protocol_status": cartpole_synthesis_protocol_status(
            cfg,
            eval_rollouts,
            test_max_steps,
            quick,
        ),
        "eval_rollouts": eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "test_max_steps": test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "num_traces": len(traces),
        "traces_output": str(traces_path),
        "adaptive_teacher_summary": summarize_adaptive_teacher_history(
            synthesis_history,
            cfg,
        ),
        "synthesis_history": summarize_synthesis_history(
            synthesis_history,
            eval_rollouts,
            test_max_steps,
            train_seed=100 + seed,
            test_seed=200 + seed,
            cfg=cfg,
        ),
        "trace_summary": summarize_traces(traces),
        "policy_description": policy.describe(),
        "probabilistic_student": summarize_student(student),
        "switch_fit_diagnostics": cartpole_switch_fit_diagnostics(traces, student),
        "train": train,
        "test": test,
    }
    trace_payload = {
        "command": command,
        "config": asdict(cfg),
        "seed": seed,
        "num_traces": len(traces),
        "metrics_output": str(metrics_path),
        "traces": serialize_traces(traces),
        "trace_history": serialize_trace_history(synthesis_history),
    }
    artifact_consistency = validate_psm_artifact_consistency(metrics, trace_payload)
    metrics["artifact_consistency"] = artifact_consistency
    trace_payload["artifact_consistency"] = artifact_consistency
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    traces_path.write_text(
        json.dumps(trace_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "policy": "Programmatic state machine",
        "seed": seed,
        "train_success": train["success_rate"],
        "test_success": test["success_rate"],
        "train_reward": train["reward_mean"],
        "test_reward": test["reward_mean"],
        "train_steps": train["steps_mean"],
        "test_steps": test["steps_mean"],
        "train_survival_seconds": train["survival_seconds_mean"],
        "test_survival_seconds": test["survival_seconds_mean"],
        "eval_rollouts": eval_rollouts,
        "test_horizon_steps": test_max_steps,
        "timesteps": 0,
        "metrics_output": str(metrics_path),
        "traces_output": str(traces_path),
        "command": metrics["command"],
        "config": asdict(cfg),
        "algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "paper_protocol_status": cartpole_synthesis_protocol_status(
            cfg,
            eval_rollouts,
            test_max_steps,
            quick,
        ),
        "policy_description": policy.describe(),
        "num_traces": len(traces),
    }


def validate_psm_artifact_consistency(
    metrics: Dict[str, Any],
    trace_payload: Dict[str, Any],
) -> Dict[str, Any]:
    config = metrics.get("config")
    if not isinstance(config, dict):
        raise ValueError("PSM metrics config must be a JSON object")
    teacher_student_iters = config.get("teacher_student_iters")
    if type(teacher_student_iters) is not int or teacher_student_iters < 1:
        raise ValueError("PSM metrics config must record a positive teacher_student_iters")
    if trace_payload.get("config") != config:
        raise ValueError("PSM trace sidecar config disagrees with metrics config")
    if trace_payload.get("command") != metrics.get("command"):
        raise ValueError("PSM trace sidecar command disagrees with metrics command")

    num_traces = metrics.get("num_traces")
    traces = trace_payload.get("traces")
    trace_history = trace_payload.get("trace_history")
    synthesis_history = metrics.get("synthesis_history")
    adaptive_teacher_summary = metrics.get("adaptive_teacher_summary")
    if type(num_traces) is not int:
        raise ValueError("PSM metrics num_traces must be an integer")
    if not isinstance(traces, list) or len(traces) != num_traces:
        raise ValueError("PSM trace sidecar trace count disagrees with metrics num_traces")
    if trace_payload.get("num_traces") != num_traces:
        raise ValueError("PSM trace sidecar num_traces disagrees with metrics num_traces")
    if not isinstance(trace_history, list) or len(trace_history) != teacher_student_iters:
        raise ValueError("PSM trace sidecar must record every teacher/student iteration")
    if not isinstance(synthesis_history, list) or len(synthesis_history) != teacher_student_iters:
        raise ValueError("PSM metrics synthesis_history must record every teacher/student iteration")
    if not isinstance(adaptive_teacher_summary, list) or len(adaptive_teacher_summary) != teacher_student_iters:
        raise ValueError("PSM metrics adaptive_teacher_summary must record every teacher/student iteration")

    expected_iterations = list(range(1, teacher_student_iters + 1))
    trace_iterations = [entry.get("iteration") if isinstance(entry, dict) else None for entry in trace_history]
    synthesis_iterations = [
        entry.get("iteration") if isinstance(entry, dict) else None for entry in synthesis_history
    ]
    adaptive_iterations = [
        entry.get("iteration") if isinstance(entry, dict) else None for entry in adaptive_teacher_summary
    ]
    if trace_iterations != expected_iterations:
        raise ValueError("PSM trace sidecar iteration sequence disagrees with config")
    if synthesis_iterations != expected_iterations:
        raise ValueError("PSM metrics synthesis_history iteration sequence disagrees with config")
    if adaptive_iterations != expected_iterations:
        raise ValueError("PSM metrics adaptive_teacher_summary iteration sequence disagrees with config")

    for index, history_entry in enumerate(trace_history):
        history_traces = history_entry.get("traces") if isinstance(history_entry, dict) else None
        history_num_traces = history_entry.get("num_traces") if isinstance(history_entry, dict) else None
        if not isinstance(history_traces, list) or history_num_traces != len(history_traces):
            raise ValueError("PSM trace sidecar history trace counts are inconsistent")
        synthesis_entry = synthesis_history[index]
        trace_summary = synthesis_entry.get("trace_summary") if isinstance(synthesis_entry, dict) else None
        if not isinstance(trace_summary, dict) or trace_summary.get("count") != history_num_traces:
            raise ValueError("PSM metrics synthesis_history trace counts disagree with trace sidecar")
        adaptive_entry = adaptive_teacher_summary[index]
        if not isinstance(adaptive_entry, dict) or adaptive_entry.get("trace_count") != history_num_traces:
            raise ValueError("PSM metrics adaptive_teacher_summary trace counts disagree with trace sidecar")

    if trace_history[-1].get("traces") != traces:
        raise ValueError("PSM final trace_history traces disagree with the top-level trace sidecar traces")
    if metrics.get("trace_summary", {}).get("count") != num_traces:
        raise ValueError("PSM metrics trace_summary count disagrees with metrics num_traces")
    final_evaluation = synthesis_history[-1].get("evaluation") if isinstance(synthesis_history[-1], dict) else None
    if (
        not isinstance(final_evaluation, dict)
        or final_evaluation.get("train") != metrics.get("train")
        or final_evaluation.get("test") != metrics.get("test")
    ):
        raise ValueError("PSM final synthesis_history evaluation disagrees with top-level metrics")

    status = metrics.get("paper_protocol_status", {})
    if not isinstance(status, dict) or status.get("synthesized_by_current_algorithm") is not True:
        raise ValueError("PSM metrics must mark runner-generated rows as current synthesized artifacts")
    return {
        "validated_by_runner": True,
        "num_traces": num_traces,
        "teacher_student_iters": teacher_student_iters,
        "trace_history_iterations": trace_iterations,
        "synthesis_history_iterations": synthesis_iterations,
        "adaptive_teacher_summary_iterations": adaptive_iterations,
        "final_trace_history_matches_traces": True,
        "final_evaluation_matches_top_level": True,
    }


def run_ppo(
    policy: str,
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    outdir: Path,
    eval_interval: int,
    quick: bool,
) -> Dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to run PPO baselines")
    artifact_stem = f"ppo_{policy}_seed{seed}"
    checkpoint_path = outdir / "checkpoints" / f"{artifact_stem}.pt"
    metrics_path = outdir / "metrics" / f"{artifact_stem}.json"
    command = " ".join(sys.argv)
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
        eval_interval=eval_interval,
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
        "train_steps": result.train_steps_mean,
        "test_steps": result.test_steps_mean,
        "train_survival_seconds": result.train_survival_seconds_mean,
        "test_survival_seconds": result.test_survival_seconds_mean,
        "eval_rollouts": eval_rollouts,
        "test_horizon_steps": test_max_steps,
        "timesteps": result.timesteps,
        "checkpoint": str(checkpoint_path),
        "metrics_output": str(metrics_path),
        "command": command,
        "config": asdict(cfg),
        "paper_protocol_status": ppo_paper_protocol_status(cfg),
    }


def run_direct_opt(
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    quick: bool,
    outdir: Path,
    parallel_threads: int,
    time_limit_seconds: float | None,
) -> Dict[str, Any]:
    cfg = DirectOptConfig(
        seed=seed,
        num_train_states=2 if quick else 10,
        random_candidates=8 if quick else 256,
        batch_size=2 if quick else 10,
        batch_refinement_rounds=1,
        local_refinement_steps=1 if quick else 2,
        restart_candidates_on_stall=1,
        parallel_threads=parallel_threads,
        time_limit_seconds=time_limit_seconds,
        eval_rollouts=eval_rollouts,
        test_max_steps=test_max_steps,
        quick=quick,
    )
    result = run_cartpole_direct_opt(cfg)
    metrics_path = outdir / "metrics" / f"direct_opt_seed{seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"command": " ".join(sys.argv), **direct_opt_metrics(result)}
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "policy": "Direct-Opt diagnostic",
        "seed": seed,
        "train_success": result.train_success_rate,
        "test_success": result.test_success_rate,
        "train_reward": result.train_reward_mean,
        "test_reward": result.test_reward_mean,
        "train_steps": result.train_steps_mean,
        "test_steps": result.test_steps_mean,
        "train_survival_seconds": result.train_survival_seconds_mean,
        "test_survival_seconds": result.test_survival_seconds_mean,
        "eval_rollouts": eval_rollouts,
        "test_horizon_steps": test_max_steps,
        "timesteps": 0,
        "metrics_output": str(metrics_path),
        "command": metrics["command"],
        "config": asdict(cfg),
        "algorithm_provenance": result.algorithm_provenance,
        "paper_protocol_status": cartpole_direct_opt_protocol_status(cfg),
        "policy_description": result.policy.describe(),
        "searched_candidates": result.searched_candidates,
    }


def load_ppo_sweep_manifest(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PPO sweep manifest must be a JSON object")
    if payload.get("artifact_kind") != "cartpole_ppo_sweep_manifest":
        raise ValueError("PPO sweep manifest must have artifact_kind=cartpole_ppo_sweep_manifest")
    status = payload.get("paper_protocol_status")
    if not isinstance(status, dict):
        raise ValueError("PPO sweep manifest must include paper_protocol_status")
    payload["manifest_path"] = str(path)
    return payload


def _truthy_bool(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value == "True")


def _int_set(values: List[Any]) -> set[int] | None:
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError):
        return None


def _manifest_seed_set(value: Any) -> set[int] | None:
    if isinstance(value, list):
        return _int_set(value)
    if isinstance(value, str):
        try:
            return {int(item) for item in value.split(",") if item != ""}
        except ValueError:
            return None
    return None


def _manifest_best_hyperparameter_rows_have_seed_coverage(
    manifest: Dict[str, Any],
    policies: List[Any],
    seeds: List[Any],
) -> bool:
    rows = manifest.get("hyperparameter_summary")
    if not isinstance(rows, list):
        return False
    expected_policies = set(policies) if isinstance(policies, list) else set()
    expected_seed_values = _int_set(seeds) if isinstance(seeds, list) else None
    if not expected_policies or expected_seed_values is None:
        return False
    expected_seed_count = len(expected_seed_values)
    best_rows_by_policy: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not _truthy_bool(row.get("is_best_hyperparam_for_policy")):
            continue
        policy = str(row.get("policy", ""))
        if policy in best_rows_by_policy:
            return False
        best_rows_by_policy[policy] = row
    if set(best_rows_by_policy) != expected_policies:
        return False
    for row in best_rows_by_policy.values():
        if not _truthy_bool(row.get("complete_seed_coverage")):
            return False
        if _manifest_seed_set(row.get("seeds_completed")) != expected_seed_values:
            return False
        if _manifest_seed_set(row.get("selected_seeds")) != expected_seed_values:
            return False
        try:
            if int(row.get("selected_seed_count")) != expected_seed_count:
                return False
            if int(row.get("seed_count")) != expected_seed_count:
                return False
        except (TypeError, ValueError):
            return False
        if str(row.get("missing_seeds", "")):
            return False
    return True


def ppo_sweep_evidence_status(manifest: Dict[str, Any] | None) -> Dict[str, Any]:
    if manifest is None:
        return {
            "manifest_path": None,
            "manifest_loaded": False,
            "paper_scale_plan": False,
            "paper_scale_execution": False,
            "paper_random_hyperparameter_search": False,
            "all_planned_jobs_completed": False,
            "jobs_planned": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "limitation": "No PPO sweep manifest was supplied to the reproduction runner.",
        }
    status = manifest["paper_protocol_status"]
    jobs_planned = int(manifest.get("jobs_planned", 0))
    jobs_completed = int(manifest.get("jobs_completed", 0))
    jobs_failed = int(manifest.get("jobs_failed", 0))
    jobs_uncapped = int(manifest.get("jobs_uncapped_for_selected_space", 0))
    policies = manifest.get("policies", [])
    seeds = manifest.get("seeds", [])
    hyperparam_mode = manifest.get("hyperparam_mode")
    hyperparam_samples = manifest.get("hyperparam_samples")
    top_level_full_policy_set = (
        isinstance(policies, list)
        and policies == ["mlp", "lstm"]
    )
    top_level_paper_seed_count = (
        isinstance(seeds, list)
        and len(seeds) == 5
        and len(set(seeds)) == 5
    )
    top_level_paper_random_samples = hyperparam_mode == "paper-random" and hyperparam_samples == 10
    paper_scale_plan = bool(status.get("paper_scale_plan"))
    raw_paper_scale_execution = bool(status.get("paper_scale_execution"))
    all_planned_jobs_completed = bool(status.get("all_planned_jobs_completed"))
    planned_job_count_matches_selected_space = bool(status.get("planned_job_count_matches_selected_space"))
    full_baseline_policy_set = bool(status.get("full_baseline_policy_set"))
    paper_seed_count = bool(status.get("paper_seed_count"))
    paper_timestep_budget = bool(status.get("paper_timestep_budget"))
    uses_paper_eval_rollouts = bool(status.get("uses_paper_eval_rollouts"))
    paper_test_horizon = bool(status.get("paper_test_horizon"))
    paper_random_hyperparameter_search = bool(status.get("paper_random_hyperparameter_search"))
    paper_random_sample_count = bool(status.get("paper_random_sample_count"))
    sampled_ranges_ok = bool(status.get("sampled_hyperparameters_follow_paper_ranges"))
    sampled_minibatches_ok = bool(status.get("sampled_hyperparameters_follow_paper_minibatch_rules"))
    best_hyperparameters_have_complete_seed_coverage = _manifest_best_hyperparameter_rows_have_seed_coverage(
        manifest,
        policies,
        seeds,
    )
    completed_job_counts_match = (
        jobs_planned > 0
        and jobs_uncapped == jobs_planned
        and jobs_completed == jobs_planned
        and jobs_failed == 0
    )
    acceptable_paper_scale_execution = (
        raw_paper_scale_execution
        and paper_scale_plan
        and all_planned_jobs_completed
        and planned_job_count_matches_selected_space
        and completed_job_counts_match
        and full_baseline_policy_set
        and top_level_full_policy_set
        and paper_seed_count
        and top_level_paper_seed_count
        and paper_timestep_budget
        and uses_paper_eval_rollouts
        and paper_test_horizon
        and paper_random_hyperparameter_search
        and paper_random_sample_count
        and top_level_paper_random_samples
        and sampled_ranges_ok
        and sampled_minibatches_ok
        and best_hyperparameters_have_complete_seed_coverage
    )
    return {
        "manifest_path": manifest.get("manifest_path"),
        "manifest_loaded": True,
        "command": manifest.get("command", ""),
        "policies": policies,
        "seeds": seeds,
        "jobs_planned": jobs_planned,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "jobs_uncapped_for_selected_space": jobs_uncapped,
        "hyperparam_mode": hyperparam_mode,
        "hyperparam_samples": hyperparam_samples,
        "paper_scale_plan": paper_scale_plan,
        "raw_paper_scale_execution": raw_paper_scale_execution,
        "paper_scale_execution": acceptable_paper_scale_execution,
        "all_planned_jobs_completed": all_planned_jobs_completed,
        "planned_job_count_matches_selected_space": planned_job_count_matches_selected_space,
        "completed_job_counts_match": completed_job_counts_match,
        "paper_random_hyperparameter_search": paper_random_hyperparameter_search,
        "paper_random_sample_count": paper_random_sample_count,
        "top_level_paper_random_samples": top_level_paper_random_samples,
        "sampled_hyperparameters_follow_paper_ranges": sampled_ranges_ok,
        "sampled_hyperparameters_follow_paper_minibatch_rules": sampled_minibatches_ok,
        "best_hyperparameters_have_complete_seed_coverage": best_hyperparameters_have_complete_seed_coverage,
        "full_baseline_policy_set": full_baseline_policy_set,
        "top_level_full_policy_set": top_level_full_policy_set,
        "paper_seed_count": paper_seed_count,
        "top_level_paper_seed_count": top_level_paper_seed_count,
        "paper_timestep_budget": paper_timestep_budget,
        "uses_paper_eval_rollouts": uses_paper_eval_rollouts,
        "paper_test_horizon": paper_test_horizon,
        "limitation": (
            "PPO hyperparameter-search evidence is accepted only when the supplied sweep manifest "
            "marks paper_scale_execution true, its protocol fields are internally consistent, and its "
            "embedded best-hyperparameter rows cover every selected seed for both PPO policies."
        ),
    }


def direct_opt_evidence_status(
    rows: List[Dict[str, Any]],
    *,
    seeds: List[int],
    include_direct_opt: bool,
) -> Dict[str, Any]:
    direct_rows = [row for row in rows if row.get("policy") == "Direct-Opt diagnostic"]
    selected_distinct_seeds = sorted(set(seeds))
    direct_row_seeds: List[int] = []
    invalid_seed_rows = 0
    for row in direct_rows:
        try:
            direct_row_seeds.append(int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            invalid_seed_rows += 1

    expected_rows = len(seeds) if include_direct_opt else 0
    records_rows_for_selected_seeds = len(direct_rows) == expected_rows
    distinct_direct_row_seeds = sorted(set(direct_row_seeds))
    covers_selected_seed_set = distinct_direct_row_seeds == selected_distinct_seeds
    protocol_statuses = [
        row.get("paper_protocol_status")
        for row in direct_rows
        if isinstance(row.get("paper_protocol_status"), dict)
    ]
    all_rows_have_protocol_status = len(protocol_statuses) == len(direct_rows)
    requirement_maps = [
        status.get("direct_opt_protocol_requirements")
        for status in protocol_statuses
        if isinstance(status.get("direct_opt_protocol_requirements"), dict)
    ]
    missing_requirement_lists = [
        status.get("missing_direct_opt_protocol_requirements")
        for status in protocol_statuses
        if isinstance(status.get("missing_direct_opt_protocol_requirements"), list)
    ]
    all_rows_have_requirement_maps = len(requirement_maps) == len(direct_rows)
    all_rows_have_expected_requirement_keys = bool(requirement_maps) and all(
        set(requirements.keys()) == DIRECT_OPT_PROTOCOL_REQUIREMENT_KEYS
        for requirements in requirement_maps
    )
    all_row_requirements_satisfied = bool(requirement_maps) and all(
        all(bool(satisfied) for satisfied in requirements.values())
        for requirements in requirement_maps
    )
    all_rows_missing_requirement_lists_empty = (
        len(missing_requirement_lists) == len(direct_rows)
        and all(missing_requirements == [] for missing_requirements in missing_requirement_lists)
    )
    all_rows_paper_scale_direct_opt_protocol = bool(protocol_statuses) and all(
        bool(status.get("paper_scale_direct_opt_protocol"))
        for status in protocol_statuses
    )

    requirements = {
        "direct_opt_requested": include_direct_opt,
        "direct_opt_rows_for_selected_seeds": records_rows_for_selected_seeds,
        "valid_direct_opt_row_seeds": invalid_seed_rows == 0,
        "direct_opt_selected_seed_coverage": covers_selected_seed_set,
        "direct_opt_protocol_status_per_row": all_rows_have_protocol_status,
        "direct_opt_protocol_requirement_map_per_row": all_rows_have_requirement_maps,
        "direct_opt_protocol_expected_requirement_keys_per_row": all_rows_have_expected_requirement_keys,
        "direct_opt_protocol_requirements_satisfied_per_row": all_row_requirements_satisfied,
        "direct_opt_missing_requirements_empty_per_row": all_rows_missing_requirement_lists_empty,
        "paper_scale_direct_opt_protocol_per_row": all_rows_paper_scale_direct_opt_protocol,
    }
    missing_requirements = [
        requirement
        for requirement, satisfied in requirements.items()
        if not satisfied
    ]
    paper_scale_direct_opt_protocol = not missing_requirements
    return {
        "requested": include_direct_opt,
        "selected_seeds": seeds,
        "distinct_selected_seeds": selected_distinct_seeds,
        "expected_rows": expected_rows,
        "rows_recorded": len(direct_rows),
        "direct_opt_row_seeds": direct_row_seeds,
        "distinct_direct_opt_row_seeds": distinct_direct_row_seeds,
        "invalid_seed_rows": invalid_seed_rows,
        "records_rows_for_selected_seeds": records_rows_for_selected_seeds,
        "covers_selected_seed_set": covers_selected_seed_set,
        "all_rows_have_protocol_status": all_rows_have_protocol_status,
        "all_rows_have_requirement_maps": all_rows_have_requirement_maps,
        "expected_requirement_keys": sorted(DIRECT_OPT_PROTOCOL_REQUIREMENT_KEYS),
        "all_rows_have_expected_requirement_keys": all_rows_have_expected_requirement_keys,
        "all_row_requirements_satisfied": all_row_requirements_satisfied,
        "all_rows_missing_requirement_lists_empty": all_rows_missing_requirement_lists_empty,
        "all_rows_paper_scale_direct_opt_protocol": all_rows_paper_scale_direct_opt_protocol,
        "direct_opt_evidence_requirements": requirements,
        "missing_direct_opt_evidence_requirements": missing_requirements,
        "paper_scale_direct_opt_protocol": paper_scale_direct_opt_protocol,
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
                "train_steps_mean": _mean([float(row["train_steps"]) for row in group]),
                "train_steps_std": _sample_std([float(row["train_steps"]) for row in group]),
                "test_steps_mean": _mean([float(row["test_steps"]) for row in group]),
                "test_steps_std": _sample_std([float(row["test_steps"]) for row in group]),
                "train_survival_seconds_mean": _mean(
                    [float(row["train_survival_seconds"]) for row in group]
                ),
                "train_survival_seconds_std": _sample_std(
                    [float(row["train_survival_seconds"]) for row in group]
                ),
                "test_survival_seconds_mean": _mean(
                    [float(row["test_survival_seconds"]) for row in group]
                ),
                "test_survival_seconds_std": _sample_std(
                    [float(row["test_survival_seconds"]) for row in group]
                ),
                "best_seed_by_train": int(best["seed"]),
                "best_train_success": float(best["train_success"]),
                "best_test_success": float(best["test_success"]),
                "best_train_reward": float(best["train_reward"]),
                "best_test_reward": float(best["test_reward"]),
                "best_train_steps": float(best["train_steps"]),
                "best_test_steps": float(best["test_steps"]),
                "best_train_survival_seconds": float(best["train_survival_seconds"]),
                "best_test_survival_seconds": float(best["test_survival_seconds"]),
                "eval_rollouts": int(best["eval_rollouts"]),
                "test_horizon_steps": int(best["test_horizon_steps"]),
                "best_timesteps": int(best["timesteps"]),
                "best_checkpoint": best.get("checkpoint", ""),
                "best_metrics_output": best.get("metrics_output", ""),
                "best_traces_output": best.get("traces_output", ""),
                "best_command": best.get("command", ""),
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
    default_psm = CartpoleSynthesisConfig()
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "results")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-rollouts", type=int, default=PAPER_EVAL_ROLLOUTS)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--include-ppo", action="store_true")
    parser.add_argument(
        "--ppo-sweep-manifest",
        type=Path,
        default=None,
        help=(
            "Path to a cartpole_ppo_sweep_manifest.json file. The runner records it as "
            "PPO hyperparameter-search evidence only if it marks paper_scale_execution true."
        ),
    )
    parser.add_argument("--include-direct-opt", action="store_true")
    parser.add_argument("--direct-opt-parallel-threads", type=int, default=1)
    parser.add_argument("--direct-opt-time-limit-seconds", type=float, default=None)
    parser.add_argument("--psm-teacher-theta-gain", type=float, default=default_psm.teacher_theta_gain)
    parser.add_argument("--psm-teacher-omega-gain", type=float, default=default_psm.teacher_omega_gain)
    parser.add_argument(
        "--psm-teacher-student-iters",
        type=int,
        default=None,
        help="Teacher/student alternations for PSM; defaults to 1 for --quick and 2 otherwise.",
    )
    parser.add_argument("--psm-student-em-iters", type=int, default=default_psm.student_em_iters)
    parser.add_argument(
        "--psm-student-switch-responsibility-passes",
        type=int,
        default=default_psm.student_switch_responsibility_passes,
    )
    parser.add_argument("--psm-teacher-student-regularizer", type=float, default=default_psm.teacher_student_regularizer)
    parser.add_argument("--psm-teacher-reward-lambda", type=float, default=default_psm.teacher_reward_lambda)
    parser.add_argument("--psm-teacher-top-rho", type=int, default=default_psm.teacher_top_rho)
    parser.add_argument("--psm-teacher-refinement-steps", type=int, default=default_psm.teacher_refinement_steps)
    parser.add_argument(
        "--psm-teacher-elite-distribution-resamples",
        type=int,
        default=default_psm.teacher_elite_distribution_resamples,
    )
    parser.add_argument(
        "--psm-teacher-elite-distribution-rounds",
        type=int,
        default=default_psm.teacher_elite_distribution_rounds,
    )
    parser.add_argument("--psm-parallel-trace-workers", type=int, default=default_psm.parallel_trace_workers)
    parser.add_argument("--psm-parallel-switch-workers", type=int, default=default_psm.parallel_switch_workers)
    parser.add_argument(
        "--ppo-eval-interval",
        type=int,
        default=None,
        help=(
            "Record PPO train/test eval_history every N timesteps. "
            "Defaults to 32 for --quick and 0, final-result only, otherwise."
        ),
    )
    parser.add_argument("--quick", action="store_true", help="Run a small diagnostic configuration for CI/local checks.")
    args = parser.parse_args()
    if args.ppo_eval_interval is None:
        args.ppo_eval_interval = 32 if args.quick else 0
    if args.psm_teacher_student_iters is None:
        args.psm_teacher_student_iters = 1 if args.quick else default_psm.teacher_student_iters
    return args


def psm_teacher_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "teacher_theta_gain": args.psm_teacher_theta_gain,
        "teacher_omega_gain": args.psm_teacher_omega_gain,
        "teacher_student_iters": args.psm_teacher_student_iters,
        "student_em_iters": args.psm_student_em_iters,
        "student_switch_responsibility_passes": args.psm_student_switch_responsibility_passes,
        "teacher_student_regularizer": args.psm_teacher_student_regularizer,
        "teacher_reward_lambda": args.psm_teacher_reward_lambda,
        "teacher_top_rho": args.psm_teacher_top_rho,
        "teacher_refinement_steps": args.psm_teacher_refinement_steps,
        "teacher_elite_distribution_resamples": args.psm_teacher_elite_distribution_resamples,
        "teacher_elite_distribution_rounds": args.psm_teacher_elite_distribution_rounds,
        "parallel_trace_workers": args.psm_parallel_trace_workers,
        "parallel_switch_workers": args.psm_parallel_switch_workers,
    }


def psm_config(
    seed: int,
    quick: bool,
    teacher_overrides: Dict[str, Any] | None = None,
) -> CartpoleSynthesisConfig:
    # The quick path is a CI/local smoke test; the non-quick path preserves the
    # larger candidate pool and trace count expected for reproduction runs.
    cfg_kwargs = {
        "num_initial_states": 4 if quick else 64,
        "candidate_rollouts": 4 if quick else 128,
        "segment_steps": 2 if quick else 1,
        "segments_per_trace": 8 if quick else 250,
        "teacher_student_iters": 1 if quick else 2,
        "seed": seed,
    }
    cfg_kwargs.update(teacher_overrides or {})
    return CartpoleSynthesisConfig(**cfg_kwargs)


def reproduction_protocol_status(
    *,
    seeds: List[int],
    eval_rollouts: int,
    test_max_steps: int,
    include_ppo: bool,
    include_direct_opt: bool,
    quick: bool,
    ppo_eval_interval: int,
    psm_status: Dict[str, Any],
    ppo_sweep_status: Dict[str, Any] | None = None,
    direct_opt_status: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    paper_test_steps = CartpoleEnv.test_env().cfg.max_steps
    distinct_seeds = sorted(set(seeds))
    five_distinct_seeds = len(seeds) == 5 and len(distinct_seeds) == 5
    uses_full_test_horizon = test_max_steps == paper_test_steps
    uses_paper_eval_rollouts = eval_rollouts == PAPER_EVAL_ROLLOUTS
    ppo_sweep_status = ppo_sweep_status or ppo_sweep_evidence_status(None)
    direct_opt_status = direct_opt_status or direct_opt_evidence_status(
        [],
        seeds=seeds,
        include_direct_opt=include_direct_opt,
    )
    ppo_hyperparameter_search = bool(ppo_sweep_status.get("paper_scale_execution"))
    includes_ppo_baseline_evidence = include_ppo or ppo_hyperparameter_search
    ppo_fixed_config_only = include_ppo and not ppo_hyperparameter_search
    full_probabilistic_adaptive_teaching = bool(psm_status.get("full_probabilistic_adaptive_teaching"))
    full_direct_opt_protocol = include_direct_opt and bool(
        direct_opt_status.get("paper_scale_direct_opt_protocol")
    )
    paper_scale_result = (
        not quick
        and five_distinct_seeds
        and uses_full_test_horizon
        and uses_paper_eval_rollouts
        and includes_ppo_baseline_evidence
        and include_direct_opt
        and ppo_hyperparameter_search
        and full_probabilistic_adaptive_teaching
        and full_direct_opt_protocol
    )
    return {
        "artifact_kind": "cartpole_reproduction_runner_manifest",
        "selected_seeds": seeds,
        "distinct_seeds": distinct_seeds,
        "uses_five_distinct_seeds": five_distinct_seeds,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": eval_rollouts,
        "uses_paper_eval_rollouts": uses_paper_eval_rollouts,
        "paper_test_horizon_steps": paper_test_steps,
        "selected_test_max_steps": test_max_steps,
        "uses_full_test_horizon": uses_full_test_horizon,
        "quick_diagnostic": quick,
        "include_ppo": include_ppo,
        "includes_ppo_baseline_evidence": includes_ppo_baseline_evidence,
        "include_direct_opt": include_direct_opt,
        "ppo_eval_interval": ppo_eval_interval,
        "ppo_fixed_config_only": ppo_fixed_config_only,
        "ppo_hyperparameter_search": ppo_hyperparameter_search,
        "ppo_sweep_evidence": ppo_sweep_status,
        "full_probabilistic_adaptive_teaching": full_probabilistic_adaptive_teaching,
        "full_direct_opt_protocol": full_direct_opt_protocol,
        "direct_opt_evidence": direct_opt_status,
        "paper_scale_result": paper_scale_result,
        "limitation": (
            "This orchestrated runner records local diagnostic rows and exact artifact paths. "
            "The paper_scale_result flag is true only when five distinct seeds, 1000-rollout "
            "full-horizon evaluation, completed PPO/PPO-LSTM hyperparameter-search evidence, "
            "full probabilistic adaptive teaching, and full Direct-Opt protocol evidence are "
            "all satisfied."
        ),
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    psm_teacher_overrides = psm_teacher_overrides_from_args(args)
    ppo_sweep_manifest = load_ppo_sweep_manifest(args.ppo_sweep_manifest)
    ppo_sweep_status = ppo_sweep_evidence_status(ppo_sweep_manifest)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.append(run_psm(seed, args.eval_rollouts, args.test_max_steps, args.quick, args.outdir, psm_teacher_overrides))
        if args.include_ppo:
            # Baselines share the same seed list and evaluation budget so their
            # raw rows remain comparable under one reproduction manifest.
            rows.append(
                run_ppo(
                    "mlp",
                    seed,
                    args.eval_rollouts,
                    args.test_max_steps,
                    args.outdir,
                    args.ppo_eval_interval,
                    args.quick,
                )
            )
            rows.append(
                run_ppo(
                    "lstm",
                    seed,
                    args.eval_rollouts,
                    args.test_max_steps,
                    args.outdir,
                    args.ppo_eval_interval,
                    args.quick,
                )
            )
        if args.include_direct_opt:
            rows.append(
                run_direct_opt(
                    seed,
                    args.eval_rollouts,
                    args.test_max_steps,
                    args.quick,
                    args.outdir,
                    args.direct_opt_parallel_threads,
                    args.direct_opt_time_limit_seconds,
                )
            )

    psm_status = cartpole_synthesis_protocol_status(
        psm_config(seeds[0] if seeds else 0, args.quick, psm_teacher_overrides),
        args.eval_rollouts,
        args.test_max_steps,
        args.quick,
        five_seed_selection=len(seeds) == 5 and len(set(seeds)) == 5,
    )
    direct_opt_status = direct_opt_evidence_status(
        rows,
        seeds=seeds,
        include_direct_opt=args.include_direct_opt,
    )
    manifest = {
        "command": " ".join(sys.argv),
        "quick": args.quick,
        "include_ppo": args.include_ppo,
        "ppo_sweep_manifest": str(args.ppo_sweep_manifest) if args.ppo_sweep_manifest is not None else None,
        "ppo_sweep_evidence": ppo_sweep_status,
        "include_direct_opt": args.include_direct_opt,
        "direct_opt_parallel_threads": args.direct_opt_parallel_threads,
        "direct_opt_time_limit_seconds": args.direct_opt_time_limit_seconds,
        "direct_opt_evidence": direct_opt_status,
        "seeds": seeds,
        "eval_rollouts": args.eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": args.eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "test_max_steps": args.test_max_steps,
        "psm_teacher_overrides": psm_teacher_overrides,
        "psm_algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "psm_paper_protocol_status": psm_status,
        "paper_protocol_status": reproduction_protocol_status(
            seeds=seeds,
            eval_rollouts=args.eval_rollouts,
            test_max_steps=args.test_max_steps,
            include_ppo=args.include_ppo,
            include_direct_opt=args.include_direct_opt,
            quick=args.quick,
            ppo_eval_interval=args.ppo_eval_interval,
            psm_status=psm_status,
            ppo_sweep_status=ppo_sweep_status,
            direct_opt_status=direct_opt_status,
        ),
        "ppo_eval_interval": args.ppo_eval_interval,
        "paper_scale_note": (
            "Without --quick, fixed PPO rows use 10^7 timesteps per seed. "
            "Supply --ppo-sweep-manifest from scripts/run_cartpole_ppo_sweep.py to attach "
            "paper-scale PPO/PPO-LSTM hyperparameter-search evidence."
        ),
        "summary_note": (
            "cartpole_summary.csv reports per-policy means and sample standard deviations over "
            "the requested seeds; with one seed, std is reported as 0. Best seed is selected by "
            "train_success, then train_reward, then lower seed."
        ),
        "survival_metric_note": (
            "Rows and summaries report mean survived simulator steps and seconds separately "
            "from reward so long-horizon survival plots do not rely on reward as an implicit proxy."
        ),
        "psm_artifact_note": (
            "Programmatic-state-machine rows include metrics_output paths under the requested "
            "output directory. PSM metrics contain the fitted probabilistic student, compact "
            "teacher-trace examples, exact config, and fixed local synthesis constants. PSM rows "
            "also include traces_output paths with the full selected teacher traces."
        ),
        "ppo_artifact_note": (
            "When --include-ppo is set, PPO rows include checkpoint and metrics_output paths "
            "under the requested output directory. PPO metrics contain eval_history entries only "
            "when ppo_eval_interval is greater than zero."
        ),
        "direct_opt_artifact_note": (
            "When --include-direct-opt is set, Direct-Opt diagnostic rows include metrics_output "
            "paths under the requested output directory. This is a bounded diagnostic baseline, "
            "not the paper's full direct optimization protocol."
        ),
    }
    write_results(rows, args.outdir, manifest)
    print(f"wrote {args.outdir / 'cartpole_results.csv'}")
    print(f"wrote {args.outdir / 'cartpole_summary.csv'}")
    print(f"wrote {args.outdir / 'cartpole_manifest.json'}")


if __name__ == "__main__":
    main()
