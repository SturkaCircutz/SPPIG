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
from cartpole_direct_opt import (  # noqa: E402
    DirectOptConfig,
    direct_opt_metrics,
    run_cartpole_direct_opt,
)
from cartpole_synthesis import (  # noqa: E402
    CartpoleSynthesisConfig,
    cartpole_synthesis_algorithm_provenance,
    cartpole_switch_fit_diagnostics,
    synthesize_cartpole_student_with_history,
)
from train_cartpole_psm import summarize_student, summarize_synthesis_history, summarize_traces  # noqa: E402

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


def run_psm(
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    quick: bool,
    outdir: Path,
    teacher_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # The quick path is a CI/local smoke test; the non-quick path preserves the
    # larger candidate pool and trace count expected for reproduction runs.
    cfg_kwargs = {
        "num_initial_states": 4 if quick else 64,
        "candidate_rollouts": 4 if quick else 128,
        "segment_steps": 2 if quick else 8,
        "segments_per_trace": 8 if quick else 32,
        "teacher_student_iters": 1 if quick else 2,
        "seed": seed,
    }
    cfg_kwargs.update(teacher_overrides or {})
    cfg = CartpoleSynthesisConfig(**cfg_kwargs)
    student, traces, synthesis_history = synthesize_cartpole_student_with_history(cfg)
    policy = student.to_deterministic_policy()
    train_env = CartpoleEnv.train_env(seed=100 + seed)
    test_env = CartpoleEnv.test_env(seed=200 + seed)
    train_results = [train_env.rollout(policy) for _ in range(eval_rollouts)]
    # The paper's test horizon is 300s; test_max_steps is only exposed so tests
    # can cap runtime without changing the environment definition itself.
    test_results = [test_env.rollout(policy, max_steps=test_max_steps) for _ in range(eval_rollouts)]
    train = _summarize_results(train_results)
    test = _summarize_results(test_results)
    metrics_path = outdir / "metrics" / f"psm_seed{seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": asdict(cfg),
        "algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "eval_rollouts": eval_rollouts,
        "test_max_steps": test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "num_traces": len(traces),
        "synthesis_history": summarize_synthesis_history(synthesis_history),
        "trace_summary": summarize_traces(traces),
        "policy_description": policy.describe(),
        "probabilistic_student": summarize_student(student),
        "switch_fit_diagnostics": cartpole_switch_fit_diagnostics(traces, student),
        "train": {"success_rate": train["success"], "reward_mean": train["reward"]},
        "test": {"success_rate": test["success"], "reward_mean": test["reward"]},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "policy": "Programmatic state machine",
        "seed": seed,
        "train_success": train["success"],
        "test_success": test["success"],
        "train_reward": train["reward"],
        "test_reward": test["reward"],
        "timesteps": 0,
        "metrics_output": str(metrics_path),
        "config": asdict(cfg),
        "algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "policy_description": policy.describe(),
        "num_traces": len(traces),
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
        "timesteps": result.timesteps,
        "checkpoint": str(checkpoint_path),
        "metrics_output": str(metrics_path),
        "config": asdict(cfg),
    }


def run_direct_opt(
    seed: int,
    eval_rollouts: int,
    test_max_steps: int,
    quick: bool,
    outdir: Path,
) -> Dict[str, Any]:
    cfg = DirectOptConfig(
        seed=seed,
        num_train_states=2 if quick else 10,
        random_candidates=8 if quick else 256,
        eval_rollouts=eval_rollouts,
        test_max_steps=test_max_steps,
        quick=quick,
    )
    result = run_cartpole_direct_opt(cfg)
    metrics_path = outdir / "metrics" / f"direct_opt_seed{seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(direct_opt_metrics(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "policy": "Direct-Opt diagnostic",
        "seed": seed,
        "train_success": result.train_success_rate,
        "test_success": result.test_success_rate,
        "train_reward": result.train_reward_mean,
        "test_reward": result.test_reward_mean,
        "timesteps": 0,
        "metrics_output": str(metrics_path),
        "config": asdict(cfg),
        "algorithm_provenance": result.algorithm_provenance,
        "policy_description": result.policy.describe(),
        "searched_candidates": result.searched_candidates,
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
    default_psm = CartpoleSynthesisConfig()
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "results")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-rollouts", type=int, default=20)
    parser.add_argument("--test-max-steps", type=int, default=15_000)
    parser.add_argument("--include-ppo", action="store_true")
    parser.add_argument("--include-direct-opt", action="store_true")
    parser.add_argument("--psm-teacher-theta-gain", type=float, default=default_psm.teacher_theta_gain)
    parser.add_argument("--psm-teacher-omega-gain", type=float, default=default_psm.teacher_omega_gain)
    parser.add_argument(
        "--psm-teacher-student-iters",
        type=int,
        default=None,
        help="Teacher/student alternations for PSM; defaults to 1 for --quick and 2 otherwise.",
    )
    parser.add_argument("--psm-teacher-student-regularizer", type=float, default=default_psm.teacher_student_regularizer)
    parser.add_argument("--psm-teacher-reward-lambda", type=float, default=default_psm.teacher_reward_lambda)
    parser.add_argument("--psm-teacher-top-rho", type=int, default=default_psm.teacher_top_rho)
    parser.add_argument("--psm-teacher-refinement-steps", type=int, default=default_psm.teacher_refinement_steps)
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
        "teacher_student_regularizer": args.psm_teacher_student_regularizer,
        "teacher_reward_lambda": args.psm_teacher_reward_lambda,
        "teacher_top_rho": args.psm_teacher_top_rho,
        "teacher_refinement_steps": args.psm_teacher_refinement_steps,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    psm_teacher_overrides = psm_teacher_overrides_from_args(args)
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
                )
            )

    manifest = {
        "command": " ".join(sys.argv),
        "quick": args.quick,
        "include_ppo": args.include_ppo,
        "include_direct_opt": args.include_direct_opt,
        "seeds": seeds,
        "eval_rollouts": args.eval_rollouts,
        "test_max_steps": args.test_max_steps,
        "psm_teacher_overrides": psm_teacher_overrides,
        "psm_algorithm_provenance": cartpole_synthesis_algorithm_provenance(),
        "ppo_eval_interval": args.ppo_eval_interval,
        "paper_scale_note": (
            "Without --quick, PPO uses 10^7 timesteps per seed. "
            "This runner records exact configs but does not perform hyperparameter search."
        ),
        "summary_note": (
            "cartpole_summary.csv reports per-policy means and sample standard deviations over "
            "the requested seeds; with one seed, std is reported as 0. Best seed is selected by "
            "train_success, then train_reward, then lower seed."
        ),
        "psm_artifact_note": (
            "Programmatic-state-machine rows include metrics_output paths under the requested "
            "output directory. PSM metrics contain the fitted probabilistic student, compact "
            "teacher-trace examples, exact config, and fixed local synthesis constants."
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
