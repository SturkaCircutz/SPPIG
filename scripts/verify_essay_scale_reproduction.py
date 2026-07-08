from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "artifacts" / "ppo_sweep_cuda_medium_core"
RESULTS_DIR = ROOT / "artifacts" / "results"
ESSAY_DIR = ROOT / "essay"
LOCAL_DIAGNOSTIC_PHRASE = "not a paper-scale reproduction"


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssertionError(f"missing JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must contain an object: {path}")
    return payload


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError as exc:
        raise AssertionError(f"missing CSV artifact: {path}") from exc
    if not rows:
        raise AssertionError(f"CSV artifact has no rows: {path}")
    return rows


def repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_float_equal(left: Any, right: Any, label: str, tolerance: float = 1e-9) -> None:
    require(abs(float(left) - float(right)) <= tolerance, f"{label} mismatch: {left!r} != {right!r}")


def metric_result(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    selected = metrics.get("selected_result")
    if isinstance(selected, dict):
        train = {
            "success_rate": selected["train_success_rate"],
            "reward_mean": selected["train_reward_mean"],
            "steps_mean": selected["train_steps_mean"],
            "survival_seconds_mean": selected["train_survival_seconds_mean"],
        }
        test = {
            "success_rate": selected["test_success_rate"],
            "reward_mean": selected["test_reward_mean"],
            "steps_mean": selected["test_steps_mean"],
            "survival_seconds_mean": selected["test_survival_seconds_mean"],
        }
        timesteps = selected.get("timesteps")
        return train, test, int(timesteps) if timesteps is not None else None
    train = metrics.get("train")
    test = metrics.get("test")
    require(isinstance(train, dict), "metrics artifact lacks train result")
    require(isinstance(test, dict), "metrics artifact lacks test result")
    return train, test, None


def require_no_paper_scale_claim(status: dict[str, Any], label: str) -> None:
    for key, value in status.items():
        if key.startswith("paper_scale") or key in {"paper_scale_plan", "paper_scale_execution"}:
            require(value is False, f"{label} must not claim {key}")
    if "uses_paper_eval_rollouts" in status:
        require(status["uses_paper_eval_rollouts"] is False, f"{label} must not claim paper eval rollouts")
    if "full_probabilistic_adaptive_teaching" in status:
        require(status["full_probabilistic_adaptive_teaching"] is False, f"{label} must not claim full probabilistic adaptive teaching")


def require_cartpole_split(status: dict[str, Any], label: str) -> None:
    if "train_horizon_seconds" in status:
        require_float_equal(status["train_horizon_seconds"], 5.0, f"{label} train horizon seconds")
    if "train_pole_length" in status:
        require_float_equal(status["train_pole_length"], 0.5, f"{label} train pole length")
    if "test_horizon_seconds" in status:
        require_float_equal(status["test_horizon_seconds"], 300.0, f"{label} test horizon seconds")
    if "test_pole_length" in status:
        require_float_equal(status["test_pole_length"], 1.0, f"{label} test pole length")
    if "paper_test_horizon" in status:
        require(status["paper_test_horizon"] is True, f"{label} must record full test horizon")
    if "uses_full_test_horizon" in status:
        require(status["uses_full_test_horizon"] is True, f"{label} must record full test horizon")


def verify_medium_sweep() -> None:
    manifest = load_json(SWEEP_DIR / "cartpole_ppo_sweep_manifest.json")
    results = load_csv(SWEEP_DIR / "cartpole_ppo_sweep_results.csv")
    summary = load_csv(SWEEP_DIR / "cartpole_ppo_sweep_summary.csv")
    hyper_summary = load_csv(SWEEP_DIR / "cartpole_ppo_sweep_hyperparam_summary.csv")
    status = manifest.get("paper_protocol_status")
    require(isinstance(status, dict), "medium sweep manifest lacks paper_protocol_status")

    require(manifest.get("artifact_kind") == "cartpole_ppo_sweep_manifest", "unexpected sweep manifest kind")
    require(manifest.get("jobs_planned") == 4, "essay-scale sweep must plan 4 jobs")
    require(manifest.get("jobs_completed") == 4, "essay-scale sweep must complete 4 jobs")
    require(manifest.get("jobs_failed") == 0, "essay-scale sweep must have zero failed jobs")
    require(status.get("paper_scale_plan") is False, "medium sweep must not claim paper-scale plan")
    require(status.get("paper_scale_execution") is False, "medium sweep must not claim paper-scale execution")
    require(status.get("paper_test_horizon") is True, "medium sweep must use full 15,000-step test horizon")
    require(status.get("uses_paper_eval_rollouts") is False, "medium sweep must not claim 1000-rollout eval")
    require(status.get("selected_eval_rollouts") == 200, "medium sweep must use 200 eval rollouts")
    require(status.get("selected_test_max_steps") == 15_000, "medium sweep must use 15,000 test steps")
    require(status.get("selected_seed_count") == 2, "medium sweep must use two selected seeds")
    require(status.get("distinct_seed_count") == 2, "medium sweep must have two completed seeds")
    require(set(status.get("distinct_policies", [])) == {"mlp", "lstm"}, "medium sweep must include MLP and LSTM")
    require(status.get("sampled_hyperparameters_follow_paper_ranges") is True, "sampled PPO hyperparameters must stay in paper ranges")
    require(status.get("ppo_lstm_minibatches_fixed_to_one") is True, "PPO-LSTM sweep must keep minibatches fixed to 1")

    require(len(results) == 4, "medium sweep results must contain four jobs")
    seen = {(row["policy"], row["seed"]) for row in results}
    require(seen == {("mlp", "0"), ("mlp", "1"), ("lstm", "0"), ("lstm", "1")}, "medium sweep jobs must cover both policies and seeds 0,1")
    require({row["policy"] for row in summary} == {"mlp", "lstm"}, "medium sweep summary must include both policies")
    require({row["policy"] for row in hyper_summary} == {"mlp", "lstm"}, "medium sweep hyperparameter summary must include both policies")

    for row in results:
        require(row["total_timesteps"] == "1000000", f"job {row['job_id']} must use 1,000,000 timesteps")
        require(row["eval_rollouts"] == "200", f"job {row['job_id']} must use 200 eval rollouts")
        require(row["test_max_steps"] == "15000", f"job {row['job_id']} must use 15,000 test steps")
        checkpoint = repo_path(row["output"])
        metrics_path = repo_path(row["metrics_output"])
        require(checkpoint.exists(), f"missing checkpoint for job {row['job_id']}: {checkpoint}")
        require(metrics_path.exists(), f"missing metrics for job {row['job_id']}: {metrics_path}")
        metrics = load_json(metrics_path)
        config = metrics.get("config")
        selected = metrics.get("selected_result")
        metric_status = metrics.get("paper_protocol_status")
        require(isinstance(config, dict), f"job {row['job_id']} metrics lacks config")
        require(isinstance(selected, dict), f"job {row['job_id']} metrics lacks selected_result")
        require(isinstance(metric_status, dict), f"job {row['job_id']} metrics lacks paper_protocol_status")
        require(config.get("policy_type") == row["policy"], f"job {row['job_id']} policy disagrees with metrics")
        require(str(config.get("seed")) == row["seed"], f"job {row['job_id']} seed disagrees with metrics")
        require(config.get("total_timesteps") == 1_000_000, f"job {row['job_id']} metrics timesteps mismatch")
        require(config.get("eval_rollouts") == 200, f"job {row['job_id']} metrics eval rollouts mismatch")
        require(config.get("eval_test_max_steps") == 15_000, f"job {row['job_id']} metrics test horizon mismatch")
        require(metric_status.get("paper_test_horizon") is True, f"job {row['job_id']} must record full test horizon")
        require(metric_status.get("uses_paper_eval_rollouts") is False, f"job {row['job_id']} must not claim paper eval rollouts")
        require(metric_status.get("paper_scale_baseline_protocol") is False, f"job {row['job_id']} must not claim paper-scale baseline")
        require_no_paper_scale_claim(metric_status, f"job {row['job_id']} metrics")
        require_cartpole_split(metric_status, f"job {row['job_id']} metrics")
        require_float_equal(selected["train_success_rate"], row["train_success"], f"job {row['job_id']} train_success")
        require_float_equal(selected["test_success_rate"], row["test_success"], f"job {row['job_id']} test_success")
        require_float_equal(selected["train_reward_mean"], row["train_reward"], f"job {row['job_id']} train_reward")
        require_float_equal(selected["test_reward_mean"], row["test_reward"], f"job {row['job_id']} test_reward")


def verify_result_bundle() -> None:
    summary_rows = load_csv(RESULTS_DIR / "cartpole_summary.csv")
    manifest = load_json(RESULTS_DIR / "cartpole_manifest.json")
    manifest_status = manifest.get("paper_protocol_status")
    require(isinstance(manifest_status, dict), "result manifest lacks paper_protocol_status")
    require(manifest.get("paper_scale_result") is False, "checked-in result bundle must not claim paper-scale result")
    require(manifest.get("local_diagnostic_only") is True, "checked-in result bundle must be marked local diagnostic")
    require(manifest.get("row_count") == len(summary_rows), "result manifest row_count must match summary rows")
    require_no_paper_scale_claim(manifest, "result manifest")
    require_cartpole_split(manifest, "result manifest")
    require_no_paper_scale_claim(manifest_status, "result manifest paper_protocol_status")
    require_cartpole_split(manifest_status, "result manifest paper_protocol_status")

    psm_row = next((row for row in summary_rows if row["policy"] == "Synthesized PSM diagnostic"), None)
    require(psm_row is not None, "result summary must include synthesized PSM diagnostic")
    require_float_equal(psm_row["train_reward_mean"], 48.05, "synthesized PSM train reward")
    require_float_equal(psm_row["test_reward_mean"], 59.85, "synthesized PSM test reward")
    require("--parallel-trace-workers 10" in psm_row["best_command"], "PSM summary must record ten teacher trace workers")

    for row in summary_rows:
        metrics_path = repo_path(row["best_metrics_output"])
        require(metrics_path.exists(), f"missing result metrics artifact for {row['policy']}: {metrics_path}")
        metrics = load_json(metrics_path)
        status = metrics.get("paper_protocol_status")
        require(isinstance(status, dict), f"{row['policy']} metrics lacks paper_protocol_status")
        require(row["best_command"] == metrics.get("command"), f"{row['policy']} command disagrees with metrics")
        require(row["test_horizon_steps"] == "15000", f"{row['policy']} must use full 15,000-step test horizon")
        require(row["eval_rollouts"] == "20", f"{row['policy']} checked-in diagnostic must use 20 eval rollouts")
        require_no_paper_scale_claim(status, f"{row['policy']} metrics")
        require_cartpole_split(status, f"{row['policy']} metrics")
        train, test, timesteps = metric_result(metrics)
        require_float_equal(row["best_train_success"], train["success_rate"], f"{row['policy']} train success")
        require_float_equal(row["best_test_success"], test["success_rate"], f"{row['policy']} test success")
        require_float_equal(row["best_train_reward"], train["reward_mean"], f"{row['policy']} train reward")
        require_float_equal(row["best_test_reward"], test["reward_mean"], f"{row['policy']} test reward")
        require_float_equal(row["best_train_steps"], train["steps_mean"], f"{row['policy']} train steps")
        require_float_equal(row["best_test_steps"], test["steps_mean"], f"{row['policy']} test steps")
        require_float_equal(row["best_train_survival_seconds"], train["survival_seconds_mean"], f"{row['policy']} train survival seconds")
        require_float_equal(row["best_test_survival_seconds"], test["survival_seconds_mean"], f"{row['policy']} test survival seconds")
        if timesteps is not None:
            require(str(timesteps) == row["best_timesteps"], f"{row['policy']} selected timesteps disagree with metrics")


def verify_essay_artifacts() -> None:
    required_files = [
        ESSAY_DIR / "project.tex",
        ESSAY_DIR / "cartpole_abstract_results.tex",
        ESSAY_DIR / "cartpole_results_table.tex",
        ESSAY_DIR / "cartpole_ppo_sweep_fragment.tex",
        ESSAY_DIR / "cartpole_policy_fragment.tex",
        ESSAY_DIR / "cartpole_figure19_reference_fragment.tex",
        ESSAY_DIR / "00README.json",
        ESSAY_DIR / "figures" / "cartpole_success_rates.png",
        ESSAY_DIR / "figures" / "cartpole_test_survival_reward.png",
        ESSAY_DIR / "figures" / "programmatic_switch_boundary.png",
        ESSAY_DIR / "figures" / "cartpole_ppo_training_curves.png",
    ]
    for path in required_files:
        require(path.exists(), f"missing essay artifact: {path}")
        require(path.stat().st_size > 0, f"essay artifact is empty: {path}")

    readme = load_json(ESSAY_DIR / "00README.json")
    listed = {source.get("filename") for source in readme.get("sources", []) if isinstance(source, dict)}
    for path in required_files:
        if path.name == "00README.json":
            continue
        rel = path.relative_to(ESSAY_DIR).as_posix()
        require(rel in listed, f"essay manifest does not list {rel}")

    table = (ESSAY_DIR / "cartpole_results_table.tex").read_text(encoding="utf-8")
    sweep_fragment = (ESSAY_DIR / "cartpole_ppo_sweep_fragment.tex").read_text(encoding="utf-8")
    abstract = (ESSAY_DIR / "cartpole_abstract_results.tex").read_text(encoding="utf-8")
    project = (ESSAY_DIR / "project.tex").read_text(encoding="utf-8")
    require(LOCAL_DIAGNOSTIC_PHRASE in table, "result table must carry local-diagnostic caveat")
    require(LOCAL_DIAGNOSTIC_PHRASE in sweep_fragment, "PPO sweep fragment must carry local-diagnostic caveat")
    require(LOCAL_DIAGNOSTIC_PHRASE in abstract, "abstract result fragment must carry local-diagnostic caveat")
    require("Synthesized PSM diagnostic & 0.00 & 0.00 & 48.0 & 59.9" in table, "result table has stale synthesized PSM row")
    require("28.4" not in table and "41.6" not in table, "result table still contains stale synthesized PSM values")
    require("Medium partial PPO/PPO-LSTM sweep summary. Completed jobs: 4/4" in sweep_fragment, "PPO sweep fragment must report completed 4/4 jobs")
    require("Paper-scale plan: false; paper-scale execution: false" in sweep_fragment, "PPO sweep fragment must not claim paper-scale run")
    require("1,000,000 timesteps per job" in sweep_fragment, "PPO sweep fragment must report medium timestep budget")
    for include in [
        "cartpole_abstract_results.tex",
        "cartpole_results_table.tex",
        "cartpole_ppo_sweep_fragment.tex",
        "cartpole_policy_fragment.tex",
        "cartpole_figure19_reference_fragment.tex",
    ]:
        require(include in project, f"project.tex does not include {include}")


def verify() -> None:
    verify_medium_sweep()
    verify_result_bundle()
    verify_essay_artifacts()


def main() -> None:
    verify()
    print("essay-scale partial reproduction artifacts verified")


if __name__ == "__main__":
    main()
