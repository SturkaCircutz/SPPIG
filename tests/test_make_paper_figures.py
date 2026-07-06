import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_paper_figures  # noqa: E402


PSM_TRACE_COMMAND = "python train.py --traces-output traces.json"
RUNNER_QUICK_COMMAND = "python scripts/run_cartpole_reproduction.py --quick"
FIXED_PSM_COMMAND = "python scripts/evaluate_cartpole_program.py"
PSM_TRACE_CONFIG = {"teacher_student_iters": 1}
PSM_TRACES = [
    {
        "reward": 1,
        "actions": [0.0],
        "mode_labels": [0],
        "observations": [[0.0, 0.0, 0.0, 0.0]],
        "theta_gain": 1.0,
        "omega_gain": 0.0,
        "segment_actions": [0.0],
        "segment_durations": [1],
        "segment_time_increments": [0.02],
        "teacher_source": "unit_test_teacher",
        "student_log_probability": -1.0,
        "teacher_objective": 1.0,
        "teacher_refinement_objective": 1.0,
    }
]
PSM_TRACE_SUMMARY = {
    "count": 1,
    "reward_mean": 1.0,
    "length_mean": 1.0,
    "teacher_source_counts": {"unit_test_teacher": 1},
    "examples": [
        {
            "reward": 1,
            "steps": 1,
            "switches": 0,
            "theta_gain": 1.0,
            "omega_gain": 0.0,
            "segment_actions": [0.0],
            "segment_durations": [1],
            "segment_time_increments": [0.02],
            "teacher_source": "unit_test_teacher",
            "student_log_probability": -1.0,
            "teacher_objective": 1.0,
            "teacher_refinement_objective": 1.0,
            "first_observation": [0.0, 0.0, 0.0, 0.0],
            "last_observation": [0.0, 0.0, 0.0, 0.0],
            "mode_prefix": [0],
        }
    ],
}


def current_synthesized_psm_algorithm_provenance() -> dict:
    return make_paper_figures.cartpole_synthesis_algorithm_provenance()


def current_synthesized_psm_status() -> dict:
    return {
        "paper_scale_result": False,
        "synthesized_by_current_algorithm": True,
        "adaptive_teaching_protocol_requirements": {
            "five_seed_selection": False,
            "full_continuous_switch_m_step": False,
            "full_cem_teacher_optimizer": False,
        },
        "missing_adaptive_teaching_protocol_requirements": [
            "five_seed_selection",
            "full_continuous_switch_m_step",
            "full_cem_teacher_optimizer",
        ],
        "probabilistic_adaptive_teaching_requirements": {
            "full_continuous_switch_m_step": False,
            "full_cem_teacher_optimizer": False,
        },
        "missing_probabilistic_adaptive_teaching_requirements": [
            "full_continuous_switch_m_step",
            "full_cem_teacher_optimizer",
        ],
    }


def minimal_objective_component_summary() -> dict:
    return {
        key: {"count": 1, "min": 1.0, "max": 1.0, "mean": 1.0}
        for key in make_paper_figures.EXPECTED_PSM_OBJECTIVE_COMPONENT_KEYS
    }


def one_iteration_objective_component_metrics() -> dict:
    adaptive_summary = {
        "iteration": 1,
        "objective_component_summary": minimal_objective_component_summary(),
    }
    return {
        "config": dict(PSM_TRACE_CONFIG),
        "num_traces": 1,
        "trace_summary": dict(PSM_TRACE_SUMMARY),
        "adaptive_teacher_summary": [adaptive_summary],
        "synthesis_history": [
            {
                "iteration": 1,
                "trace_summary": dict(PSM_TRACE_SUMMARY),
                "adaptive_teacher_summary": adaptive_summary,
            }
        ],
    }


def one_iteration_trace_payload(command: str = PSM_TRACE_COMMAND, **overrides) -> dict:
    payload = {
        "command": command,
        "config": dict(PSM_TRACE_CONFIG),
        "num_traces": 1,
        "traces": list(PSM_TRACES),
        "trace_history": [
            {
                "iteration": 1,
                "num_traces": 1,
                "traces": list(PSM_TRACES),
            }
        ],
    }
    payload.update(overrides)
    return payload


def one_iteration_metrics_payload(**overrides) -> dict:
    payload = {
        "command": PSM_TRACE_COMMAND,
        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
        "paper_protocol_status": current_synthesized_psm_status(),
        **one_iteration_objective_component_metrics(),
    }
    payload.update(overrides)
    return payload


def artifact_row(
    policy: str,
    metrics_path: str,
    command: str,
    eval_rollouts: str = "20",
    test_horizon_steps: str = "15000",
    **extra: str,
) -> dict[str, str]:
    row = {
        "policy": policy,
        "metrics_output": metrics_path,
        "command": command,
        "eval_rollouts": eval_rollouts,
        "test_horizon_steps": test_horizon_steps,
    }
    row.update(extra)
    return row


def synthesized_psm_row(metrics_path: str) -> dict[str, str]:
    return artifact_row("Synthesized PSM diagnostic", metrics_path, PSM_TRACE_COMMAND)


def runner_psm_row(metrics_path: str) -> dict[str, str]:
    return artifact_row("Programmatic state machine", metrics_path, RUNNER_QUICK_COMMAND)


class MakePaperFiguresTest(unittest.TestCase):
    def test_checked_in_results_reference_existing_artifacts(self):
        rows = make_paper_figures.read_results()

        make_paper_figures.require_result_artifacts(rows)
        self.assertTrue(all(row.get("best_metrics_output") or row.get("metrics_output") for row in rows))
        self.assertTrue(all(int(float(row["test_horizon_steps"])) == 15000 for row in rows))
        self.assertTrue(all(row["eval_rollouts"] == "20" for row in rows))

    def test_checked_in_result_manifest_matches_summary(self):
        summary_path = os.path.join(ROOT, "artifacts", "results", "cartpole_summary.csv")
        manifest_path = os.path.join(ROOT, "artifacts", "results", "cartpole_manifest.json")
        self.assertTrue(os.path.exists(summary_path))
        self.assertTrue(os.path.exists(manifest_path))

        with open(summary_path, newline="", encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertFalse(manifest["paper_scale_result"])
        self.assertTrue(manifest["local_diagnostic_only"])
        self.assertIn("10^7-timestep", manifest["limitation"])
        bundle_status = manifest["paper_protocol_status"]
        self.assertEqual(
            bundle_status["artifact_kind"],
            "checked_in_local_cartpole_diagnostic_bundle_status",
        )
        self.assertEqual(bundle_status["selected_seeds"], [0])
        self.assertEqual(bundle_status["distinct_seeds"], [0])
        self.assertFalse(bundle_status["uses_five_distinct_seeds"])
        self.assertEqual(bundle_status["paper_eval_rollouts"], 1000)
        self.assertEqual(bundle_status["selected_eval_rollouts"], 20)
        self.assertFalse(bundle_status["uses_paper_eval_rollouts"])
        self.assertEqual(bundle_status["paper_test_horizon_steps"], 15000)
        self.assertTrue(bundle_status["uses_full_test_horizon"])
        self.assertTrue(bundle_status["includes_ppo_checkpoint_reevaluation"])
        self.assertTrue(bundle_status["includes_ppo_lstm_checkpoint_reevaluation"])
        self.assertTrue(bundle_status["includes_direct_opt_diagnostic"])
        self.assertTrue(bundle_status["includes_synthesized_psm_diagnostic"])
        self.assertFalse(bundle_status["ppo_hyperparameter_search"])
        self.assertTrue(bundle_status["ppo_lstm_is_warm_started"])
        self.assertFalse(bundle_status["full_probabilistic_adaptive_teaching"])
        self.assertFalse(bundle_status["full_direct_opt_protocol"])
        self.assertFalse(bundle_status["paper_scale_result"])
        self.assertEqual(manifest["row_count"], len(summary))
        self.assertEqual(manifest["summary_csv"], "artifacts/results/cartpole_summary.csv")
        self.assertEqual({row["policy"] for row in summary}, set(manifest["policies"]))
        self.assertTrue(all(row["best_metrics_output"] for row in summary))
        self.assertTrue(all(os.path.exists(os.path.join(ROOT, row["best_metrics_output"])) for row in summary))
        for row in summary:
            with open(os.path.join(ROOT, row["best_metrics_output"]), encoding="utf-8") as handle:
                metrics = json.load(handle)
            manifest_summary_row = next(item for item in manifest["summary"] if item["policy"] == row["policy"])
            self.assertEqual(row["best_command"], metrics["command"])
            self.assertEqual(row["best_command"], manifest["reproduction_commands"][row["policy"]])
            self.assertEqual(manifest_summary_row["best_command"], row["best_command"])
        for row in manifest["rows"]:
            with open(os.path.join(ROOT, row["metrics_output"]), encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(row["command"], metrics["command"])
            self.assertEqual(row["command"], manifest["reproduction_commands"][row["policy"]])
            self.assertNotIn("metrics_command", row)
        fixed_psm_row = next(row for row in manifest["rows"] if row["policy"] == "Programmatic state machine")
        with open(os.path.join(ROOT, fixed_psm_row["metrics_output"]), encoding="utf-8") as handle:
            fixed_psm_metrics = json.load(handle)
        self.assertEqual(fixed_psm_row["paper_protocol_status"], fixed_psm_metrics["paper_protocol_status"])
        self.assertFalse(fixed_psm_row["paper_protocol_status"]["synthesized_by_current_algorithm"])
        self.assertFalse(fixed_psm_row["paper_protocol_status"]["paper_scale_fixed_program_result"])
        synthesized_psm_row = next(row for row in manifest["rows"] if row["policy"] == "Synthesized PSM diagnostic")
        with open(os.path.join(ROOT, synthesized_psm_row["metrics_output"]), encoding="utf-8") as handle:
            synthesized_psm_metrics = json.load(handle)
        self.assertEqual(
            synthesized_psm_row["paper_protocol_status"],
            synthesized_psm_metrics["paper_protocol_status"],
        )
        self.assertEqual(synthesized_psm_row["train"], synthesized_psm_metrics["train"])
        self.assertEqual(synthesized_psm_row["test"], synthesized_psm_metrics["test"])
        self.assertTrue(
            synthesized_psm_row["paper_protocol_status"]["synthesized_by_current_algorithm"]
        )
        self.assertEqual(synthesized_psm_row["traces_output"], synthesized_psm_metrics["traces_output"])
        self.assertTrue(os.path.exists(os.path.join(ROOT, synthesized_psm_row["traces_output"])))
        self.assertIn("--traces-output", synthesized_psm_row["command"])
        self.assertIn("--traces-output", manifest["reproduction_commands"]["Synthesized PSM diagnostic"])
        self.assertIn("student_fit_history", synthesized_psm_metrics["synthesis_history"][0])
        self.assertEqual(
            synthesized_psm_row["adaptive_teacher_summary"],
            synthesized_psm_metrics["adaptive_teacher_summary"],
        )
        self.assertEqual(
            synthesized_psm_row["synthesis_history"],
            synthesized_psm_metrics["synthesis_history"],
        )
        for adaptive_summary in synthesized_psm_metrics["adaptive_teacher_summary"]:
            self.assertIn("objective_component_summary", adaptive_summary)
        self.assertNotIn("artifact_status", synthesized_psm_metrics)
        self.assertNotIn("artifact_status", synthesized_psm_row)
        self.assertEqual(
            synthesized_psm_metrics["algorithm_provenance"]["probabilistic_student"]["mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertEqual(
            synthesized_psm_row["algorithm_provenance"],
            synthesized_psm_metrics["algorithm_provenance"],
        )
        self.assertIn("current probabilistic adaptive-teaching diagnostic", synthesized_psm_row["notes"])
        self.assertIn("PPO MLP", manifest["reproduction_commands"])
        self.assertIn("--test-max-steps 15000", manifest["reproduction_commands"]["PPO MLP"])

    def test_essay_manifest_lists_generated_cartpole_artifacts(self):
        with open(os.path.join(ROOT, "essay", "00README.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)

        filenames = {source["filename"] for source in manifest["sources"]}

        self.assertIn("cartpole_abstract_results.tex", filenames)
        self.assertIn("cartpole_results_table.tex", filenames)
        self.assertIn("cartpole_policy_fragment.tex", filenames)
        self.assertIn("cartpole_figure19_reference_fragment.tex", filenames)
        self.assertIn("figures/cartpole_success_rates.png", filenames)
        self.assertIn("figures/cartpole_test_survival_reward.png", filenames)
        self.assertIn("figures/programmatic_switch_boundary.png", filenames)
        self.assertEqual(
            manifest["process"]["regenerate_generated_artifacts"],
            ".venv/bin/python scripts/make_paper_figures.py",
        )

    def test_read_results_prefers_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "cartpole_results.csv")
            summary_path = os.path.join(tmpdir, "cartpole_summary.csv")
            with open(results_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["policy", "train_success", "test_success", "test_reward"])
                writer.writeheader()
                writer.writerow(
                    {
                        "policy": "raw",
                        "train_success": "0.0",
                        "test_success": "0.0",
                        "test_reward": "1.0",
                    }
                )
            with open(summary_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["policy", "train_success_mean", "test_success_mean", "test_reward_mean"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "policy": "summary",
                        "train_success_mean": "1.0",
                        "test_success_mean": "0.5",
                        "test_reward_mean": "10.0",
                    }
                )

            original_results = make_paper_figures.RESULTS_CSV
            original_summary = make_paper_figures.SUMMARY_CSV
            try:
                make_paper_figures.RESULTS_CSV = results_path
                make_paper_figures.SUMMARY_CSV = summary_path
                rows = make_paper_figures.read_results()
            finally:
                make_paper_figures.RESULTS_CSV = original_results
                make_paper_figures.SUMMARY_CSV = original_summary

        self.assertEqual(rows[0]["policy"], "summary")
        self.assertEqual(make_paper_figures.metric(rows[0], "test_reward"), 10.0)
        self.assertIsNone(make_paper_figures.metric_or_none(rows[0], "test_steps"))

    def test_read_results_falls_back_to_raw_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "cartpole_results.csv")
            summary_path = os.path.join(tmpdir, "missing_summary.csv")
            with open(results_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["policy", "train_success", "test_success", "test_reward"])
                writer.writeheader()
                writer.writerow(
                    {
                        "policy": "raw",
                        "train_success": "0.25",
                        "test_success": "0.0",
                        "test_reward": "2.0",
                    }
                )

            original_results = make_paper_figures.RESULTS_CSV
            original_summary = make_paper_figures.SUMMARY_CSV
            try:
                make_paper_figures.RESULTS_CSV = results_path
                make_paper_figures.SUMMARY_CSV = summary_path
                rows = make_paper_figures.read_results()
            finally:
                make_paper_figures.RESULTS_CSV = original_results
                make_paper_figures.SUMMARY_CSV = original_summary

        self.assertEqual(rows[0]["policy"], "raw")
        self.assertEqual(make_paper_figures.metric(rows[0], "train_success"), 0.25)

    def test_require_result_artifacts_accepts_metrics_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": "python train.py --metrics-output metrics.json",
                        "paper_protocol_status": {"paper_scale_result": False},
                        "selected_result": {},
                    },
                    handle,
                )

            make_paper_figures.require_result_artifacts(
                [artifact_row("PPO MLP", metrics_path, "python train.py --metrics-output metrics.json")]
            )

    def test_require_result_artifacts_accepts_synthesized_psm_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            make_paper_figures.require_result_artifacts(
                [synthesized_psm_row(metrics_path)]
            )

    def test_require_result_artifacts_rejects_synthesized_psm_trace_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": 1},
                        "num_traces": 2,
                        "traces": [{"reward": 1}],
                        "trace_history": [{"iteration": 1, "num_traces": 1, "traces": [{"reward": 1}]}],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_empty_objective_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            empty_summary = {"iteration": 1, "objective_component_summary": {}}
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "adaptive_teacher_summary": [empty_summary],
                        "synthesis_history": [
                            {"iteration": 1, "adaptive_teacher_summary": empty_summary}
                        ],
                        "traces_output": traces_path,
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaisesRegex(ValueError, "objective components"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_accepts_runner_named_synthesized_psm_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": RUNNER_QUICK_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(command=RUNNER_QUICK_COMMAND), handle)

            make_paper_figures.require_result_artifacts(
                [runner_psm_row(metrics_path)]
            )

    def test_require_result_artifacts_rejects_synthesized_psm_missing_current_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": RUNNER_QUICK_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": {"paper_scale_result": False},
                        "synthesis_history": [{"iteration": 1}],
                        "traces_output": traces_path,
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaisesRegex(ValueError, "current-synthesis protocol status"):
                make_paper_figures.require_result_artifacts([runner_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_missing_protocol_requirements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": {
                            "paper_scale_result": False,
                            "synthesized_by_current_algorithm": True,
                        },
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaisesRegex(ValueError, "current-synthesis protocol status"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_accepts_fixed_psm_without_synthesis_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": FIXED_PSM_COMMAND,
                        "paper_protocol_status": {
                            "paper_scale_result": False,
                            "synthesized_by_current_algorithm": False,
                        },
                    },
                    handle,
                )

            make_paper_figures.require_result_artifacts(
                [artifact_row("Programmatic state machine", metrics_path, FIXED_PSM_COMMAND)]
            )

    def test_require_result_artifacts_rejects_synthesized_psm_boolean_trace_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": 1},
                        "num_traces": True,
                        "traces": [{"reward": 1}],
                        "trace_history": [{"iteration": 1, "num_traces": 1, "traces": [{"reward": 1}]}],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_history_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": 1},
                        "num_traces": 1,
                        "traces": [{"reward": 1}],
                        "trace_history": [{"iteration": 1, "num_traces": 2, "traces": [{"reward": 1}]}],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_boolean_iteration_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": True},
                        "num_traces": 1,
                        "traces": [{"reward": 1}],
                        "trace_history": [{"iteration": 1, "num_traces": 1, "traces": [{"reward": 1}]}],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_missing_history_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": 2},
                        "num_traces": 1,
                        "traces": [{"reward": 1}],
                        "trace_history": [{"iteration": 1, "num_traces": 1, "traces": [{"reward": 1}]}],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_history_sequence_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"teacher_student_iters": 2},
                        "num_traces": 1,
                        "traces": [{"reward": 1}],
                        "trace_history": [
                            {"iteration": 1, "num_traces": 1, "traces": [{"reward": 0}]},
                            {"iteration": 3, "num_traces": 1, "traces": [{"reward": 1}]},
                        ],
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_missing_synthesized_psm_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": os.path.join(tmpdir, "missing_traces.json"),
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )

            with self.assertRaises(FileNotFoundError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_runner_named_synthesized_psm_missing_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": RUNNER_QUICK_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )

            with self.assertRaises(FileNotFoundError):
                make_paper_figures.require_result_artifacts([runner_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_stale_synthesized_psm_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump({"config": {"teacher_student_iters": 1}, "num_traces": 1, "traces": [{"reward": 1}]}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_runner_named_synthesized_psm_stale_trace_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": RUNNER_QUICK_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump({"config": {"teacher_student_iters": 1}, "num_traces": 1, "traces": [{"reward": 1}]}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([runner_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_stale_synthesized_psm_algorithm_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            stale_provenance = current_synthesized_psm_algorithm_provenance()
            stale_provenance["teacher_search"] = {
                **stale_provenance["teacher_search"],
                "selected_trace_candidate_pool_diagnostics": "stale_partial_diagnostics",
            }
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": stale_provenance,
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_trace_command_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(command="python stale.py --traces-output traces.json"), handle)

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_trace_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(config={"teacher_student_iters": 1, "seed": 7}), handle)

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_trace_summary_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            stale_traces = [
                {
                    "reward": 1,
                    "actions": [0.0, 0.0],
                    "mode_labels": [0, 0],
                    "observations": [[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]],
                    "theta_gain": 1.0,
                    "omega_gain": 0.0,
                    "segment_actions": [0.0],
                    "segment_durations": [2],
                    "segment_time_increments": [0.02],
                    "teacher_source": "unit_test_teacher",
                    "student_log_probability": -1.0,
                    "teacher_objective": 1.0,
                    "teacher_refinement_objective": 1.0,
                }
            ]
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    one_iteration_trace_payload(
                        traces=stale_traces,
                        trace_history=[{"iteration": 1, "num_traces": 1, "traces": stale_traces}],
                    ),
                    handle,
                )

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_trace_example_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            stale_traces = [
                {
                    **PSM_TRACES[0],
                    "segment_actions": [0.5],
                }
            ]
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": PSM_TRACE_COMMAND,
                        "algorithm_provenance": current_synthesized_psm_algorithm_provenance(),
                        "paper_protocol_status": current_synthesized_psm_status(),
                        "traces_output": traces_path,
                        **one_iteration_objective_component_metrics(),
                    },
                    handle,
                )
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    one_iteration_trace_payload(
                        traces=stale_traces,
                        trace_history=[{"iteration": 1, "num_traces": 1, "traces": stale_traces}],
                    ),
                    handle,
                )

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_missing_top_level_trace_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            trace_summary = dict(PSM_TRACE_SUMMARY)
            trace_summary["examples"] = []
            metrics = one_iteration_metrics_payload(
                traces_output=traces_path,
                trace_summary=trace_summary,
            )
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_truncated_top_level_trace_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            traces = [
                PSM_TRACES[0],
                {
                    **PSM_TRACES[0],
                    "observations": [[0.1, 0.0, 0.0, 0.0]],
                    "segment_actions": [0.1],
                },
            ]
            trace_summary = {
                "count": 2,
                "reward_mean": 1.0,
                "length_mean": 1.0,
                "teacher_source_counts": {"unit_test_teacher": 2},
                "examples": [PSM_TRACE_SUMMARY["examples"][0]],
            }
            metrics = one_iteration_metrics_payload(
                traces_output=traces_path,
                num_traces=2,
                trace_summary=trace_summary,
            )
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(
                    one_iteration_trace_payload(
                        num_traces=2,
                        traces=traces,
                        trace_history=[{"iteration": 1, "num_traces": 2, "traces": traces}],
                    ),
                    handle,
                )

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_synthesized_psm_missing_iteration_trace_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            history_trace_summary = dict(PSM_TRACE_SUMMARY)
            history_trace_summary["examples"] = []
            metrics = one_iteration_metrics_payload(
                traces_output=traces_path,
                synthesis_history=[
                    {
                        "iteration": 1,
                        "trace_summary": history_trace_summary,
                        "adaptive_teacher_summary": one_iteration_objective_component_metrics()[
                            "adaptive_teacher_summary"
                        ][0],
                    }
                ],
            )
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with open(traces_path, "w", encoding="utf-8") as handle:
                json.dump(one_iteration_trace_payload(), handle)

            with self.assertRaisesRegex(ValueError, "disagree with metrics provenance"):
                make_paper_figures.require_result_artifacts([synthesized_psm_row(metrics_path)])

    def test_require_result_artifacts_rejects_missing_protocol_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "python train.py", "selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_missing_command_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"paper_protocol_status": {"paper_scale_result": False}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_empty_command_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "  ", "paper_protocol_status": {"paper_scale_result": False}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_missing_row_command_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": "python train.py --metrics-output metrics.json",
                        "paper_protocol_status": {"paper_scale_result": False},
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_legacy_metrics_command_without_row_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            command = "python train.py --metrics-output metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": command,
                        "paper_protocol_status": {"paper_scale_result": False},
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "metrics_command": command,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_row_command_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "command": "python train.py --metrics-output metrics.json",
                        "paper_protocol_status": {"paper_scale_result": False},
                    },
                    handle,
                )

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "command": "python other.py --metrics-output metrics.json",
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_missing_eval_rollout_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "python train.py", "selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [{"policy": "PPO MLP", "metrics_output": metrics_path, "test_horizon_steps": "15000"}]
                )

    def test_require_result_artifacts_rejects_paper_scale_rows_without_1000_rollouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "python train.py", "paper_protocol_status": {"paper_scale_result": False}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "15000",
                            "paper_scale_result": "true",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_missing_test_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "python train.py", "selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [{"policy": "PPO MLP", "metrics_output": metrics_path, "eval_rollouts": "20"}]
                )

    def test_require_result_artifacts_rejects_short_test_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"command": "python train.py", "selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [
                        {
                            "policy": "PPO MLP",
                            "metrics_output": metrics_path,
                            "eval_rollouts": "20",
                            "test_horizon_steps": "100",
                        }
                    ]
                )

    def test_require_result_artifacts_rejects_missing_outputs(self):
        with self.assertRaises(FileNotFoundError):
            make_paper_figures.require_result_artifacts(
                [
                    {
                        "policy": "PPO MLP",
                        "metrics_output": "missing.json",
                        "eval_rollouts": "20",
                        "test_horizon_steps": "15000",
                    }
                ]
            )

    def test_write_results_table_uses_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = os.path.join(tmpdir, "table.tex")
            make_paper_figures.write_results_table(
                [
                    {
                        "policy": "Programmatic state machine",
                        "train_success_mean": "1.0",
                        "test_success_mean": "0.0",
                        "train_reward_mean": "250.0",
                        "test_reward_mean": "1560.6",
                    }
                ],
                table_path,
            )
            with open(table_path, encoding="utf-8") as handle:
                table = handle.read()

        self.assertIn("Generated by scripts/make_paper_figures.py", table)
        self.assertIn("Local diagnostic artifacts only", table)
        self.assertIn(r"10\textsuperscript{7}-timestep, five-seed, 1000-rollout PPO/PPO-LSTM protocol", table)
        self.assertIn("Programmatic PSM & 1.00 & 0.00 & 250.0 & 1560.6", table)
        self.assertIn("\\bottomrule", table)

    def test_plot_survival_rewards_prefers_explicit_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "figures")
            os.makedirs(outdir)
            original_out_dir = make_paper_figures.OUT_DIR
            try:
                make_paper_figures.OUT_DIR = outdir
                make_paper_figures.plot_survival_rewards(
                    [
                        {
                            "policy": "PPO MLP",
                            "test_reward_mean": "900.0",
                            "test_steps_mean": "901.0",
                        }
                    ]
                )
            finally:
                make_paper_figures.OUT_DIR = original_out_dir

            outpath = os.path.join(outdir, "cartpole_test_survival_reward.png")
            self.assertTrue(os.path.exists(outpath))
            self.assertGreater(os.path.getsize(outpath), 0)

    def test_write_abstract_results_uses_result_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "abstract.tex")
            wrote = make_paper_figures.write_abstract_results(
                [
                    {
                        "policy": "PPO MLP",
                        "train_success_mean": "1.0",
                        "test_success_mean": "0.0",
                        "test_reward_mean": "910.6",
                    },
                    {
                        "policy": "Programmatic state machine",
                        "train_success_mean": "1.0",
                        "test_success_mean": "0.0",
                        "test_reward_mean": "1560.6",
                    },
                ],
                outpath,
            )
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertTrue(wrote)
        self.assertIn("Local diagnostic artifacts only", fragment)
        self.assertIn("feed-forward PPO reaches 100\\% training success", fragment)
        self.assertNotIn("20 rollouts", fragment)
        self.assertIn("obtains 0\\% success", fragment)
        self.assertIn("mean test reward 910.6", fragment)
        self.assertIn("fixed programmatic state machine reaches 100\\% training success", fragment)
        self.assertIn("obtains 0\\% full-horizon test success", fragment)
        self.assertIn("mean test reward 1560.6", fragment)

    def test_write_abstract_results_records_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "abstract.tex")
            wrote = make_paper_figures.write_abstract_results(
                [{"policy": "PPO MLP", "train_success": "1.0", "test_success": "0.0", "test_reward": "10.0"}],
                outpath,
            )
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertFalse(wrote)
        self.assertIn("required result rows were unavailable", fragment)

    def test_read_ppo_metric_files_skips_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            good_path = os.path.join(tmpdir, "good_metrics.json")
            empty_path = os.path.join(tmpdir, "empty_metrics.json")
            with open(good_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"policy_type": "mlp", "seed": 0},
                        "eval_history": [
                            {
                                "timesteps": 32,
                                "train_success_rate": 0.5,
                                "test_success_rate": 0.25,
                            }
                        ],
                    },
                    handle,
                )
            with open(empty_path, "w", encoding="utf-8") as handle:
                json.dump({"config": {"policy_type": "mlp"}, "eval_history": []}, handle)

            metric_files = make_paper_figures.read_ppo_metric_files([os.path.join(tmpdir, "*_metrics.json")])

        self.assertEqual(len(metric_files), 1)
        self.assertEqual(make_paper_figures.metric_label(metric_files[0]), "MLP seed 0")

    def test_default_ppo_metric_globs_include_runner_metrics_dir(self):
        runner_metrics_pattern = os.path.join("artifacts", "results", "metrics", "*.json")

        self.assertTrue(
            any(pattern.endswith(runner_metrics_pattern) for pattern in make_paper_figures.PPO_METRICS_GLOBS)
        )

    def test_default_psm_metric_globs_include_runner_metrics_dir(self):
        runner_metrics_pattern = os.path.join("artifacts", "results", "metrics", "psm_seed*.json")

        self.assertTrue(
            any(pattern.endswith(runner_metrics_pattern) for pattern in make_paper_figures.PSM_METRICS_GLOBS)
        )

    def test_default_figure19_metric_globs_include_reference_metrics(self):
        reference_pattern = os.path.join("artifacts", "results", "metrics", "figure19*.json")

        self.assertTrue(
            any(pattern.endswith(reference_pattern) for pattern in make_paper_figures.FIGURE19_METRICS_GLOBS)
        )

    def test_read_figure19_metric_files_requires_reference_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            good_path = os.path.join(tmpdir, "figure19_reference.json")
            synthesized_path = os.path.join(tmpdir, "figure19_synthesized.json")
            with open(good_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "paper_protocol_status": {
                            "policy_source": "paper_figure19_manual_transcription",
                            "synthesized_by_current_algorithm": False,
                        },
                        "program_parameters": {"figure": "SPPIG paper Figure 19"},
                    },
                    handle,
                )
            with open(synthesized_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "paper_protocol_status": {
                            "policy_source": "fixed_two_mode_program_parameters",
                            "synthesized_by_current_algorithm": False,
                        },
                        "program_parameters": {"figure": "SPPIG paper Figure 19"},
                    },
                    handle,
                )

            metric_files = make_paper_figures.read_figure19_metric_files(
                [os.path.join(tmpdir, "figure19*.json")]
            )

        self.assertEqual(len(metric_files), 1)
        self.assertEqual(metric_files[0]["path"], good_path)

    def test_parse_linear_switch_from_policy_description(self):
        parsed = make_paper_figures.parse_linear_switch(
            "m0 action=-10.000; m1 action=10.000; mode=1 if 12.500*theta + 0.750*omega >= 0.250, else mode=0"
        )

        self.assertEqual(parsed, (12.5, 0.75, 0.25))

    def test_linear_switch_latex_formats_negative_omega_weight(self):
        latex = make_paper_figures.linear_switch_latex((12.5, -0.75, 0.25))

        self.assertEqual(latex, "12.5\\theta_t - 0.75\\dot{\\theta}_t \\ge 0.25")
        self.assertEqual(
            make_paper_figures.linear_switch_mathtext((12.5, -0.75, 0.25)),
            "12.5\\theta - 0.75\\dot{\\theta} \\geq 0.25",
        )
        self.assertFalse(make_paper_figures.mode1_region_is_above_boundary((12.5, -0.75, 0.25)))

    def test_read_psm_metric_files_requires_policy_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            good_path = os.path.join(tmpdir, "psm_seed0.json")
            empty_path = os.path.join(tmpdir, "psm_seed1.json")
            with open(good_path, "w", encoding="utf-8") as handle:
                json.dump({"policy_description": "mode=1 if 5.000*theta + 0.500*omega >= 0.000"}, handle)
            with open(empty_path, "w", encoding="utf-8") as handle:
                json.dump({"config": {}}, handle)

            metric_files = make_paper_figures.read_psm_metric_files([os.path.join(tmpdir, "psm_seed*.json")])

        self.assertEqual(len(metric_files), 1)
        self.assertEqual(metric_files[0]["path"], good_path)

    def test_read_psm_metric_files_prefers_fixed_program_result_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            smoke_path = os.path.join(tmpdir, "cartpole_psm_smoke_metrics.json")
            fixed_path = os.path.join(tmpdir, "psm_seed0_fixed_program_full_horizon.json")
            with open(smoke_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "policy_description": (
                            "m0 action=-1.000; m1 action=1.000; "
                            "mode=1 if -1.000*theta + -0.500*omega >= 0.033, else mode=0"
                        )
                    },
                    handle,
                )
            with open(fixed_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "policy_description": (
                            "m0 action=-10.000; m1 action=10.000; "
                            "mode=1 if 10.000*theta + 1.000*omega >= 0.000, else mode=0"
                        ),
                        "paper_protocol_status": {
                            "policy_source": "fixed_two_mode_program_parameters",
                            "uses_full_test_horizon": True,
                        },
                    },
                    handle,
                )

            metric_files = make_paper_figures.read_psm_metric_files(
                [
                    os.path.join(tmpdir, "cartpole_psm*_metrics.json"),
                    os.path.join(tmpdir, "psm_seed*.json"),
                ]
            )

        self.assertEqual([metric_file["path"] for metric_file in metric_files], [fixed_path, smoke_path])

    def test_plot_switch_boundary_uses_psm_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "switch.png")
            wrote = make_paper_figures.plot_switch_boundary(
                [
                    {
                        "path": "synthetic.json",
                        "payload": {
                            "policy_description": (
                                "m0 action=-10.000; m1 action=10.000; "
                                "mode=1 if 12.500*theta + 0.750*omega >= 0.250, else mode=0"
                            )
                        },
                    }
                ],
                outpath,
            )

            self.assertTrue(wrote)
            self.assertTrue(os.path.exists(outpath))
            self.assertGreater(os.path.getsize(outpath), 0)

    def test_write_policy_fragment_uses_linear_psm_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "policy.tex")
            wrote = make_paper_figures.write_policy_fragment(
                [
                    {
                        "path": "synthetic.json",
                        "payload": {
                            "policy_description": (
                                "m0 action=-10.000; m1 action=10.000; "
                                "mode=1 if 12.500*theta + 0.750*omega >= 0.250, else mode=0"
                            )
                        },
                    }
                ],
                outpath,
            )
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertTrue(wrote)
        self.assertIn("Generated by scripts/make_paper_figures.py", fragment)
        self.assertIn("+10, & 12.5\\theta_t + 0.75\\dot{\\theta}_t \\ge 0.25", fragment)

    def test_write_figure19_reference_fragment_uses_manual_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "figure19.tex")
            wrote = make_paper_figures.write_figure19_reference_fragment(
                [
                    {
                        "path": "figure19_reference.json",
                        "payload": {
                            "paper_protocol_status": {
                                "policy_source": "paper_figure19_manual_transcription",
                                "synthesized_by_current_algorithm": False,
                            },
                            "program_parameters": {
                                "figure": "SPPIG paper Figure 19",
                                "start": {"m1": "omega >= 0.02", "m2": "omega < 0.02"},
                                "modes": {
                                    "m1": {
                                        "action": -3.3,
                                        "switch_to_m2": "omega >= 0.46 and theta >= -0.06",
                                    },
                                    "m2": {"action": 3.98, "switch_to_m1": "omega < -0.49"},
                                },
                            },
                        },
                    }
                ],
                outpath,
            )
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertTrue(wrote)
        self.assertIn("Generated by scripts/make_paper_figures.py", fragment)
        self.assertIn(r"m_0 &\to m_1 \text{ if } \dot{\theta}_t \ge 0.02", fragment)
        self.assertIn(r"a_{m_1} &= -3.3", fragment)
        self.assertIn(r"\dot{\theta}_t \ge 0.46 \wedge \theta_t \ge -0.06", fragment)
        self.assertIn("manual visual transcription", fragment)
        self.assertIn(r"\texttt{synthesized\_by\_current\_algorithm=false}", fragment)
        self.assertIn(r"paper\_figure19\_manual\_transcription", fragment)

    def test_write_figure19_reference_fragment_records_missing_reference_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "figure19.tex")
            wrote = make_paper_figures.write_figure19_reference_fragment([], outpath)
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertFalse(wrote)
        self.assertIn("no Figure 19 reference metrics artifact was available", fragment)

    def test_write_policy_fragment_records_missing_linear_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "policy.tex")
            wrote = make_paper_figures.write_policy_fragment(
                [
                    {
                        "path": "synthetic.json",
                        "payload": {"policy_description": "mode=1 if o[2] >= 0.000, else mode=0"},
                    }
                ],
                outpath,
            )
            with open(outpath, encoding="utf-8") as handle:
                fragment = handle.read()

        self.assertFalse(wrote)
        self.assertIn("no linear PSM metrics artifact was available", fragment)

    def test_plot_switch_boundary_skips_non_linear_switch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "switch.png")
            wrote = make_paper_figures.plot_switch_boundary(
                [
                    {
                        "path": "synthetic.json",
                        "payload": {"policy_description": "mode=1 if o[2] >= 0.000, else mode=0"},
                    }
                ],
                outpath,
            )

            self.assertFalse(wrote)
            self.assertFalse(os.path.exists(outpath))

    def test_plot_ppo_training_curves_writes_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "curve.png")
            wrote = make_paper_figures.plot_ppo_training_curves(
                [
                    {
                        "path": "synthetic.json",
                        "payload": {
                            "config": {"policy_type": "mlp", "seed": 0},
                            "eval_history": [
                                {
                                    "timesteps": 32,
                                    "train_success_rate": 0.5,
                                    "test_success_rate": 0.25,
                                },
                                {
                                    "timesteps": 64,
                                    "train_success_rate": 1.0,
                                    "test_success_rate": 0.5,
                                },
                            ],
                        },
                    }
                ],
                outpath,
            )

            self.assertTrue(wrote)
            self.assertTrue(os.path.exists(outpath))
            self.assertGreater(os.path.getsize(outpath), 0)


if __name__ == "__main__":
    unittest.main()
