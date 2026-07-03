import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_paper_figures  # noqa: E402


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
        self.assertEqual(manifest["row_count"], len(summary))
        self.assertEqual(manifest["summary_csv"], "artifacts/results/cartpole_summary.csv")
        self.assertEqual({row["policy"] for row in summary}, set(manifest["policies"]))
        self.assertTrue(all(row["best_metrics_output"] for row in summary))
        self.assertTrue(all(os.path.exists(os.path.join(ROOT, row["best_metrics_output"])) for row in summary))
        fixed_psm_row = next(row for row in manifest["rows"] if row["policy"] == "Programmatic state machine")
        with open(os.path.join(ROOT, fixed_psm_row["metrics_output"]), encoding="utf-8") as handle:
            fixed_psm_metrics = json.load(handle)
        self.assertEqual(fixed_psm_row["paper_protocol_status"], fixed_psm_metrics["paper_protocol_status"])
        self.assertFalse(fixed_psm_row["paper_protocol_status"]["synthesized_by_current_algorithm"])
        self.assertFalse(fixed_psm_row["paper_protocol_status"]["paper_scale_fixed_program_result"])
        synthesized_psm_row = next(row for row in manifest["rows"] if row["policy"] == "Synthesized PSM diagnostic")
        with open(os.path.join(ROOT, synthesized_psm_row["metrics_output"]), encoding="utf-8") as handle:
            synthesized_psm_metrics = json.load(handle)
        self.assertNotIn("artifact_status", synthesized_psm_metrics)
        self.assertNotIn("artifact_status", synthesized_psm_row)
        self.assertEqual(
            synthesized_psm_metrics["algorithm_provenance"]["probabilistic_student"]["mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertEqual(
            synthesized_psm_row["algorithm_provenance"]["probabilistic_student"]["mode_update_order"],
            synthesized_psm_metrics["algorithm_provenance"]["probabilistic_student"]["mode_update_order"],
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
                json.dump({"paper_protocol_status": {"paper_scale_result": False}, "selected_result": {}}, handle)

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

    def test_require_result_artifacts_rejects_missing_protocol_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"selected_result": {}}, handle)

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

    def test_require_result_artifacts_rejects_missing_eval_rollout_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [{"policy": "PPO MLP", "metrics_output": metrics_path, "test_horizon_steps": "15000"}]
                )

    def test_require_result_artifacts_rejects_paper_scale_rows_without_1000_rollouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"selected_result": {}}, handle)

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
                json.dump({"selected_result": {}}, handle)

            with self.assertRaises(ValueError):
                make_paper_figures.require_result_artifacts(
                    [{"policy": "PPO MLP", "metrics_output": metrics_path, "eval_rollouts": "20"}]
                )

    def test_require_result_artifacts_rejects_short_test_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump({"selected_result": {}}, handle)

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
