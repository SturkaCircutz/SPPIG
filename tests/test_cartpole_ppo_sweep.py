import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "run_cartpole_ppo_sweep.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_cartpole_ppo_sweep import (  # noqa: E402
    PAPER_NMINIBATCHES,
    PAPER_HYPERPARAMETER_SAMPLES,
    PLAN_FIELDS,
    build_jobs,
    count_uncapped_jobs,
    parse_args,
    paper_protocol_status,
    read_existing_results,
    resumable_result_for_job,
    summarize_results,
)


class CartpolePPOSweepTest(unittest.TestCase):
    def test_summarize_results_selects_best_train_per_policy(self):
        summary = summarize_results(
            [
                {
                    "job_id": 0,
                    "policy": "mlp",
                    "seed": 0,
                    "train_success": 0.5,
                    "test_success": 1.0,
                    "train_reward": 100.0,
                    "test_reward": 200.0,
                    "selected_timesteps": 32,
                    "minibatches": 1,
                    "learning_rate": 0.001,
                    "entropy_coef": 0.0,
                    "update_epochs": 3,
                    "clip_range": 0.1,
                    "output": "a.pt",
                    "metrics_output": "a.json",
                },
                {
                    "job_id": 1,
                    "policy": "mlp",
                    "seed": 1,
                    "train_success": 1.0,
                    "test_success": 0.0,
                    "train_reward": 250.0,
                    "test_reward": 50.0,
                    "selected_timesteps": 64,
                    "minibatches": 8,
                    "learning_rate": 0.0003,
                    "entropy_coef": 0.01,
                    "update_epochs": 8,
                    "clip_range": 0.2,
                    "output": "b.pt",
                    "metrics_output": "b.json",
                },
            ]
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["policy"], "mlp")
        self.assertEqual(summary[0]["jobs_completed"], 2)
        self.assertEqual(summary[0]["best_job_id"], 1)
        self.assertEqual(summary[0]["best_minibatches"], 8)
        self.assertAlmostEqual(summary[0]["best_learning_rate"], 0.0003)

    def test_build_jobs_uses_paper_minibatch_rule_for_lstm(self):
        original_argv = sys.argv
        try:
            sys.argv = [
                SCRIPT,
                "--policies",
                "mlp,lstm",
                "--hyperparam-mode",
                "grid",
                "--seeds",
                "0",
                "--learning-rates",
                "0.001",
                "--nminibatches",
                ",".join(str(value) for value in PAPER_NMINIBATCHES),
                "--ent-coefs",
                "0.0",
                "--update-epochs",
                "3",
                "--clip-ranges",
                "0.1",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        jobs = build_jobs(args)
        mlp_jobs = [job for job in jobs if job["policy"] == "mlp"]
        lstm_jobs = [job for job in jobs if job["policy"] == "lstm"]

        self.assertEqual(len(mlp_jobs), len(PAPER_NMINIBATCHES))
        self.assertEqual(len(lstm_jobs), 1)
        self.assertEqual(lstm_jobs[0]["minibatches"], 1)
        self.assertEqual(mlp_jobs[0]["total_timesteps"], 10_000_000)
        self.assertEqual(count_uncapped_jobs(args), len(PAPER_NMINIBATCHES) + 1)

    def test_build_jobs_defaults_to_paper_random_hyperparameter_samples(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--policies", "mlp,lstm", "--seeds", "0"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        jobs = build_jobs(args)
        mlp_jobs = [job for job in jobs if job["policy"] == "mlp"]
        lstm_jobs = [job for job in jobs if job["policy"] == "lstm"]

        self.assertEqual(len(mlp_jobs), PAPER_HYPERPARAMETER_SAMPLES)
        self.assertEqual(len(lstm_jobs), PAPER_HYPERPARAMETER_SAMPLES)
        self.assertEqual({job["hyperparam_mode"] for job in jobs}, {"paper-random"})
        self.assertEqual({int(job["minibatches"]) for job in lstm_jobs}, {1})
        self.assertTrue(all(5e-6 <= float(job["learning_rate"]) <= 0.003 for job in jobs))
        self.assertEqual(count_uncapped_jobs(args), 2 * PAPER_HYPERPARAMETER_SAMPLES)

    def test_dry_run_writes_plan_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--dry-run",
                    "--quick",
                    "--max-configs",
                    "2",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            plan_path = os.path.join(tmpdir, "cartpole_ppo_sweep_plan.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json")
            self.assertTrue(os.path.exists(plan_path))
            self.assertTrue(os.path.exists(manifest_path))

            with open(plan_path, newline="", encoding="utf-8") as handle:
                plan_rows = list(csv.DictReader(handle))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(len(plan_rows), 2)
        self.assertEqual(plan_rows[0]["hyperparam_mode"], "paper-random")
        self.assertTrue(manifest["dry_run"])
        self.assertTrue(manifest["quick"])
        self.assertEqual(manifest["jobs_planned"], 2)
        self.assertEqual(manifest["hyperparam_mode"], "paper-random")
        self.assertEqual(manifest["hyperparam_samples"], 10)
        self.assertEqual(manifest["paper_space"]["hyperparameter_samples"], 10)
        self.assertGreater(manifest["jobs_uncapped_for_selected_space"], manifest["jobs_planned"])
        self.assertEqual(manifest["jobs_completed"], 0)
        self.assertEqual(manifest["paper_space"]["timesteps"], 10_000_000)
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_plan"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_execution"])
        self.assertTrue(manifest["paper_protocol_status"]["quick_diagnostic"])
        self.assertTrue(manifest["paper_protocol_status"]["dry_run_only"])
        self.assertTrue(manifest["paper_protocol_status"]["truncated_by_max_configs"])

    def test_paper_protocol_status_identifies_full_dry_run_plan(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertTrue(status["paper_timestep_budget"])
        self.assertTrue(status["paper_test_horizon"])
        self.assertTrue(status["paper_seed_count"])
        self.assertTrue(status["full_baseline_policy_set"])
        self.assertEqual(status["hyperparam_mode"], "paper-random")
        self.assertTrue(status["paper_random_hyperparameter_search"])
        self.assertTrue(status["paper_random_sample_count"])
        self.assertTrue(status["paper_random_learning_rate_values_within_reported_interval"])
        self.assertTrue(status["learning_rate_values_within_reported_interval"])
        self.assertFalse(status["grid_hyperparameter_search"])
        self.assertFalse(status["full_reported_mlp_grid"])
        self.assertTrue(status["ppo_lstm_minibatches_fixed_to_one"])
        self.assertTrue(status["paper_scale_plan"])
        self.assertFalse(status["paper_scale_execution"])

    def test_paper_protocol_status_requires_full_test_horizon(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--test-max-steps", "1000"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertFalse(status["paper_test_horizon"])
        self.assertEqual(status["selected_test_max_steps"], 1000)
        self.assertFalse(status["paper_scale_plan"])
        self.assertFalse(status["paper_scale_execution"])

    def test_paper_protocol_status_rejects_grid_mode_as_paper_scale_plan(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--hyperparam-mode", "grid"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertTrue(status["grid_hyperparameter_search"])
        self.assertTrue(status["full_reported_mlp_grid"])
        self.assertTrue(status["full_default_learning_rate_grid"])
        self.assertFalse(status["paper_random_hyperparameter_search"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_requires_ten_random_hyperparameter_samples(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--hyperparam-samples", "9"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertFalse(status["paper_random_sample_count"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_duplicate_seed_list(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--seeds", "0,0,0,0,0"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args, jobs_planned=10, jobs_completed=10, jobs_failed=0)

        self.assertFalse(status["paper_seed_count"])
        self.assertFalse(status["paper_scale_plan"])
        self.assertFalse(status["paper_scale_execution"])

    def test_paper_protocol_status_requires_completed_jobs_for_execution(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT]
            args = parse_args()
        finally:
            sys.argv = original_argv

        expected_jobs = count_uncapped_jobs(args)

        incomplete_status = paper_protocol_status(
            args,
            jobs_planned=expected_jobs,
            jobs_completed=expected_jobs - 1,
            jobs_failed=0,
        )
        failed_status = paper_protocol_status(
            args,
            jobs_planned=expected_jobs,
            jobs_completed=expected_jobs - 1,
            jobs_failed=1,
        )
        mismatched_count_status = paper_protocol_status(
            args,
            jobs_planned=10,
            jobs_completed=10,
            jobs_failed=0,
        )
        completed_status = paper_protocol_status(
            args,
            jobs_planned=expected_jobs,
            jobs_completed=expected_jobs,
            jobs_failed=0,
        )

        self.assertTrue(incomplete_status["paper_scale_plan"])
        self.assertTrue(incomplete_status["planned_job_count_matches_selected_space"])
        self.assertFalse(incomplete_status["all_planned_jobs_completed"])
        self.assertFalse(incomplete_status["paper_scale_execution"])
        self.assertFalse(failed_status["all_planned_jobs_completed"])
        self.assertFalse(failed_status["paper_scale_execution"])
        self.assertFalse(mismatched_count_status["planned_job_count_matches_selected_space"])
        self.assertTrue(mismatched_count_status["all_planned_jobs_completed"])
        self.assertFalse(mismatched_count_status["paper_scale_execution"])
        self.assertTrue(completed_status["all_planned_jobs_completed"])
        self.assertTrue(completed_status["planned_job_count_matches_selected_space"])
        self.assertTrue(completed_status["paper_scale_execution"])

    def test_paper_protocol_status_rejects_empty_learning_rate_list(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--hyperparam-mode", "grid", "--learning-rates", ""]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertFalse(status["grid_learning_rate_values_within_reported_interval"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_reduced_learning_rate_grid(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--hyperparam-mode", "grid", "--learning-rates", "0.001"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertTrue(status["grid_learning_rate_values_within_reported_interval"])
        self.assertFalse(status["full_default_learning_rate_grid"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_partial_policy_set(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--policies", "mlp"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertTrue(status["includes_ppo_mlp"])
        self.assertFalse(status["includes_ppo_lstm"])
        self.assertFalse(status["full_baseline_policy_set"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_duplicate_policy_entries(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--policies", "mlp,lstm,lstm"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertTrue(status["includes_ppo_mlp"])
        self.assertTrue(status["includes_ppo_lstm"])
        self.assertFalse(status["full_baseline_policy_set"])
        self.assertFalse(status["paper_scale_plan"])

    def test_quick_execution_writes_results_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--max-configs",
                    "1",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            results_path = os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv")
            summary_path = os.path.join(tmpdir, "cartpole_ppo_sweep_summary.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json")
            self.assertTrue(os.path.exists(results_path))
            self.assertTrue(os.path.exists(summary_path))

            with open(results_path, newline="", encoding="utf-8") as handle:
                result_rows = list(csv.DictReader(handle))
            with open(summary_path, newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(result_rows[0]["hyperparam_mode"], "paper-random")
        self.assertEqual(result_rows[0]["hyperparam_sample"], "0")
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0]["best_job_id"], "0")
        self.assertEqual(manifest["hyperparam_mode"], "paper-random")
        self.assertEqual(manifest["hyperparam_samples"], 10)
        self.assertEqual(manifest["jobs_completed"], 1)
        self.assertEqual(manifest["jobs_failed"], 0)
        self.assertEqual(manifest["jobs_skipped_existing"], 0)
        self.assertEqual(manifest["jobs_run_this_invocation"], 1)
        self.assertIn("selection_rule", manifest)
        self.assertIn("summary", manifest["artifacts"])

    def test_resume_skips_matching_completed_jobs_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--max-configs",
                    "1",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )
            with open(os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv"), newline="", encoding="utf-8") as handle:
                first_rows = list(csv.DictReader(handle))

            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--resume",
                    "--max-configs",
                    "2",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with open(os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], first_rows[0])
        self.assertEqual(rows[1]["job_id"], "1")
        self.assertTrue(manifest["resume"])
        self.assertEqual(manifest["jobs_planned"], 2)
        self.assertEqual(manifest["jobs_completed"], 2)
        self.assertEqual(manifest["jobs_failed"], 0)
        self.assertEqual(manifest["jobs_skipped_existing"], 1)
        self.assertEqual(manifest["jobs_run_this_invocation"], 1)

    def test_resume_rejects_rows_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_argv = sys.argv
            try:
                sys.argv = [
                    SCRIPT,
                    "--quick",
                    "--max-configs",
                    "1",
                    "--outdir",
                    tmpdir,
                ]
                args = parse_args()
            finally:
                sys.argv = original_argv
            job = build_jobs(args)[0]
            row = {
                **{field: str(job[field]) for field in PLAN_FIELDS},
                "train_success": "0.0",
                "test_success": "0.0",
                "train_reward": "1.0",
                "test_reward": "1.0",
                "selected_timesteps": "64",
            }
            results_path = os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv")
            with open(results_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            existing = read_existing_results(os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv"))

        self.assertIsNone(resumable_result_for_job(job, existing))

    def test_continue_on_error_records_failed_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--continue-on-error",
                    "--policies",
                    "bad",
                    "--max-configs",
                    "1",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            failures_path = os.path.join(tmpdir, "cartpole_ppo_sweep_failures.csv")
            results_path = os.path.join(tmpdir, "cartpole_ppo_sweep_results.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json")
            self.assertTrue(os.path.exists(failures_path))
            self.assertTrue(os.path.exists(results_path))
            with open(failures_path, newline="", encoding="utf-8") as handle:
                failures = list(csv.DictReader(handle))
            with open(results_path, newline="", encoding="utf-8") as handle:
                results = list(csv.DictReader(handle))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(results, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["policy"], "bad")
        self.assertEqual(failures[0]["error_type"], "ValueError")
        self.assertIn("policy_type", failures[0]["error_message"])
        self.assertTrue(manifest["continue_on_error"])
        self.assertEqual(manifest["jobs_completed"], 0)
        self.assertEqual(manifest["jobs_failed"], 1)
        self.assertFalse(manifest["paper_protocol_status"]["all_planned_jobs_completed"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_execution"])

    def test_default_job_failure_stops_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--policies",
                    "bad",
                    "--max-configs",
                    "1",
                    "--outdir",
                    tmpdir,
                ],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )

        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
