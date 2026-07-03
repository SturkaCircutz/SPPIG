import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


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
    sampled_hyperparameter_manifest,
    summarize_hyperparameter_configs,
    summarize_results,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def survival_fields(train_steps=250.0, test_steps=50.0):
    return {
        "train_steps": train_steps,
        "test_steps": test_steps,
        "train_survival_seconds": train_steps * 0.02,
        "test_survival_seconds": test_steps * 0.02,
    }


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
                    **survival_fields(100.0, 200.0),
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
                    **survival_fields(250.0, 50.0),
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
        self.assertEqual(summary[0]["best_test_steps"], 50.0)
        self.assertEqual(summary[0]["best_test_survival_seconds"], 1.0)
        self.assertEqual(summary[0]["best_minibatches"], 8)
        self.assertAlmostEqual(summary[0]["best_learning_rate"], 0.0003)

    def test_summarize_hyperparameter_configs_aggregates_completed_seeds(self):
        rows = [
            {
                "job_id": 0,
                "policy": "mlp",
                "seed": 0,
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 0,
                "train_success": 1.0,
                "test_success": 0.0,
                "train_reward": 200.0,
                "test_reward": 50.0,
                **survival_fields(200.0, 50.0),
                "selected_timesteps": 64,
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
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 0,
                "train_success": 0.0,
                "test_success": 1.0,
                "train_reward": 100.0,
                "test_reward": 70.0,
                **survival_fields(100.0, 70.0),
                "selected_timesteps": 64,
                "minibatches": 1,
                "learning_rate": 0.001,
                "entropy_coef": 0.0,
                "update_epochs": 3,
                "clip_range": 0.1,
                "output": "b.pt",
                "metrics_output": "b.json",
            },
            {
                "job_id": 2,
                "policy": "mlp",
                "seed": 0,
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 1,
                "train_success": 0.75,
                "test_success": 0.5,
                "train_reward": 175.0,
                "test_reward": 60.0,
                **survival_fields(175.0, 60.0),
                "selected_timesteps": 64,
                "minibatches": 4,
                "learning_rate": 0.0003,
                "entropy_coef": 0.01,
                "update_epochs": 8,
                "clip_range": 0.2,
                "output": "c.pt",
                "metrics_output": "c.json",
            },
        ]

        summary = summarize_hyperparameter_configs(rows, selected_seeds=[0, 1])

        self.assertEqual(len(summary), 2)
        first = summary[0]
        second = summary[1]
        self.assertEqual(first["hyperparam_sample"], 0)
        self.assertEqual(first["jobs_completed"], 2)
        self.assertEqual(first["seed_count"], 2)
        self.assertEqual(first["seeds_completed"], "0,1")
        self.assertEqual(first["selected_seed_count"], 2)
        self.assertEqual(first["selected_seeds"], "0,1")
        self.assertEqual(first["missing_seeds"], "")
        self.assertTrue(first["complete_seed_coverage"])
        self.assertAlmostEqual(first["train_success_mean"], 0.5)
        self.assertAlmostEqual(first["train_success_std"], 0.7071067811865476)
        self.assertAlmostEqual(first["test_steps_mean"], 60.0)
        self.assertAlmostEqual(first["test_survival_seconds_mean"], 1.2)
        self.assertEqual(first["best_job_id"], 0)
        self.assertTrue(first["is_best_hyperparam_for_policy"])
        self.assertEqual(second["hyperparam_sample"], 1)
        self.assertEqual(second["seed_count"], 1)
        self.assertEqual(second["selected_seed_count"], 2)
        self.assertEqual(second["missing_seeds"], "1")
        self.assertFalse(second["complete_seed_coverage"])
        self.assertFalse(second["is_best_hyperparam_for_policy"])

    def test_summarize_hyperparameter_configs_prefers_complete_seed_coverage(self):
        rows = [
            {
                "job_id": 0,
                "policy": "mlp",
                "seed": 0,
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 0,
                "train_success": 0.4,
                "test_success": 0.0,
                "train_reward": 100.0,
                "test_reward": 50.0,
                **survival_fields(100.0, 50.0),
                "selected_timesteps": 64,
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
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 0,
                "train_success": 0.4,
                "test_success": 0.0,
                "train_reward": 100.0,
                "test_reward": 50.0,
                **survival_fields(100.0, 50.0),
                "selected_timesteps": 64,
                "minibatches": 1,
                "learning_rate": 0.001,
                "entropy_coef": 0.0,
                "update_epochs": 3,
                "clip_range": 0.1,
                "output": "b.pt",
                "metrics_output": "b.json",
            },
            {
                "job_id": 2,
                "policy": "mlp",
                "seed": 0,
                "hyperparam_mode": "paper-random",
                "hyperparam_sample": 1,
                "train_success": 1.0,
                "test_success": 0.0,
                "train_reward": 250.0,
                "test_reward": 50.0,
                **survival_fields(250.0, 50.0),
                "selected_timesteps": 64,
                "minibatches": 4,
                "learning_rate": 0.0003,
                "entropy_coef": 0.01,
                "update_epochs": 8,
                "clip_range": 0.2,
                "output": "c.pt",
                "metrics_output": "c.json",
            },
        ]

        summary = summarize_hyperparameter_configs(rows, selected_seeds=[0, 1])
        by_sample = {row["hyperparam_sample"]: row for row in summary}

        self.assertTrue(by_sample[0]["complete_seed_coverage"])
        self.assertFalse(by_sample[1]["complete_seed_coverage"])
        self.assertEqual(by_sample[1]["missing_seeds"], "1")
        self.assertTrue(by_sample[0]["is_best_hyperparam_for_policy"])
        self.assertFalse(by_sample[1]["is_best_hyperparam_for_policy"])

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

    def test_sampled_hyperparameter_manifest_records_each_policy_config_once(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--policies", "mlp,lstm", "--seeds", "0,1"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        jobs = build_jobs(args)
        samples = sampled_hyperparameter_manifest(args)

        self.assertEqual(len(samples), 2 * PAPER_HYPERPARAMETER_SAMPLES)
        self.assertEqual({sample["hyperparam_mode"] for sample in samples}, {"paper-random"})
        self.assertEqual({sample["minibatches"] for sample in samples if sample["policy"] == "lstm"}, {1})
        for sample in samples:
            matching_jobs = [
                job
                for job in jobs
                if job["policy"] == sample["policy"]
                and job["hyperparam_sample"] == sample["hyperparam_sample"]
            ]
            self.assertEqual(len(matching_jobs), 2)
            for field in ("minibatches", "learning_rate", "entropy_coef", "update_epochs", "clip_range"):
                self.assertTrue(all(job[field] == sample[field] for job in matching_jobs))

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
        self.assertEqual(len(manifest["sampled_hyperparameters"]), 20)
        self.assertEqual({row["policy"] for row in manifest["sampled_hyperparameters"]}, {"mlp", "lstm"})
        self.assertEqual(
            {row["minibatches"] for row in manifest["sampled_hyperparameters"] if row["policy"] == "lstm"},
            {1},
        )
        self.assertEqual(manifest["paper_space"]["hyperparameter_samples"], 10)
        self.assertGreater(manifest["jobs_uncapped_for_selected_space"], manifest["jobs_planned"])
        self.assertEqual(manifest["jobs_completed"], 0)
        self.assertEqual(manifest["paper_space"]["timesteps"], 10_000_000)
        self.assertEqual(manifest["paper_space"]["eval_rollouts"], 1000)
        self.assertTrue(manifest["paper_space"]["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(manifest["paper_space"]["space_spec"]["action_dimension"], 1)
        self.assertEqual(manifest["paper_space"]["space_spec"]["observation_dimension"], 4)
        self.assertEqual(manifest["paper_space"]["space_spec"]["initial_state_distribution"]["low"], -0.05)
        self.assertEqual(manifest["paper_protocol_status"]["selected_eval_rollouts"], 1)
        self.assertFalse(manifest["paper_protocol_status"]["uses_paper_eval_rollouts"])
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
        self.assertEqual(status["paper_eval_rollouts"], 1000)
        self.assertEqual(status["selected_eval_rollouts"], 1000)
        self.assertTrue(status["uses_paper_eval_rollouts"])
        self.assertEqual(status["selected_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(status["distinct_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(status["selected_seed_count"], 5)
        self.assertEqual(status["distinct_seed_count"], 5)
        self.assertEqual(status["selected_policies"], ["mlp", "lstm"])
        self.assertEqual(status["distinct_policies"], ["lstm", "mlp"])
        self.assertTrue(status["paper_seed_count"])
        self.assertTrue(status["full_baseline_policy_set"])
        self.assertEqual(status["hyperparam_mode"], "paper-random")
        self.assertTrue(status["paper_random_hyperparameter_search"])
        self.assertTrue(status["paper_random_sample_count"])
        self.assertTrue(status["requested_paper_random_sample_count"])
        self.assertTrue(status["generated_paper_random_sample_count"])
        self.assertTrue(status["sampled_hyperparameters_follow_paper_ranges"])
        self.assertTrue(status["sampled_hyperparameters_follow_paper_minibatch_rules"])
        self.assertTrue(status["sampled_learning_rate_values_within_reported_interval"])
        self.assertTrue(status["paper_random_learning_rate_values_within_reported_interval"])
        self.assertTrue(status["learning_rate_values_within_reported_interval"])
        self.assertFalse(status["grid_hyperparameter_search"])
        self.assertFalse(status["full_reported_mlp_grid"])
        self.assertTrue(status["ppo_lstm_minibatches_fixed_to_one"])
        self.assertTrue(status["paper_scale_plan"])
        self.assertFalse(status["paper_scale_execution"])

    def test_paper_protocol_status_rejects_sampled_values_outside_paper_ranges(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        def bad_configs(_args, policy):
            config = {
                "minibatches": 1,
                "learning_rate": 0.004,
                "entropy_coef": 0.02,
                "update_epochs": 37,
                "clip_range": 0.4,
            }
            return [dict(config) for _ in range(PAPER_HYPERPARAMETER_SAMPLES)]

        with patch("run_cartpole_ppo_sweep.hyperparameter_configs", side_effect=bad_configs):
            status = paper_protocol_status(args)

        self.assertFalse(status["sampled_learning_rate_values_within_reported_interval"])
        self.assertFalse(status["sampled_hyperparameters_follow_paper_ranges"])
        self.assertFalse(status["paper_random_learning_rate_values_within_reported_interval"])
        self.assertFalse(status["learning_rate_values_within_reported_interval"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_sampled_lstm_minibatch_violation(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        def bad_lstm_configs(_args, policy):
            minibatches = 2 if policy == "lstm" else 1
            return [
                {
                    "minibatches": minibatches,
                    "learning_rate": 1e-4,
                    "entropy_coef": 0.01,
                    "update_epochs": 8,
                    "clip_range": 0.2,
                }
                for _ in range(PAPER_HYPERPARAMETER_SAMPLES)
            ]

        with patch("run_cartpole_ppo_sweep.hyperparameter_configs", side_effect=bad_lstm_configs):
            status = paper_protocol_status(args)

        self.assertTrue(status["sampled_hyperparameters_follow_paper_ranges"])
        self.assertFalse(status["sampled_hyperparameters_follow_paper_minibatch_rules"])
        self.assertFalse(status["ppo_lstm_minibatches_fixed_to_one"])
        self.assertFalse(status["paper_scale_plan"])

    def test_paper_protocol_status_rejects_generated_sample_count_mismatch(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        def too_few_configs(_args, _policy):
            return [
                {
                    "minibatches": 1,
                    "learning_rate": 1e-4,
                    "entropy_coef": 0.01,
                    "update_epochs": 8,
                    "clip_range": 0.2,
                }
                for _ in range(PAPER_HYPERPARAMETER_SAMPLES - 1)
            ]

        with patch("run_cartpole_ppo_sweep.hyperparameter_configs", side_effect=too_few_configs):
            status = paper_protocol_status(args)

        self.assertTrue(status["requested_paper_random_sample_count"])
        self.assertFalse(status["generated_paper_random_sample_count"])
        self.assertFalse(status["paper_random_sample_count"])
        self.assertFalse(status["paper_scale_plan"])

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

    def test_paper_protocol_status_requires_1000_eval_rollouts(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--dry-run", "--eval-rollouts", "20"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        status = paper_protocol_status(args)

        self.assertEqual(status["paper_eval_rollouts"], 1000)
        self.assertEqual(status["selected_eval_rollouts"], 20)
        self.assertFalse(status["uses_paper_eval_rollouts"])
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

        self.assertEqual(status["selected_seeds"], [0, 0, 0, 0, 0])
        self.assertEqual(status["distinct_seeds"], [0])
        self.assertEqual(status["selected_seed_count"], 5)
        self.assertEqual(status["distinct_seed_count"], 1)
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
        self.assertEqual(status["selected_policies"], ["mlp", "lstm", "lstm"])
        self.assertEqual(status["distinct_policies"], ["lstm", "mlp"])
        self.assertFalse(status["full_baseline_policy_set"])
        self.assertFalse(status["paper_scale_plan"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for PPO sweep execution")
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
            hyperparam_summary_path = os.path.join(tmpdir, "cartpole_ppo_sweep_hyperparam_summary.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json")
            self.assertTrue(os.path.exists(results_path))
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(hyperparam_summary_path))

            with open(results_path, newline="", encoding="utf-8") as handle:
                result_rows = list(csv.DictReader(handle))
            with open(summary_path, newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with open(hyperparam_summary_path, newline="", encoding="utf-8") as handle:
                hyperparam_summary_rows = list(csv.DictReader(handle))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(result_rows[0]["hyperparam_mode"], "paper-random")
        self.assertEqual(result_rows[0]["hyperparam_sample"], "0")
        self.assertIn("test_steps", result_rows[0])
        self.assertIn("test_survival_seconds", result_rows[0])
        self.assertGreater(float(result_rows[0]["test_steps"]), 0.0)
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0]["best_job_id"], "0")
        self.assertIn("best_test_steps", summary_rows[0])
        self.assertIn("best_test_survival_seconds", summary_rows[0])
        self.assertEqual(len(hyperparam_summary_rows), 1)
        self.assertEqual(hyperparam_summary_rows[0]["hyperparam_sample"], "0")
        self.assertEqual(hyperparam_summary_rows[0]["selected_seed_count"], "5")
        self.assertEqual(hyperparam_summary_rows[0]["selected_seeds"], "0,1,2,3,4")
        self.assertEqual(hyperparam_summary_rows[0]["missing_seeds"], "1,2,3,4")
        self.assertEqual(hyperparam_summary_rows[0]["complete_seed_coverage"], "False")
        self.assertIn("test_steps_mean", hyperparam_summary_rows[0])
        self.assertIn("test_survival_seconds_mean", hyperparam_summary_rows[0])
        self.assertEqual(hyperparam_summary_rows[0]["is_best_hyperparam_for_policy"], "True")
        self.assertEqual(manifest["hyperparam_mode"], "paper-random")
        self.assertEqual(manifest["hyperparam_samples"], 10)
        self.assertEqual(manifest["jobs_completed"], 1)
        self.assertEqual(manifest["jobs_failed"], 0)
        self.assertEqual(manifest["jobs_skipped_existing"], 0)
        self.assertEqual(manifest["jobs_run_this_invocation"], 1)
        self.assertIn("selection_rule", manifest)
        self.assertIn("hyperparameter_selection_rule", manifest)
        self.assertIn("summary", manifest["artifacts"])
        self.assertIn("hyperparameter_summary", manifest["artifacts"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for PPO sweep execution")
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
            with open(
                os.path.join(tmpdir, "cartpole_ppo_sweep_hyperparam_summary.csv"),
                newline="",
                encoding="utf-8",
            ) as handle:
                hyperparam_summary = list(csv.DictReader(handle))
            with open(os.path.join(tmpdir, "cartpole_ppo_sweep_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(hyperparam_summary), 2)
        self.assertEqual(rows[0], first_rows[0])
        self.assertEqual(rows[1]["job_id"], "1")
        self.assertEqual(hyperparam_summary[0]["complete_seed_coverage"], "False")
        self.assertEqual(hyperparam_summary[0]["missing_seeds"], "1,2,3,4")
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
                "train_steps": "1.0",
                "test_steps": "1.0",
                "train_survival_seconds": "0.02",
                "test_survival_seconds": "0.02",
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
