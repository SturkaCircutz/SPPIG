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

from run_cartpole_ppo_sweep import PAPER_NMINIBATCHES, build_jobs, parse_args, summarize_results  # noqa: E402


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
        self.assertTrue(manifest["dry_run"])
        self.assertTrue(manifest["quick"])
        self.assertEqual(manifest["jobs_planned"], 2)
        self.assertEqual(manifest["jobs_completed"], 0)
        self.assertEqual(manifest["paper_space"]["timesteps"], 10_000_000)

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

            with open(summary_path, newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0]["best_job_id"], "0")
        self.assertEqual(manifest["jobs_completed"], 1)
        self.assertIn("selection_rule", manifest)
        self.assertIn("summary", manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
