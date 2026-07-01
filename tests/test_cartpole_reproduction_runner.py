import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "run_cartpole_reproduction.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_cartpole_reproduction import summarize_rows  # noqa: E402


class CartpoleReproductionRunnerTest(unittest.TestCase):
    def test_summary_rows_report_mean_std_and_best_train_seed(self):
        summary = summarize_rows(
            [
                {
                    "policy": "Programmatic state machine",
                    "seed": 1,
                    "train_success": 0.5,
                    "test_success": 0.25,
                    "train_reward": 100.0,
                    "test_reward": 200.0,
                    "timesteps": 0,
                },
                {
                    "policy": "Programmatic state machine",
                    "seed": 0,
                    "train_success": 1.0,
                    "test_success": 0.75,
                    "train_reward": 250.0,
                    "test_reward": 900.0,
                    "timesteps": 0,
                },
            ]
        )

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["policy"], "Programmatic state machine")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["train_success_mean"], 0.75)
        self.assertAlmostEqual(row["train_success_std"], 0.3535533905932738)
        self.assertAlmostEqual(row["test_reward_mean"], 550.0)
        self.assertEqual(row["best_seed_by_train"], 0)
        self.assertAlmostEqual(row["best_test_success"], 0.75)

    def test_quick_runner_writes_results_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--seeds",
                    "0",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            csv_path = os.path.join(tmpdir, "cartpole_results.csv")
            summary_path = os.path.join(tmpdir, "cartpole_summary.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_manifest.json")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(manifest_path))

            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["policy"], "Programmatic state machine")
            self.assertEqual(rows[0]["seed"], "0")

            with open(summary_path, newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["policy"], "Programmatic state machine")
            self.assertEqual(summary[0]["n"], "1")
            self.assertEqual(summary[0]["best_seed_by_train"], "0")
            self.assertEqual(summary[0]["train_success_std"], "0.0")

            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["quick"])
            self.assertEqual(manifest["seeds"], [0])
            self.assertEqual(manifest["test_max_steps"], 20)
            self.assertIn("rows", manifest)
            self.assertIn("summary", manifest)
            self.assertIn("summary_note", manifest)


if __name__ == "__main__":
    unittest.main()
