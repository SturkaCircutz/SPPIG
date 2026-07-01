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

from run_cartpole_ppo_sweep import PAPER_NMINIBATCHES, build_jobs, parse_args  # noqa: E402


class CartpolePPOSweepTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
