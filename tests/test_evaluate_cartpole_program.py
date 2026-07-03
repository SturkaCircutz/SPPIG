import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "evaluate_cartpole_program.py")


class EvaluateCartpoleProgramTest(unittest.TestCase):
    def test_script_writes_fixed_program_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "program_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--theta-weight",
                    "10",
                    "--omega-weight",
                    "1",
                    "--eval-rollouts",
                    "2",
                    "--test-max-steps",
                    "20",
                    "--metrics-output",
                    metrics_path,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertIn("mode=1 if 10.000*theta + 1.000*omega >= 0.000", metrics["policy_description"])
        self.assertEqual(metrics["program_parameters"]["theta_weight"], 10.0)
        self.assertEqual(metrics["program_parameters"]["omega_weight"], 1.0)
        self.assertEqual(metrics["eval_rollouts"], 2)
        self.assertEqual(metrics["paper_eval_rollouts"], 1000)
        self.assertFalse(metrics["uses_paper_eval_rollouts"])
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["space_spec"]["initial_state_distribution"]["low"], -0.05)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("train", metrics)
        self.assertIn("test", metrics)
        self.assertIn("steps_mean", metrics["train"])
        self.assertIn("survival_seconds_mean", metrics["train"])
        self.assertIn("steps_mean", metrics["test"])
        self.assertIn("survival_seconds_mean", metrics["test"])

    def test_fixed_program_matches_checked_in_result_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "program_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--theta-weight",
                    "10",
                    "--omega-weight",
                    "1",
                    "--eval-rollouts",
                    "20",
                    "--test-max-steps",
                    "15000",
                    "--metrics-output",
                    metrics_path,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertEqual(metrics["train"]["success_rate"], 1.0)
        self.assertEqual(metrics["test"]["success_rate"], 0.2)
        self.assertEqual(metrics["train"]["reward_mean"], 250.0)
        self.assertAlmostEqual(metrics["test"]["reward_mean"], 6275.35)
        self.assertEqual(metrics["train"]["steps_mean"], 250.0)
        self.assertAlmostEqual(metrics["test"]["steps_mean"], 6275.35)
        self.assertEqual(metrics["train"]["survival_seconds_mean"], 5.0)
        self.assertAlmostEqual(metrics["test"]["survival_seconds_mean"], 125.507)


if __name__ == "__main__":
    unittest.main()
