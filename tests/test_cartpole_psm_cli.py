import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_psm.py")


class CartpolePSMCliTest(unittest.TestCase):
    def test_cli_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "psm_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--num-initial-states",
                    "2",
                    "--candidate-rollouts",
                    "2",
                    "--segment-steps",
                    "2",
                    "--segments-per-trace",
                    "4",
                    "--eval-rollouts",
                    "1",
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

        self.assertEqual(metrics["config"]["num_initial_states"], 2)
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("policy_description", metrics)
        self.assertEqual(metrics["trace_summary"]["count"], metrics["num_traces"])
        self.assertGreaterEqual(metrics["trace_summary"]["reward_mean"], 0.0)
        self.assertLessEqual(len(metrics["trace_summary"]["examples"]), 3)
        self.assertIn("mode_prefix", metrics["trace_summary"]["examples"][0])
        self.assertIn("theta_gain", metrics["trace_summary"]["examples"][0])
        self.assertIn("segment_durations", metrics["trace_summary"]["examples"][0])
        self.assertIn("probabilistic_student", metrics)
        self.assertIn("action_distributions", metrics["probabilistic_student"])
        self.assertIn("switch_parameter_distributions", metrics["probabilistic_student"])
        self.assertGreaterEqual(metrics["probabilistic_student"]["responsibility_summary"]["segments"], 1)
        self.assertIn("success_rate", metrics["train"])
        self.assertIn("reward_mean", metrics["test"])


if __name__ == "__main__":
    unittest.main()
