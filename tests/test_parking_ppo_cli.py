import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_parking_ppo.py")
sys.path.insert(0, os.path.join(ROOT, "src"))


class ParkingPpoCliTest(unittest.TestCase):
    def test_cli_writes_two_action_ppo_metrics_and_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            traces_path = os.path.join(tmpdir, "traces.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--train-n",
                    "2",
                    "--test-n",
                    "2",
                    "--updates",
                    "1",
                    "--rollouts-per-update",
                    "2",
                    "--epochs",
                    "1",
                    "--seed",
                    "0",
                    "--outdir",
                    tmpdir,
                    "--metrics-output",
                    metrics_path,
                    "--traces-output",
                    traces_path,
                    "--verify",
                ],
                check=True,
                cwd=ROOT,
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            with open(traces_path, encoding="utf-8") as handle:
                traces = json.load(handle)

        self.assertEqual(metrics["artifact_kind"], "parking_ppo_training_metrics")
        self.assertEqual(traces["artifact_kind"], "parking_ppo_training_traces")
        self.assertEqual(metrics["action_spec"]["components"], ["velocity", "steering"])
        self.assertEqual(metrics["action_spec"]["dimension"], 2)
        self.assertEqual(traces["action_spec"]["dimension"], 2)
        self.assertEqual(metrics["algorithm"]["name"], "numpy_linear_ppo")
        self.assertEqual(metrics["train_action_lengths"], [2])
        self.assertEqual(metrics["test_action_lengths"], [2])
        self.assertTrue(metrics["history"])
        self.assertIn("policy_parameters", metrics)
        self.assertEqual(len(traces["train_tasks"]), 2)
        self.assertEqual(len(traces["test_tasks"]), 2)
        self.assertTrue(traces["ppo_test_traces"])

        action_lengths = {
            len(action)
            for trace in traces["ppo_test_traces"]
            for action in trace["actions"]
        }
        self.assertEqual(action_lengths, {2})

        metrics_text = json.dumps(metrics).lower()
        self.assertNotIn("lateral_rate", metrics_text)
        self.assertNotIn("center_lateral", metrics_text)

    def test_parking_ppo_module_exports_runner_helpers(self):
        import train_parking_ppo

        self.assertTrue(callable(train_parking_ppo.main))
        self.assertTrue(callable(train_parking_ppo.run_experiment))
        self.assertTrue(callable(train_parking_ppo.verify_metrics))


if __name__ == "__main__":
    unittest.main()
