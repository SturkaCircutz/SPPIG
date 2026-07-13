import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_parking_psm.py")
sys.path.insert(0, os.path.join(ROOT, "src"))


class ParkingTrainingCliTest(unittest.TestCase):
    def test_cli_writes_parking_metrics_and_traces(self):
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
                    "--teacher-iters",
                    "1",
                    "--outer-iters",
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

        self.assertEqual(metrics["artifact_kind"], "parking_psm_training_metrics")
        self.assertEqual(traces["artifact_kind"], "parking_psm_training_traces")
        self.assertEqual(metrics["config"]["train_n"], 2)
        self.assertEqual(metrics["config"]["test_n"], 2)
        self.assertEqual(metrics["student_fit"]["method"], "em_style_segment_assignment")
        self.assertIn("front_threshold", metrics["learned_thresholds"])
        self.assertIn("center_lateral_gain", metrics["learned_thresholds"])
        self.assertGreater(metrics["student_train"]["success_rate"], 0.0)
        self.assertGreater(
            metrics["student_test"]["success_rate"],
            metrics["baseline_test"]["success_rate"],
        )
        self.assertEqual(len(traces["train_tasks"]), 2)
        self.assertEqual(len(traces["test_tasks"]), 2)
        self.assertEqual(len(traces["teacher_traces"]), 2)
        self.assertTrue(traces["student_test_traces"])
        self.assertIn("slot_length", traces["train_tasks"][0])
        self.assertIn("actions", traces["teacher_traces"][0])

        metrics_text = json.dumps(metrics).lower()
        self.assertNotIn("pole_length", metrics_text)
        self.assertNotIn("classic_control", metrics_text)

    def test_parking_training_module_exports_runner_helpers(self):
        import train_parking_psm

        self.assertTrue(callable(train_parking_psm.main))
        self.assertTrue(callable(train_parking_psm.run_experiment))
        self.assertTrue(callable(train_parking_psm.verify_metrics))


if __name__ == "__main__":
    unittest.main()
