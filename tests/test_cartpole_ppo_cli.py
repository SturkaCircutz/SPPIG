import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_ppo.py")

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class CartpolePPOCliTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_cli_writes_checkpoint_and_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "cartpole_ppo.pt")
            metrics_path = os.path.join(tmpdir, "metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--policy",
                    "mlp",
                    "--timesteps",
                    "64",
                    "--rollout-steps",
                    "32",
                    "--num-envs",
                    "1",
                    "--update-epochs",
                    "1",
                    "--minibatches",
                    "1",
                    "--hidden-size",
                    "8",
                    "--eval-interval",
                    "32",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--output",
                    checkpoint_path,
                    "--metrics-output",
                    metrics_path,
                ],
                check=True,
                cwd=ROOT,
            )

            self.assertTrue(os.path.exists(checkpoint_path))
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertEqual(metrics["config"]["hidden_size"], 8)
        self.assertGreaterEqual(len(metrics["eval_history"]), 1)
        self.assertEqual(len(metrics["update_history"]), 2)
        self.assertEqual(metrics["update_history"][0]["rollout_steps"], 32)
        self.assertIn("selected_result", metrics)
        self.assertIn("train_steps_mean", metrics["selected_result"])
        self.assertIn("test_steps_mean", metrics["selected_result"])
        self.assertIn("train_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("test_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("test_steps_mean", metrics["eval_history"][0])


if __name__ == "__main__":
    unittest.main()
