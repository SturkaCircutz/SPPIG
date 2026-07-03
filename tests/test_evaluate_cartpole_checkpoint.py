import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "evaluate_cartpole_checkpoint.py")

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class EvaluateCartpoleCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_script_writes_full_horizon_checkpoint_metrics(self):
        checkpoint_path = os.path.join(ROOT, "artifacts", "progress_mlp_128k_seed0.pt")
        if not os.path.exists(checkpoint_path):
            self.skipTest("local PPO checkpoint artifact is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "checkpoint_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--checkpoint",
                    checkpoint_path,
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

        self.assertEqual(metrics["checkpoint"], checkpoint_path)
        self.assertEqual(metrics["checkpoint_config"]["policy_type"], "mlp")
        self.assertEqual(metrics["eval_rollouts"], 2)
        self.assertEqual(metrics["paper_eval_rollouts"], 1000)
        self.assertFalse(metrics["uses_paper_eval_rollouts"])
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["space_spec"]["initial_state_distribution"]["high"], 0.05)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("selected_result", metrics)
        self.assertIn("train_steps_mean", metrics["selected_result"])
        self.assertIn("test_steps_mean", metrics["selected_result"])
        self.assertIn("train_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("test_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("command", metrics)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_script_loads_lstm_checkpoint(self):
        checkpoint_path = os.path.join(ROOT, "artifacts", "debug_lstm_pretrain_ppo_64k_lr1e5.pt")
        if not os.path.exists(checkpoint_path):
            self.skipTest("local PPO-LSTM checkpoint artifact is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "checkpoint_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--checkpoint",
                    checkpoint_path,
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

        self.assertEqual(metrics["checkpoint_config"]["policy_type"], "lstm")
        self.assertIn("selected_result", metrics)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_mlp_checkpoint_matches_checked_in_full_horizon_row(self):
        checkpoint_path = os.path.join(ROOT, "artifacts", "progress_mlp_128k_seed0.pt")
        if not os.path.exists(checkpoint_path):
            self.skipTest("local PPO checkpoint artifact is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "checkpoint_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--checkpoint",
                    checkpoint_path,
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

        self.assertEqual(metrics["selected_result"]["train_success_rate"], 1.0)
        self.assertEqual(metrics["selected_result"]["test_success_rate"], 0.0)
        self.assertEqual(metrics["selected_result"]["train_reward_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_reward_mean"], 910.6)
        self.assertEqual(metrics["selected_result"]["train_steps_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_steps_mean"], 910.6)
        self.assertEqual(metrics["selected_result"]["train_survival_seconds_mean"], 5.0)
        self.assertAlmostEqual(metrics["selected_result"]["test_survival_seconds_mean"], 18.212)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_lstm_warm_start_checkpoint_matches_checked_in_full_horizon_row(self):
        checkpoint_path = os.path.join(ROOT, "artifacts", "debug_lstm_pretrain_ppo_64k_lr1e5.pt")
        if not os.path.exists(checkpoint_path):
            self.skipTest("local PPO-LSTM checkpoint artifact is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "checkpoint_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--checkpoint",
                    checkpoint_path,
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

        self.assertEqual(metrics["selected_result"]["train_success_rate"], 1.0)
        self.assertEqual(metrics["selected_result"]["test_success_rate"], 0.0)
        self.assertEqual(metrics["selected_result"]["train_reward_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_reward_mean"], 912.25)
        self.assertEqual(metrics["selected_result"]["train_steps_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_steps_mean"], 912.25)
        self.assertEqual(metrics["selected_result"]["train_survival_seconds_mean"], 5.0)
        self.assertAlmostEqual(metrics["selected_result"]["test_survival_seconds_mean"], 18.245)


if __name__ == "__main__":
    unittest.main()
