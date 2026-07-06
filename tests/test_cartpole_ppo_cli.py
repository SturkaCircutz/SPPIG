import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_ppo.py")
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from cartpole_env import PAPER_EVAL_ROLLOUTS  # noqa: E402
import train_cartpole_ppo  # noqa: E402


PAPER_PPO_TIMESTEPS = train_cartpole_ppo.PAPER_PPO_TIMESTEPS


class FakePPOConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CartpolePPOCliTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_cli_paper_timestep_constant_matches_runtime(self):
        from ppo_cartpole import PAPER_PPO_TIMESTEPS as runtime_paper_timesteps

        self.assertEqual(PAPER_PPO_TIMESTEPS, runtime_paper_timesteps)

    def test_cli_defaults_to_paper_timestep_budget_without_running(self):
        captured = {}

        def fake_train(cfg, output=None):
            captured["cfg"] = cfg
            captured["output"] = output

            class Result:
                timesteps = cfg.total_timesteps
                train_success_rate = 0.0
                test_success_rate = 0.0
                train_reward_mean = 0.0
                test_reward_mean = 0.0

            return None, Result()

        def fake_load_ppo_runtime():
            return FakePPOConfig, fake_train

        with patch.object(sys, "argv", [SCRIPT]), patch.object(
            train_cartpole_ppo, "_load_ppo_runtime", fake_load_ppo_runtime
        ):
            train_cartpole_ppo.main()

        self.assertEqual(captured["cfg"].total_timesteps, PAPER_PPO_TIMESTEPS)
        self.assertEqual(captured["cfg"].eval_rollouts, PAPER_EVAL_ROLLOUTS)
        self.assertEqual(captured["cfg"].device, "auto")
        self.assertEqual(captured["output"], "artifacts/cartpole_ppo.pt")

    def test_cli_passes_requested_device_without_running(self):
        captured = {}

        def fake_train(cfg, output=None):
            captured["cfg"] = cfg
            captured["output"] = output

            class Result:
                timesteps = cfg.total_timesteps
                train_success_rate = 0.0
                test_success_rate = 0.0
                train_reward_mean = 0.0
                test_reward_mean = 0.0

            return None, Result()

        def fake_load_ppo_runtime():
            return FakePPOConfig, fake_train

        with patch.object(sys, "argv", [SCRIPT, "--device", "cuda:0"]), patch.object(
            train_cartpole_ppo, "_load_ppo_runtime", fake_load_ppo_runtime
        ):
            train_cartpole_ppo.main()

        self.assertEqual(captured["cfg"].device, "cuda:0")

    def test_cli_help_does_not_require_torch(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertIn("--timesteps", result.stdout)
        self.assertIn("--policy", result.stdout)
        self.assertIn("--device", result.stdout)

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
                    "--device",
                    "cpu",
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
        self.assertEqual(metrics["config"]["device"], "cpu")
        self.assertEqual(metrics["torch_device"]["selected"], "cpu")
        self.assertGreaterEqual(len(metrics["eval_history"]), 1)
        self.assertEqual(len(metrics["update_history"]), 2)
        self.assertEqual(metrics["update_history"][0]["rollout_steps"], 32)
        self.assertEqual(metrics["update_history"][0]["optimizer_minibatch_updates"], 1)
        for field in (
            "policy_loss_mean",
            "value_loss_mean",
            "entropy_mean",
            "loss_mean",
            "approx_kl_mean",
            "clip_fraction_mean",
        ):
            self.assertIsInstance(metrics["update_history"][0][field], float)
        self.assertIn("selected_result", metrics)
        self.assertIn("train_steps_mean", metrics["selected_result"])
        self.assertIn("test_steps_mean", metrics["selected_result"])
        self.assertIn("train_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("test_survival_seconds_mean", metrics["selected_result"])
        self.assertIn("test_steps_mean", metrics["eval_history"][0])
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["paper_protocol_status"]["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["paper_protocol_status"]["paper_eval_rollouts"], 1000)
        self.assertEqual(metrics["paper_protocol_status"]["selected_eval_rollouts"], 1)
        self.assertFalse(metrics["paper_protocol_status"]["uses_paper_eval_rollouts"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_timestep_budget"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_test_horizon"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_scale_baseline_protocol"])


if __name__ == "__main__":
    unittest.main()
