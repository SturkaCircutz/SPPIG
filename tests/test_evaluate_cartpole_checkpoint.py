import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SCRIPT = os.path.join(ROOT, "scripts", "evaluate_cartpole_checkpoint.py")

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


class EvaluateCartpoleCheckpointTest(unittest.TestCase):
    def test_checkpoint_reevaluation_protocol_status_distinguishes_checkpoint_from_reeval(self):
        from evaluate_cartpole_checkpoint import checkpoint_reevaluation_protocol_status

        status = checkpoint_reevaluation_protocol_status(
            {
                "policy_type": "mlp",
                "total_timesteps": 131072,
                "eval_test_max_steps": 1000,
            },
            eval_rollouts=20,
            test_max_steps=15000,
        )

        self.assertEqual(status["artifact_kind"], "ppo_checkpoint_reevaluation")
        self.assertEqual(status["policy_type"], "mlp")
        self.assertEqual(status["paper_timestep_budget"], 10_000_000)
        self.assertFalse(status["checkpoint_uses_paper_timestep_budget"])
        self.assertEqual(status["checkpoint_eval_test_max_steps"], 1000)
        self.assertFalse(status["checkpoint_eval_used_full_test_horizon"])
        self.assertTrue(status["reevaluation_uses_full_test_horizon"])
        self.assertEqual(status["selected_eval_rollouts"], 20)
        self.assertFalse(status["uses_paper_eval_rollouts"])
        self.assertEqual(status["checkpoint_pretrain_steps"], 0)
        self.assertIsNone(status["checkpoint_pretrain_teacher_policy"])
        self.assertEqual(status["checkpoint_pretrain_teacher_policy_status"], "not_applicable_no_pretraining")
        self.assertTrue(status["checkpoint_pretrain_teacher_policy_recorded"])
        self.assertTrue(status["checkpoint_pretrain_teacher_policy_matches_current_implementation"])
        self.assertEqual(status["checkpoint_pretrain_teacher_mode_order_status"], "not_applicable_no_pretraining")
        self.assertTrue(status["checkpoint_pretrain_teacher_mode_order_recorded"])
        self.assertTrue(status["checkpoint_pretrain_teacher_mode_order_matches_current_implementation"])
        self.assertFalse(status["paper_scale_checkpoint_result"])

    def test_checkpoint_status_flags_missing_warm_start_teacher_order(self):
        from evaluate_cartpole_checkpoint import checkpoint_reevaluation_protocol_status

        status = checkpoint_reevaluation_protocol_status(
            {
                "policy_type": "lstm",
                "total_timesteps": 65536,
                "eval_test_max_steps": 1000,
                "pretrain_steps": 500,
            },
            eval_rollouts=20,
            test_max_steps=15000,
        )

        self.assertEqual(status["checkpoint_pretrain_steps"], 500)
        self.assertEqual(status["current_pretrain_teacher_policy"], "BangBangCartpolePSM")
        self.assertIsNone(status["checkpoint_pretrain_teacher_policy"])
        self.assertEqual(
            status["checkpoint_pretrain_teacher_policy_status"],
            "missing_from_checkpoint_config",
        )
        self.assertFalse(status["checkpoint_pretrain_teacher_policy_recorded"])
        self.assertFalse(status["checkpoint_pretrain_teacher_policy_matches_current_implementation"])
        self.assertEqual(
            status["current_pretrain_teacher_mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertIsNone(status["checkpoint_pretrain_teacher_mode_update_order"])
        self.assertEqual(
            status["checkpoint_pretrain_teacher_mode_order_status"],
            "missing_from_checkpoint_config",
        )
        self.assertFalse(status["checkpoint_pretrain_teacher_mode_order_recorded"])
        self.assertFalse(status["checkpoint_pretrain_teacher_mode_order_matches_current_implementation"])

    def test_checkpoint_status_accepts_recorded_warm_start_teacher_order(self):
        from evaluate_cartpole_checkpoint import checkpoint_reevaluation_protocol_status

        status = checkpoint_reevaluation_protocol_status(
            {
                "policy_type": "lstm",
                "total_timesteps": 65536,
                "eval_test_max_steps": 15000,
                "pretrain_steps": 500,
                "pretrain_teacher_policy": "BangBangCartpolePSM",
                "pretrain_teacher_mode_update_order": "act_with_current_mode_then_update_next_mode",
            },
            eval_rollouts=1000,
            test_max_steps=15000,
        )

        self.assertEqual(status["checkpoint_pretrain_teacher_policy_status"], "recorded_matches_current_implementation")
        self.assertEqual(status["checkpoint_pretrain_teacher_policy"], "BangBangCartpolePSM")
        self.assertTrue(status["checkpoint_pretrain_teacher_policy_recorded"])
        self.assertTrue(status["checkpoint_pretrain_teacher_policy_matches_current_implementation"])
        self.assertEqual(status["checkpoint_pretrain_teacher_mode_order_status"], "recorded_matches_current_implementation")
        self.assertEqual(
            status["checkpoint_pretrain_teacher_mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertTrue(status["checkpoint_pretrain_teacher_mode_order_recorded"])
        self.assertTrue(status["checkpoint_pretrain_teacher_mode_order_matches_current_implementation"])

    def test_checkpoint_status_flags_recorded_warm_start_teacher_mismatch(self):
        from evaluate_cartpole_checkpoint import checkpoint_reevaluation_protocol_status

        status = checkpoint_reevaluation_protocol_status(
            {
                "policy_type": "lstm",
                "total_timesteps": 65536,
                "eval_test_max_steps": 15000,
                "pretrain_steps": 500,
                "pretrain_teacher_policy": "OtherTeacher",
                "pretrain_teacher_mode_update_order": "update_mode_before_acting",
            },
            eval_rollouts=1000,
            test_max_steps=15000,
        )

        self.assertEqual(
            status["checkpoint_pretrain_teacher_policy_status"],
            "recorded_mismatch_current_implementation",
        )
        self.assertTrue(status["checkpoint_pretrain_teacher_policy_recorded"])
        self.assertFalse(status["checkpoint_pretrain_teacher_policy_matches_current_implementation"])
        self.assertEqual(
            status["checkpoint_pretrain_teacher_mode_order_status"],
            "recorded_mismatch_current_implementation",
        )
        self.assertTrue(status["checkpoint_pretrain_teacher_mode_order_recorded"])
        self.assertFalse(status["checkpoint_pretrain_teacher_mode_order_matches_current_implementation"])

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
        status = metrics["paper_protocol_status"]
        self.assertEqual(status["artifact_kind"], "ppo_checkpoint_reevaluation")
        self.assertEqual(status["policy_type"], "mlp")
        self.assertFalse(status["checkpoint_uses_paper_timestep_budget"])
        self.assertFalse(status["reevaluation_uses_full_test_horizon"])
        self.assertFalse(status["uses_paper_eval_rollouts"])
        self.assertFalse(status["paper_scale_checkpoint_result"])
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
        self.assertEqual(
            metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_mode_order_status"],
            "missing_from_checkpoint_config",
        )
        self.assertEqual(
            metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_policy_status"],
            "missing_from_checkpoint_config",
        )
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
        self.assertTrue(metrics["paper_protocol_status"]["reevaluation_uses_full_test_horizon"])
        self.assertFalse(metrics["paper_protocol_status"]["uses_paper_eval_rollouts"])
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
        self.assertTrue(metrics["paper_protocol_status"]["reevaluation_uses_full_test_horizon"])
        self.assertFalse(metrics["paper_protocol_status"]["checkpoint_uses_paper_timestep_budget"])
        self.assertEqual(metrics["paper_protocol_status"]["checkpoint_pretrain_steps"], 500)
        self.assertFalse(metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_policy_recorded"])
        self.assertFalse(metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_mode_order_recorded"])
        self.assertEqual(
            metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_policy_status"],
            "missing_from_checkpoint_config",
        )
        self.assertEqual(
            metrics["paper_protocol_status"]["checkpoint_pretrain_teacher_mode_order_status"],
            "missing_from_checkpoint_config",
        )
        self.assertEqual(metrics["selected_result"]["test_success_rate"], 0.0)
        self.assertEqual(metrics["selected_result"]["train_reward_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_reward_mean"], 912.25)
        self.assertEqual(metrics["selected_result"]["train_steps_mean"], 250.0)
        self.assertEqual(metrics["selected_result"]["test_steps_mean"], 912.25)
        self.assertEqual(metrics["selected_result"]["train_survival_seconds_mean"], 5.0)
        self.assertAlmostEqual(metrics["selected_result"]["test_survival_seconds_mean"], 18.245)


if __name__ == "__main__":
    unittest.main()
