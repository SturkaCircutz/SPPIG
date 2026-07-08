import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import verify_essay_scale_reproduction as verifier  # noqa: E402


class VerifyEssayScaleReproductionTest(unittest.TestCase):
    def test_checked_in_essay_scale_bundle_verifies(self):
        verifier.verify()

    def test_essay_artifact_check_rejects_stale_synthesized_psm_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            essay_dir = Path(tmpdir)
            (essay_dir / "figures").mkdir()
            for name in [
                "cartpole_success_rates.png",
                "cartpole_test_survival_reward.png",
                "programmatic_switch_boundary.png",
                "cartpole_ppo_training_curves.png",
            ]:
                (essay_dir / "figures" / name).write_bytes(b"png")
            (essay_dir / "project.tex").write_text(
                "\n".join(
                    [
                        r"\input{cartpole_abstract_results.tex}",
                        r"\input{cartpole_results_table.tex}",
                        r"\input{cartpole_ppo_sweep_fragment.tex}",
                        r"\input{cartpole_policy_fragment.tex}",
                        r"\input{cartpole_figure19_reference_fragment.tex}",
                    ]
                ),
                encoding="utf-8",
            )
            (essay_dir / "cartpole_abstract_results.tex").write_text(
                "not a paper-scale reproduction\n",
                encoding="utf-8",
            )
            (essay_dir / "cartpole_results_table.tex").write_text(
                "not a paper-scale reproduction\n"
                r"Synthesized PSM diagnostic & 0.00 & 0.00 & 28.4 & 41.6 \\" "\n",
                encoding="utf-8",
            )
            (essay_dir / "cartpole_ppo_sweep_fragment.tex").write_text(
                "not a paper-scale reproduction\n"
                "Medium partial PPO/PPO-LSTM sweep summary. Completed jobs: 4/4. "
                "Paper-scale plan: false; paper-scale execution: false.\n"
                "1,000,000 timesteps per job\n",
                encoding="utf-8",
            )
            (essay_dir / "cartpole_policy_fragment.tex").write_text("policy\n", encoding="utf-8")
            (essay_dir / "cartpole_figure19_reference_fragment.tex").write_text(
                "figure19\n",
                encoding="utf-8",
            )
            (essay_dir / "00README.json").write_text(
                """{
  "sources": [
    {"filename": "project.tex"},
    {"filename": "cartpole_abstract_results.tex"},
    {"filename": "cartpole_results_table.tex"},
    {"filename": "cartpole_ppo_sweep_fragment.tex"},
    {"filename": "cartpole_policy_fragment.tex"},
    {"filename": "cartpole_figure19_reference_fragment.tex"},
    {"filename": "figures/cartpole_success_rates.png"},
    {"filename": "figures/cartpole_test_survival_reward.png"},
    {"filename": "figures/programmatic_switch_boundary.png"},
    {"filename": "figures/cartpole_ppo_training_curves.png"}
  ]
}
""",
                encoding="utf-8",
            )

            with patch.object(verifier, "ESSAY_DIR", essay_dir):
                with self.assertRaisesRegex(AssertionError, "stale synthesized PSM"):
                    verifier.verify_essay_artifacts()

    def test_result_bundle_check_rejects_psm_command_without_teacher_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            metrics_dir = results_dir / "metrics"
            metrics_dir.mkdir()
            metrics_path = metrics_dir / "psm.json"
            metrics_path.write_text(
                """{
  "command": "src/train_cartpole_psm.py --num-initial-states 10",
  "paper_protocol_status": {
    "paper_scale_result": false,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  },
  "train": {"reward_mean": 48.05},
  "test": {"reward_mean": 59.85}
}
""",
                encoding="utf-8",
            )
            with (results_dir / "cartpole_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "policy",
                        "train_reward_mean",
                        "test_reward_mean",
                        "best_command",
                        "best_metrics_output",
                        "test_horizon_steps",
                        "eval_rollouts",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "policy": "Synthesized PSM diagnostic",
                        "train_reward_mean": "48.05",
                        "test_reward_mean": "59.85",
                        "best_command": "src/train_cartpole_psm.py --num-initial-states 10",
                        "best_metrics_output": str(metrics_path),
                        "test_horizon_steps": "15000",
                        "eval_rollouts": "20",
                    }
                )
            (results_dir / "cartpole_manifest.json").write_text(
                """{
  "paper_scale_result": false,
  "local_diagnostic_only": true,
  "row_count": 1,
  "paper_protocol_status": {
    "paper_scale_result": false,
    "uses_full_test_horizon": true
  }
}
""",
                encoding="utf-8",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "ten teacher trace workers"):
                    verifier.verify_result_bundle()

    def write_minimal_result_bundle(
        self,
        tmpdir: str,
        metrics_status: str,
        train_reward: str = "48.05",
        manifest_extra: str = "",
    ) -> Path:
        results_dir = Path(tmpdir)
        metrics_dir = results_dir / "metrics"
        metrics_dir.mkdir()
        metrics_path = metrics_dir / "psm.json"
        metrics_path.write_text(
            f"""{{
  "command": "src/train_cartpole_psm.py --parallel-trace-workers 10",
  "paper_protocol_status": {metrics_status},
  "train": {{
    "success_rate": 0.0,
    "reward_mean": {train_reward},
    "steps_mean": {train_reward},
    "survival_seconds_mean": 0.961
  }},
  "test": {{
    "success_rate": 0.0,
    "reward_mean": 59.85,
    "steps_mean": 59.85,
    "survival_seconds_mean": 1.197
  }}
}}
""",
            encoding="utf-8",
        )
        with (results_dir / "cartpole_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "policy",
                    "train_reward_mean",
                    "test_reward_mean",
                    "best_command",
                    "best_metrics_output",
                    "test_horizon_steps",
                    "eval_rollouts",
                    "best_train_success",
                    "best_test_success",
                    "best_train_reward",
                    "best_test_reward",
                    "best_train_steps",
                    "best_test_steps",
                    "best_train_survival_seconds",
                    "best_test_survival_seconds",
                    "best_timesteps",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "policy": "Synthesized PSM diagnostic",
                    "train_reward_mean": "48.05",
                    "test_reward_mean": "59.85",
                    "best_command": "src/train_cartpole_psm.py --parallel-trace-workers 10",
                    "best_metrics_output": str(metrics_path),
                    "test_horizon_steps": "15000",
                    "eval_rollouts": "20",
                    "best_train_success": "0.0",
                    "best_test_success": "0.0",
                    "best_train_reward": "48.05",
                    "best_test_reward": "59.85",
                    "best_train_steps": "48.05",
                    "best_test_steps": "59.85",
                    "best_train_survival_seconds": "0.961",
                    "best_test_survival_seconds": "1.197",
                    "best_timesteps": "0",
                }
            )
        (results_dir / "cartpole_manifest.json").write_text(
            f"""{{
  "paper_scale_result": false,
  "local_diagnostic_only": true,
  "train_horizon_seconds": 5.0,
  "train_pole_length": 0.5,
  "test_horizon_seconds": 300.0,
  "test_pole_length": 1.0,
  "row_count": 1,
  "paper_protocol_status": {{
    "paper_scale_result": false,
    "uses_full_test_horizon": true
  }}{manifest_extra}
}}
""",
            encoding="utf-8",
        )
        return results_dir

    def test_result_bundle_check_rejects_nested_paper_scale_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": true,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  }""",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "paper_scale_result"):
                    verifier.verify_result_bundle()

    def test_result_bundle_check_rejects_paper_scale_execution_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": false,
    "paper_scale_execution": true,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  }""",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "paper_scale_execution"):
                    verifier.verify_result_bundle()

    def test_result_bundle_check_rejects_paper_eval_rollout_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": false,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true,
    "uses_paper_eval_rollouts": true
  }""",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "paper eval rollouts"):
                    verifier.verify_result_bundle()

    def test_result_bundle_check_rejects_metric_value_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": false,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  }""",
                train_reward="47.0",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "train reward"):
                    verifier.verify_result_bundle()

    def test_result_bundle_check_rejects_cartpole_split_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": false,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 6.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  }""",
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "train horizon seconds"):
                    verifier.verify_result_bundle()

    def test_result_bundle_check_rejects_top_level_manifest_split_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = self.write_minimal_result_bundle(
                tmpdir,
                """{
    "paper_scale_result": false,
    "full_probabilistic_adaptive_teaching": false,
    "train_horizon_seconds": 5.0,
    "train_pole_length": 0.5,
    "test_horizon_seconds": 300.0,
    "test_pole_length": 1.0,
    "uses_full_test_horizon": true
  }""",
                manifest_extra=',\n  "test_pole_length": 0.5',
            )

            with patch.object(verifier, "RESULTS_DIR", results_dir):
                with self.assertRaisesRegex(AssertionError, "test pole length"):
                    verifier.verify_result_bundle()


if __name__ == "__main__":
    unittest.main()
