import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evaluate_cartpole_ppo_sweep_checkpoints as sweep_eval  # noqa: E402


class EvaluateCartpolePPOSweepCheckpointsTest(unittest.TestCase):
    def write_results_csv(self, tmpdir: str) -> tuple[Path, Path]:
        checkpoint_path = Path(tmpdir) / "model.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        results_path = Path(tmpdir) / "results.csv"
        with results_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "job_id",
                    "policy",
                    "seed",
                    "output",
                    "eval_rollouts",
                    "test_max_steps",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "job_id": "0",
                    "policy": "mlp",
                    "seed": "3",
                    "output": str(checkpoint_path),
                    "eval_rollouts": "7",
                    "test_max_steps": "15000",
                }
            )
        return results_path, checkpoint_path

    def fake_metrics(self, checkpoint_path: Path, eval_rollouts: int, test_max_steps: int, command: str):
        return {
            "command": command,
            "checkpoint": str(checkpoint_path),
            "checkpoint_config": {
                "policy_type": "mlp",
                "total_timesteps": 1_000_000,
            },
            "checkpoint_result": {"timesteps": 1_000_000},
            "eval_rollouts": eval_rollouts,
            "test_max_steps": test_max_steps,
            "paper_protocol_status": {
                "checkpoint_training_reused": True,
                "training_launched": False,
                "checkpoint_total_timesteps": 1_000_000,
                "checkpoint_uses_paper_timestep_budget": False,
                "reevaluation_uses_full_test_horizon": test_max_steps == 15000,
                "uses_paper_eval_rollouts": eval_rollouts == 1000,
                "paper_scale_checkpoint_result": False,
            },
            "selected_result": {
                "train_success_rate": 0.25,
                "test_success_rate": 0.5,
                "train_reward_mean": 100.0,
                "test_reward_mean": 200.0,
                "train_steps_mean": 100.0,
                "test_steps_mean": 200.0,
                "train_survival_seconds_mean": 2.0,
                "test_survival_seconds_mean": 4.0,
            },
        }

    def test_batch_reevaluation_uses_checkpoint_rows_without_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path, checkpoint_path = self.write_results_csv(tmpdir)
            outdir = Path(tmpdir) / "reeval"
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate, patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                return_value={"policy_type": "mlp", "seed": 3},
            ):
                summary_rows = sweep_eval.reevaluate_sweep_checkpoints(args, command="batch --results-csv results.csv")

            metrics_path = outdir / "metrics" / "00000_mlp_seed3_reeval.json"
            manifest_path = outdir / sweep_eval.MANIFEST_FILENAME
            summary_path = outdir / sweep_eval.SUMMARY_FILENAME
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        evaluate.assert_called_once()
        called_checkpoint, called_rollouts, called_test_steps, _ = evaluate.call_args.args
        self.assertEqual(called_checkpoint, checkpoint_path)
        self.assertEqual(called_rollouts, 7)
        self.assertEqual(called_test_steps, 15000)
        self.assertEqual(metrics["sweep_source_row"]["job_id"], "0")
        self.assertEqual(summary_rows[0]["metrics_output"], str(metrics_path))
        self.assertEqual(csv_rows[0]["checkpoint"], str(checkpoint_path))
        self.assertEqual(csv_rows[0]["test_success"], "0.5")
        self.assertEqual(manifest["jobs_completed"], 1)
        self.assertEqual(manifest["command"], "batch --results-csv results.csv")
        self.assertEqual(metrics["command"], "batch --results-csv results.csv")
        self.assertFalse(manifest["training_launched"])
        self.assertTrue(manifest["paper_protocol_status"]["checkpoint_training_reused"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_checkpoint_result"])

    def test_batch_reevaluation_allows_rollout_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path, _ = self.write_results_csv(tmpdir)
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                    "--eval-rollouts",
                    "2",
                    "--test-max-steps",
                    "20",
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate, patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                return_value={"policy_type": "mlp", "seed": 3},
            ):
                sweep_eval.reevaluate_sweep_checkpoints(args)

        _, called_rollouts, called_test_steps, _ = evaluate.call_args.args
        self.assertEqual(called_rollouts, 2)
        self.assertEqual(called_test_steps, 20)

    def test_batch_reevaluation_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["job_id", "policy", "seed", "output", "eval_rollouts", "test_max_steps"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "0",
                        "output": str(Path(tmpdir) / "missing.pt"),
                        "eval_rollouts": "1",
                        "test_max_steps": "20",
                    }
                )
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                ]
            )

            with self.assertRaises(FileNotFoundError):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["job_id", "policy", "seed", "eval_rollouts", "test_max_steps"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "0",
                        "eval_rollouts": "1",
                        "test_max_steps": "20",
                    }
                )
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing required columns: output"):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_empty_results_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_id",
                        "policy",
                        "seed",
                        "output",
                        "eval_rollouts",
                        "test_max_steps",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "has no checkpoint rows"):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_nonpositive_row_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_id",
                        "policy",
                        "seed",
                        "output",
                        "eval_rollouts",
                        "test_max_steps",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "0",
                        "output": str(checkpoint_path),
                        "eval_rollouts": "0",
                        "test_max_steps": "20",
                    }
                )
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "nonpositive eval_rollouts"):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_nonpositive_override_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path, _ = self.write_results_csv(tmpdir)
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                    "--test-max-steps",
                    "0",
                ]
            )

            with self.assertRaisesRegex(ValueError, "nonpositive test_max_steps"):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_fractional_integer_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_id",
                        "policy",
                        "seed",
                        "output",
                        "eval_rollouts",
                        "test_max_steps",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "0",
                        "output": str(checkpoint_path),
                        "eval_rollouts": "1.9",
                        "test_max_steps": "20",
                    }
                )
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(Path(tmpdir) / "reeval"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "lacks integer eval_rollouts"):
                sweep_eval.reevaluate_sweep_checkpoints(args)

    def test_batch_reevaluation_rejects_checkpoint_policy_mismatch_before_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path, _ = self.write_results_csv(tmpdir)
            outdir = Path(tmpdir) / "reeval"
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                return_value={"policy_type": "lstm", "seed": 3},
            ), patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate:
                with self.assertRaisesRegex(ValueError, "does not match checkpoint policy_type"):
                    sweep_eval.reevaluate_sweep_checkpoints(args)

            evaluate.assert_not_called()
            self.assertFalse((outdir / "metrics").exists())
            self.assertFalse((outdir / sweep_eval.MANIFEST_FILENAME).exists())

    def test_batch_reevaluation_rejects_missing_checkpoint_seed_before_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path, _ = self.write_results_csv(tmpdir)
            outdir = Path(tmpdir) / "reeval"
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                return_value={"policy_type": "mlp"},
            ), patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate:
                with self.assertRaisesRegex(ValueError, "lacks seed provenance"):
                    sweep_eval.reevaluate_sweep_checkpoints(args)

            evaluate.assert_not_called()
            self.assertFalse((outdir / "metrics").exists())
            self.assertFalse((outdir / sweep_eval.MANIFEST_FILENAME).exists())

    def test_batch_reevaluation_validates_all_rows_before_writing_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_checkpoint = Path(tmpdir) / "first.pt"
            first_checkpoint.write_bytes(b"checkpoint")
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_id",
                        "policy",
                        "seed",
                        "output",
                        "eval_rollouts",
                        "test_max_steps",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "3",
                        "output": str(first_checkpoint),
                        "eval_rollouts": "7",
                        "test_max_steps": "15000",
                    }
                )
                writer.writerow(
                    {
                        "job_id": "1",
                        "policy": "mlp",
                        "seed": "4",
                        "output": str(Path(tmpdir) / "missing.pt"),
                        "eval_rollouts": "7",
                        "test_max_steps": "15000",
                    }
                )
            outdir = Path(tmpdir) / "reeval"
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                return_value={"policy_type": "mlp", "seed": 3},
            ), patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate:
                with self.assertRaises(FileNotFoundError):
                    sweep_eval.reevaluate_sweep_checkpoints(args)

            evaluate.assert_not_called()
            self.assertFalse((outdir / "metrics").exists())
            self.assertFalse((outdir / sweep_eval.SUMMARY_FILENAME).exists())

    def test_batch_reevaluation_validates_later_row_job_id_before_writing_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_checkpoint = Path(tmpdir) / "first.pt"
            second_checkpoint = Path(tmpdir) / "second.pt"
            first_checkpoint.write_bytes(b"checkpoint")
            second_checkpoint.write_bytes(b"checkpoint")
            results_path = Path(tmpdir) / "results.csv"
            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_id",
                        "policy",
                        "seed",
                        "output",
                        "eval_rollouts",
                        "test_max_steps",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_id": "0",
                        "policy": "mlp",
                        "seed": "3",
                        "output": str(first_checkpoint),
                        "eval_rollouts": "7",
                        "test_max_steps": "15000",
                    }
                )
                writer.writerow(
                    {
                        "job_id": "1.5",
                        "policy": "mlp",
                        "seed": "4",
                        "output": str(second_checkpoint),
                        "eval_rollouts": "7",
                        "test_max_steps": "15000",
                    }
                )
            outdir = Path(tmpdir) / "reeval"
            args = sweep_eval.parse_args(
                [
                    "--results-csv",
                    str(results_path),
                    "--outdir",
                    str(outdir),
                ]
            )

            with patch(
                "evaluate_cartpole_ppo_sweep_checkpoints._load_checkpoint_config",
                side_effect=[
                    {"policy_type": "mlp", "seed": 3},
                    {"policy_type": "mlp", "seed": 4},
                ],
            ), patch(
                "evaluate_cartpole_ppo_sweep_checkpoints.evaluate_checkpoint_metrics",
                side_effect=self.fake_metrics,
            ) as evaluate:
                with self.assertRaisesRegex(ValueError, "lacks integer job_id"):
                    sweep_eval.reevaluate_sweep_checkpoints(args)

            evaluate.assert_not_called()
            self.assertFalse((outdir / "metrics").exists())
            self.assertFalse((outdir / sweep_eval.SUMMARY_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
