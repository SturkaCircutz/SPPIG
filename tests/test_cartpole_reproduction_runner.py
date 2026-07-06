import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "run_cartpole_reproduction.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_cartpole_reproduction  # noqa: E402
from run_cartpole_reproduction import (  # noqa: E402
    HAS_TORCH,
    direct_opt_evidence_status,
    load_ppo_sweep_manifest,
    ppo_sweep_evidence_status,
    reproduction_protocol_status,
    run_ppo,
    run_psm,
    summarize_rows,
    validate_psm_artifact_consistency,
)


@dataclass
class FakePPOConfig:
    policy_type: str
    total_timesteps: int
    rollout_steps: int
    update_epochs: int
    minibatches: int
    hidden_size: int
    num_envs: int
    eval_rollouts: int
    eval_test_max_steps: int
    eval_interval: int
    seed: int
    initial_log_std: float
    metrics_output: str


class CartpoleReproductionRunnerTest(unittest.TestCase):
    def test_reproduction_protocol_status_keeps_fixed_config_runs_non_paper_scale(self):
        status = reproduction_protocol_status(
            seeds=[0, 1, 2, 3, 4],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=True,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": False},
        )

        self.assertEqual(status["artifact_kind"], "cartpole_reproduction_runner_manifest")
        self.assertEqual(status["selected_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(status["distinct_seeds"], [0, 1, 2, 3, 4])
        self.assertTrue(status["uses_five_distinct_seeds"])
        self.assertTrue(status["uses_paper_eval_rollouts"])
        self.assertTrue(status["uses_full_test_horizon"])
        self.assertTrue(status["include_ppo"])
        self.assertTrue(status["includes_ppo_baseline_evidence"])
        self.assertTrue(status["include_direct_opt"])
        self.assertTrue(status["ppo_fixed_config_only"])
        self.assertFalse(status["ppo_hyperparameter_search"])
        self.assertFalse(status["ppo_sweep_evidence"]["manifest_loaded"])
        self.assertFalse(status["full_probabilistic_adaptive_teaching"])
        self.assertFalse(status["full_direct_opt_protocol"])
        self.assertFalse(status["direct_opt_evidence"]["paper_scale_direct_opt_protocol"])
        self.assertIn(
            "paper_scale_direct_opt_protocol_per_row",
            status["direct_opt_evidence"]["missing_direct_opt_evidence_requirements"],
        )
        self.assertFalse(status["paper_scale_result"])
        self.assertIn("completed PPO/PPO-LSTM hyperparameter-search evidence", status["limitation"])

    def test_reproduction_protocol_status_accepts_completed_ppo_sweep_evidence_only(self):
        dry_run_evidence = {
            "manifest_loaded": True,
            "paper_scale_plan": True,
            "paper_scale_execution": False,
            "paper_random_hyperparameter_search": True,
            "paper_random_sample_count": True,
            "sampled_hyperparameters_follow_paper_ranges": True,
            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
            "all_planned_jobs_completed": False,
            "planned_job_count_matches_selected_space": True,
            "full_baseline_policy_set": True,
            "paper_seed_count": True,
            "paper_timestep_budget": True,
            "uses_paper_eval_rollouts": True,
            "paper_test_horizon": True,
            "jobs_planned": 100,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "jobs_uncapped_for_selected_space": 100,
        }
        dry_run_status = reproduction_protocol_status(
            seeds=[0, 1, 2, 3, 4],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=True,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": True},
            ppo_sweep_status=dry_run_evidence,
        )
        self.assertFalse(dry_run_status["ppo_hyperparameter_search"])
        self.assertTrue(dry_run_status["ppo_fixed_config_only"])
        self.assertFalse(dry_run_status["paper_scale_result"])

        completed_evidence = {
            **dry_run_evidence,
            "paper_scale_execution": True,
            "all_planned_jobs_completed": True,
            "jobs_completed": 100,
        }
        completed_status = reproduction_protocol_status(
            seeds=[0, 1, 2, 3, 4],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=False,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": True},
            ppo_sweep_status=completed_evidence,
        )
        self.assertTrue(completed_status["ppo_hyperparameter_search"])
        self.assertTrue(completed_status["includes_ppo_baseline_evidence"])
        self.assertFalse(completed_status["ppo_fixed_config_only"])
        self.assertFalse(completed_status["paper_scale_result"])

    def test_direct_opt_evidence_status_requires_complete_row_protocol_evidence(self):
        rows = [
            {
                "policy": "Direct-Opt diagnostic",
                "seed": 0,
                "paper_protocol_status": {
                    "direct_opt_protocol_requirements": {
                        "paper_batch_size_and_batch_refinement": True,
                        "full_continuous_one_hot_switch_grammar": False,
                    },
                    "missing_direct_opt_protocol_requirements": [
                        "full_continuous_one_hot_switch_grammar",
                    ],
                    "paper_scale_direct_opt_protocol": False,
                },
            },
            {
                "policy": "Programmatic state machine",
                "seed": 0,
            },
        ]

        status = direct_opt_evidence_status(rows, seeds=[0], include_direct_opt=True)

        self.assertTrue(status["requested"])
        self.assertEqual(status["rows_recorded"], 1)
        self.assertTrue(status["records_rows_for_selected_seeds"])
        self.assertTrue(status["covers_selected_seed_set"])
        self.assertTrue(status["all_rows_have_protocol_status"])
        self.assertTrue(status["all_rows_have_requirement_maps"])
        self.assertFalse(status["all_row_requirements_satisfied"])
        self.assertFalse(status["all_rows_missing_requirement_lists_empty"])
        self.assertFalse(status["all_rows_paper_scale_direct_opt_protocol"])
        self.assertFalse(status["paper_scale_direct_opt_protocol"])
        self.assertIn(
            "direct_opt_protocol_requirements_satisfied_per_row",
            status["missing_direct_opt_evidence_requirements"],
        )
        self.assertIn(
            "direct_opt_missing_requirements_empty_per_row",
            status["missing_direct_opt_evidence_requirements"],
        )
        self.assertIn(
            "paper_scale_direct_opt_protocol_per_row",
            status["missing_direct_opt_evidence_requirements"],
        )

    def test_direct_opt_evidence_status_rejects_bare_overclaimed_protocol_flag(self):
        status = direct_opt_evidence_status(
            [
                {
                    "policy": "Direct-Opt diagnostic",
                    "seed": 0,
                    "paper_protocol_status": {
                        "paper_scale_direct_opt_protocol": True,
                    },
                },
            ],
            seeds=[0],
            include_direct_opt=True,
        )

        self.assertTrue(status["all_rows_paper_scale_direct_opt_protocol"])
        self.assertFalse(status["all_rows_have_requirement_maps"])
        self.assertFalse(status["all_row_requirements_satisfied"])
        self.assertFalse(status["all_rows_missing_requirement_lists_empty"])
        self.assertFalse(status["paper_scale_direct_opt_protocol"])
        self.assertIn(
            "direct_opt_protocol_requirement_map_per_row",
            status["missing_direct_opt_evidence_requirements"],
        )

    def test_direct_opt_evidence_status_rejects_partial_requirement_map(self):
        status = direct_opt_evidence_status(
            [
                {
                    "policy": "Direct-Opt diagnostic",
                    "seed": 0,
                    "paper_protocol_status": {
                        "direct_opt_protocol_requirements": {
                            "paper_eval_rollouts": True,
                        },
                        "missing_direct_opt_protocol_requirements": [],
                        "paper_scale_direct_opt_protocol": True,
                    },
                },
            ],
            seeds=[0],
            include_direct_opt=True,
        )

        self.assertTrue(status["all_rows_have_requirement_maps"])
        self.assertFalse(status["all_rows_have_expected_requirement_keys"])
        self.assertFalse(status["paper_scale_direct_opt_protocol"])
        self.assertIn(
            "direct_opt_protocol_expected_requirement_keys_per_row",
            status["missing_direct_opt_evidence_requirements"],
        )

    def test_reproduction_protocol_status_accepts_complete_direct_opt_row_evidence(self):
        ppo_sweep_status = {
            "manifest_loaded": True,
            "paper_scale_plan": True,
            "paper_scale_execution": True,
            "paper_random_hyperparameter_search": True,
            "paper_random_sample_count": True,
            "sampled_hyperparameters_follow_paper_ranges": True,
            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
            "all_planned_jobs_completed": True,
            "planned_job_count_matches_selected_space": True,
            "full_baseline_policy_set": True,
            "paper_seed_count": True,
            "paper_timestep_budget": True,
            "uses_paper_eval_rollouts": True,
            "paper_test_horizon": True,
            "jobs_planned": 100,
            "jobs_completed": 100,
            "jobs_failed": 0,
            "jobs_uncapped_for_selected_space": 100,
        }
        direct_opt_status = direct_opt_evidence_status(
            [
                {
                    "policy": "Direct-Opt diagnostic",
                    "seed": seed,
                    "paper_protocol_status": {
                        "direct_opt_protocol_requirements": {
                            "paper_batch_size_and_batch_refinement": True,
                            "paper_parallel_threads": True,
                            "paper_time_limit": True,
                            "full_continuous_one_hot_switch_grammar": True,
                            "full_initial_state_distribution": True,
                            "full_test_horizon": True,
                            "paper_eval_rollouts": True,
                        },
                        "missing_direct_opt_protocol_requirements": [],
                        "paper_scale_direct_opt_protocol": True,
                    },
                }
                for seed in [0, 1, 2, 3, 4]
            ],
            seeds=[0, 1, 2, 3, 4],
            include_direct_opt=True,
        )

        status = reproduction_protocol_status(
            seeds=[0, 1, 2, 3, 4],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=False,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": True},
            ppo_sweep_status=ppo_sweep_status,
            direct_opt_status=direct_opt_status,
        )

        self.assertTrue(status["full_direct_opt_protocol"])
        self.assertTrue(status["direct_opt_evidence"]["paper_scale_direct_opt_protocol"])
        self.assertTrue(status["paper_scale_result"])

    def test_ppo_sweep_evidence_status_requires_manifest_protocol_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "cartpole_ppo_sweep_manifest",
                        "jobs_planned": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "paper_protocol_status"):
                load_ppo_sweep_manifest(path)

    def test_ppo_sweep_evidence_status_requires_sweep_artifact_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "other_manifest",
                        "paper_protocol_status": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "artifact_kind"):
                load_ppo_sweep_manifest(path)

    def test_ppo_sweep_evidence_status_records_completed_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "cartpole_ppo_sweep_manifest",
                        "command": "python scripts/run_cartpole_ppo_sweep.py",
                        "policies": ["mlp", "lstm"],
                        "seeds": [0, 1, 2, 3, 4],
                        "jobs_planned": 100,
                        "jobs_completed": 100,
                        "jobs_failed": 0,
                        "jobs_uncapped_for_selected_space": 100,
                        "hyperparam_mode": "paper-random",
                        "hyperparam_samples": 10,
                        "paper_protocol_status": {
                            "paper_scale_plan": True,
                            "paper_scale_execution": True,
                            "paper_random_hyperparameter_search": True,
                            "paper_random_sample_count": True,
                            "all_planned_jobs_completed": True,
                            "planned_job_count_matches_selected_space": True,
                            "full_baseline_policy_set": True,
                            "paper_seed_count": True,
                            "paper_timestep_budget": True,
                            "uses_paper_eval_rollouts": True,
                            "paper_test_horizon": True,
                            "sampled_hyperparameters_follow_paper_ranges": True,
                            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_ppo_sweep_manifest(path)
            status = ppo_sweep_evidence_status(manifest)

        self.assertEqual(status["manifest_path"], str(path))
        self.assertTrue(status["manifest_loaded"])
        self.assertTrue(status["paper_scale_execution"])
        self.assertTrue(status["raw_paper_scale_execution"])
        self.assertTrue(status["paper_random_hyperparameter_search"])
        self.assertEqual(status["jobs_planned"], 100)
        self.assertEqual(status["jobs_completed"], 100)

    def test_ppo_sweep_evidence_status_rejects_inconsistent_execution_claim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "cartpole_ppo_sweep_manifest",
                        "jobs_planned": 100,
                        "jobs_completed": 99,
                        "jobs_failed": 0,
                        "jobs_uncapped_for_selected_space": 100,
                        "paper_protocol_status": {
                            "paper_scale_plan": True,
                            "paper_scale_execution": True,
                            "paper_random_hyperparameter_search": True,
                            "paper_random_sample_count": True,
                            "all_planned_jobs_completed": False,
                            "planned_job_count_matches_selected_space": True,
                            "full_baseline_policy_set": True,
                            "paper_seed_count": True,
                            "paper_timestep_budget": True,
                            "uses_paper_eval_rollouts": True,
                            "paper_test_horizon": True,
                            "sampled_hyperparameters_follow_paper_ranges": True,
                            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_ppo_sweep_manifest(path)
            status = ppo_sweep_evidence_status(manifest)

        self.assertTrue(status["raw_paper_scale_execution"])
        self.assertFalse(status["all_planned_jobs_completed"])
        self.assertFalse(status["completed_job_counts_match"])
        self.assertFalse(status["paper_scale_execution"])

    def test_ppo_sweep_evidence_status_cross_checks_top_level_manifest_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "cartpole_ppo_sweep_manifest",
                        "policies": ["mlp"],
                        "seeds": [0, 0, 0, 0, 0],
                        "jobs_planned": 100,
                        "jobs_completed": 100,
                        "jobs_failed": 0,
                        "jobs_uncapped_for_selected_space": 100,
                        "hyperparam_mode": "grid",
                        "hyperparam_samples": 10,
                        "paper_protocol_status": {
                            "paper_scale_plan": True,
                            "paper_scale_execution": True,
                            "paper_random_hyperparameter_search": True,
                            "paper_random_sample_count": True,
                            "all_planned_jobs_completed": True,
                            "planned_job_count_matches_selected_space": True,
                            "full_baseline_policy_set": True,
                            "paper_seed_count": True,
                            "paper_timestep_budget": True,
                            "uses_paper_eval_rollouts": True,
                            "paper_test_horizon": True,
                            "sampled_hyperparameters_follow_paper_ranges": True,
                            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_ppo_sweep_manifest(path)
            status = ppo_sweep_evidence_status(manifest)

        self.assertTrue(status["raw_paper_scale_execution"])
        self.assertFalse(status["top_level_full_policy_set"])
        self.assertFalse(status["top_level_paper_seed_count"])
        self.assertFalse(status["top_level_paper_random_samples"])
        self.assertFalse(status["paper_scale_execution"])

    def test_quick_runner_records_ppo_sweep_manifest_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_path = Path(tmpdir) / "cartpole_ppo_sweep_manifest.json"
            sweep_path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "cartpole_ppo_sweep_manifest",
                        "command": "python scripts/run_cartpole_ppo_sweep.py",
                        "policies": ["mlp", "lstm"],
                        "seeds": [0, 1, 2, 3, 4],
                        "jobs_planned": 100,
                        "jobs_completed": 100,
                        "jobs_failed": 0,
                        "jobs_uncapped_for_selected_space": 100,
                        "hyperparam_mode": "paper-random",
                        "hyperparam_samples": 10,
                        "paper_protocol_status": {
                            "paper_scale_plan": True,
                            "paper_scale_execution": True,
                            "paper_random_hyperparameter_search": True,
                            "paper_random_sample_count": True,
                            "all_planned_jobs_completed": True,
                            "planned_job_count_matches_selected_space": True,
                            "full_baseline_policy_set": True,
                            "paper_seed_count": True,
                            "paper_timestep_budget": True,
                            "uses_paper_eval_rollouts": True,
                            "paper_test_horizon": True,
                            "sampled_hyperparameters_follow_paper_ranges": True,
                            "sampled_hyperparameters_follow_paper_minibatch_rules": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--ppo-sweep-manifest",
                    str(sweep_path),
                    "--seeds",
                    "0",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(Path(tmpdir) / "cartpole_manifest.json", encoding="utf-8") as handle:
                manifest = json.load(handle)

        evidence = manifest["paper_protocol_status"]["ppo_sweep_evidence"]
        self.assertEqual(manifest["ppo_sweep_manifest"], str(sweep_path))
        self.assertEqual(manifest["ppo_sweep_evidence"], evidence)
        self.assertTrue(evidence["manifest_loaded"])
        self.assertTrue(evidence["paper_scale_execution"])
        self.assertFalse(manifest["include_ppo"])
        self.assertTrue(manifest["paper_protocol_status"]["ppo_hyperparameter_search"])
        self.assertTrue(manifest["paper_protocol_status"]["includes_ppo_baseline_evidence"])
        self.assertFalse(manifest["paper_protocol_status"]["ppo_fixed_config_only"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_result"])

    def test_reproduction_protocol_status_rejects_duplicate_seed_coverage(self):
        status = reproduction_protocol_status(
            seeds=[0, 0, 1, 2, 3],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=True,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": True},
        )

        self.assertEqual(status["distinct_seeds"], [0, 1, 2, 3])
        self.assertFalse(status["uses_five_distinct_seeds"])
        self.assertFalse(status["paper_scale_result"])

    def test_reproduction_protocol_status_rejects_extra_duplicate_seed_coverage(self):
        status = reproduction_protocol_status(
            seeds=[0, 0, 1, 2, 3, 4],
            eval_rollouts=1000,
            test_max_steps=15000,
            include_ppo=True,
            include_direct_opt=True,
            quick=False,
            ppo_eval_interval=0,
            psm_status={"full_probabilistic_adaptive_teaching": True},
        )

        self.assertEqual(status["distinct_seeds"], [0, 1, 2, 3, 4])
        self.assertFalse(status["uses_five_distinct_seeds"])
        self.assertFalse(status["paper_scale_result"])

    def test_runner_rejects_manifest_level_psm_duplicate_plus_five_seed_selection(self):
        captured: dict[str, object] = {}
        original_argv = sys.argv

        def fake_run_psm(seed, eval_rollouts, test_max_steps, quick, outdir, teacher_overrides):
            return {
                "policy": "Programmatic state machine",
                "seed": seed,
                "train_success": 0.0,
                "test_success": 0.0,
                "train_reward": 1.0,
                "test_reward": 1.0,
                "train_steps": 1.0,
                "test_steps": 1.0,
                "train_survival_seconds": 0.02,
                "test_survival_seconds": 0.02,
                "eval_rollouts": eval_rollouts,
                "test_horizon_steps": test_max_steps,
                "timesteps": 0,
                "metrics_output": f"metrics/psm_seed{seed}.json",
                "traces_output": f"traces/psm_seed{seed}.json",
                "command": "python mocked-runner",
                "paper_protocol_status": {
                    "synthesized_by_current_algorithm": True,
                    "five_seed_selection": False,
                },
            }

        def capture_write_results(rows, outdir, manifest):
            captured["rows"] = rows
            captured["manifest"] = manifest

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = [
                    SCRIPT,
                    "--seeds",
                    "0,0,1,2,3,4",
                    "--eval-rollouts",
                    "1000",
                    "--test-max-steps",
                    "15000",
                    "--outdir",
                    tmpdir,
                ]
                with patch.object(run_cartpole_reproduction, "run_psm", side_effect=fake_run_psm), patch.object(
                    run_cartpole_reproduction,
                    "write_results",
                    side_effect=capture_write_results,
                ):
                    run_cartpole_reproduction.main()
        finally:
            sys.argv = original_argv

        manifest = captured["manifest"]
        self.assertEqual(manifest["seeds"], [0, 0, 1, 2, 3, 4])
        self.assertFalse(manifest["psm_paper_protocol_status"]["five_seed_selection"])
        self.assertFalse(manifest["paper_protocol_status"]["uses_five_distinct_seeds"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_result"])

    def test_runner_records_manifest_level_psm_five_seed_selection(self):
        captured: dict[str, object] = {}
        original_argv = sys.argv

        def fake_run_psm(seed, eval_rollouts, test_max_steps, quick, outdir, teacher_overrides):
            return {
                "policy": "Programmatic state machine",
                "seed": seed,
                "train_success": 0.0,
                "test_success": 0.0,
                "train_reward": 1.0,
                "test_reward": 1.0,
                "train_steps": 1.0,
                "test_steps": 1.0,
                "train_survival_seconds": 0.02,
                "test_survival_seconds": 0.02,
                "eval_rollouts": eval_rollouts,
                "test_horizon_steps": test_max_steps,
                "timesteps": 0,
                "metrics_output": f"metrics/psm_seed{seed}.json",
                "traces_output": f"traces/psm_seed{seed}.json",
                "command": "python mocked-runner",
                "paper_protocol_status": {
                    "synthesized_by_current_algorithm": True,
                    "five_seed_selection": False,
                    "adaptive_teaching_protocol_requirements": {
                        "five_seed_selection": False,
                    },
                    "missing_adaptive_teaching_protocol_requirements": [
                        "five_seed_selection",
                    ],
                },
            }

        def capture_write_results(rows, outdir, manifest):
            captured["rows"] = rows
            captured["manifest"] = manifest

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = [
                    SCRIPT,
                    "--seeds",
                    "0,1,2,3,4",
                    "--eval-rollouts",
                    "1000",
                    "--test-max-steps",
                    "15000",
                    "--outdir",
                    tmpdir,
                ]
                with patch.object(run_cartpole_reproduction, "run_psm", side_effect=fake_run_psm), patch.object(
                    run_cartpole_reproduction,
                    "write_results",
                    side_effect=capture_write_results,
                ):
                    run_cartpole_reproduction.main()
        finally:
            sys.argv = original_argv

        manifest = captured["manifest"]
        self.assertEqual(manifest["seeds"], [0, 1, 2, 3, 4])
        psm_status = manifest["psm_paper_protocol_status"]
        self.assertTrue(psm_status["five_seed_selection"])
        self.assertTrue(psm_status["adaptive_teaching_protocol_requirements"]["five_seed_selection"])
        self.assertNotIn(
            "five_seed_selection",
            psm_status["missing_adaptive_teaching_protocol_requirements"],
        )
        self.assertFalse(captured["rows"][0]["paper_protocol_status"]["five_seed_selection"])
        self.assertFalse(manifest["paper_protocol_status"]["paper_scale_result"])

    def test_summary_rows_report_mean_std_and_best_train_seed(self):
        summary = summarize_rows(
            [
                {
                    "policy": "Programmatic state machine",
                    "seed": 1,
                    "train_success": 0.5,
                    "test_success": 0.25,
                    "train_reward": 100.0,
                    "test_reward": 200.0,
                    "train_steps": 100.0,
                    "test_steps": 200.0,
                    "train_survival_seconds": 2.0,
                    "test_survival_seconds": 4.0,
                    "eval_rollouts": 2,
                    "test_horizon_steps": 15000,
                    "timesteps": 0,
                    "command": "python runner.py --seed 1",
                },
                {
                    "policy": "Programmatic state machine",
                    "seed": 0,
                    "train_success": 1.0,
                    "test_success": 0.75,
                    "train_reward": 250.0,
                    "test_reward": 900.0,
                    "train_steps": 250.0,
                    "test_steps": 900.0,
                    "train_survival_seconds": 5.0,
                    "test_survival_seconds": 18.0,
                    "eval_rollouts": 2,
                    "test_horizon_steps": 15000,
                    "timesteps": 0,
                    "command": "python runner.py --seed 0",
                },
            ]
        )

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["policy"], "Programmatic state machine")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["train_success_mean"], 0.75)
        self.assertAlmostEqual(row["train_success_std"], 0.3535533905932738)
        self.assertAlmostEqual(row["test_reward_mean"], 550.0)
        self.assertAlmostEqual(row["test_steps_mean"], 550.0)
        self.assertAlmostEqual(row["test_survival_seconds_mean"], 11.0)
        self.assertEqual(row["best_seed_by_train"], 0)
        self.assertAlmostEqual(row["best_test_success"], 0.75)
        self.assertAlmostEqual(row["best_test_steps"], 900.0)
        self.assertAlmostEqual(row["best_test_survival_seconds"], 18.0)
        self.assertEqual(row["eval_rollouts"], 2)
        self.assertEqual(row["test_horizon_steps"], 15000)
        self.assertEqual(row["best_command"], "python runner.py --seed 0")

    def test_run_ppo_row_records_command_without_reloading_metrics(self):
        original_argv = sys.argv
        try:
            sys.argv = [SCRIPT, "--quick", "--include-ppo"]
            result = SimpleNamespace(
                train_success_rate=1.0,
                test_success_rate=0.0,
                train_reward_mean=250.0,
                test_reward_mean=20.0,
                train_steps_mean=250.0,
                test_steps_mean=20.0,
                train_survival_seconds_mean=5.0,
                test_survival_seconds_mean=0.4,
                timesteps=64,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch.object(run_cartpole_reproduction, "HAS_TORCH", True), patch.object(
                    run_cartpole_reproduction,
                    "PPOConfig",
                    FakePPOConfig,
                    create=True,
                ), patch.object(
                    run_cartpole_reproduction,
                    "train_ppo_cartpole",
                    return_value=(object(), result),
                    create=True,
                ), patch.object(
                    run_cartpole_reproduction,
                    "ppo_paper_protocol_status",
                    return_value={"paper_scale_baseline_protocol": False},
                    create=True,
                ):
                    row = run_ppo(
                        "mlp",
                        seed=0,
                        eval_rollouts=1,
                        test_max_steps=20,
                        outdir=Path(tmpdir),
                        eval_interval=32,
                        quick=True,
                    )
        finally:
            sys.argv = original_argv

        self.assertEqual(row["command"], f"{SCRIPT} --quick --include-ppo")
        self.assertEqual(row["policy"], "PPO MLP")
        self.assertEqual(row["timesteps"], 64)
        self.assertEqual(row["paper_protocol_status"]["paper_scale_baseline_protocol"], False)

    def test_quick_runner_writes_results_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--seeds",
                    "0",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--psm-teacher-theta-gain",
                    "12.5",
                    "--psm-teacher-omega-gain",
                    "0.75",
                    "--psm-teacher-student-iters",
                    "1",
                    "--psm-student-em-iters",
                    "2",
                    "--psm-student-switch-responsibility-passes",
                    "2",
                    "--psm-teacher-student-regularizer",
                    "0.5",
                    "--psm-teacher-reward-lambda",
                    "100",
                    "--psm-teacher-top-rho",
                    "1",
                    "--psm-teacher-refinement-steps",
                    "1",
                    "--psm-teacher-elite-distribution-resamples",
                    "3",
                    "--psm-teacher-elite-distribution-rounds",
                    "2",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            csv_path = os.path.join(tmpdir, "cartpole_results.csv")
            summary_path = os.path.join(tmpdir, "cartpole_summary.csv")
            manifest_path = os.path.join(tmpdir, "cartpole_manifest.json")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(manifest_path))

            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["policy"], "Programmatic state machine")
            self.assertEqual(rows[0]["seed"], "0")
            self.assertEqual(rows[0]["test_horizon_steps"], "20")
            self.assertIn("train_steps", rows[0])
            self.assertIn("test_steps", rows[0])
            self.assertIn("train_survival_seconds", rows[0])
            self.assertIn("test_survival_seconds", rows[0])
            self.assertEqual(rows[0]["eval_rollouts"], "1")
            self.assertGreater(float(rows[0]["train_steps"]), 0.0)
            self.assertGreater(float(rows[0]["test_steps"]), 0.0)
            self.assertTrue(os.path.exists(rows[0]["metrics_output"]))
            self.assertTrue(os.path.exists(rows[0]["traces_output"]))
            with open(rows[0]["metrics_output"], encoding="utf-8") as handle:
                psm_metrics = json.load(handle)
            with open(rows[0]["traces_output"], encoding="utf-8") as handle:
                psm_traces = json.load(handle)
            self.assertIn("run_cartpole_reproduction.py", psm_metrics["command"])
            self.assertIn("--psm-teacher-theta-gain 12.5", psm_metrics["command"])
            self.assertEqual(rows[0]["command"], psm_metrics["command"])
            self.assertEqual(psm_traces["command"], psm_metrics["command"])
            self.assertEqual(psm_metrics["config"]["teacher_theta_gain"], 12.5)
            self.assertEqual(psm_metrics["config"]["parallel_trace_workers"], 1)
            self.assertEqual(psm_metrics["config"]["parallel_switch_workers"], 1)
            self.assertEqual(psm_metrics["algorithm_provenance"]["switch_timing"]["std_steps"], 2.0)
            self.assertEqual(
                psm_metrics["algorithm_provenance"]["switch_timing"]["duration_units"],
                "segment_elapsed_time_normalized_to_default_cartpole_dt",
            )
            self.assertEqual(psm_metrics["paper_test_horizon_steps"], 15000)
            self.assertEqual(psm_metrics["paper_eval_rollouts"], 1000)
            self.assertFalse(psm_metrics["uses_paper_eval_rollouts"])
            self.assertTrue(psm_metrics["reward_spec"]["reward_equals_survived_steps"])
            self.assertEqual(psm_metrics["space_spec"]["action_dimension"], 1)
            self.assertEqual(psm_metrics["space_spec"]["observation_dimension"], 4)
            self.assertIn("steps_mean", psm_metrics["train"])
            self.assertIn("survival_seconds_mean", psm_metrics["train"])
            self.assertIn("steps_mean", psm_metrics["test"])
            self.assertIn("survival_seconds_mean", psm_metrics["test"])
            self.assertEqual(psm_metrics["traces_output"], rows[0]["traces_output"])
            self.assertEqual(psm_traces["metrics_output"], rows[0]["metrics_output"])
            self.assertEqual(psm_metrics["artifact_consistency"], psm_traces["artifact_consistency"])
            artifact_consistency = psm_metrics["artifact_consistency"]
            self.assertTrue(artifact_consistency["validated_by_runner"])
            self.assertEqual(artifact_consistency["num_traces"], psm_metrics["num_traces"])
            self.assertEqual(artifact_consistency["teacher_student_iters"], 1)
            self.assertEqual(artifact_consistency["trace_history_iterations"], [1])
            self.assertEqual(artifact_consistency["synthesis_history_iterations"], [1])
            self.assertEqual(artifact_consistency["adaptive_teacher_summary_iterations"], [1])
            self.assertTrue(artifact_consistency["final_trace_history_matches_traces"])
            self.assertTrue(artifact_consistency["final_evaluation_matches_top_level"])
            self.assertEqual(psm_traces["num_traces"], psm_metrics["num_traces"])
            self.assertEqual(len(psm_traces["traces"]), psm_metrics["num_traces"])
            self.assertEqual(
                len(psm_traces["trace_history"]),
                psm_metrics["config"]["teacher_student_iters"],
            )
            self.assertEqual(psm_traces["trace_history"][-1]["traces"], psm_traces["traces"])
            self.assertIn("observations", psm_traces["traces"][0])
            self.assertIn("actions", psm_traces["traces"][0])
            self.assertIn("mode_labels", psm_traces["traces"][0])
            self.assertIn("segment_actions", psm_traces["traces"][0])
            self.assertIn("segment_durations", psm_traces["traces"][0])
            self.assertIn("segment_time_increments", psm_traces["traces"][0])
            self.assertIn("teacher_candidate_pool_diagnostics", psm_traces["traces"][0])
            candidate_pool = psm_traces["traces"][0]["teacher_candidate_pool_diagnostics"]
            self.assertTrue(candidate_pool["not_full_paper_cem"])
            self.assertEqual(candidate_pool["effective_candidate_rollouts"], psm_metrics["config"]["candidate_rollouts"])
            self.assertEqual(candidate_pool["sampled_candidate_count"], psm_metrics["config"]["candidate_rollouts"])
            self.assertEqual(candidate_pool["top_rho"], psm_metrics["config"]["teacher_top_rho"])
            self.assertIn("selection_pool_refinement_objective", candidate_pool)
            psm_status = psm_metrics["paper_protocol_status"]
            self.assertTrue(psm_status["cartpole_environment"])
            self.assertEqual(psm_status["train_horizon_seconds"], 5.0)
            self.assertEqual(psm_status["train_pole_length"], 0.5)
            self.assertEqual(psm_status["test_horizon_seconds"], 300.0)
            self.assertEqual(psm_status["test_pole_length"], 1.0)
            self.assertTrue(psm_status["quick_diagnostic"])
            self.assertFalse(psm_status["uses_full_test_horizon"])
            self.assertEqual(psm_status["paper_eval_rollouts"], 1000)
            self.assertFalse(psm_status["uses_paper_eval_rollouts"])
            self.assertTrue(psm_status["reward_spec"]["reward_equals_survived_steps"])
            self.assertEqual(psm_status["space_spec"]["action_dimension"], 1)
            self.assertEqual(psm_status["space_spec"]["initial_state_distribution"]["high"], 0.05)
            self.assertTrue(psm_status["transition_specific_switch_conditions"])
            self.assertTrue(psm_status["synthesized_by_current_algorithm"])
            self.assertFalse(psm_status["full_probabilistic_adaptive_teaching"])
            self.assertFalse(psm_status["full_continuous_switch_m_step"])
            self.assertFalse(psm_status["full_cem_teacher_optimizer"])
            self.assertFalse(psm_status["paper_scale_result"])
            self.assertEqual(psm_status["student_em_iters"], 2)
            self.assertEqual(psm_status["student_switch_responsibility_passes"], 2)
            self.assertEqual(psm_status["teacher_candidate_rollouts"], 4)
            self.assertEqual(psm_status["effective_teacher_candidate_rollouts"], 4)
            self.assertEqual(psm_status["selected_teacher_top_rho"], 1)
            self.assertEqual(psm_status["effective_teacher_top_rho"], 1)
            self.assertEqual(psm_status["paper_teacher_top_rho"], 10)
            self.assertFalse(psm_status["uses_paper_teacher_top_rho"])
            self.assertEqual(psm_status["selected_teacher_parallel_trace_workers"], 1)
            self.assertEqual(psm_status["effective_teacher_parallel_trace_workers"], 1)
            self.assertEqual(psm_status["effective_teacher_parallel_trace_initial_states"], 4)
            self.assertEqual(psm_status["effective_teacher_parallel_trace_slots"], 1)
            self.assertEqual(psm_status["paper_teacher_parallel_threads"], 10)
            self.assertFalse(psm_status["uses_parallel_teacher_trace_optimization"])
            self.assertFalse(psm_status["uses_paper_teacher_parallel_threads"])
            self.assertEqual(psm_status["selected_student_parallel_switch_workers"], 1)
            self.assertEqual(psm_status["effective_student_parallel_switch_workers"], 1)
            self.assertEqual(psm_status["student_transition_switch_fit_count"], 2)
            self.assertEqual(psm_status["effective_student_parallel_switch_slots"], 1)
            self.assertEqual(psm_status["paper_student_parallel_threads"], 10)
            self.assertFalse(psm_status["uses_parallel_student_switch_optimization"])
            self.assertFalse(psm_status["uses_paper_student_parallel_threads"])
            self.assertTrue(psm_status["teacher_candidate_rollouts_cover_selected_top_rho"])
            self.assertFalse(psm_status["teacher_candidate_rollouts_cover_paper_top_rho"])
            self.assertFalse(psm_status["teacher_cem_phase_matches_paper_rho"])
            self.assertFalse(
                psm_status["probabilistic_adaptive_teaching_requirements"][
                    "teacher_cem_phase_matches_paper_rho"
                ]
            )
            self.assertFalse(
                psm_status["probabilistic_adaptive_teaching_requirements"][
                    "uses_paper_teacher_parallel_threads"
                ]
            )
            self.assertFalse(
                psm_status["probabilistic_adaptive_teaching_requirements"][
                    "uses_paper_student_parallel_worker_limit"
                ]
            )
            self.assertFalse(
                psm_status["probabilistic_adaptive_teaching_requirements"][
                    "full_continuous_switch_m_step"
                ]
            )
            self.assertIn(
                "teacher_cem_phase_matches_paper_rho",
                psm_status["missing_probabilistic_adaptive_teaching_requirements"],
            )
            self.assertIn(
                "uses_paper_teacher_parallel_threads",
                psm_status["missing_probabilistic_adaptive_teaching_requirements"],
            )
            self.assertIn(
                "uses_paper_student_parallel_worker_limit",
                psm_status["missing_probabilistic_adaptive_teaching_requirements"],
            )
            self.assertIn(
                "full_continuous_switch_m_step",
                psm_status["missing_probabilistic_adaptive_teaching_requirements"],
            )
            self.assertFalse(psm_status["adaptive_teaching_protocol_requirements"]["full_test_horizon"])
            self.assertFalse(psm_status["adaptive_teaching_protocol_requirements"]["paper_eval_rollouts"])
            self.assertFalse(psm_status["adaptive_teaching_protocol_requirements"]["five_seed_selection"])
            self.assertIn(
                "full_test_horizon",
                psm_status["missing_adaptive_teaching_protocol_requirements"],
            )
            self.assertIn(
                "paper_eval_rollouts",
                psm_status["missing_adaptive_teaching_protocol_requirements"],
            )
            self.assertIn(
                "five_seed_selection",
                psm_status["missing_adaptive_teaching_protocol_requirements"],
            )
            self.assertEqual(psm_status["teacher_elite_distribution_resamples"], 3)
            self.assertEqual(psm_status["teacher_elite_distribution_rounds"], 2)
            self.assertIn("probabilistic_student", psm_metrics)
            self.assertEqual(len(psm_metrics["adaptive_teacher_summary"]), 1)
            adaptive_summary = psm_metrics["adaptive_teacher_summary"][0]
            self.assertEqual(adaptive_summary["iteration"], 1)
            self.assertEqual(adaptive_summary["trace_count"], psm_metrics["num_traces"])
            self.assertEqual(
                adaptive_summary["teacher_sampling_model"],
                "bootstrap_probabilistic_prior",
            )
            self.assertIn("teacher_source_counts", adaptive_summary)
            self.assertEqual(
                adaptive_summary["teacher_reward_lambda"],
                psm_metrics["config"]["teacher_reward_lambda"],
            )
            self.assertEqual(
                adaptive_summary["teacher_student_regularizer"],
                psm_metrics["config"]["teacher_student_regularizer"],
            )
            self.assertLessEqual(adaptive_summary["recorded_student_log_probability_fraction"], 1.0)
            self.assertIn("recorded_teacher_objective_mean", adaptive_summary)
            self.assertEqual(adaptive_summary["recorded_teacher_objective_direct_count"], psm_metrics["num_traces"])
            self.assertEqual(
                adaptive_summary["recorded_teacher_refinement_objective_count"],
                psm_metrics["num_traces"],
            )
            self.assertEqual(len(psm_metrics["synthesis_history"]), 1)
            self.assertEqual(psm_metrics["synthesis_history"][0]["iteration"], 1)
            self.assertEqual(
                psm_metrics["synthesis_history"][0]["adaptive_teacher_summary"],
                adaptive_summary,
            )
            self.assertIn("evaluation", psm_metrics["synthesis_history"][0])
            self.assertEqual(psm_metrics["synthesis_history"][0]["evaluation"]["train"], psm_metrics["train"])
            self.assertEqual(psm_metrics["synthesis_history"][0]["evaluation"]["test"], psm_metrics["test"])
            self.assertEqual(
                psm_metrics["synthesis_history"][0]["trace_summary"]["count"],
                psm_metrics["num_traces"],
            )
            self.assertIn(
                "switch_fit_diagnostics",
                psm_metrics["synthesis_history"][0],
            )
            self.assertIn("switch_fit_diagnostics", psm_metrics)
            self.assertIn(
                "fixed_local_reference_switch",
                psm_metrics["switch_fit_diagnostics"]["candidates"],
            )
            self.assertEqual(psm_metrics["trace_summary"]["count"], psm_metrics["num_traces"])
            self.assertIn("teacher_objective", psm_metrics["trace_summary"]["examples"][0])
            self.assertIn("teacher_refinement_objective", psm_metrics["trace_summary"]["examples"][0])

            with open(summary_path, newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["policy"], "Programmatic state machine")
            self.assertEqual(summary[0]["n"], "1")
            self.assertEqual(summary[0]["best_seed_by_train"], "0")
            self.assertEqual(summary[0]["train_success_std"], "0.0")
            self.assertEqual(summary[0]["eval_rollouts"], "1")
            self.assertEqual(summary[0]["test_horizon_steps"], "20")
            self.assertIn("test_steps_mean", summary[0])
            self.assertIn("test_survival_seconds_mean", summary[0])
            self.assertEqual(summary[0]["best_traces_output"], rows[0]["traces_output"])
            self.assertEqual(summary[0]["best_command"], psm_metrics["command"])

            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["quick"])
            self.assertEqual(manifest["seeds"], [0])
            self.assertEqual(manifest["paper_eval_rollouts"], 1000)
            self.assertFalse(manifest["uses_paper_eval_rollouts"])
            self.assertTrue(manifest["reward_spec"]["reward_equals_survived_steps"])
            self.assertEqual(manifest["space_spec"]["action_dimension"], 1)
            self.assertEqual(manifest["space_spec"]["observation_dimension"], 4)
            self.assertEqual(manifest["test_max_steps"], 20)
            self.assertIn("traces_output", manifest["rows"][0])
            self.assertTrue(os.path.exists(manifest["rows"][0]["traces_output"]))
            self.assertIn("full selected teacher traces", manifest["psm_artifact_note"])
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_theta_gain"], 12.5)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_omega_gain"], 0.75)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_student_iters"], 1)
            self.assertEqual(manifest["psm_teacher_overrides"]["student_em_iters"], 2)
            self.assertEqual(manifest["psm_teacher_overrides"]["student_switch_responsibility_passes"], 2)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_student_regularizer"], 0.5)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_reward_lambda"], 100.0)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_top_rho"], 1)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_refinement_steps"], 1)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_elite_distribution_resamples"], 3)
            self.assertEqual(manifest["psm_teacher_overrides"]["teacher_elite_distribution_rounds"], 2)
            self.assertEqual(manifest["psm_teacher_overrides"]["parallel_trace_workers"], 1)
            self.assertEqual(manifest["psm_teacher_overrides"]["parallel_switch_workers"], 1)
            manifest_psm_status = manifest["psm_paper_protocol_status"]
            self.assertEqual(manifest_psm_status, psm_status)
            manifest_status = manifest["paper_protocol_status"]
            self.assertEqual(manifest_status["artifact_kind"], "cartpole_reproduction_runner_manifest")
            self.assertEqual(manifest_status["selected_seeds"], [0])
            self.assertEqual(manifest_status["distinct_seeds"], [0])
            self.assertFalse(manifest_status["uses_five_distinct_seeds"])
            self.assertEqual(manifest_status["paper_eval_rollouts"], 1000)
            self.assertEqual(manifest_status["selected_eval_rollouts"], 1)
            self.assertFalse(manifest_status["uses_paper_eval_rollouts"])
            self.assertEqual(manifest_status["paper_test_horizon_steps"], 15000)
            self.assertEqual(manifest_status["selected_test_max_steps"], 20)
            self.assertFalse(manifest_status["uses_full_test_horizon"])
            self.assertTrue(manifest_status["quick_diagnostic"])
            self.assertFalse(manifest_status["include_ppo"])
            self.assertFalse(manifest_status["include_direct_opt"])
            self.assertFalse(manifest_status["ppo_fixed_config_only"])
            self.assertFalse(manifest_status["ppo_hyperparameter_search"])
            self.assertFalse(manifest_status["full_probabilistic_adaptive_teaching"])
            self.assertFalse(manifest_status["full_direct_opt_protocol"])
            self.assertFalse(manifest_status["paper_scale_result"])
            provenance = manifest["psm_algorithm_provenance"]
            self.assertEqual(provenance["probabilistic_student"]["default_em_iters"], 4)
            self.assertEqual(provenance["probabilistic_student"]["default_switch_responsibility_passes"], 1)
            self.assertEqual(provenance["teacher_search"]["paper_parallel_threads"], 10)
            self.assertEqual(
                provenance["teacher_search"]["local_parallel_trace_workers"],
                "configurable_via_parallel_trace_workers",
            )
            self.assertEqual(
                provenance["probabilistic_student"]["responsibility_evidence"],
                "action_likelihood_initialization_then_directed_switch_forward_backward_action_refits",
            )
            self.assertTrue(provenance["probabilistic_student"]["switch_responsibility_passes_are_per_em_iteration"])
            self.assertEqual(
                provenance["probabilistic_student"]["switch_condition_m_step_schedule"],
                "once_per_student_em_iteration_after_configured_eq10_eq11_passes",
            )
            self.assertEqual(
                provenance["probabilistic_student"]["initial_switch_before_first_timing_e_step"],
                "fixed_bootstrap_not_data_fit",
            )
            self.assertEqual(
                provenance["probabilistic_student"]["directed_switch_e_step_schedule"],
                "uses_latest_transition_specific_switches_after_first_bounded_m_step",
            )
            self.assertEqual(provenance["probabilistic_student"]["rollout_parameter_resampling"], "on_mode_entry")
            self.assertEqual(
                provenance["probabilistic_student"]["transition_specific_switches"],
                "separate_fitted_conditions_for_0_to_1_and_1_to_0",
            )
            self.assertEqual(provenance["probabilistic_student"]["initial_mode"], 0)
            self.assertEqual(provenance["probabilistic_student"]["initial_mode_prior"], "fixed_mode_0")
            self.assertEqual(provenance["switch_timing"]["std_steps"], 2.0)
            self.assertEqual(
                provenance["switch_timing"]["duration_units"],
                "segment_elapsed_time_normalized_to_default_cartpole_dt",
            )
            self.assertEqual(
                provenance["switch_timing"]["depth2_boolean_probability"],
                "shared_threshold_rectangle_union",
            )
            self.assertEqual(provenance["switch_timing"]["coordinate_refinement_steps"], 3)
            self.assertAlmostEqual(
                provenance["switch_timing"]["coordinate_log_std_initial_step"],
                math.log(2.0),
            )
            self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_refinement_steps"], 2)
            self.assertEqual(
                provenance["switch_timing"]["transition_specific_m_step"],
                "bounded_separate_0_to_1_and_1_to_0_switch_fits",
            )
            self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_mean_step_fraction"], 0.5)
            self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_log_std_step"], 0.25)
            self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_epsilon_fraction"], 0.25)
            self.assertEqual(
                provenance["switch_timing"]["finite_difference_gradient_backtracking_factors"],
                [1.0, 0.5, 0.25, 0.125],
            )
            self.assertEqual(provenance["switch_search"]["boolean_tree_depth"], 2)
            self.assertTrue(
                provenance["switch_search"]["greedy_second_predicate_expands_switch_and_no_switch_leaves"]
            )
            self.assertEqual(provenance["switch_search"]["greedy_second_predicate_prefilter_top_k"], 32)
            self.assertIn(50.0, provenance["switch_search"]["oblique_theta_weights"])
            self.assertEqual(provenance["teacher_search"]["duration_refinement_deltas"], [-1, 1])
            self.assertEqual(
                provenance["teacher_search"]["action_refinement_max_candidates_per_segment"],
                2,
            )
            self.assertEqual(
                provenance["teacher_search"]["action_refinement_step_fraction"],
                0.25,
            )
            self.assertEqual(provenance["teacher_search"]["action_gradient_step_fraction"], 0.10)
            self.assertEqual(provenance["teacher_search"]["action_gradient_epsilon_fraction"], 0.05)
            self.assertEqual(provenance["teacher_search"]["gain_gradient_step_fraction"], 0.05)
            self.assertEqual(provenance["teacher_search"]["gain_gradient_epsilon_fraction"], 0.025)
            self.assertEqual(provenance["teacher_search"]["duration_gradient_step"], 1)
            self.assertEqual(provenance["teacher_search"]["duration_gradient_epsilon"], 1)
            self.assertEqual(
                provenance["teacher_search"]["finite_difference_gradient_backtracking_factors"],
                [1.0, 0.5, 0.25, 0.125],
            )
            self.assertEqual(
                provenance["teacher_search"]["finite_difference_candidates_per_refinement_iteration"],
                {
                    "teacher_gain_schedule": 1,
                    "action_schedule": 1,
                    "duration_schedule": 1,
                    "time_increment_schedule": 1,
                    "joint_gain_action_duration_time_increment_schedule": 1,
                },
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_local_refinement"],
                "mode_preserving_duration_time_increment_continuous_action_gain_and_finite_difference_schedule_search",
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_segment_budget"],
                "preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood",
            )
            self.assertEqual(
                provenance["teacher_search"]["teacher_rollout_horizon"],
                "min_environment_max_steps_and_configured_loop_free_horizon",
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_fraction_after_first_iteration"],
                1.0,
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_probability"],
                "forward_marginalized_action_and_switch_timing_likelihood",
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_switch_timing"],
                "uses_transition_specific_switches_when_available",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_recombination"],
                "top_rho_segment_mode_action_duration_time_increment_centroid",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_recombination_candidate_count"],
                "at_most_one_when_elites_have_loop_free_schedules",
            )
            self.assertEqual(provenance["teacher_search"]["default_elite_distribution_resamples"], 1)
            self.assertEqual(provenance["teacher_search"]["default_elite_distribution_rounds"], 1)
            self.assertEqual(provenance["teacher_search"]["elite_distribution_mean_candidate_per_round"], 1)
            self.assertEqual(provenance["teacher_search"]["elite_distribution_min_action_std"], 0.001)
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_phase"],
                "bounded_cem_style_distribution_refit_top_rho_refresh",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_update"],
                "fit_objective_weighted_gaussian_schedule_distribution_from_current_top_rho_each_round",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_weighting"],
                "softmax_teacher_objective_when_student_available_else_uniform",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_selection_objective"],
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_student_log_probability",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_fit_diagnostics"],
                "serialized_on_distribution_mean_and_sample_traces_with_source_weights_objectives_and_gaussian_parameters",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_refinement_selected_trace_diagnostics"],
                "serialized_on_selected_teacher_traces_with_refreshed_elite_count_sources_objectives_distances_and_kernel_terms",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_refinement_kernel_weighting"],
                "normalized_student_probability_weights_times_exp_negative_loop_free_distance",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_refinement_elite_set"],
                "refreshed_top_rho_after_distribution_rounds",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_refinement_objective"],
                "reward_plus_top_rho_log_probability_distance_kernel",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distance_metric"],
                "normalized_l2_over_teacher_gains_segment_modes_actions_durations_and_time_increments",
            )
            self.assertEqual(provenance["teacher_search"]["elite_distance_action_scale"], "max_abs_segment_action_floor_1")
            self.assertEqual(
                provenance["teacher_search"]["bootstrap_source"],
                "probabilistic_student_prior",
            )
            self.assertEqual(provenance["teacher_search"]["bootstrap_action_std"], 10.0)
            self.assertEqual(
                provenance["teacher_search"]["bootstrap_switch_mean"],
                {"theta_weight": 1.0, "omega_weight": 0.25, "threshold": 0.0},
            )
            self.assertIn("rows", manifest)
            self.assertIn("summary", manifest)
            self.assertIn("summary_note", manifest)
            self.assertIn("survival_metric_note", manifest)
            self.assertIn("psm_artifact_note", manifest)
            config = manifest["rows"][0]["config"]
            self.assertEqual(config["teacher_theta_gain"], 12.5)
            self.assertEqual(config["teacher_omega_gain"], 0.75)
            self.assertEqual(config["teacher_student_iters"], 1)
            self.assertEqual(config["student_em_iters"], 2)
            self.assertEqual(config["student_switch_responsibility_passes"], 2)
            self.assertEqual(config["teacher_student_regularizer"], 0.5)
            self.assertEqual(config["teacher_reward_lambda"], 100.0)
            self.assertEqual(config["teacher_top_rho"], 1)
            self.assertEqual(config["teacher_refinement_steps"], 1)
            self.assertEqual(config["teacher_elite_distribution_resamples"], 3)
            self.assertEqual(config["teacher_elite_distribution_rounds"], 2)
            row_provenance = manifest["rows"][0]["algorithm_provenance"]
            self.assertEqual(manifest["rows"][0]["paper_protocol_status"], psm_status)
            self.assertEqual(row_provenance["probabilistic_student"]["default_em_iters"], 4)
            self.assertEqual(row_provenance["probabilistic_student"]["default_switch_responsibility_passes"], 1)
            self.assertEqual(
                row_provenance["probabilistic_student"]["responsibility_evidence"],
                "action_likelihood_initialization_then_directed_switch_forward_backward_action_refits",
            )
            self.assertTrue(
                row_provenance["probabilistic_student"]["switch_responsibility_passes_are_per_em_iteration"]
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["switch_condition_m_step_schedule"],
                "once_per_student_em_iteration_after_configured_eq10_eq11_passes",
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["initial_switch_before_first_timing_e_step"],
                "fixed_bootstrap_not_data_fit",
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["directed_switch_e_step_schedule"],
                "uses_latest_transition_specific_switches_after_first_bounded_m_step",
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["rollout_parameter_resampling"],
                "on_mode_entry",
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["transition_specific_switches"],
                "separate_fitted_conditions_for_0_to_1_and_1_to_0",
            )
            self.assertEqual(row_provenance["probabilistic_student"]["initial_mode"], 0)
            self.assertEqual(row_provenance["probabilistic_student"]["initial_mode_prior"], "fixed_mode_0")
            self.assertEqual(row_provenance["probabilistic_student"]["min_gaussian_std"], 1e-3)
            self.assertEqual(
                row_provenance["switch_timing"]["duration_units"],
                "segment_elapsed_time_normalized_to_default_cartpole_dt",
            )
            self.assertEqual(
                row_provenance["switch_timing"]["depth2_boolean_probability"],
                "shared_threshold_rectangle_union",
            )
            self.assertEqual(row_provenance["switch_timing"]["coordinate_refinement_steps"], 3)
            self.assertEqual(
                row_provenance["switch_timing"]["transition_specific_m_step"],
                "bounded_separate_0_to_1_and_1_to_0_switch_fits",
            )
            self.assertEqual(row_provenance["switch_timing"]["coordinate_step_decay"], 0.5)
            self.assertEqual(row_provenance["switch_timing"]["finite_difference_gradient_refinement_steps"], 2)
            self.assertEqual(row_provenance["switch_timing"]["finite_difference_gradient_mean_step_fraction"], 0.5)
            self.assertEqual(row_provenance["switch_timing"]["finite_difference_gradient_log_std_step"], 0.25)
            self.assertEqual(row_provenance["switch_timing"]["finite_difference_gradient_epsilon_fraction"], 0.25)
            self.assertEqual(
                row_provenance["switch_timing"]["finite_difference_gradient_backtracking_factors"],
                [1.0, 0.5, 0.25, 0.125],
            )
            self.assertEqual(row_provenance["switch_search"]["max_threshold_candidates"], 64)
            self.assertEqual(row_provenance["teacher_search"]["gain_sample_std_fraction"], 0.10)
            self.assertEqual(
                row_provenance["teacher_search"]["action_refinement_max_candidates_per_segment"],
                2,
            )
            self.assertEqual(
                row_provenance["teacher_search"]["action_refinement_step_fraction"],
                0.25,
            )
            self.assertEqual(row_provenance["teacher_search"]["action_gradient_step_fraction"], 0.10)
            self.assertEqual(row_provenance["teacher_search"]["action_gradient_epsilon_fraction"], 0.05)
            self.assertEqual(row_provenance["teacher_search"]["gain_gradient_step_fraction"], 0.05)
            self.assertEqual(row_provenance["teacher_search"]["gain_gradient_epsilon_fraction"], 0.025)
            self.assertEqual(row_provenance["teacher_search"]["duration_gradient_step"], 1)
            self.assertEqual(row_provenance["teacher_search"]["duration_gradient_epsilon"], 1)
            self.assertEqual(
                row_provenance["teacher_search"]["finite_difference_gradient_backtracking_factors"],
                [1.0, 0.5, 0.25, 0.125],
            )
            self.assertEqual(
                row_provenance["teacher_search"]["finite_difference_candidates_per_refinement_iteration"],
                {
                    "teacher_gain_schedule": 1,
                    "action_schedule": 1,
                    "duration_schedule": 1,
                    "time_increment_schedule": 1,
                    "joint_gain_action_duration_time_increment_schedule": 1,
                },
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_local_refinement"],
                "mode_preserving_duration_time_increment_continuous_action_gain_and_finite_difference_schedule_search",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_segment_budget"],
                "preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["teacher_rollout_horizon"],
                "min_environment_max_steps_and_configured_loop_free_horizon",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_fraction_after_first_iteration"],
                1.0,
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_probability"],
                "forward_marginalized_action_and_switch_timing_likelihood",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_switch_timing"],
                "uses_transition_specific_switches_when_available",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_recombination"],
                "top_rho_segment_mode_action_duration_time_increment_centroid",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_recombination_candidate_count"],
                "at_most_one_when_elites_have_loop_free_schedules",
            )
            self.assertEqual(row_provenance["teacher_search"]["default_elite_distribution_resamples"], 1)
            self.assertEqual(row_provenance["teacher_search"]["default_elite_distribution_rounds"], 1)
            self.assertEqual(row_provenance["teacher_search"]["elite_distribution_mean_candidate_per_round"], 1)
            self.assertEqual(row_provenance["teacher_search"]["elite_distribution_min_action_std"], 0.001)
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_phase"],
                "bounded_cem_style_distribution_refit_top_rho_refresh",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_update"],
                "fit_objective_weighted_gaussian_schedule_distribution_from_current_top_rho_each_round",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_weighting"],
                "softmax_teacher_objective_when_student_available_else_uniform",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_selection_objective"],
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_student_log_probability",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_fit_diagnostics"],
                "serialized_on_distribution_mean_and_sample_traces_with_source_weights_objectives_and_gaussian_parameters",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_refinement_selected_trace_diagnostics"],
                "serialized_on_selected_teacher_traces_with_refreshed_elite_count_sources_objectives_distances_and_kernel_terms",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_refinement_kernel_weighting"],
                "normalized_student_probability_weights_times_exp_negative_loop_free_distance",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_refinement_elite_set"],
                "refreshed_top_rho_after_distribution_rounds",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_refinement_objective"],
                "reward_plus_top_rho_log_probability_distance_kernel",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distance_metric"],
                "normalized_l2_over_teacher_gains_segment_modes_actions_durations_and_time_increments",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distance_action_scale"],
                "max_abs_segment_action_floor_1",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["bootstrap_source"],
                "probabilistic_student_prior",
            )
            self.assertEqual(row_provenance["teacher_search"]["bootstrap_action_std"], 10.0)
            self.assertEqual(
                row_provenance["teacher_search"]["bootstrap_switch_mean"],
                {"theta_weight": 1.0, "omega_weight": 0.25, "threshold": 0.0},
            )
            self.assertTrue(os.path.exists(manifest["rows"][0]["metrics_output"]))

    def test_psm_artifact_consistency_rejects_trace_sidecar_mismatch(self):
        metrics = {
            "command": "python runner.py",
            "config": {"teacher_student_iters": 1},
            "num_traces": 1,
            "trace_summary": {"count": 1},
            "adaptive_teacher_summary": [{"iteration": 1, "trace_count": 1}],
            "synthesis_history": [
                {
                    "iteration": 1,
                    "trace_summary": {"count": 1},
                    "evaluation": {
                        "train": {"success_rate": 1.0},
                        "test": {"success_rate": 0.0},
                    },
                }
            ],
            "paper_protocol_status": {"synthesized_by_current_algorithm": True},
            "train": {"success_rate": 1.0},
            "test": {"success_rate": 0.0},
        }
        trace_payload = {
            "command": "python runner.py",
            "config": {"teacher_student_iters": 1},
            "num_traces": 1,
            "traces": [{"actions": [1.0]}],
            "trace_history": [
                {
                    "iteration": 1,
                    "num_traces": 1,
                    "traces": [{"actions": [-1.0]}],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "final trace_history traces disagree"):
            validate_psm_artifact_consistency(metrics, trace_payload)

    def test_nonquick_psm_profile_uses_full_training_horizon_loop_free_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = run_psm(
                seed=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=False,
                outdir=Path(tmpdir),
                teacher_overrides={
                    "num_initial_states": 1,
                    "candidate_rollouts": 1,
                    "teacher_student_iters": 1,
                    "teacher_top_rho": 1,
                    "teacher_refinement_steps": 0,
                    "parallel_trace_workers": 2,
                    "parallel_switch_workers": 2,
                },
            )

        config = row["config"]
        self.assertEqual(config["segment_steps"], 1)
        self.assertEqual(config["segments_per_trace"], 250)
        self.assertEqual(config["segment_steps"] * config["segments_per_trace"], 250)
        self.assertEqual(config["parallel_trace_workers"], 2)
        self.assertEqual(config["parallel_switch_workers"], 2)
        self.assertEqual(row["paper_protocol_status"]["selected_teacher_parallel_trace_workers"], 2)
        self.assertEqual(row["paper_protocol_status"]["effective_teacher_parallel_trace_initial_states"], 1)
        self.assertEqual(row["paper_protocol_status"]["effective_teacher_parallel_trace_slots"], 1)
        self.assertFalse(row["paper_protocol_status"]["uses_parallel_teacher_trace_optimization"])
        self.assertEqual(row["paper_protocol_status"]["selected_student_parallel_switch_workers"], 2)
        self.assertEqual(row["paper_protocol_status"]["effective_student_parallel_switch_slots"], 2)
        self.assertTrue(row["paper_protocol_status"]["uses_parallel_student_switch_optimization"])

    def test_quick_runner_can_include_direct_opt_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--include-direct-opt",
                    "--seeds",
                    "0",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(os.path.join(tmpdir, "cartpole_results.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["policy"] for row in rows], ["Programmatic state machine", "Direct-Opt diagnostic"])
            direct_row = rows[1]
            self.assertIn("test_steps", direct_row)
            self.assertIn("test_survival_seconds", direct_row)
            self.assertGreater(float(direct_row["test_steps"]), 0.0)
            self.assertTrue(os.path.exists(direct_row["metrics_output"]))

            with open(direct_row["metrics_output"], encoding="utf-8") as handle:
                direct_metrics = json.load(handle)
            self.assertIn("run_cartpole_reproduction.py", direct_metrics["command"])
            self.assertIn("--include-direct-opt", direct_metrics["command"])
            self.assertEqual(direct_metrics["algorithm_provenance"]["paper_baseline"], "Direct-Opt")
            self.assertTrue(direct_metrics["algorithm_provenance"]["not_paper_scale"])
            self.assertEqual(direct_metrics["algorithm_provenance"]["batch_refinement"], "seed_each_batch_from_best_so_far_and_restart_on_stall")
            self.assertEqual(
                direct_metrics["algorithm_provenance"]["search_stopping"],
                "stop_after_training_solution_or_parallel_chunk_or_time_limit",
            )
            self.assertEqual(
                direct_metrics["algorithm_provenance"]["switch_search_space"],
                "linear_theta_omega_grid_plus_bounded_boolean_tree_predicates_plus_bounded_continuous_one_hot_leaf_depth2_mixtures",
            )
            self.assertEqual(direct_metrics["algorithm_provenance"]["boolean_tree_depth"], 2)
            self.assertIn("one-hot", direct_metrics["algorithm_provenance"]["one_hot_switch_encoding"])
            self.assertEqual(direct_metrics["algorithm_provenance"]["paper_time_limit_seconds"], 7200)
            self.assertEqual(
                direct_metrics["algorithm_provenance"]["local_parallel_threads"],
                "configurable_via_parallel_threads",
            )
            direct_status = direct_metrics["paper_protocol_status"]
            self.assertEqual(direct_status["paper_baseline"], "Direct-Opt")
            self.assertEqual(direct_status["selected_parallel_threads"], 1)
            self.assertIsNone(direct_status["selected_time_limit_seconds"])
            self.assertFalse(direct_status["uses_paper_batch_size"])
            self.assertFalse(direct_status["uses_paper_parallel_threads"])
            self.assertFalse(direct_status["uses_paper_time_limit"])
            self.assertFalse(direct_status["full_continuous_one_hot_switch_grammar"])
            self.assertTrue(direct_status["bounded_continuous_one_hot_switch_relaxation"])
            self.assertTrue(direct_status["stops_when_training_solution_found"])
            self.assertTrue(direct_status["optimizes_combined_reward_over_selected_initial_states"])
            self.assertTrue(direct_status["optimizes_combined_reward_over_all_selected_initial_states"])
            self.assertFalse(direct_status["optimizes_full_initial_state_distribution"])
            self.assertFalse(direct_status["direct_opt_protocol_requirements"]["paper_batch_size_and_batch_refinement"])
            self.assertIn(
                "paper_batch_size_and_batch_refinement",
                direct_status["missing_direct_opt_protocol_requirements"],
            )
            self.assertIn(
                "full_continuous_one_hot_switch_grammar",
                direct_status["missing_direct_opt_protocol_requirements"],
            )
            self.assertFalse(direct_status["uses_full_test_horizon"])
            self.assertFalse(direct_status["uses_paper_eval_rollouts"])
            self.assertFalse(direct_status["paper_scale_direct_opt_protocol"])
            self.assertEqual(direct_metrics["config"]["quick"], True)
            self.assertEqual(direct_metrics["config"]["batch_size"], 2)
            self.assertEqual(direct_metrics["config"]["parallel_threads"], 1)
            self.assertIsNone(direct_metrics["config"]["time_limit_seconds"])
            self.assertEqual(direct_metrics["config"]["batch_refinement_rounds"], 1)
            self.assertEqual(direct_metrics["config"]["local_refinement_steps"], 1)
            self.assertEqual(direct_metrics["search_diagnostics"]["batch_count"], 1)
            self.assertEqual(direct_metrics["search_diagnostics"]["parallel_threads"], 1)
            self.assertFalse(direct_metrics["search_diagnostics"]["uses_parallel_candidate_evaluation"])
            self.assertIsNone(direct_metrics["search_diagnostics"]["time_limit_seconds"])
            self.assertFalse(direct_metrics["search_diagnostics"]["time_limit_reached"])
            self.assertTrue(direct_metrics["search_diagnostics"]["solution_found"])
            self.assertEqual(direct_metrics["search_diagnostics"]["solution_found_phase"], "grid")
            self.assertEqual(direct_metrics["search_diagnostics"]["grid_candidates"], 86)
            self.assertEqual(direct_metrics["search_diagnostics"]["batch_refinement_candidates"], 0)
            self.assertEqual(direct_metrics["search_diagnostics"]["boolean_stump_candidates"], 0)
            self.assertEqual(direct_metrics["search_diagnostics"]["boolean_depth2_candidates"], 0)
            self.assertEqual(direct_metrics["search_diagnostics"]["continuous_one_hot_leaf_candidates"], 0)
            self.assertEqual(direct_metrics["search_diagnostics"]["continuous_one_hot_depth2_candidates"], 0)
            self.assertEqual(direct_metrics["search_diagnostics"]["continuous_one_hot_candidates"], 0)
            self.assertEqual(
                direct_metrics["search_diagnostics"]["boolean_candidates_with_one_hot_metadata"],
                direct_metrics["search_diagnostics"]["boolean_stump_candidates"]
                + direct_metrics["search_diagnostics"]["boolean_depth2_candidates"],
            )
            self.assertEqual(
                direct_metrics["search_diagnostics"]["boolean_candidates_with_appendix_b3_vertex_metadata"],
                direct_metrics["search_diagnostics"]["boolean_candidates_with_one_hot_metadata"],
            )
            self.assertEqual(
                direct_metrics["search_diagnostics"]["evaluated_candidates_units"],
                "candidate_evaluation_calls",
            )
            self.assertGreater(
                direct_metrics["search_diagnostics"]["train_rollout_evaluations"],
                direct_metrics["search_diagnostics"]["candidate_evaluation_calls"],
            )
            self.assertEqual(direct_metrics["search_diagnostics"]["batch_local_evaluations"], 0)
            self.assertIn("steps_mean", direct_metrics["train"])
            self.assertIn("survival_seconds_mean", direct_metrics["train"])
            self.assertIn("steps_mean", direct_metrics["test"])
            self.assertIn("survival_seconds_mean", direct_metrics["test"])

            with open(os.path.join(tmpdir, "cartpole_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["include_direct_opt"])
            self.assertEqual(manifest["direct_opt_parallel_threads"], 1)
            self.assertIsNone(manifest["direct_opt_time_limit_seconds"])
            direct_manifest_row = manifest["rows"][1]
            self.assertEqual(direct_manifest_row["algorithm_provenance"]["baseline"], "direct_opt")
            self.assertEqual(direct_manifest_row["paper_protocol_status"], direct_status)
            direct_evidence = manifest["direct_opt_evidence"]
            self.assertEqual(direct_evidence, manifest["paper_protocol_status"]["direct_opt_evidence"])
            self.assertTrue(direct_evidence["requested"])
            self.assertEqual(direct_evidence["rows_recorded"], 1)
            self.assertTrue(direct_evidence["records_rows_for_selected_seeds"])
            self.assertTrue(direct_evidence["covers_selected_seed_set"])
            self.assertFalse(direct_evidence["paper_scale_direct_opt_protocol"])
            self.assertFalse(manifest["paper_protocol_status"]["full_direct_opt_protocol"])
            self.assertIn(
                "paper_scale_direct_opt_protocol_per_row",
                direct_evidence["missing_direct_opt_evidence_requirements"],
            )
            self.assertIn("direct_opt_artifact_note", manifest)

    def test_quick_runner_passes_direct_opt_parallel_threads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--seeds",
                    "0",
                    "--include-direct-opt",
                    "--direct-opt-parallel-threads",
                    "2",
                    "--direct-opt-time-limit-seconds",
                    "7200",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(os.path.join(tmpdir, "cartpole_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            direct_row = next(row for row in manifest["rows"] if row["policy"] == "Direct-Opt diagnostic")
            with open(direct_row["metrics_output"], encoding="utf-8") as handle:
                direct_metrics = json.load(handle)

        self.assertEqual(manifest["direct_opt_parallel_threads"], 2)
        self.assertEqual(manifest["direct_opt_time_limit_seconds"], 7200)
        self.assertEqual(direct_metrics["config"]["parallel_threads"], 2)
        self.assertEqual(direct_metrics["config"]["time_limit_seconds"], 7200)
        self.assertEqual(direct_metrics["search_diagnostics"]["parallel_threads"], 2)
        self.assertTrue(direct_metrics["search_diagnostics"]["uses_parallel_candidate_evaluation"])
        self.assertEqual(direct_row["paper_protocol_status"]["selected_parallel_threads"], 2)
        self.assertEqual(direct_row["paper_protocol_status"]["selected_time_limit_seconds"], 7200)
        self.assertTrue(direct_row["paper_protocol_status"]["uses_paper_time_limit"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for PPO artifact checks")
    def test_quick_runner_with_ppo_writes_checkpoints_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--include-ppo",
                    "--seeds",
                    "0",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--ppo-eval-interval",
                    "32",
                    "--outdir",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(os.path.join(tmpdir, "cartpole_results.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            ppo_rows = [row for row in rows if row["policy"] in {"PPO MLP", "PPO-LSTM"}]
            self.assertEqual(len(ppo_rows), 2)
            for row in ppo_rows:
                self.assertIn("test_steps", row)
                self.assertIn("test_survival_seconds", row)
                self.assertGreater(float(row["test_steps"]), 0.0)
                self.assertTrue(os.path.exists(row["checkpoint"]))
                self.assertTrue(os.path.exists(row["metrics_output"]))
                with open(row["metrics_output"], encoding="utf-8") as handle:
                    metrics = json.load(handle)
                self.assertIn("run_cartpole_reproduction.py", metrics["command"])
                self.assertIn("--include-ppo", metrics["command"])
                self.assertEqual(row["command"], metrics["command"])
                self.assertEqual(metrics["config"]["eval_test_max_steps"], 20)
                self.assertEqual(metrics["config"]["eval_interval"], 32)
                self.assertGreaterEqual(len(metrics["eval_history"]), 1)
                self.assertGreaterEqual(len(metrics["update_history"]), 1)
                self.assertIn("horizon_truncations", metrics["update_history"][0])
                self.assertIn("selected_result", metrics)
                self.assertIn("test_steps_mean", metrics["selected_result"])
                self.assertIn("test_survival_seconds_mean", metrics["selected_result"])
                self.assertIn("test_steps_mean", metrics["eval_history"][0])

            with open(os.path.join(tmpdir, "cartpole_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertIn("ppo_artifact_note", manifest)
            self.assertEqual(manifest["ppo_eval_interval"], 32)
            manifest_ppo_rows = [row for row in manifest["rows"] if row["policy"] in {"PPO MLP", "PPO-LSTM"}]
            self.assertEqual(len(manifest_ppo_rows), 2)
            for row in manifest_ppo_rows:
                self.assertTrue(os.path.exists(row["checkpoint"]))
                self.assertTrue(os.path.exists(row["metrics_output"]))
                with open(row["metrics_output"], encoding="utf-8") as handle:
                    metrics = json.load(handle)
                self.assertEqual(row["paper_protocol_status"], metrics["paper_protocol_status"])
                self.assertFalse(row["paper_protocol_status"]["paper_timestep_budget"])
                self.assertFalse(row["paper_protocol_status"]["paper_test_horizon"])
                self.assertFalse(row["paper_protocol_status"]["paper_scale_baseline_protocol"])


if __name__ == "__main__":
    unittest.main()
