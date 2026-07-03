import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "scripts", "run_cartpole_reproduction.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_cartpole_reproduction import HAS_TORCH, run_psm, summarize_rows  # noqa: E402


class CartpoleReproductionRunnerTest(unittest.TestCase):
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
            with open(rows[0]["metrics_output"], encoding="utf-8") as handle:
                psm_metrics = json.load(handle)
            self.assertEqual(psm_metrics["config"]["teacher_theta_gain"], 12.5)
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
            self.assertFalse(psm_status["full_probabilistic_adaptive_teaching"])
            self.assertFalse(psm_status["full_continuous_switch_m_step"])
            self.assertFalse(psm_status["full_cem_teacher_optimizer"])
            self.assertFalse(psm_status["paper_scale_result"])
            self.assertEqual(psm_status["student_em_iters"], 2)
            self.assertEqual(psm_status["student_switch_responsibility_passes"], 2)
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
            manifest_psm_status = manifest["psm_paper_protocol_status"]
            self.assertEqual(manifest_psm_status, psm_status)
            provenance = manifest["psm_algorithm_provenance"]
            self.assertEqual(provenance["probabilistic_student"]["default_em_iters"], 4)
            self.assertEqual(provenance["probabilistic_student"]["default_switch_responsibility_passes"], 1)
            self.assertEqual(
                provenance["probabilistic_student"]["responsibility_evidence"],
                "action_likelihood_initialization_then_alternating_switch_timing_forward_backward",
            )
            self.assertTrue(provenance["probabilistic_student"]["switch_responsibility_passes_are_per_em_iteration"])
            self.assertEqual(provenance["probabilistic_student"]["rollout_parameter_resampling"], "on_mode_entry")
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
                },
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_local_refinement"],
                "duration_time_increment_continuous_action_and_finite_difference_schedule_search",
            )
            self.assertEqual(
                provenance["teacher_search"]["student_sample_segment_budget"],
                "chunk_sampled_actions_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood",
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
                provenance["teacher_search"]["elite_recombination"],
                "top_rho_segment_action_duration_time_increment_centroid",
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
                "bounded_cem_style_top_rho_refresh",
            )
            self.assertEqual(
                provenance["teacher_search"]["elite_distribution_selection_objective"],
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_student_log_probability",
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
                "l2_over_teacher_gains_segment_actions_durations_and_time_increments",
            )
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
                "action_likelihood_initialization_then_alternating_switch_timing_forward_backward",
            )
            self.assertTrue(
                row_provenance["probabilistic_student"]["switch_responsibility_passes_are_per_em_iteration"]
            )
            self.assertEqual(
                row_provenance["probabilistic_student"]["rollout_parameter_resampling"],
                "on_mode_entry",
            )
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
                },
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_local_refinement"],
                "duration_time_increment_continuous_action_and_finite_difference_schedule_search",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["student_sample_segment_budget"],
                "chunk_sampled_actions_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood",
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
                row_provenance["teacher_search"]["elite_recombination"],
                "top_rho_segment_action_duration_time_increment_centroid",
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
                "bounded_cem_style_top_rho_refresh",
            )
            self.assertEqual(
                row_provenance["teacher_search"]["elite_distribution_selection_objective"],
                "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_student_log_probability",
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
                "l2_over_teacher_gains_segment_actions_durations_and_time_increments",
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
                },
            )

        config = row["config"]
        self.assertEqual(config["segment_steps"], 1)
        self.assertEqual(config["segments_per_trace"], 250)
        self.assertEqual(config["segment_steps"] * config["segments_per_trace"], 250)

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
            self.assertEqual(direct_metrics["algorithm_provenance"]["paper_baseline"], "Direct-Opt")
            self.assertTrue(direct_metrics["algorithm_provenance"]["not_paper_scale"])
            self.assertEqual(direct_metrics["algorithm_provenance"]["batch_refinement"], "seed_each_batch_from_best_so_far_and_restart_on_stall")
            self.assertEqual(
                direct_metrics["algorithm_provenance"]["switch_search_space"],
                "linear_theta_omega_grid_plus_bounded_boolean_tree_predicates_with_one_hot_metadata",
            )
            self.assertEqual(direct_metrics["algorithm_provenance"]["boolean_tree_depth"], 2)
            self.assertIn("one-hot", direct_metrics["algorithm_provenance"]["one_hot_switch_encoding"])
            self.assertEqual(direct_metrics["algorithm_provenance"]["paper_time_limit_seconds"], 7200)
            direct_status = direct_metrics["paper_protocol_status"]
            self.assertEqual(direct_status["paper_baseline"], "Direct-Opt")
            self.assertFalse(direct_status["uses_paper_batch_size"])
            self.assertFalse(direct_status["uses_paper_parallel_threads"])
            self.assertFalse(direct_status["uses_paper_time_limit"])
            self.assertFalse(direct_status["full_continuous_one_hot_switch_grammar"])
            self.assertTrue(direct_status["optimizes_combined_reward_over_selected_initial_states"])
            self.assertFalse(direct_status["uses_full_test_horizon"])
            self.assertFalse(direct_status["uses_paper_eval_rollouts"])
            self.assertFalse(direct_status["paper_scale_direct_opt_protocol"])
            self.assertEqual(direct_metrics["config"]["quick"], True)
            self.assertEqual(direct_metrics["config"]["batch_size"], 2)
            self.assertEqual(direct_metrics["config"]["batch_refinement_rounds"], 1)
            self.assertEqual(direct_metrics["config"]["local_refinement_steps"], 1)
            self.assertEqual(direct_metrics["search_diagnostics"]["batch_count"], 1)
            self.assertEqual(direct_metrics["search_diagnostics"]["batch_refinement_candidates"], 1)
            self.assertEqual(direct_metrics["search_diagnostics"]["boolean_stump_candidates"], 24)
            self.assertGreater(direct_metrics["search_diagnostics"]["boolean_depth2_candidates"], 0)
            self.assertEqual(
                direct_metrics["search_diagnostics"]["boolean_candidates_with_one_hot_metadata"],
                direct_metrics["search_diagnostics"]["boolean_stump_candidates"]
                + direct_metrics["search_diagnostics"]["boolean_depth2_candidates"],
            )
            self.assertGreater(direct_metrics["search_diagnostics"]["batch_local_evaluations"], 0)
            self.assertIn("steps_mean", direct_metrics["train"])
            self.assertIn("survival_seconds_mean", direct_metrics["train"])
            self.assertIn("steps_mean", direct_metrics["test"])
            self.assertIn("survival_seconds_mean", direct_metrics["test"])

            with open(os.path.join(tmpdir, "cartpole_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["include_direct_opt"])
            direct_manifest_row = manifest["rows"][1]
            self.assertEqual(direct_manifest_row["algorithm_provenance"]["baseline"], "direct_opt")
            self.assertEqual(direct_manifest_row["paper_protocol_status"], direct_status)
            self.assertIn("direct_opt_artifact_note", manifest)

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
