import json
import math
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_psm.py")
sys.path.insert(0, os.path.join(ROOT, "src"))

from cartpole_synthesis import Depth2Switch, GaussianScalar, ProbabilisticCartpoleStudent  # noqa: E402
from train_cartpole_psm import summarize_student  # noqa: E402


class CartpolePSMCliTest(unittest.TestCase):
    def test_summarize_student_reports_responsibility_confidence(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 1.0),
                1: GaussianScalar(10.0, 1.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.9, 0.1), (0.4, 0.6), (0.2, 0.8)],
        )

        summary = summarize_student(student)["responsibility_summary"]

        self.assertEqual(summary["segments"], 3)
        self.assertEqual(summary["hard_mode_0_count"], 1)
        self.assertEqual(summary["hard_mode_1_count"], 2)
        self.assertEqual(summary["ambiguous_segment_count"], 1)
        self.assertEqual(summary["ambiguous_segment_threshold"], 0.75)
        self.assertAlmostEqual(summary["mean_max_responsibility"], (0.9 + 0.6 + 0.8) / 3.0)
        self.assertAlmostEqual(summary["min_max_responsibility"], 0.6)
        expected_entropy = (
            -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
            -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
            -(0.2 * math.log(0.2) + 0.8 * math.log(0.8))
        ) / 3.0
        self.assertAlmostEqual(summary["mean_entropy_nats"], expected_entropy)
        self.assertAlmostEqual(summary["max_entropy_nats"], -(0.4 * math.log(0.4) + 0.6 * math.log(0.6)))

    def test_cli_default_teacher_profile_matches_cartpole_training_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "psm_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--num-initial-states",
                    "1",
                    "--candidate-rollouts",
                    "1",
                    "--teacher-student-iters",
                    "1",
                    "--teacher-top-rho",
                    "1",
                    "--teacher-refinement-steps",
                    "0",
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

        self.assertEqual(metrics["config"]["segment_steps"], 1)
        self.assertEqual(metrics["config"]["segments_per_trace"], 250)
        self.assertEqual(
            metrics["config"]["segment_steps"] * metrics["config"]["segments_per_trace"],
            250,
        )

    def test_cli_records_parallel_trace_worker_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "psm_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--num-initial-states",
                    "2",
                    "--candidate-rollouts",
                    "1",
                    "--segments-per-trace",
                    "2",
                    "--teacher-student-iters",
                    "1",
                    "--teacher-top-rho",
                    "1",
                    "--teacher-refinement-steps",
                    "0",
                    "--parallel-trace-workers",
                    "2",
                    "--parallel-switch-workers",
                    "2",
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

        status = metrics["paper_protocol_status"]
        self.assertEqual(metrics["config"]["parallel_trace_workers"], 2)
        self.assertEqual(metrics["config"]["parallel_switch_workers"], 2)
        self.assertEqual(status["selected_teacher_parallel_trace_workers"], 2)
        self.assertEqual(status["effective_teacher_parallel_trace_workers"], 2)
        self.assertEqual(status["effective_teacher_parallel_trace_initial_states"], 2)
        self.assertEqual(status["effective_teacher_parallel_trace_slots"], 2)
        self.assertEqual(status["paper_teacher_parallel_threads"], 10)
        self.assertTrue(status["uses_parallel_teacher_trace_optimization"])
        self.assertFalse(status["uses_paper_teacher_parallel_threads"])
        self.assertEqual(status["selected_student_parallel_switch_workers"], 2)
        self.assertEqual(status["effective_student_parallel_switch_workers"], 2)
        self.assertEqual(status["student_transition_switch_fit_count"], 2)
        self.assertEqual(status["effective_student_parallel_switch_slots"], 2)
        self.assertEqual(status["paper_student_parallel_threads"], 10)
        self.assertTrue(status["uses_parallel_student_switch_optimization"])
        self.assertFalse(status["uses_paper_student_parallel_threads"])

    def test_cli_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "psm_metrics.json")
            traces_path = os.path.join(tmpdir, "psm_traces.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--num-initial-states",
                    "2",
                    "--candidate-rollouts",
                    "2",
                    "--segment-steps",
                    "2",
                    "--segments-per-trace",
                    "4",
                    "--teacher-theta-gain",
                    "12.5",
                    "--teacher-omega-gain",
                    "0.75",
                    "--teacher-student-iters",
                    "2",
                    "--student-em-iters",
                    "2",
                    "--student-switch-responsibility-passes",
                    "2",
                    "--teacher-student-regularizer",
                    "0.5",
                    "--teacher-reward-lambda",
                    "100",
                    "--teacher-top-rho",
                    "1",
                    "--teacher-refinement-steps",
                    "1",
                    "--teacher-elite-distribution-resamples",
                    "3",
                    "--teacher-elite-distribution-rounds",
                    "2",
                    "--eval-rollouts",
                    "1",
                    "--test-max-steps",
                    "20",
                    "--metrics-output",
                    metrics_path,
                    "--traces-output",
                    traces_path,
                ],
                check=True,
                cwd=ROOT,
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
            with open(traces_path, encoding="utf-8") as handle:
                trace_artifact = json.load(handle)

        self.assertEqual(metrics["config"]["num_initial_states"], 2)
        self.assertEqual(metrics["config"]["teacher_theta_gain"], 12.5)
        self.assertEqual(metrics["config"]["teacher_omega_gain"], 0.75)
        self.assertEqual(metrics["config"]["teacher_student_iters"], 2)
        self.assertEqual(metrics["config"]["student_em_iters"], 2)
        self.assertEqual(metrics["config"]["student_switch_responsibility_passes"], 2)
        self.assertEqual(metrics["config"]["teacher_student_regularizer"], 0.5)
        self.assertEqual(metrics["config"]["teacher_reward_lambda"], 100.0)
        self.assertEqual(metrics["config"]["teacher_top_rho"], 1)
        self.assertEqual(metrics["config"]["teacher_refinement_steps"], 1)
        self.assertEqual(metrics["config"]["teacher_elite_distribution_resamples"], 3)
        self.assertEqual(metrics["config"]["teacher_elite_distribution_rounds"], 2)
        self.assertEqual(metrics["config"]["parallel_trace_workers"], 1)
        self.assertEqual(metrics["config"]["parallel_switch_workers"], 1)
        provenance = metrics["algorithm_provenance"]
        self.assertEqual(provenance["probabilistic_student"]["default_em_iters"], 4)
        self.assertEqual(provenance["probabilistic_student"]["default_switch_responsibility_passes"], 1)
        self.assertEqual(
            provenance["probabilistic_student"]["responsibility_evidence"],
            "action_likelihood_initialization_then_alternating_switch_timing_forward_backward",
        )
        self.assertTrue(provenance["probabilistic_student"]["switch_responsibility_passes_are_per_em_iteration"])
        self.assertEqual(provenance["probabilistic_student"]["rollout_parameter_resampling"], "on_mode_entry")
        self.assertEqual(provenance["probabilistic_student"]["initial_mode"], 0)
        self.assertEqual(provenance["probabilistic_student"]["initial_mode_prior"], "fixed_mode_0")
        self.assertEqual(provenance["probabilistic_student"]["min_gaussian_std"], 1e-3)
        self.assertEqual(provenance["switch_timing"]["std_steps"], 2.0)
        self.assertEqual(
            provenance["switch_timing"]["duration_units"],
            "segment_elapsed_time_normalized_to_default_cartpole_dt",
        )
        self.assertTrue(provenance["switch_timing"]["final_segment_stay_evidence"])
        self.assertTrue(provenance["switch_timing"]["scalar_threshold_uses_shared_sample"])
        self.assertEqual(
            provenance["switch_timing"]["depth2_boolean_probability"],
            "shared_threshold_rectangle_union",
        )
        self.assertEqual(provenance["switch_timing"]["std_refinement_multipliers"], [0.5, 1.0, 2.0])
        self.assertEqual(provenance["switch_timing"]["coordinate_refinement_steps"], 3)
        self.assertEqual(provenance["switch_timing"]["coordinate_mean_step_fraction"], 0.25)
        self.assertAlmostEqual(
            provenance["switch_timing"]["coordinate_log_std_initial_step"],
            math.log(2.0),
        )
        self.assertEqual(provenance["switch_timing"]["coordinate_step_decay"], 0.5)
        self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_refinement_steps"], 2)
        self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_mean_step_fraction"], 0.5)
        self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_log_std_step"], 0.25)
        self.assertEqual(provenance["switch_timing"]["finite_difference_gradient_epsilon_fraction"], 0.25)
        self.assertEqual(
            provenance["switch_timing"]["finite_difference_gradient_backtracking_factors"],
            [1.0, 0.5, 0.25, 0.125],
        )
        self.assertTrue(provenance["switch_timing"]["structure_rescore_uses_pair_posteriors"])
        self.assertEqual(
            provenance["probabilistic_student"]["transition_specific_switches"],
            "separate_fitted_conditions_for_0_to_1_and_1_to_0",
        )
        self.assertEqual(
            provenance["switch_timing"]["transition_specific_m_step"],
            "bounded_separate_0_to_1_and_1_to_0_switch_fits",
        )
        self.assertEqual(provenance["switch_search"]["boolean_tree_depth"], 2)
        self.assertTrue(provenance["switch_search"]["greedy_second_predicate_expands_switch_and_no_switch_leaves"])
        self.assertEqual(provenance["switch_search"]["greedy_second_predicate_prefilter_top_k"], 32)
        self.assertIn(50.0, provenance["switch_search"]["oblique_theta_weights"])
        self.assertEqual(provenance["switch_search"]["max_threshold_candidates"], 64)
        self.assertEqual(provenance["switch_search"]["distribution_rescore_top_k"], 32)
        self.assertEqual(
            provenance["switch_search"]["prefilter_objective_order"][1],
            "eq12_style_timing_loss",
        )
        self.assertEqual(
            provenance["switch_search"]["selection_objective_order"][0],
            "responsibility_weighted_label_loss",
        )
        self.assertEqual(
            provenance["switch_search"]["selection_objective_order"][1],
            "bounded_eq12_style_distribution_loss",
        )
        self.assertEqual(
            provenance["switch_search"]["structure_label_objective"],
            "responsibility_weighted_expected_label_loss_when_available_else_hard_label_mistakes",
        )
        self.assertEqual(
            provenance["switch_search"]["structure_label_observations"],
            "nonboundary_segment_observations_boundary_observations_scored_by_timing_loss",
        )
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
        self.assertTrue(provenance["teacher_search"]["elite_distribution_samples_teacher_gains"])
        self.assertEqual(
            provenance["teacher_search"]["student_sample_fraction_after_first_iteration"],
            1.0,
        )
        self.assertEqual(
            provenance["teacher_search"]["student_sample_probability"],
            "forward_marginalized_action_and_switch_timing_likelihood",
        )
        self.assertEqual(
            provenance["teacher_search"]["student_sample_local_refinement"],
            "mode_preserving_duration_time_increment_continuous_action_gain_and_finite_difference_schedule_search",
        )
        self.assertEqual(
            provenance["teacher_search"]["student_sample_segment_budget"],
            "preserve_sampled_mode_action_runs_split_by_max_segment_duration_then_reroll_loop_free_trace_and_recompute_likelihood",
        )
        self.assertEqual(provenance["teacher_search"]["candidate_rollout_count"], "configurable_via_candidate_rollouts")
        self.assertEqual(provenance["teacher_search"]["paper_top_rho"], 10)
        self.assertEqual(
            provenance["teacher_search"]["top_rho_selection"],
            "sort_by_teacher_objective_and_keep_teacher_top_rho_elites",
        )
        self.assertEqual(
            provenance["teacher_search"]["phase_one_objective"],
            "teacher_reward_lambda_times_reward_plus_teacher_student_regularizer_times_log_p_trace_under_student",
        )
        self.assertEqual(
            provenance["teacher_search"]["teacher_rollout_horizon"],
            "min_environment_max_steps_and_configured_loop_free_horizon",
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
            provenance["teacher_search"]["elite_distribution_parameters"],
            [
                "teacher_gain_schedule",
                "segment_action_schedule",
                "integer_segment_duration_schedule",
                "segment_time_increment_schedule",
                "majority_segment_mode_schedule",
            ],
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
            provenance["teacher_search"]["selected_trace_objective_metrics"],
            ["teacher_objective", "teacher_refinement_objective"],
        )
        self.assertEqual(
            provenance["teacher_search"]["elite_distance_metric"],
            "normalized_l2_over_teacher_gains_segment_modes_actions_durations_and_time_increments",
        )
        self.assertEqual(provenance["teacher_search"]["elite_distance_action_scale"], "max_abs_segment_action_floor_1")
        self.assertEqual(
            provenance["teacher_search"]["elite_distance_duration_scale_floor"],
            1.0,
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
        self.assertEqual(provenance["teacher_search"]["bootstrap_switch_std"], 1.0)
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["paper_eval_rollouts"], 1000)
        self.assertFalse(metrics["uses_paper_eval_rollouts"])
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["space_spec"]["initial_state_distribution"]["low"], -0.05)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertEqual(metrics["traces_output"], traces_path)
        self.assertEqual(trace_artifact["num_traces"], metrics["num_traces"])
        self.assertEqual(trace_artifact["config"], metrics["config"])
        self.assertEqual(len(trace_artifact["traces"]), metrics["num_traces"])
        self.assertEqual(len(trace_artifact["trace_history"]), metrics["config"]["teacher_student_iters"])
        self.assertEqual([entry["iteration"] for entry in trace_artifact["trace_history"]], [1, 2])
        self.assertEqual(
            trace_artifact["trace_history"][-1]["num_traces"],
            metrics["num_traces"],
        )
        self.assertEqual(
            trace_artifact["trace_history"][-1]["traces"],
            trace_artifact["traces"],
        )
        first_trace = trace_artifact["traces"][0]
        self.assertIn("observations", first_trace)
        self.assertIn("actions", first_trace)
        self.assertIn("mode_labels", first_trace)
        self.assertIn("segment_actions", first_trace)
        self.assertIn("segment_durations", first_trace)
        self.assertIn("segment_time_increments", first_trace)
        self.assertIn("teacher_objective", first_trace)
        self.assertIn("teacher_refinement_objective", first_trace)
        self.assertIn("teacher_refinement_elite_summary", first_trace)
        first_elite_summary = first_trace["teacher_refinement_elite_summary"]
        self.assertEqual(first_elite_summary["top_rho"], metrics["config"]["teacher_top_rho"])
        self.assertGreaterEqual(first_elite_summary["elite_count"], 1)
        self.assertIn("source_counts", first_elite_summary)
        self.assertIn("selected_distance_to_nearest_elite", first_elite_summary)
        self.assertIn("elite_probability_weights", first_elite_summary)
        self.assertAlmostEqual(sum(first_elite_summary["elite_probability_weights"]), 1.0)
        self.assertIn("selected_kernel_component_weights", first_elite_summary)
        self.assertAlmostEqual(sum(first_elite_summary["selected_kernel_component_weights"]), 1.0)
        serialized_distribution_fits = [
            trace.get("elite_distribution_fit")
            for history_entry in trace_artifact["trace_history"]
            for trace in history_entry["traces"]
            if trace.get("elite_distribution_fit") is not None
        ]
        self.assertTrue(serialized_distribution_fits)
        first_distribution_fit = serialized_distribution_fits[0]
        self.assertIn(
            first_distribution_fit["weighting"],
            {"softmax_teacher_objective", "uniform_no_student"},
        )
        self.assertAlmostEqual(sum(first_distribution_fit["source_weights"]), 1.0)
        self.assertEqual(
            len(first_distribution_fit["source_weights"]),
            first_distribution_fit["source_count"],
        )
        self.assertEqual(
            len(first_distribution_fit["source_teacher_objectives"]),
            first_distribution_fit["source_count"],
        )
        self.assertIn("theta_gain", first_distribution_fit)
        self.assertIn("segments", first_distribution_fit)
        self.assertIn("action", first_distribution_fit["segments"][0])
        self.assertEqual(len(metrics["adaptive_teacher_summary"]), 2)
        first_teacher_summary = metrics["adaptive_teacher_summary"][0]
        second_teacher_summary = metrics["adaptive_teacher_summary"][1]
        self.assertEqual(first_teacher_summary["iteration"], 1)
        self.assertEqual(
            first_teacher_summary["teacher_sampling_model"],
            "bootstrap_probabilistic_prior",
        )
        self.assertEqual(
            second_teacher_summary["teacher_sampling_model"],
            "previous_iteration_student",
        )
        self.assertEqual(first_teacher_summary["trace_count"], 2)
        self.assertEqual(first_teacher_summary["candidate_rollouts"], 2)
        self.assertEqual(first_teacher_summary["effective_candidate_rollouts"], 2)
        self.assertEqual(first_teacher_summary["selected_top_rho"], 1)
        self.assertEqual(first_teacher_summary["effective_top_rho"], 1)
        self.assertEqual(first_teacher_summary["paper_top_rho"], 10)
        self.assertFalse(first_teacher_summary["uses_paper_top_rho"])
        self.assertTrue(first_teacher_summary["candidate_rollouts_cover_selected_top_rho"])
        self.assertFalse(first_teacher_summary["candidate_rollouts_cover_paper_top_rho"])
        self.assertFalse(first_teacher_summary["cem_phase_matches_paper_rho"])
        self.assertIn("teacher_source_counts", first_teacher_summary)
        self.assertIn("refinement_elite_summary", first_teacher_summary)
        self.assertEqual(first_teacher_summary["refinement_elite_summary"]["count"], 2)
        self.assertEqual(
            first_teacher_summary["refinement_elite_summary"]["elite_count"]["count"],
            2,
        )
        first_components = first_teacher_summary["objective_component_summary"]
        self.assertEqual(first_components["reward_term"]["count"], 2)
        self.assertIn("direct_objective", first_components)
        self.assertEqual(
            first_components["direct_objective"]["count"],
            first_teacher_summary["recorded_teacher_objective_direct_count"],
        )
        self.assertEqual(
            first_components["refinement_objective"]["count"],
            first_teacher_summary["recorded_teacher_refinement_objective_count"],
        )
        self.assertIn("refinement_minus_direct_objective", first_components)
        self.assertEqual(
            first_teacher_summary["teacher_reward_lambda"],
            metrics["config"]["teacher_reward_lambda"],
        )
        self.assertEqual(
            first_teacher_summary["teacher_student_regularizer"],
            metrics["config"]["teacher_student_regularizer"],
        )
        self.assertIn("teacher_reward_lambda * reward", first_teacher_summary["teacher_objective_formula"])
        self.assertGreaterEqual(first_teacher_summary["recorded_student_log_probability_count"], 0)
        self.assertLessEqual(first_teacher_summary["recorded_student_log_probability_fraction"], 1.0)
        self.assertIn("recorded_teacher_objective_mean", first_teacher_summary)
        self.assertEqual(first_teacher_summary["recorded_teacher_objective_direct_count"], 2)
        self.assertEqual(first_teacher_summary["recorded_teacher_refinement_objective_count"], 2)
        self.assertIn("recorded_teacher_refinement_objective_mean", first_teacher_summary)
        self.assertEqual(second_teacher_summary["trace_count"], 2)
        self.assertGreaterEqual(second_teacher_summary["recorded_student_log_probability_count"], 1)
        self.assertGreater(second_teacher_summary["recorded_student_log_probability_fraction"], 0.0)
        second_components = second_teacher_summary["objective_component_summary"]
        self.assertEqual(
            second_components["student_log_probability"]["count"],
            second_teacher_summary["recorded_student_log_probability_count"],
        )
        self.assertEqual(
            second_components["student_regularizer_term"]["count"],
            second_teacher_summary["recorded_student_log_probability_count"],
        )
        self.assertEqual(
            second_components["direct_objective_formula_residual"]["count"],
            second_teacher_summary["recorded_student_log_probability_count"],
        )
        if second_components["direct_objective_formula_residual"]["count"]:
            self.assertAlmostEqual(
                second_components["direct_objective_formula_residual"]["mean"],
                0.0,
            )
        self.assertEqual(second_teacher_summary["recorded_teacher_objective_direct_count"], 2)
        self.assertEqual(second_teacher_summary["recorded_teacher_refinement_objective_count"], 2)
        status = metrics["paper_protocol_status"]
        self.assertTrue(status["cartpole_environment"])
        self.assertEqual(status["train_horizon_seconds"], 5.0)
        self.assertEqual(status["train_pole_length"], 0.5)
        self.assertEqual(status["test_horizon_seconds"], 300.0)
        self.assertEqual(status["test_pole_length"], 1.0)
        self.assertEqual(status["paper_test_horizon_steps"], 15000)
        self.assertFalse(status["uses_full_test_horizon"])
        self.assertEqual(status["paper_eval_rollouts"], 1000)
        self.assertFalse(status["uses_paper_eval_rollouts"])
        self.assertTrue(status["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(status["space_spec"]["action_dimension"], 1)
        self.assertEqual(status["space_spec"]["observation_dimension"], 4)
        self.assertTrue(status["uses_paper_reward_scale"])
        self.assertTrue(status["gaussian_action_parameter_distributions"])
        self.assertTrue(status["gaussian_switch_parameter_distributions"])
        self.assertTrue(status["transition_specific_switch_conditions"])
        self.assertTrue(status["resamples_parameters_on_mode_entry"])
        self.assertEqual(status["student_em_iters"], 2)
        self.assertEqual(status["student_switch_responsibility_passes"], 2)
        self.assertEqual(status["teacher_candidate_rollouts"], 2)
        self.assertEqual(status["effective_teacher_candidate_rollouts"], 2)
        self.assertEqual(status["selected_teacher_top_rho"], 1)
        self.assertEqual(status["effective_teacher_top_rho"], 1)
        self.assertEqual(status["paper_teacher_top_rho"], 10)
        self.assertFalse(status["uses_paper_teacher_top_rho"])
        self.assertEqual(status["selected_teacher_parallel_trace_workers"], 1)
        self.assertEqual(status["effective_teacher_parallel_trace_workers"], 1)
        self.assertEqual(status["effective_teacher_parallel_trace_initial_states"], 2)
        self.assertEqual(status["effective_teacher_parallel_trace_slots"], 1)
        self.assertEqual(status["paper_teacher_parallel_threads"], 10)
        self.assertFalse(status["uses_parallel_teacher_trace_optimization"])
        self.assertFalse(status["uses_paper_teacher_parallel_threads"])
        self.assertEqual(status["selected_student_parallel_switch_workers"], 1)
        self.assertEqual(status["effective_student_parallel_switch_workers"], 1)
        self.assertEqual(status["student_transition_switch_fit_count"], 2)
        self.assertEqual(status["effective_student_parallel_switch_slots"], 1)
        self.assertEqual(status["paper_student_parallel_threads"], 10)
        self.assertFalse(status["uses_parallel_student_switch_optimization"])
        self.assertFalse(status["uses_paper_student_parallel_threads"])
        self.assertTrue(status["teacher_candidate_rollouts_cover_selected_top_rho"])
        self.assertFalse(status["teacher_candidate_rollouts_cover_paper_top_rho"])
        self.assertFalse(status["teacher_cem_phase_matches_paper_rho"])
        self.assertEqual(status["teacher_elite_distribution_resamples"], 3)
        self.assertEqual(status["teacher_elite_distribution_rounds"], 2)
        self.assertTrue(status["synthesized_by_current_algorithm"])
        self.assertFalse(status["full_probabilistic_adaptive_teaching"])
        self.assertFalse(status["full_continuous_switch_m_step"])
        self.assertFalse(status["full_cem_teacher_optimizer"])
        self.assertFalse(status["paper_scale_result"])
        self.assertIn("Local bounded Cartpole PSM diagnostic", status["limitation"])
        self.assertIn("policy_description", metrics)
        self.assertEqual(len(metrics["synthesis_history"]), 2)
        for index, entry in enumerate(metrics["synthesis_history"], start=1):
            self.assertEqual(entry["iteration"], index)
            self.assertEqual(
                entry["adaptive_teacher_summary"],
                metrics["adaptive_teacher_summary"][index - 1],
            )
            self.assertEqual(len(entry["student_fit_history"]), 5)
            self.assertEqual(entry["student_fit_history"][0]["em_iteration"], 1)
            self.assertEqual(entry["student_fit_history"][0]["responsibility_pass"], 0)
            self.assertEqual(
                entry["student_fit_history"][0]["phase"],
                "action_likelihood_initialization",
            )
            self.assertEqual(entry["student_fit_history"][-1]["em_iteration"], 2)
            self.assertEqual(entry["student_fit_history"][-1]["responsibility_pass"], 2)
            self.assertEqual(
                entry["student_fit_history"][-1]["phase"],
                "switch_timing_refinement",
            )
            self.assertIn("action_distributions", entry["student_fit_history"][-1])
            self.assertIn("switch_parameter_distributions", entry["student_fit_history"][-1])
            self.assertIn("mean_entropy_nats", entry["student_fit_history"][-1]["responsibility_summary"])
            pair_summary = entry["student_fit_history"][-1]["switch_pair_responsibility_summary"]
            self.assertIn("transition_mass", pair_summary)
            self.assertIn("stay_mass", pair_summary)
            self.assertGreaterEqual(pair_summary["pairs"], 0)
            self.assertAlmostEqual(
                pair_summary["transition_mass"] + pair_summary["stay_mass"],
                pair_summary["pairs"],
            )
            self.assertIn("evaluation", entry)
            self.assertIn("success_rate", entry["evaluation"]["train"])
            self.assertIn("reward_mean", entry["evaluation"]["train"])
            self.assertIn("success_rate", entry["evaluation"]["test"])
            self.assertIn("reward_mean", entry["evaluation"]["test"])
        history_entry = metrics["synthesis_history"][-1]
        self.assertEqual(history_entry["trace_summary"]["count"], metrics["num_traces"])
        self.assertEqual(history_entry["evaluation"]["train"], metrics["train"])
        self.assertEqual(history_entry["evaluation"]["test"], metrics["test"])
        self.assertIn("teacher_source_counts", history_entry["trace_summary"])
        self.assertIn("probabilistic_student", history_entry)
        self.assertEqual(
            history_entry["student_fit_history"][-1]["responsibility_summary"],
            history_entry["probabilistic_student"]["responsibility_summary"],
        )
        self.assertIn("switch_fit_diagnostics", history_entry)
        self.assertEqual(
            history_entry["switch_fit_diagnostics"]["diagnostic_scope"],
            "local_teacher_trace_fit",
        )
        self.assertEqual(metrics["trace_summary"]["count"], metrics["num_traces"])
        self.assertGreaterEqual(metrics["trace_summary"]["reward_mean"], 0.0)
        self.assertIn("teacher_source_counts", metrics["trace_summary"])
        self.assertLessEqual(len(metrics["trace_summary"]["examples"]), 3)
        self.assertIn("mode_prefix", metrics["trace_summary"]["examples"][0])
        self.assertIn("theta_gain", metrics["trace_summary"]["examples"][0])
        self.assertIn("segment_actions", metrics["trace_summary"]["examples"][0])
        self.assertIn("segment_durations", metrics["trace_summary"]["examples"][0])
        self.assertIn("segment_time_increments", metrics["trace_summary"]["examples"][0])
        self.assertIn("teacher_source", metrics["trace_summary"]["examples"][0])
        self.assertIn("student_log_probability", metrics["trace_summary"]["examples"][0])
        self.assertIn("teacher_objective", metrics["trace_summary"]["examples"][0])
        self.assertIn("teacher_refinement_objective", metrics["trace_summary"]["examples"][0])
        self.assertIn("teacher_refinement_elite_summary", metrics["trace_summary"]["examples"][0])
        trace_summary_distribution_fits = [
            example.get("elite_distribution_fit")
            for example in metrics["trace_summary"]["examples"]
            if example.get("elite_distribution_fit") is not None
        ]
        if trace_summary_distribution_fits:
            self.assertEqual(
                trace_summary_distribution_fits[0]["source_count"],
                len(trace_summary_distribution_fits[0]["source_weights"]),
            )
        self.assertIn("probabilistic_student", metrics)
        self.assertIn("action_distributions", metrics["probabilistic_student"])
        self.assertIn("switch_parameter_distributions", metrics["probabilistic_student"])
        self.assertIn("transition_switches", metrics["probabilistic_student"])
        self.assertIn("0->1", metrics["probabilistic_student"]["transition_switches"])
        self.assertIn("1->0", metrics["probabilistic_student"]["transition_switches"])
        self.assertIn("transition_switch_parameter_distributions", metrics["probabilistic_student"])
        self.assertIn("switch_pair_responsibility_summary", metrics["probabilistic_student"])
        final_pair_summary = metrics["synthesis_history"][-1]["student_fit_history"][-1][
            "switch_pair_responsibility_summary"
        ]
        self.assertEqual(
            metrics["probabilistic_student"]["switch_pair_responsibility_summary"],
            final_pair_summary,
        )
        self.assertGreaterEqual(metrics["probabilistic_student"]["responsibility_summary"]["segments"], 1)
        responsibility_summary = metrics["probabilistic_student"]["responsibility_summary"]
        self.assertIn("hard_mode_0_count", responsibility_summary)
        self.assertIn("hard_mode_1_count", responsibility_summary)
        self.assertIn("ambiguous_segment_count", responsibility_summary)
        self.assertEqual(responsibility_summary["ambiguous_segment_threshold"], 0.75)
        self.assertIn("mean_max_responsibility", responsibility_summary)
        self.assertIn("min_max_responsibility", responsibility_summary)
        self.assertIn("mean_entropy_nats", responsibility_summary)
        self.assertIn("max_entropy_nats", responsibility_summary)
        diagnostics = metrics["switch_fit_diagnostics"]
        self.assertEqual(diagnostics["diagnostic_scope"], "local_teacher_trace_fit")
        self.assertTrue(diagnostics["not_paper_reproduction"])
        self.assertEqual(diagnostics["selection_objective_order"][0], "responsibility_weighted_label_loss")
        self.assertEqual(
            diagnostics["selection_objective_order"][1],
            "bounded_eq12_style_distribution_loss",
        )
        self.assertEqual(diagnostics["distribution_rescore_top_k"], 32)
        self.assertEqual(diagnostics["prefilter_objective_order"][1], "eq12_style_timing_loss")
        self.assertTrue(diagnostics["responsibility_segment_count_match"])
        self.assertTrue(diagnostics["transition_specific_switch_conditions"])
        self.assertIn("0->1", diagnostics["transition_specific_switches"])
        self.assertIn("1->0", diagnostics["transition_specific_switches"])
        self.assertIn(
            "bounded_eq12_style_distribution_loss",
            diagnostics["transition_specific_switches"]["0->1"],
        )
        self.assertIn(
            "directed_weighted_label_loss",
            diagnostics["transition_specific_switches"]["0->1"],
        )
        self.assertEqual(
            diagnostics["transition_specific_switches"]["0->1"]["objective_tuple"][0],
            diagnostics["transition_specific_switches"]["0->1"]["directed_weighted_label_loss"],
        )
        self.assertEqual(diagnostics["num_trace_steps"], diagnostics["example_count"])
        self.assertEqual(diagnostics["num_segments"], diagnostics["segment_count"])
        self.assertEqual(diagnostics["segment_count"], metrics["probabilistic_student"]["responsibility_summary"]["segments"])
        self.assertGreater(diagnostics["example_count"], 0)
        self.assertIn("selected_student_switch", diagnostics["candidates"])
        self.assertIn("fixed_local_reference_switch", diagnostics["candidates"])
        selected = diagnostics["candidates"]["selected_student_switch"]
        self.assertIn("hard_label_mistakes", selected)
        self.assertIn("responsibility_weighted_label_loss", selected)
        self.assertIn("bounded_eq12_style_distribution_loss", selected)
        self.assertIn("eq12_style_timing_loss", selected)
        self.assertIn("deterministic_objective_tuple", selected)
        self.assertIn("objective_boundary_alignment", selected)
        self.assertIn("boundary_alignment", selected)
        self.assertEqual(selected["boundary_alignment"]["num_boundaries"], diagnostics["num_boundaries"])
        self.assertLessEqual(
            selected["boundary_alignment"]["enabled_boundary_count"],
            selected["boundary_alignment"]["num_boundaries"],
        )
        self.assertEqual(selected["objective_tuple"][0], selected["responsibility_weighted_label_loss"])
        self.assertEqual(selected["objective_tuple"][1], selected["bounded_eq12_style_distribution_loss"])
        self.assertEqual(selected["deterministic_objective_tuple"][1], selected["eq12_style_timing_loss"])
        self.assertIn("not paper-scale reproduction results", diagnostics["note"])
        self.assertIn("success_rate", metrics["train"])
        self.assertIn("reward_mean", metrics["test"])


if __name__ == "__main__":
    unittest.main()
