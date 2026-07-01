import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_psm.py")


class CartpolePSMCliTest(unittest.TestCase):
    def test_cli_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "psm_metrics.json")
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
                    "1",
                    "--teacher-student-regularizer",
                    "0.5",
                    "--teacher-reward-lambda",
                    "100",
                    "--teacher-top-rho",
                    "1",
                    "--teacher-refinement-steps",
                    "1",
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

        self.assertEqual(metrics["config"]["num_initial_states"], 2)
        self.assertEqual(metrics["config"]["teacher_theta_gain"], 12.5)
        self.assertEqual(metrics["config"]["teacher_omega_gain"], 0.75)
        self.assertEqual(metrics["config"]["teacher_student_iters"], 1)
        self.assertEqual(metrics["config"]["teacher_student_regularizer"], 0.5)
        self.assertEqual(metrics["config"]["teacher_reward_lambda"], 100.0)
        self.assertEqual(metrics["config"]["teacher_top_rho"], 1)
        self.assertEqual(metrics["config"]["teacher_refinement_steps"], 1)
        provenance = metrics["algorithm_provenance"]
        self.assertEqual(provenance["probabilistic_student"]["em_iters"], 4)
        self.assertEqual(provenance["probabilistic_student"]["min_gaussian_std"], 1e-3)
        self.assertEqual(provenance["switch_timing"]["std_steps"], 2.0)
        self.assertTrue(provenance["switch_timing"]["scalar_threshold_uses_shared_sample"])
        self.assertEqual(provenance["switch_timing"]["std_refinement_multipliers"], [0.5, 1.0, 2.0])
        self.assertEqual(provenance["switch_search"]["boolean_tree_depth"], 2)
        self.assertIn(50.0, provenance["switch_search"]["oblique_theta_weights"])
        self.assertEqual(provenance["switch_search"]["max_threshold_candidates"], 64)
        self.assertEqual(provenance["switch_search"]["distribution_rescore_top_k"], 128)
        self.assertEqual(
            provenance["switch_search"]["prefilter_objective_order"][1],
            "eq12_style_timing_loss",
        )
        self.assertEqual(
            provenance["switch_search"]["selection_objective_order"][0],
            "hard_label_mistakes",
        )
        self.assertEqual(
            provenance["switch_search"]["selection_objective_order"][1],
            "bounded_eq12_style_distribution_loss",
        )
        self.assertEqual(provenance["teacher_search"]["duration_refinement_deltas"], [-1, 1])
        self.assertEqual(
            provenance["teacher_search"]["action_refinement_candidates_per_segment"],
            1,
        )
        self.assertEqual(
            provenance["teacher_search"]["student_sample_fraction_after_first_iteration"],
            0.5,
        )
        self.assertEqual(
            provenance["teacher_search"]["student_sample_local_refinement"],
            "duration_and_action_coordinate_search",
        )
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("policy_description", metrics)
        self.assertEqual(len(metrics["synthesis_history"]), 1)
        history_entry = metrics["synthesis_history"][0]
        self.assertEqual(history_entry["iteration"], 1)
        self.assertEqual(history_entry["trace_summary"]["count"], metrics["num_traces"])
        self.assertIn("teacher_source_counts", history_entry["trace_summary"])
        self.assertIn("probabilistic_student", history_entry)
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
        self.assertIn("teacher_source", metrics["trace_summary"]["examples"][0])
        self.assertIn("student_log_probability", metrics["trace_summary"]["examples"][0])
        self.assertIn("probabilistic_student", metrics)
        self.assertIn("action_distributions", metrics["probabilistic_student"])
        self.assertIn("switch_parameter_distributions", metrics["probabilistic_student"])
        self.assertGreaterEqual(metrics["probabilistic_student"]["responsibility_summary"]["segments"], 1)
        diagnostics = metrics["switch_fit_diagnostics"]
        self.assertEqual(diagnostics["diagnostic_scope"], "local_teacher_trace_fit")
        self.assertTrue(diagnostics["not_paper_reproduction"])
        self.assertEqual(diagnostics["selection_objective_order"][0], "hard_label_mistakes")
        self.assertEqual(
            diagnostics["selection_objective_order"][1],
            "bounded_eq12_style_distribution_loss",
        )
        self.assertEqual(diagnostics["distribution_rescore_top_k"], 128)
        self.assertEqual(diagnostics["prefilter_objective_order"][1], "eq12_style_timing_loss")
        self.assertTrue(diagnostics["responsibility_segment_count_match"])
        self.assertEqual(diagnostics["num_trace_steps"], diagnostics["example_count"])
        self.assertEqual(diagnostics["num_segments"], diagnostics["segment_count"])
        self.assertEqual(diagnostics["segment_count"], metrics["probabilistic_student"]["responsibility_summary"]["segments"])
        self.assertGreater(diagnostics["example_count"], 0)
        self.assertIn("selected_student_switch", diagnostics["candidates"])
        self.assertIn("fixed_local_reference_switch", diagnostics["candidates"])
        selected = diagnostics["candidates"]["selected_student_switch"]
        self.assertIn("hard_label_mistakes", selected)
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
        self.assertEqual(selected["objective_tuple"][0], selected["hard_label_mistakes"])
        self.assertEqual(selected["objective_tuple"][1], selected["bounded_eq12_style_distribution_loss"])
        self.assertEqual(selected["deterministic_objective_tuple"][1], selected["eq12_style_timing_loss"])
        self.assertIn("not paper-scale reproduction results", diagnostics["note"])
        self.assertIn("success_rate", metrics["train"])
        self.assertIn("reward_mean", metrics["test"])


if __name__ == "__main__":
    unittest.main()
