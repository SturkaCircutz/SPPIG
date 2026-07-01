import json
import math
import os
import random
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from cartpole_env import BangBangCartpolePSM, CartpoleEnv, evaluate_cartpole_policy
from cartpole_synthesis import (
    BooleanTreeSwitch,
    CartpoleSegment,
    CartpoleSynthesisConfig,
    CartpoleTrace,
    Depth2Switch,
    GaussianScalar,
    ObservationPredicate,
    ProbabilisticCartpoleStudent,
    cartpole_switch_fit_diagnostics,
    fit_probabilistic_cartpole_student,
    synthesize_cartpole_policy,
    synthesize_cartpole_student,
    synthesize_cartpole_student_with_history,
    _eq12_switch_log_likelihood,
    _action_refinement_candidates,
    _boolean_tree_candidates,
    _fit_switch_parameter_distributions,
    _gaussian_threshold_pass_probability,
    _greedy_boolean_tree_candidates,
    _duration_refinement_candidates,
    _elite_kernel_log_probability,
    _mode_responsibilities,
    _mode_run_lengths,
    _mode_run_actions,
    _optimize_loop_free_trace,
    _loop_free_trace_distance,
    _refine_responsibilities_with_switch_timing,
    _refine_loop_free_trace,
    _refine_switch_distribution_means,
    _switch_distribution_std_candidates,
    _switch_distribution_timing_loss,
    _rollout_student_sampled_trace,
    _rollout_with_teacher_gains,
    _sample_switch,
    _single_threshold_transition_probability,
    _switch_cost,
    _switch_structure_rescore_candidates,
    _switch_structure_cost,
    _teacher_candidate_traces,
    _switch_timing_loss,
    _teacher_objective,
    _teacher_refinement_objective,
)

if HAS_TORCH:
    from ppo_cartpole import (
        LSTMActorCritic,
        MLPActorCritic,
        PPOConfig,
        _collect_rollout,
        _update_lstm,
        train_ppo_cartpole,
    )


class CartpolePaperTest(unittest.TestCase):
    def test_cartpole_train_test_split_matches_paper_row(self):
        train_env = CartpoleEnv.train_env()
        test_env = CartpoleEnv.test_env()

        self.assertEqual(train_env.cfg.pole_length, 0.5)
        self.assertEqual(train_env.cfg.horizon_seconds, 5.0)
        self.assertEqual(test_env.cfg.pole_length, 1.0)
        self.assertEqual(test_env.cfg.horizon_seconds, 300.0)
        self.assertEqual(len(train_env.reset()), 4)

    def test_programmatic_policy_evaluates(self):
        metrics = evaluate_cartpole_policy(
            BangBangCartpolePSM(),
            train_rollouts=2,
            test_rollouts=2,
            test_max_steps=100,
        )

        self.assertIn("train_success_rate", metrics)
        self.assertIn("test_success_rate", metrics)

    def test_cartpole_synthesis_returns_two_mode_policy(self):
        policy, traces = synthesize_cartpole_policy(
            CartpoleSynthesisConfig(
                num_initial_states=2,
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=4,
            teacher_student_iters=1,
            seed=3,
        )
        )

        self.assertEqual(len(traces), 2)
        self.assertIn("m0", policy.describe())
        self.assertIn("m1", policy.describe())

    def test_cartpole_synthesis_can_return_probabilistic_student(self):
        student, traces = synthesize_cartpole_student(
            CartpoleSynthesisConfig(
                num_initial_states=2,
                candidate_rollouts=4,
                segment_steps=2,
                segments_per_trace=4,
                teacher_student_iters=1,
                seed=5,
            )
        )

        self.assertEqual(len(traces), 2)
        self.assertIsInstance(student, ProbabilisticCartpoleStudent)
        self.assertIn("N(", student.describe())
        self.assertIn("m0", student.to_deterministic_policy().describe())

    def test_cartpole_synthesis_history_records_each_teacher_student_iteration(self):
        student, traces, history = synthesize_cartpole_student_with_history(
            CartpoleSynthesisConfig(
                num_initial_states=2,
                candidate_rollouts=4,
                segment_steps=2,
                segments_per_trace=4,
                teacher_student_iters=2,
                seed=5,
            )
        )

        self.assertEqual(len(history), 2)
        self.assertEqual([entry.iteration for entry in history], [1, 2])
        self.assertEqual(history[-1].student.describe(), student.describe())
        self.assertEqual(history[-1].traces, traces)
        self.assertEqual(len(history[0].traces), 2)
        self.assertGreaterEqual(len(history[0].student.responsibilities), 1)

    def test_cartpole_probabilistic_student_uses_gaussian_modes(self):
        cfg = CartpoleSynthesisConfig(
            num_initial_states=4,
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=8,
            teacher_student_iters=1,
            seed=4,
        )
        _, traces = synthesize_cartpole_policy(cfg)

        student = fit_probabilistic_cartpole_student(traces, cfg)

        self.assertEqual(set(student.action_distributions), {0, 1})
        self.assertLess(student.action_distributions[0].mean, 0.0)
        self.assertGreater(student.action_distributions[1].mean, 0.0)
        self.assertGreater(student.action_distributions[0].std, 0.0)
        self.assertGreater(student.action_distributions[1].std, 0.0)
        self.assertGreater(student.switch_threshold_distribution.std, 0.0)
        self.assertTrue(student.switch_parameter_distributions)
        for distribution in student.switch_parameter_distributions:
            self.assertGreater(distribution.std, 0.0)
        self.assertTrue(student.responsibilities)
        for left_weight, right_weight in student.responsibilities:
            self.assertAlmostEqual(left_weight + right_weight, 1.0)
            self.assertGreaterEqual(left_weight, 0.0)
            self.assertGreaterEqual(right_weight, 0.0)

    def test_cartpole_responsibility_refinement_uses_switch_timing(self):
        first_segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.4, 0.0],
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
            ],
            action_parameter=-0.9,
            duration=3,
            hard_mode=0,
        )
        second_segment = CartpoleSegment(
            observations=[[0.0, 0.0, -0.2, 0.0]],
            action_parameter=0.1,
            duration=1,
            hard_mode=1,
        )
        action_distributions = {
            0: GaussianScalar(-1.0, 1.0),
            1: GaussianScalar(1.0, 1.0),
        }
        action_only_second = _mode_responsibilities(
            second_segment.action_parameter,
            action_distributions,
        )

        responsibilities = _refine_responsibilities_with_switch_timing(
            [[first_segment, second_segment]],
            action_distributions,
            Depth2Switch(1.0, 0.0, 0.0),
            [GaussianScalar(0.0, 0.05)],
        )

        self.assertGreater(action_only_second[1], action_only_second[0])
        self.assertGreater(responsibilities[1][0], action_only_second[0])
        for left_weight, right_weight in responsibilities:
            self.assertAlmostEqual(left_weight + right_weight, 1.0)

    def test_cartpole_probabilistic_student_projects_to_policy(self):
        cfg = CartpoleSynthesisConfig(
            num_initial_states=3,
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=8,
            teacher_student_iters=1,
            seed=6,
        )
        _, traces = synthesize_cartpole_policy(cfg)
        student = fit_probabilistic_cartpole_student(traces, cfg)
        policy = student.to_deterministic_policy()

        metrics = evaluate_cartpole_policy(
            policy,
            train_rollouts=2,
            test_rollouts=1,
            test_max_steps=100,
        )

        self.assertIn("train_success_rate", metrics)
        self.assertIn("test_success_rate", metrics)
        self.assertIn("N(", student.describe())

    def test_cartpole_switch_timing_loss_prefers_segment_boundary(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.3, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        boundary_switch = Depth2Switch(1.0, 0.0, 0.0)
        early_switch = Depth2Switch(1.0, 0.0, -0.3)

        self.assertLess(
            _switch_timing_loss(boundary_switch, [[segment, next_segment]], responsibilities),
            _switch_timing_loss(early_switch, [[segment, next_segment]], responsibilities),
        )

    def test_cartpole_switch_fit_diagnostics_reports_boundary_alignment(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=4.0,
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.1)],
            responsibilities=[(1.0, 0.0), (0.0, 1.0)],
        )

        diagnostics = cartpole_switch_fit_diagnostics([trace], student)

        self.assertTrue(diagnostics["not_paper_reproduction"])
        self.assertEqual(diagnostics["diagnostic_scope"], "local_teacher_trace_fit")
        self.assertTrue(diagnostics["responsibility_segment_count_match"])
        self.assertEqual(diagnostics["num_trace_steps"], 4)
        self.assertEqual(diagnostics["num_segments"], 2)
        self.assertEqual(diagnostics["num_boundaries"], 1)
        selected = diagnostics["candidates"]["selected_student_switch"]
        alignment = selected["boundary_alignment"]
        self.assertEqual(alignment["num_boundaries"], 1)
        self.assertEqual(alignment["at_boundary_count"], 1)
        self.assertEqual(alignment["early_switch_count"], 0)
        self.assertEqual(selected["timing_loss_per_boundary"], selected["timing_loss_total"])
        self.assertEqual(
            selected["objective_tuple"][1],
            selected["bounded_eq12_style_distribution_loss"],
        )
        self.assertEqual(
            selected["deterministic_objective_tuple"][1],
            selected["eq12_style_timing_loss"],
        )
        self.assertEqual(
            selected["objective_boundary_alignment"]["num_boundaries"],
            diagnostics["num_boundaries"],
        )

    def test_cartpole_switch_fit_diagnostics_excludes_never_enabled_delta_sentinel(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=4.0,
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 1.0),
            switch_threshold_distribution=GaussianScalar(1.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(1.0, 0.1)],
            responsibilities=[(1.0, 0.0), (0.0, 1.0)],
        )

        diagnostics = cartpole_switch_fit_diagnostics([trace], student)

        alignment = diagnostics["candidates"]["selected_student_switch"]["boundary_alignment"]
        self.assertEqual(alignment["num_boundaries"], 1)
        self.assertEqual(alignment["enabled_boundary_count"], 0)
        self.assertEqual(alignment["never_enabled_count"], 1)
        self.assertIsNone(alignment["first_enabled_minus_duration_mean"])
        self.assertIsNone(alignment["first_enabled_minus_duration_min"])
        self.assertIsNone(alignment["first_enabled_minus_duration_max"])

    def test_cartpole_eq12_likelihood_rewards_transition_at_duration(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        boundary_switch = Depth2Switch(1.0, 0.0, 0.0)
        early_switch = Depth2Switch(1.0, 0.0, -0.3)

        self.assertGreater(
            _eq12_switch_log_likelihood(boundary_switch, segment, (1.0, 0.0), (0.0, 1.0)),
            _eq12_switch_log_likelihood(early_switch, segment, (1.0, 0.0), (0.0, 1.0)),
        )

    def test_cartpole_eq12_likelihood_penalizes_early_transition_when_staying(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        late_switch = Depth2Switch(1.0, 0.0, 0.3)
        early_switch = Depth2Switch(1.0, 0.0, -0.3)

        self.assertGreater(
            _eq12_switch_log_likelihood(late_switch, segment, (1.0, 0.0), (1.0, 0.0)),
            _eq12_switch_log_likelihood(early_switch, segment, (1.0, 0.0), (1.0, 0.0)),
        )

    def test_cartpole_switch_distribution_refinement_improves_timing_likelihood(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.3, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        initial_switch = Depth2Switch(1.0, 0.0, -0.3)

        refined_switch, refined = _refine_switch_distribution_means(
            initial_switch,
            [GaussianScalar(-0.3, 0.2)],
            segments_by_trace,
            responsibilities,
        )

        def mistakes(switch):
            return sum(
                int(switch.decide(observation) != segment.hard_mode)
                for trace_segments in segments_by_trace
                for segment in trace_segments
                for observation in segment.observations
            )

        self.assertEqual(len(refined), 1)
        self.assertGreater(refined[0].std, 0.0)
        self.assertLess(
            _switch_timing_loss(refined_switch, segments_by_trace, responsibilities),
            _switch_timing_loss(initial_switch, segments_by_trace, responsibilities),
        )
        self.assertLessEqual(mistakes(refined_switch), mistakes(initial_switch))

    def test_cartpole_switch_distribution_refinement_can_improve_probabilistic_std(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.15, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.25, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        switch = Depth2Switch(1.0, 0.0, 0.0)
        initial = [GaussianScalar(0.0, 1.0)]

        refined_switch, refined = _refine_switch_distribution_means(
            switch,
            initial,
            segments_by_trace,
            responsibilities,
        )

        self.assertLess(
            _switch_distribution_timing_loss(refined_switch, refined, segments_by_trace, responsibilities),
            _switch_distribution_timing_loss(switch, initial, segments_by_trace, responsibilities),
        )
        self.assertLess(refined[0].std, initial[0].std)

    def test_cartpole_switch_coordinate_refinement_polishes_grid_solution(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.15, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.25, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        switch = Depth2Switch(1.0, 0.0, 0.0)
        initial = [GaussianScalar(0.0, 1.0)]
        grid_best_std = min(
            _switch_distribution_std_candidates(
                initial[0],
                switch,
                0,
                segments_by_trace,
            )
        )

        _, refined = _refine_switch_distribution_means(
            switch,
            initial,
            segments_by_trace,
            responsibilities,
        )

        self.assertLess(refined[0].std, grid_best_std)

    def test_cartpole_switch_distribution_refinement_keeps_std_finite(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=2,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.3, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )

        _, refined = _refine_switch_distribution_means(
            Depth2Switch(1.0, 0.0, 0.0),
            [GaussianScalar(0.0, 0.0)],
            [[segment, next_segment]],
            [(1.0, 0.0), (0.0, 1.0)],
        )

        self.assertTrue(math.isfinite(refined[0].std))
        self.assertGreaterEqual(refined[0].std, 1e-3)

    def test_cartpole_switch_std_refinement_uses_boundary_variance_candidate(self):
        segments_by_trace = [
            [
                CartpoleSegment(
                    observations=[[0.0, 0.0, -0.4, 0.0]],
                    action_parameter=-10.0,
                    duration=1,
                    hard_mode=0,
                ),
                CartpoleSegment(
                    observations=[[0.0, 0.0, 0.2, 0.0]],
                    action_parameter=10.0,
                    duration=1,
                    hard_mode=1,
                ),
            ],
            [
                CartpoleSegment(
                    observations=[[0.0, 0.0, 0.4, 0.0]],
                    action_parameter=-10.0,
                    duration=1,
                    hard_mode=0,
                ),
                CartpoleSegment(
                    observations=[[0.0, 0.0, 0.5, 0.0]],
                    action_parameter=10.0,
                    duration=1,
                    hard_mode=1,
                ),
            ],
        ]

        candidates = _switch_distribution_std_candidates(
            GaussianScalar(0.0, 1.0),
            Depth2Switch(1.0, 0.0, 0.0),
            0,
            segments_by_trace,
        )

        self.assertIn(0.4, candidates)
        self.assertIn(0.5, candidates)
        self.assertIn(2.0, candidates)

    def test_cartpole_switch_parameter_refinement_rejects_more_label_mistakes(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.3, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        switch = Depth2Switch(1.0, 0.0, 0.0)

        refined_switch, _ = _refine_switch_distribution_means(
            switch,
            [GaussianScalar(0.0, 1.0)],
            segments_by_trace,
            responsibilities,
        )

        examples = [
            (observation, trace_segment.hard_mode)
            for trace_segments in segments_by_trace
            for trace_segment in trace_segments
            for observation in trace_segment.observations
        ]
        self.assertLessEqual(
            _switch_cost(refined_switch, examples)[0],
            _switch_cost(switch, examples)[0],
        )

    def test_cartpole_switch_distribution_timing_loss_rejects_responsibility_mismatch(self):
        segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )

        with self.assertRaises(ValueError):
            _switch_distribution_timing_loss(
                Depth2Switch(1.0, 0.0, 0.0),
                [GaussianScalar(0.0, 1.0)],
                [[segment]],
                [],
            )

    def test_cartpole_switch_structure_cost_uses_refined_distribution_timing(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.15, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.25, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]
        examples = [
            (observation, trace_segment.hard_mode)
            for trace_segments in segments_by_trace
            for trace_segment in trace_segments
            for observation in trace_segment.observations
        ]
        aligned = Depth2Switch(1.0, 0.0, 0.0)
        wrong_feature = Depth2Switch(0.0, 1.0, 0.0)

        aligned_cost = _switch_structure_cost(aligned, examples, segments_by_trace, responsibilities)
        wrong_feature_cost = _switch_structure_cost(wrong_feature, examples, segments_by_trace, responsibilities)

        self.assertLess(aligned_cost[1], wrong_feature_cost[1])
        self.assertNotEqual(
            wrong_feature_cost[1],
            _switch_cost(wrong_feature, examples, segments_by_trace, responsibilities)[1],
        )

    def test_cartpole_switch_structure_cost_falls_back_without_timing_evidence(self):
        examples = [
            ([0.0, 0.0, -0.1, 0.0], 0),
            ([0.0, 0.0, 0.1, 0.0], 1),
        ]
        switch = Depth2Switch(1.0, 0.0, 0.0)

        self.assertEqual(
            _switch_structure_cost(switch, examples),
            _switch_cost(switch, examples),
        )

    def test_cartpole_switch_structure_rescore_candidates_keeps_best_prefiltered_subset(self):
        examples = [
            ([0.0, 0.0, -0.2, 0.0], 0),
            ([0.0, 0.0, -0.1, 0.0], 0),
            ([0.0, 0.0, 0.1, 0.0], 1),
            ([0.0, 0.0, 0.2, 0.0], 1),
        ]
        switches = [
            Depth2Switch(1.0, 0.0, threshold / 100.0)
            for threshold in range(130)
        ]

        selected = _switch_structure_rescore_candidates(switches, examples, [], [])

        self.assertEqual(len(selected), 128)
        self.assertIn(Depth2Switch(1.0, 0.0, 0.0), selected)
        self.assertNotIn(Depth2Switch(1.0, 0.0, 1.29), selected)

    def test_cartpole_boolean_tree_switch_supports_depth_two_conjunction(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 1.0),
        )

        self.assertEqual(switch.decide([0.0, 0.0, 0.1, 0.5]), 1)
        self.assertEqual(switch.decide([0.0, 0.0, -0.1, 0.5]), 0)
        self.assertEqual(switch.decide([0.0, 0.0, 0.1, 2.0]), 0)
        self.assertIn("and", switch.describe())

    def test_cartpole_boolean_tree_candidates_include_depth_two(self):
        examples = [
            ([0.0, 0.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 1.0], 1),
        ]

        candidates = _boolean_tree_candidates(examples)

        self.assertTrue(any(candidate.second is not None for candidate in candidates))

    def test_cartpole_greedy_boolean_tree_expansion_improves_stump(self):
        examples = [
            ([0.0, 0.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 1.0], 1),
            ([0.0, 0.0, 0.0, 1.0], 0),
        ]

        candidates = _greedy_boolean_tree_candidates(examples)
        stump_costs = [
            _switch_cost(candidate, examples)[0]
            for candidate in candidates
            if candidate.second is None
        ]
        expanded_costs = [
            _switch_cost(candidate, examples)[0]
            for candidate in candidates
            if candidate.second is not None
        ]

        self.assertTrue(expanded_costs)
        self.assertLess(min(expanded_costs), min(stump_costs))

    def test_cartpole_boolean_tree_switch_has_gaussian_parameter_per_predicate(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 1.0),
        )
        segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.1, 0.5]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.2, 0.4]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )

        distributions = _fit_switch_parameter_distributions(
            switch,
            [[segment, next_segment]],
            [(1.0, 0.0), (0.0, 1.0)],
        )

        self.assertEqual(len(distributions), 2)
        self.assertGreater(distributions[0].std, 0.0)
        self.assertGreater(distributions[1].std, 0.0)

    def test_cartpole_sampled_switch_uses_gaussian_thresholds(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 1.0),
        )

        sampled = _sample_switch(
            switch,
            [GaussianScalar(0.25, 0.0), GaussianScalar(0.75, 0.0)],
            random.Random(0),
        )

        self.assertIsInstance(sampled, BooleanTreeSwitch)
        self.assertEqual(sampled.first.threshold, 0.25)
        self.assertEqual(sampled.second.threshold, 0.75)

    def test_cartpole_sampled_depth2_switch_preserves_predicate_count(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 1.0),
        )

        sampled = _sample_switch(
            switch,
            [GaussianScalar(0.25, 0.0), GaussianScalar(0.75, 0.0)],
            random.Random(0),
        )

        self.assertIsNotNone(sampled.second)

    def test_cartpole_probabilistic_student_samples_policy_parameters(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-9.0, 0.0),
                1: GaussianScalar(9.0, 0.0),
            },
            switch=BooleanTreeSwitch(ObservationPredicate(2, ">=", 0.0)),
            switch_threshold_distribution=GaussianScalar(0.1, 0.0),
            switch_parameter_distributions=[GaussianScalar(0.1, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )

        policy = student.sample_policy(random.Random(0))

        self.assertEqual(policy.left_force, -9.0)
        self.assertEqual(policy.right_force, 9.0)
        self.assertEqual(policy.switch.first.threshold, 0.1)

    def test_cartpole_switch_probability_uses_gaussian_threshold_distribution(self):
        distribution = GaussianScalar(0.0, 0.1)

        self.assertGreater(
            _gaussian_threshold_pass_probability(0.2, distribution, ">="),
            _gaussian_threshold_pass_probability(-0.2, distribution, ">="),
        )
        self.assertGreater(
            _gaussian_threshold_pass_probability(-0.2, distribution, "<="),
            _gaussian_threshold_pass_probability(0.2, distribution, "<="),
        )

    def test_cartpole_switch_transition_probability_uses_shared_threshold_sample(self):
        distribution = GaussianScalar(0.0, 0.1)
        values = [-0.3, -0.2, 0.2]

        self.assertGreater(
            _single_threshold_transition_probability(values, distribution, ">=", 3),
            _single_threshold_transition_probability(values, distribution, ">=", 1),
        )

    def test_cartpole_teacher_objective_defaults_to_reward(self):
        cfg = CartpoleSynthesisConfig()
        trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[10.0],
            mode_labels=[1],
            reward=7.0,
        )

        self.assertEqual(cfg.teacher_reward_lambda, 100.0)
        self.assertEqual(_teacher_objective(trace, None, cfg), 700.0)

    def test_cartpole_teacher_reward_lambda_is_configurable(self):
        cfg = CartpoleSynthesisConfig(teacher_reward_lambda=2.0)
        trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[10.0],
            mode_labels=[1],
            reward=7.0,
        )

        self.assertEqual(_teacher_objective(trace, None, cfg), 14.0)

    def test_cartpole_teacher_objective_uses_student_regularizer(self):
        cfg = CartpoleSynthesisConfig(teacher_student_regularizer=10.0)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        matching_trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[10.0],
            mode_labels=[1],
            reward=1.0,
        )
        mismatched_trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[0.0],
            mode_labels=[1],
            reward=2.0,
        )

        self.assertGreater(
            _teacher_objective(matching_trace, student, cfg),
            _teacher_objective(mismatched_trace, student, cfg),
        )

    def test_cartpole_teacher_regularizer_uses_switch_timing_likelihood(self):
        cfg = CartpoleSynthesisConfig(teacher_student_regularizer=1.0)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        boundary_aligned = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=1.0,
        )
        early_switching = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.5, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=1.0,
        )

        self.assertGreater(
            _teacher_objective(boundary_aligned, student, cfg),
            _teacher_objective(early_switching, student, cfg),
        )

    def test_cartpole_teacher_regularizer_uses_switch_distribution_uncertainty(self):
        cfg = CartpoleSynthesisConfig(teacher_student_regularizer=1.0)
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=1.0,
        )
        precise_student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.05),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.05)],
            responsibilities=[(0.5, 0.5)],
        )
        diffuse_student = ProbabilisticCartpoleStudent(
            action_distributions=precise_student.action_distributions,
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )

        self.assertGreater(
            _teacher_objective(trace, precise_student, cfg),
            _teacher_objective(trace, diffuse_student, cfg),
        )

    def test_cartpole_teacher_elite_distance_matches_loop_free_parameters(self):
        reference = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(2, 3),
        )
        same = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(2, 3),
        )
        different = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(-9.0, 8.0),
            segment_durations=(2, 5),
        )

        self.assertEqual(_loop_free_trace_distance(reference, same), 0.0)
        self.assertGreater(_loop_free_trace_distance(reference, different), 0.0)

    def test_cartpole_teacher_elite_kernel_uses_normalized_top_rho_distance(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        elite = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(2, 3),
            student_log_probability=-5.0,
        )
        close = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(2, 3),
        )
        far = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            segment_actions=(-8.0, 8.0),
            segment_durations=(4, 5),
        )
        cfg = CartpoleSynthesisConfig(teacher_reward_lambda=0.0, teacher_student_regularizer=1.0)

        self.assertAlmostEqual(_elite_kernel_log_probability(close, student, [elite]), 0.0)
        self.assertGreater(
            _elite_kernel_log_probability(close, student, [elite]),
            _elite_kernel_log_probability(far, student, [elite]),
        )
        self.assertGreater(
            _teacher_refinement_objective(close, student, cfg, [elite]),
            _teacher_refinement_objective(far, student, cfg, [elite]),
        )

    def test_cartpole_mode_run_lengths_records_sampled_trace_segments(self):
        self.assertEqual(_mode_run_lengths([0, 0, 1, 1, 1, 0]), (2, 3, 1))
        self.assertEqual(_mode_run_lengths([]), ())

    def test_cartpole_mode_run_actions_records_sampled_trace_action_sequence(self):
        self.assertEqual(
            _mode_run_actions([-10.0, -10.0, 10.0, 10.0, -10.0], [0, 0, 1, 1, 0]),
            (-10.0, 10.0, -10.0),
        )
        self.assertEqual(_mode_run_actions([], []), ())
        with self.assertRaises(ValueError):
            _mode_run_actions([10.0], [1, 0])

    def test_cartpole_teacher_can_sample_candidates_from_probabilistic_student(self):
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=4,
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.0),
                1: GaussianScalar(10.0, 0.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )

        trace = _rollout_student_sampled_trace(
            [0.0, 0.0, 0.05, 0.0],
            CartpoleEnv.train_env(seed=0).cfg,
            cfg,
            student,
            random.Random(0),
        )

        self.assertEqual(trace.teacher_source, "student_sample")
        self.assertIsNotNone(trace.student_log_probability)
        self.assertEqual(sum(trace.segment_durations), len(trace.actions))
        self.assertEqual(len(trace.segment_actions), len(trace.segment_durations))
        self.assertTrue(set(trace.mode_labels).issubset({0, 1}))

    def test_cartpole_teacher_candidate_pool_includes_student_samples_after_first_iteration(self):
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=4,
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.0),
                1: GaussianScalar(10.0, 0.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )

        candidates = _teacher_candidate_traces(
            [0.0, 0.0, 0.05, 0.0],
            CartpoleEnv.train_env(seed=0).cfg,
            cfg,
            random.Random(1),
            student,
        )

        self.assertEqual(len(candidates), 4)
        self.assertTrue(any(trace.teacher_source == "student_sample" for trace in candidates))
        self.assertTrue(any(trace.teacher_source == "gain_sample" for trace in candidates))

    def test_cartpole_teacher_can_refine_student_sampled_trace(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3, teacher_refinement_steps=1)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        trace = _rollout_student_sampled_trace(
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            student,
            random.Random(0),
        )

        refined = _refine_loop_free_trace(trace, [0.0, 0.0, 0.05, 0.0], env.cfg, cfg, student)

        self.assertGreaterEqual(
            _teacher_refinement_objective(refined, student, cfg, [trace]),
            _teacher_refinement_objective(trace, student, cfg, [trace]),
        )
        self.assertIsNotNone(refined.student_log_probability)
        self.assertIn(refined.teacher_source, {"student_sample", "student_sample_refined"})

    def test_cartpole_teacher_optimization_can_return_refined_student_sample(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=3,
            teacher_top_rho=4,
            teacher_refinement_steps=1,
            teacher_reward_lambda=100.0,
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )

        trace = _optimize_loop_free_trace(
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            random.Random(0),
            student,
        )

        self.assertIn(
            trace.teacher_source,
            {"gain_sample", "gain_refined", "student_sample", "student_sample_refined"},
        )
        self.assertGreaterEqual(trace.reward, 1.0)

    def test_cartpole_teacher_refinement_does_not_reduce_objective(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=4,
            teacher_refinement_steps=2,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.05, 0.0]],
            actions=[10.0],
            mode_labels=[1],
            reward=1.0,
            theta_gain=1.0,
            omega_gain=0.0,
        )

        refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_rollout_records_segment_durations(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3)

        trace = _rollout_with_teacher_gains(
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
        )

        self.assertEqual(trace.segment_durations, (2, 2, 2))
        self.assertEqual(trace.segment_actions, (10.0, 10.0, 10.0))
        self.assertEqual(len(trace.segment_actions), len(trace.segment_durations))

    def test_cartpole_teacher_rollout_records_only_started_loop_free_segments(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3)

        trace = _rollout_with_teacher_gains(
            [3.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
        )

        self.assertEqual(len(trace.segment_actions), len(trace.segment_durations))
        self.assertLessEqual(len(trace.segment_actions), cfg.segments_per_trace)

    def test_cartpole_teacher_duration_refinement_preserves_action_sequence(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, -10.0, 10.0),
        )

        candidates = _duration_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate.segment_actions == trace.segment_actions for candidate in candidates)
        )
        self.assertTrue(
            all(len(candidate.segment_actions) == len(candidate.segment_durations) for candidate in candidates)
        )

    def test_cartpole_teacher_action_refinement_changes_one_action_at_a_time(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0, 10.0),
        )

        candidates = _action_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertEqual(len(candidates), len(trace.segment_actions))
        for candidate in candidates:
            changed = sum(
                int(left != right)
                for left, right in zip(candidate.segment_actions, trace.segment_actions)
            )
            self.assertEqual(changed, 1)
            self.assertEqual(candidate.segment_durations, trace.segment_durations)

    def test_cartpole_teacher_action_refinement_does_not_reduce_objective(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=3,
            teacher_refinement_steps=1,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
        )

        refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_duration_refinement_does_not_reduce_objective(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=3)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
        )

        best = max(
            [trace, *_duration_refinement_candidates(trace, initial_state, env.cfg, cfg)],
            key=lambda candidate: _teacher_objective(candidate, None, cfg),
        )

        self.assertGreaterEqual(
            _teacher_objective(best, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_smoke_train_mlp(self):
        _, result = train_ppo_cartpole(
            PPOConfig(
                policy_type="mlp",
                total_timesteps=64,
                rollout_steps=32,
                update_epochs=1,
                minibatches=1,
                hidden_size=8,
                num_envs=1,
                seed=1,
            )
        )

        self.assertEqual(result.timesteps, 64)
        self.assertGreaterEqual(result.train_success_rate, 0.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_writes_eval_history_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "ppo_metrics.json")
            _, result = train_ppo_cartpole(
                PPOConfig(
                    policy_type="mlp",
                    total_timesteps=64,
                    rollout_steps=32,
                    update_epochs=1,
                    minibatches=1,
                    hidden_size=8,
                    num_envs=1,
                    seed=2,
                    eval_interval=32,
                    eval_rollouts=1,
                    eval_test_max_steps=20,
                    metrics_output=metrics_path,
                )
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertEqual(metrics["config"]["policy_type"], "mlp")
        self.assertGreaterEqual(len(metrics["eval_history"]), 1)
        self.assertEqual(len(metrics["update_history"]), 2)
        self.assertEqual(metrics["update_history"][0]["update"], 1)
        self.assertEqual(metrics["update_history"][0]["timesteps"], 32)
        self.assertEqual(metrics["update_history"][0]["rollout_steps"], 32)
        self.assertIn("reward_mean", metrics["update_history"][0])
        self.assertIn("horizon_truncations", metrics["update_history"][0])
        self.assertIn("failure_terminations", metrics["update_history"][0])
        self.assertEqual(metrics["selected_result"]["timesteps"], result.timesteps)
        self.assertIn("selection_rule", metrics)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_rollout_truncates_at_paper_training_horizon(self):
        env = CartpoleEnv.train_env(seed=0)
        model = MLPActorCritic(hidden_size=8, initial_log_std=-20.0)
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.zero_()
        obs = torch.tensor([env.reset([0.0, 0.0, 0.0, 0.0])], dtype=torch.float32)
        cfg = PPOConfig(rollout_steps=env.cfg.max_steps, num_envs=1)

        rollout = _collect_rollout(
            [env],
            model,
            obs,
            torch.zeros(1, dtype=torch.long),
            None,
            cfg,
        )

        self.assertEqual(env.cfg.max_steps, 250)
        self.assertEqual(rollout.dones[-1, 0].item(), 1.0)
        self.assertEqual(rollout.horizon_truncations[-1, 0].item(), 1.0)
        self.assertEqual(rollout.failure_terminations.sum().item(), 0.0)
        self.assertEqual(rollout.next_episode_steps[0].item(), 0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_rollout_counts_failures_separately_from_horizon_truncations(self):
        env = CartpoleEnv.train_env(seed=0)
        model = MLPActorCritic(hidden_size=8, initial_log_std=-20.0)
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.zero_()
        obs = torch.tensor([env.reset([0.0, 0.0, 0.3, 0.0])], dtype=torch.float32)
        cfg = PPOConfig(rollout_steps=1, num_envs=1)

        rollout = _collect_rollout(
            [env],
            model,
            obs,
            torch.zeros(1, dtype=torch.long),
            None,
            cfg,
        )

        self.assertEqual(rollout.dones[0, 0].item(), 1.0)
        self.assertEqual(rollout.failure_terminations[0, 0].item(), 1.0)
        self.assertEqual(rollout.horizon_truncations.sum().item(), 0.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_stores_raw_continuous_actions_for_log_probs(self):
        env = CartpoleEnv.train_env(seed=0)
        model = MLPActorCritic(hidden_size=8, initial_log_std=2.0)
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.zero_()
            model.actor[-1].bias.fill_(20.0)
        obs = torch.tensor([env.reset([0.0, 0.0, 0.0, 0.0])], dtype=torch.float32)
        cfg = PPOConfig(rollout_steps=8, num_envs=1)

        rollout = _collect_rollout(
            [env],
            model,
            obs,
            torch.zeros(1, dtype=torch.long),
            None,
            cfg,
        )

        self.assertTrue(torch.any(rollout.actions.abs() > env.cfg.force_limit))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_lstm_update_replays_rollout_initial_state(self):
        env = CartpoleEnv.train_env(seed=0)
        model = LSTMActorCritic(hidden_size=8, initial_log_std=-2.0)
        obs = torch.tensor([env.reset([0.0, 0.0, 0.01, 0.0])], dtype=torch.float32)
        initial_state = model.initial_state(1)
        initial_state = (
            initial_state[0] + 0.5,
            initial_state[1] - 0.25,
        )
        cfg = PPOConfig(rollout_steps=4, num_envs=1, update_epochs=1)
        rollout = _collect_rollout(
            [env],
            model,
            obs,
            torch.zeros(1, dtype=torch.long),
            initial_state,
            cfg,
        )

        calls = []
        original = model.sequence_action_and_value

        def spy(obs, action=None, dones=None, initial_state=None):
            calls.append(initial_state)
            return original(obs, action, dones, initial_state)

        model.sequence_action_and_value = spy
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        _update_lstm(model, optimizer, rollout, cfg)

        self.assertIsNotNone(calls[0])
        self.assertTrue(torch.equal(calls[0][0], rollout.initial_lstm_state[0]))
        self.assertTrue(torch.equal(calls[0][1], rollout.initial_lstm_state[1]))


if __name__ == "__main__":
    unittest.main()
