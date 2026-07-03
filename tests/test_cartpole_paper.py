import json
import math
import os
import random
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from cartpole_env import (
    BangBangCartpolePSM,
    CartpoleConfig,
    CartpoleEnv,
    PaperFigure19CartpolePSM,
    evaluate_cartpole_policy,
)
from cartpole_env import PAPER_EVAL_ROLLOUTS
from cartpole_env import (
    CARTPOLE_RESET_HIGH,
    CARTPOLE_RESET_LOW,
    STANDARD_CARTPOLE_REWARD_PER_ALIVE_STEP,
    cartpole_paper_figure19_policy_spec,
    cartpole_reward_spec,
    cartpole_space_spec,
)
from cartpole_synthesis import (
    BooleanTreeSwitch,
    CartpoleSegment,
    CartpoleSynthesisConfig,
    CartpoleTrace,
    Depth2Switch,
    GaussianScalar,
    ObservationPredicate,
    ProbabilisticCartpoleStudent,
    SynthesizedCartpolePSM,
    cartpole_switch_fit_diagnostics,
    fit_probabilistic_cartpole_student,
    fit_probabilistic_cartpole_student_with_history,
    synthesize_cartpole_policy,
    synthesize_cartpole_student,
    synthesize_cartpole_student_with_history,
    _eq12_switch_log_likelihood,
    _action_refinement_candidates,
    _boolean_tree_candidates,
    _best_switch,
    _bootstrap_probabilistic_student,
    _fit_switch_parameter_distributions,
    _gaussian_threshold_pass_probability,
    _greedy_boolean_tree_candidates,
    _condition_initial_mode_responsibilities,
    _duration_refinement_candidates,
    _action_gradient_refinement_candidate,
    _current_student_log_probability,
    _elite_centroid_trace,
    _elite_distribution_sample_trace,
    _elite_distribution_sample_traces,
    _elite_distribution_mean_trace,
    _elite_schedule_weights,
    _fit_elite_schedule_distribution,
    _fit_student_switch,
    _elite_loop_free_schedules,
    _refresh_teacher_elites_with_distribution,
    _duration_gradient_refinement_candidate,
    _elite_kernel_log_probability,
    _limit_loop_free_trace_segment_budget,
    _mode_responsibilities,
    _mode_run_lengths,
    _mode_run_actions,
    _optimize_loop_free_trace,
    _loop_free_trace_distance,
    _prefilter_switches_by_label_mistakes,
    _refine_responsibilities_and_switch_pairs_with_timing,
    _refine_responsibilities_with_switch_timing,
    _refine_loop_free_trace,
    _refine_switch_distribution_means,
    _switch_cache_key,
    _switch_distribution_std_candidates,
    _switch_distribution_timing_loss,
    _gradient_switch_parameter_candidate_distributions,
    _gain_gradient_refinement_candidate,
    _rollout_student_sampled_trace,
    _rollout_with_teacher_gains,
    _sample_switch,
    _schedule_gradient_refinement_candidate,
    _scalar_switch_timing_pairs,
    _scalar_timing_pair_probabilities,
    _single_threshold_transition_probability,
    _switch_responsibility_pair_log_potentials,
    _switch_cost,
    _segments_from_traces,
    _switch_structure_rescore_candidates,
    _switch_structure_cost,
    _switch_structure_objective_cache_key,
    _switch_selector_transition_probabilities,
    _switch_transition_and_stay_probabilities,
    _switch_transition_probability_at_duration,
    _switch_no_transition_probability_before_duration,
    _switch_example_cache,
    _depth2_switch_candidates_with_mistakes,
    _teacher_candidate_traces,
    _top_teacher_elites,
    _switch_timing_loss,
    _switch_timing_pairs,
    _teacher_objective,
    _teacher_refinement_objective,
    _trace_log_probability,
    _time_increment_refinement_candidates,
    _time_increment_gradient_refinement_candidate,
    _teacher_schedule_segments,
    _anchored_rectangle_union_probability,
    _predicate_pair_disabled_cumulative_probabilities,
    _predicate_pair_disabled_rectangles,
    _predicate_pair_enabled_cumulative_probabilities,
    _predicate_pair_enabled_rectangles,
)

if HAS_TORCH:
    from ppo_cartpole import (
        LSTMActorCritic,
        MLPActorCritic,
        PAPER_PPO_TIMESTEPS,
        PPOConfig,
        _collect_rollout,
        _update_lstm,
        ppo_paper_protocol_status,
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

    def test_cartpole_reward_matches_openai_classic_control_step_reward(self):
        env = CartpoleEnv.train_env(seed=0)
        env.reset([0.0, 0.0, 0.0, 0.0])

        _, reward, done = env.step(0.0)
        spec = cartpole_reward_spec()

        self.assertEqual(reward, STANDARD_CARTPOLE_REWARD_PER_ALIVE_STEP)
        self.assertFalse(done)
        self.assertEqual(spec["reward_per_alive_step"], 1.0)
        self.assertEqual(spec["termination_reward"], 0.0)
        self.assertTrue(spec["reward_equals_survived_steps"])
        self.assertIn("OpenAI", spec["source"])

    def test_cartpole_space_spec_records_action_observation_and_reset_contract(self):
        env = CartpoleEnv.train_env(seed=0)
        spec = cartpole_space_spec(env.cfg)

        self.assertIn("action_dimension", spec["paper_specified_fields"])
        self.assertIn("observation_dimension", spec["paper_specified_fields"])
        self.assertIn("initial_state_distribution", spec["local_provenance_fields"])
        self.assertEqual(spec["action_dimension_source"], "paper_figure_8")
        self.assertEqual(spec["observation_dimension_source"], "paper_figure_8")
        self.assertEqual(spec["action_dimension"], 1)
        self.assertEqual(spec["action_space"]["type"], "continuous_scalar_force")
        self.assertEqual(spec["action_space"]["low"], -10.0)
        self.assertEqual(spec["action_space"]["high"], 10.0)
        self.assertEqual(spec["action_space"]["source"], "local_cartpole_env_implementation")
        self.assertEqual(spec["observation_dimension"], 4)
        self.assertEqual(
            spec["observation_space"]["features"],
            ["x", "cart_velocity", "theta", "omega"],
        )
        self.assertEqual(spec["observation_space"]["source"], "local_cartpole_env_implementation")
        reset_spec = spec["initial_state_distribution"]
        self.assertEqual(reset_spec["type"], "independent_uniform")
        self.assertEqual(reset_spec["low"], CARTPOLE_RESET_LOW)
        self.assertEqual(reset_spec["high"], CARTPOLE_RESET_HIGH)
        self.assertEqual(reset_spec["source"], "local_cartpole_env_reset")
        self.assertIn("not separately specified by the paper", spec["note"])
        for _ in range(8):
            obs = env.reset()
            self.assertEqual(len(obs), 4)
            self.assertTrue(all(CARTPOLE_RESET_LOW <= value <= CARTPOLE_RESET_HIGH for value in obs))

    def test_programmatic_policy_evaluates(self):
        metrics = evaluate_cartpole_policy(
            BangBangCartpolePSM(),
            train_rollouts=2,
            test_rollouts=2,
            test_max_steps=100,
        )

        self.assertIn("train_success_rate", metrics)
        self.assertIn("test_success_rate", metrics)

    def test_bangbang_cartpole_psm_acts_before_mode_transition(self):
        policy = BangBangCartpolePSM(force=10.0)

        policy.reset()
        first_action = policy.act([0.0, 0.0, -0.1, 0.0])
        second_action = policy.act([0.0, 0.0, 0.1, 0.0])

        self.assertEqual(first_action, 10.0)
        self.assertEqual(second_action, -10.0)
        self.assertEqual(policy.mode, "push_right")
        self.assertIn("act with current mode", policy.describe())

    def test_paper_figure19_cartpole_policy_transcribes_visible_switches(self):
        spec = cartpole_paper_figure19_policy_spec()
        policy = PaperFigure19CartpolePSM()

        policy.reset()
        first_action = policy.act([0.0, 0.0, 0.0, 0.03])
        second_action = policy.act([0.0, 0.0, -0.05, 0.47])
        third_action = policy.act([0.0, 0.0, 0.0, -0.50])

        self.assertEqual(spec["source"], "manual_visual_inspection_of_pdf_page_21_figure_19")
        self.assertEqual(spec["modes"]["m1"]["action"], -3.3)
        self.assertEqual(spec["modes"]["m2"]["action"], 3.98)
        self.assertEqual(first_action, -3.3)
        self.assertEqual(second_action, -3.3)
        self.assertEqual(third_action, 3.98)
        self.assertEqual(policy.mode, "m1")
        self.assertIn("omega >= 0.020", policy.describe())

    def test_paper_figure19_cartpole_policy_start_can_select_second_mode(self):
        policy = PaperFigure19CartpolePSM()

        policy.reset()
        action = policy.act([0.0, 0.0, 0.0, 0.0])

        self.assertEqual(action, 3.98)
        self.assertEqual(policy.mode, "m2")

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

    def test_cartpole_default_loop_free_teacher_spans_training_horizon(self):
        cfg = CartpoleSynthesisConfig()
        env = CartpoleEnv.train_env(seed=0)

        self.assertEqual(cfg.segment_steps, 1)
        self.assertEqual(cfg.segments_per_trace, env.cfg.max_steps)
        self.assertEqual(cfg.segment_steps * cfg.segments_per_trace, env.cfg.max_steps)

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
        self.assertGreaterEqual(len(history[0].student_fit_history), 1)
        self.assertEqual(
            history[-1].student_fit_history[-1].responsibilities,
            student.responsibilities,
        )

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
        self.assertGreater(
            student.action_distributions[1].mean,
            student.action_distributions[0].mean,
        )
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

    def test_cartpole_student_segments_follow_teacher_loop_free_schedule(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, -0.05, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.05, 0.0],
            ],
            actions=[2.0, 2.0, 4.0, 4.0],
            mode_labels=[1, 1, 1, 1],
            reward=4.0,
            segment_actions=(2.0, 4.0),
            segment_durations=(2, 2),
        )

        segments = _segments_from_traces([trace])[0]

        self.assertEqual(len(segments), 2)
        self.assertEqual([segment.duration for segment in segments], [2, 2])
        self.assertEqual([segment.action_parameter for segment in segments], [2.0, 4.0])
        self.assertEqual([segment.switch_timing_duration for segment in segments], [2.0, 2.0])
        self.assertEqual([segment.timing_step_scale for segment in segments], [1.0, 1.0])

    def test_cartpole_student_segments_use_elapsed_time_increment_duration(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=2.5,
            segment_actions=(-10.0, 10.0),
            segment_durations=(3, 1),
            segment_time_increments=(0.01, 0.02),
        )

        segments = _segments_from_traces([trace])[0]

        self.assertEqual([segment.duration for segment in segments], [3, 1])
        self.assertEqual([segment.switch_timing_duration for segment in segments], [1.5, 1.0])
        self.assertEqual([segment.timing_step_scale for segment in segments], [0.5, 1.0])

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

    def test_cartpole_initial_segment_responsibility_is_fixed_to_mode_zero(self):
        segments_by_trace = [
            [
                CartpoleSegment(
                    observations=[[0.0, 0.0, 0.1, 0.0]],
                    action_parameter=10.0,
                    duration=1,
                    hard_mode=1,
                ),
                CartpoleSegment(
                    observations=[[0.0, 0.0, -0.1, 0.0]],
                    action_parameter=-10.0,
                    duration=1,
                    hard_mode=0,
                ),
            ],
            [
                CartpoleSegment(
                    observations=[[0.0, 0.0, 0.2, 0.0]],
                    action_parameter=10.0,
                    duration=1,
                    hard_mode=1,
                ),
            ],
        ]

        conditioned = _condition_initial_mode_responsibilities(
            segments_by_trace,
            [(0.1, 0.9), (0.8, 0.2), (0.25, 0.75)],
        )

        self.assertEqual(conditioned, [(1.0, 0.0), (0.8, 0.2), (1.0, 0.0)])

    def test_cartpole_switch_timing_responsibilities_are_directed_by_next_mode(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
            ],
            action_parameter=10.0,
            duration=3,
            hard_mode=1,
        )

        pair = _switch_responsibility_pair_log_potentials(
            Depth2Switch(1.0, 0.0, 0.0),
            [GaussianScalar(0.0, 0.05)],
            segment,
        )

        self.assertGreater(pair[1][0], pair[0][1])
        self.assertGreater(pair[1][1], pair[0][0])

    def test_cartpole_student_switch_responsibility_passes_are_configurable(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.4, 0.0],
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.2, 0.0],
            ],
            actions=[-0.9, -0.9, -0.9, 0.1],
            mode_labels=[0, 0, 0, 1],
            reward=4.0,
            segment_actions=(-0.9, 0.1),
            segment_durations=(3, 1),
        )
        action_only_cfg = CartpoleSynthesisConfig(
            student_em_iters=1,
            student_switch_responsibility_passes=0,
        )
        timing_cfg = CartpoleSynthesisConfig(
            student_em_iters=1,
            student_switch_responsibility_passes=1,
        )

        action_only = fit_probabilistic_cartpole_student([trace], action_only_cfg)
        timing_refined = fit_probabilistic_cartpole_student([trace], timing_cfg)

        self.assertEqual(len(action_only.responsibilities), 2)
        self.assertEqual(len(timing_refined.responsibilities), 2)
        self.assertLess(timing_refined.responsibilities[1][0], action_only.responsibilities[1][0])
        self.assertGreater(timing_refined.responsibilities[1][1], action_only.responsibilities[1][1])
        self.assertLess(timing_refined.action_distributions[0].std, action_only.action_distributions[0].std)

    def test_cartpole_switch_timing_e_step_returns_adjacent_pair_posteriors(self):
        trace_segments = [
            CartpoleSegment(
                observations=[
                    [0.0, 0.0, -0.2, 0.0],
                    [0.0, 0.0, -0.1, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                ],
                action_parameter=-10.0,
                duration=3,
                hard_mode=0,
            ),
            CartpoleSegment(
                observations=[[0.0, 0.0, 0.3, 0.0]],
                action_parameter=10.0,
                duration=1,
                hard_mode=1,
            ),
        ]
        action_distributions = {
            0: GaussianScalar(-10.0, 0.1),
            1: GaussianScalar(10.0, 0.1),
        }

        responsibilities, pair_responsibilities = _refine_responsibilities_and_switch_pairs_with_timing(
            [trace_segments],
            action_distributions,
            Depth2Switch(1.0, 0.0, 0.0),
            [GaussianScalar(0.0, 0.001)],
        )

        self.assertEqual(len(responsibilities), 2)
        self.assertEqual(len(pair_responsibilities), 1)
        stay_off, off_to_on, on_to_off, stay_on = pair_responsibilities[0]
        self.assertAlmostEqual(stay_off + off_to_on + on_to_off + stay_on, 1.0)
        self.assertGreater(off_to_on, 0.99)
        self.assertLess(on_to_off, 0.01)

    def test_cartpole_switch_pair_posteriors_are_not_marginal_products(self):
        trace_segments = [
            CartpoleSegment(
                observations=[[0.0, 0.0, -0.2, 0.0]],
                action_parameter=0.0,
                duration=1,
                hard_mode=0,
            ),
            CartpoleSegment(
                observations=[[0.0, 0.0, -0.1, 0.0]],
                action_parameter=0.0,
                duration=1,
                hard_mode=0,
            ),
            CartpoleSegment(
                observations=[[0.0, 0.0, 0.2, 0.0]],
                action_parameter=0.0,
                duration=1,
                hard_mode=1,
            ),
        ]
        action_distributions = {
            0: GaussianScalar(0.0, 10.0),
            1: GaussianScalar(0.0, 10.0),
        }

        responsibilities, pair_responsibilities = _refine_responsibilities_and_switch_pairs_with_timing(
            [trace_segments],
            action_distributions,
            Depth2Switch(1.0, 0.0, 0.0),
            [GaussianScalar(0.0, 0.1)],
        )
        _, off_to_on, _, _ = pair_responsibilities[1]

        self.assertNotAlmostEqual(
            off_to_on,
            responsibilities[1][0] * responsibilities[2][1],
        )

    def test_cartpole_switch_timing_pairs_use_forward_backward_pair_posteriors(self):
        segment = CartpoleSegment(
            observations=[[0.0, 0.0, -0.2, 0.0]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.2, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )

        pair = _switch_timing_pairs(
            [[segment, next_segment]],
            [(0.9, 0.1), (0.2, 0.8)],
            [(0.05, 0.90, 0.03, 0.02)],
        )[0]

        self.assertAlmostEqual(pair.stay_off_weight, 0.05)
        self.assertAlmostEqual(pair.off_to_on_weight, 0.90)
        self.assertAlmostEqual(pair.on_to_off_weight, 0.03)
        self.assertAlmostEqual(pair.stay_on_weight, 0.02)

    def test_cartpole_student_alternates_switch_responsibility_passes_per_em_iteration(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.4, 0.0],
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-1.0, -1.0, -1.0, 1.0, 1.0],
            mode_labels=[0, 0, 0, 1, 1],
            reward=5.0,
            segment_actions=(-1.0, 1.0),
            segment_durations=(3, 2),
        )
        cfg = CartpoleSynthesisConfig(
            student_em_iters=3,
            student_switch_responsibility_passes=2,
        )

        with patch(
            "cartpole_synthesis._refine_responsibilities_and_switch_pairs_with_timing",
            wraps=_refine_responsibilities_and_switch_pairs_with_timing,
        ) as refine_mock:
            student = fit_probabilistic_cartpole_student([trace], cfg)

        self.assertEqual(refine_mock.call_count, 6)
        self.assertEqual(len(student.responsibilities), 2)
        for left_weight, right_weight in student.responsibilities:
            self.assertAlmostEqual(left_weight + right_weight, 1.0)

    def test_cartpole_student_fit_history_records_inner_em_steps(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.4, 0.0],
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-1.0, -1.0, -1.0, 1.0, 1.0],
            mode_labels=[0, 0, 0, 1, 1],
            reward=5.0,
            segment_actions=(-1.0, 1.0),
            segment_durations=(3, 2),
        )
        cfg = CartpoleSynthesisConfig(
            student_em_iters=2,
            student_switch_responsibility_passes=2,
        )

        student, fit_history = fit_probabilistic_cartpole_student_with_history([trace], cfg)

        self.assertEqual(
            [(step.em_iteration, step.responsibility_pass, step.phase) for step in fit_history],
            [
                (1, 0, "action_likelihood_initialization"),
                (1, 1, "switch_timing_refinement"),
                (1, 2, "switch_timing_refinement"),
                (2, 1, "switch_timing_refinement"),
                (2, 2, "switch_timing_refinement"),
            ],
        )
        self.assertEqual(fit_history[-1].responsibilities, student.responsibilities)
        self.assertEqual(fit_history[-1].switch.describe(), student.switch.describe())
        self.assertFalse(fit_history[0].switch_pair_responsibilities)
        self.assertEqual(len(fit_history[-1].switch_pair_responsibilities), 1)
        for step in fit_history:
            self.assertEqual(len(step.responsibilities), 2)
            self.assertEqual(set(step.action_distributions), {0, 1})
            self.assertTrue(step.switch_parameter_distributions)
            for left_weight, right_weight in step.responsibilities:
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
        self.assertEqual(alignment["elapsed_at_boundary_count"], 1)
        self.assertEqual(alignment["elapsed_early_switch_count"], 0)
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
        self.assertEqual(alignment["elapsed_at_boundary_count"], 0)
        self.assertEqual(alignment["elapsed_early_switch_count"], 0)
        self.assertEqual(alignment["elapsed_late_switch_count"], 0)
        self.assertIsNone(alignment["first_enabled_minus_duration_mean"])
        self.assertIsNone(alignment["first_enabled_minus_duration_min"])
        self.assertIsNone(alignment["first_enabled_minus_duration_max"])
        self.assertIsNone(alignment["first_enabled_elapsed_minus_duration_mean"])
        self.assertIsNone(alignment["first_enabled_elapsed_minus_duration_min"])
        self.assertIsNone(alignment["first_enabled_elapsed_minus_duration_max"])

    def test_cartpole_switch_fit_diagnostics_reports_elapsed_boundary_alignment(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0, 10.0],
            mode_labels=[0, 0, 0, 1],
            reward=2.5,
            segment_actions=(-10.0, 10.0),
            segment_durations=(3, 1),
            segment_time_increments=(0.01, 0.02),
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.15),
            switch_threshold_distribution=GaussianScalar(0.15, 0.1),
            switch_parameter_distributions=[GaussianScalar(0.15, 0.1)],
            responsibilities=[(1.0, 0.0), (0.0, 1.0)],
        )

        alignment = cartpole_switch_fit_diagnostics([trace], student)["candidates"]["selected_student_switch"][
            "boundary_alignment"
        ]

        self.assertEqual(alignment["at_boundary_count"], 1)
        self.assertEqual(alignment["elapsed_at_boundary_count"], 1)
        self.assertAlmostEqual(alignment["first_enabled_elapsed_minus_duration_mean"], 0.0)

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

    def test_cartpole_eq12_likelihood_uses_elapsed_time_increment_duration(self):
        default_timing = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        half_dt_timing = CartpoleSegment(
            observations=default_timing.observations,
            action_parameter=default_timing.action_parameter,
            duration=default_timing.duration,
            timing_duration=1.5,
            timing_step_scale=0.5,
            hard_mode=default_timing.hard_mode,
        )
        early_switch = Depth2Switch(1.0, 0.0, 0.0)
        late_switch = Depth2Switch(1.0, 0.0, 0.25)

        self.assertGreater(
            _eq12_switch_log_likelihood(late_switch, half_dt_timing, (1.0, 0.0), (0.0, 1.0)),
            _eq12_switch_log_likelihood(early_switch, half_dt_timing, (1.0, 0.0), (0.0, 1.0)),
        )
        self.assertGreater(
            _eq12_switch_log_likelihood(late_switch, default_timing, (1.0, 0.0), (0.0, 1.0)),
            _eq12_switch_log_likelihood(early_switch, default_timing, (1.0, 0.0), (0.0, 1.0)),
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

    def test_cartpole_switch_timing_loss_penalizes_final_segment_early_transition(self):
        final_segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        segments_by_trace = [[final_segment]]
        responsibilities = [(1.0, 0.0)]

        self.assertGreater(
            _switch_timing_loss(Depth2Switch(1.0, 0.0, 0.0), segments_by_trace, responsibilities),
            _switch_timing_loss(Depth2Switch(1.0, 0.0, 1.0), segments_by_trace, responsibilities),
        )

    def test_cartpole_switch_distribution_timing_loss_penalizes_final_segment_early_transition(self):
        final_segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        segments_by_trace = [[final_segment]]
        responsibilities = [(1.0, 0.0)]

        self.assertGreater(
            _switch_distribution_timing_loss(
                Depth2Switch(1.0, 0.0, 0.0),
                [GaussianScalar(0.0, 0.001)],
                segments_by_trace,
                responsibilities,
            ),
            _switch_distribution_timing_loss(
                Depth2Switch(1.0, 0.0, 1.0),
                [GaussianScalar(1.0, 0.001)],
                segments_by_trace,
                responsibilities,
            ),
        )

    def test_cartpole_eq12_likelihood_is_directed_for_selector_off_transition(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, -0.2, 0.0],
            ],
            action_parameter=10.0,
            duration=3,
            hard_mode=1,
        )
        switch = Depth2Switch(1.0, 0.0, 0.0)

        self.assertGreater(
            _eq12_switch_log_likelihood(switch, segment, (0.0, 1.0), (1.0, 0.0)),
            _eq12_switch_log_likelihood(switch, segment, (1.0, 0.0), (0.0, 1.0)),
        )

    def test_cartpole_scalar_timing_pair_uses_separate_enable_disable_extrema(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, 0.6, 0.0],
                [0.0, 0.0, -0.4, 0.0],
                [0.0, 0.0, -0.2, 0.0],
            ],
            action_parameter=10.0,
            duration=3,
            hard_mode=1,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, -0.2, 0.0]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )
        timing_pair = _switch_timing_pairs([[segment, next_segment]], [(0.0, 1.0), (1.0, 0.0)])
        scalar_pair = _scalar_switch_timing_pairs(Depth2Switch(1.0, 0.0, 0.0), timing_pair)[0]

        self.assertAlmostEqual(scalar_pair.previous_enable_extreme, 0.6)
        self.assertAlmostEqual(scalar_pair.previous_disable_extreme, -0.4)
        _, on_to_off, _, stay_on = _scalar_timing_pair_probabilities(
            GaussianScalar(0.0, 0.05),
            scalar_pair,
        )

        self.assertLess(on_to_off, 0.01)
        self.assertLess(stay_on, 0.01)
        self.assertGreater(scalar_pair.previous_enable_extreme, scalar_pair.previous_disable_extreme)

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

    def test_cartpole_switch_distribution_timing_loss_uses_elapsed_duration(self):
        segments_by_trace = [[
            CartpoleSegment(
                observations=[
                    [0.0, 0.0, -0.2, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                ],
                action_parameter=-10.0,
                duration=3,
                timing_duration=1.5,
                timing_step_scale=0.5,
                hard_mode=0,
            ),
            CartpoleSegment(
                observations=[[0.0, 0.0, 0.3, 0.0]],
                action_parameter=10.0,
                duration=1,
                hard_mode=1,
            ),
        ]]
        responsibilities = [(1.0, 0.0), (0.0, 1.0)]

        self.assertLess(
            _switch_distribution_timing_loss(
                Depth2Switch(1.0, 0.0, 0.25),
                [GaussianScalar(0.25, 0.001)],
                segments_by_trace,
                responsibilities,
            ),
            _switch_distribution_timing_loss(
                Depth2Switch(1.0, 0.0, 0.0),
                [GaussianScalar(0.0, 0.001)],
                segments_by_trace,
                responsibilities,
            ),
        )

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

        with patch("cartpole_synthesis.SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS", 0):
            _, refined = _refine_switch_distribution_means(
                switch,
                initial,
                segments_by_trace,
                responsibilities,
            )

        self.assertLess(refined[0].std, grid_best_std)

    def test_cartpole_switch_gradient_refinement_polishes_coordinate_solution(self):
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

        with patch("cartpole_synthesis.SWITCH_PARAMETER_GRADIENT_REFINEMENT_STEPS", 0):
            coordinate_switch, coordinate = _refine_switch_distribution_means(
                switch,
                initial,
                segments_by_trace,
                responsibilities,
            )
        refined_switch, refined = _refine_switch_distribution_means(
            switch,
            initial,
            segments_by_trace,
            responsibilities,
        )

        self.assertLess(
            _switch_distribution_timing_loss(refined_switch, refined, segments_by_trace, responsibilities),
            _switch_distribution_timing_loss(
                coordinate_switch,
                coordinate,
                segments_by_trace,
                responsibilities,
            ),
        )
        self.assertGreaterEqual(refined[0].std, 1e-3)

    def test_cartpole_switch_gradient_candidate_supports_backtracking(self):
        distribution = GaussianScalar(0.0, 1.0)
        full_step = _gradient_switch_parameter_candidate_distributions(
            [distribution],
            [2.0],
            [(4.0, 4.0)],
            4.0,
            1.0,
        )[0]
        half_step = _gradient_switch_parameter_candidate_distributions(
            [distribution],
            [2.0],
            [(4.0, 4.0)],
            4.0,
            0.5,
        )[0]

        self.assertAlmostEqual(full_step.mean, -1.0)
        self.assertAlmostEqual(half_step.mean, -0.5)
        self.assertLess(full_step.std, half_step.std)
        self.assertLess(half_step.std, distribution.std)

    def test_cartpole_switch_mistake_cache_key_preserves_submillithresholds(self):
        lower = Depth2Switch(1.0, 0.0, 0.0004)
        higher = Depth2Switch(1.0, 0.0, 0.00049)
        examples = [
            ([0.0, 0.0, 0.00045, 0.0], 1),
        ]

        self.assertEqual(lower.describe(), higher.describe())
        self.assertNotEqual(_switch_cache_key(lower), _switch_cache_key(higher))
        self.assertLess(
            _switch_cost(lower, examples)[0],
            _switch_cost(higher, examples)[0],
        )

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

    def test_cartpole_switch_structure_cost_uses_pair_posteriors_for_distribution_fit(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.3, 0.0],
                [0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            action_parameter=-10.0,
            duration=3,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.4, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(0.5, 0.5), (0.5, 0.5)]
        examples = [
            (observation, trace_segment.hard_mode)
            for trace_segments in segments_by_trace
            for trace_segment in trace_segments
            for observation in trace_segment.observations
        ]
        switch = Depth2Switch(1.0, 0.0, 0.0)
        transition_pair = [(0.0, 1.0, 0.0, 0.0)]
        stay_pair = [(1.0, 0.0, 0.0, 0.0)]

        transition_cost = _switch_structure_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            transition_pair,
        )
        stay_cost = _switch_structure_cost(
            switch,
            examples,
            segments_by_trace,
            responsibilities,
            stay_pair,
        )
        marginal_cost = _switch_structure_cost(switch, examples, segments_by_trace, responsibilities)

        self.assertNotEqual(transition_cost[3], stay_cost[3])
        self.assertNotEqual(stay_cost[3], marginal_cost[3])

    def test_cartpole_switch_structure_cache_key_includes_pair_posteriors(self):
        switch = Depth2Switch(1.0, 0.0, 0.0)
        transition_pair = [(0.0, 1.0, 0.0, 0.0)]
        stay_pair = [(1.0, 0.0, 0.0, 0.0)]

        self.assertNotEqual(
            _switch_structure_objective_cache_key(switch, transition_pair),
            _switch_structure_objective_cache_key(switch, stay_pair),
        )
        self.assertNotIn(
            "pair_posteriors",
            _switch_structure_objective_cache_key(switch, None),
        )

    def test_cartpole_switch_structure_cache_key_uses_exact_switch_parameters(self):
        first = Depth2Switch(1.0, 0.0, 0.0004)
        second = Depth2Switch(1.0, 0.0, 0.00049)

        self.assertEqual(first.describe(), second.describe())
        self.assertNotEqual(
            _switch_structure_objective_cache_key(first, None),
            _switch_structure_objective_cache_key(second, None),
        )

    def test_cartpole_switch_structure_rescoring_receives_pair_posteriors_from_m_step(self):
        segment = CartpoleSegment(
            observations=[[0.0, 0.0, -0.1, 0.0]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, 0.1, 0.0]],
            action_parameter=10.0,
            duration=1,
            hard_mode=1,
        )
        trace = CartpoleTrace(
            observations=segment.observations + next_segment.observations,
            actions=[-10.0, 10.0],
            mode_labels=[0, 1],
            reward=2.0,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(0.5, 0.5), (0.5, 0.5)]
        pair_posteriors = [(0.0, 1.0, 0.0, 0.0)]

        with patch("cartpole_synthesis._learn_depth2_switch", return_value=Depth2Switch(1.0, 0.0, 0.0)) as learn:
            _fit_student_switch([trace], segments_by_trace, responsibilities, pair_posteriors)

        self.assertEqual(learn.call_args.args[3], pair_posteriors)

    def test_cartpole_switch_structure_cost_uses_soft_responsibility_label_loss(self):
        segment = CartpoleSegment(
            observations=[
                [0.0, 0.0, -0.2, 0.0],
                [0.0, 0.0, -0.1, 0.0],
            ],
            action_parameter=-10.0,
            duration=2,
            hard_mode=0,
        )
        next_segment = CartpoleSegment(
            observations=[[0.0, 0.0, -0.05, 0.0]],
            action_parameter=-10.0,
            duration=1,
            hard_mode=0,
        )
        segments_by_trace = [[segment, next_segment]]
        responsibilities = [(0.1, 0.9), (0.1, 0.9)]
        examples = [
            (observation, trace_segment.hard_mode)
            for trace_segments in segments_by_trace
            for trace_segment in trace_segments
            for observation in trace_segment.observations
        ]
        hard_aligned = Depth2Switch(1.0, 0.0, 0.0)
        soft_aligned = Depth2Switch(1.0, 0.0, -0.3)

        hard_cost = _switch_structure_cost(hard_aligned, examples, segments_by_trace, responsibilities)
        soft_cost = _switch_structure_cost(soft_aligned, examples, segments_by_trace, responsibilities)

        self.assertLess(_switch_cost(hard_aligned, examples)[0], _switch_cost(soft_aligned, examples)[0])
        self.assertGreater(hard_cost[0], soft_cost[0])

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

        self.assertEqual(len(selected), 32)
        self.assertIn(Depth2Switch(1.0, 0.0, 0.0), selected)
        self.assertNotIn(Depth2Switch(1.0, 0.0, 1.29), selected)

    def test_cartpole_switch_structure_rescore_candidates_caps_distribution_scoring(self):
        examples = [
            ([0.0, 0.0, -0.2, 0.0], 0),
            ([0.0, 0.0, -0.1, 0.0], 0),
        ]
        switches = [
            Depth2Switch(1.0, 0.0, threshold / 100.0)
            for threshold in range(80)
        ]

        cache = {}
        with patch(
            "cartpole_synthesis._fit_switch_structure_objective",
            return_value=(Depth2Switch(1.0, 0.0, 0.0), 0.0, 0.0, 1, "stub"),
        ) as fit_objective:
            selected = _switch_structure_rescore_candidates(switches, examples, [], [], cache=cache)
            for switch in selected:
                _switch_structure_cost(switch, examples, [], [], cache=cache)

        self.assertEqual(len(selected), 32)
        self.assertEqual(fit_objective.call_count, 32)

    def test_cartpole_best_switch_reuses_rescore_cache(self):
        examples = [
            ([0.0, 0.0, -0.2, 0.0], 0),
            ([0.0, 0.0, -0.1, 0.0], 0),
        ]
        switches = [
            BooleanTreeSwitch(ObservationPredicate(2, ">=", threshold / 100.0))
            for threshold in range(80)
        ]

        with patch(
            "cartpole_synthesis._fit_switch_structure_objective",
            return_value=(switches[0], 0.0, 0.0, 1, "stub"),
        ) as fit_objective:
            selected = _best_switch(switches, examples, [], [])

        self.assertIn(selected, switches)
        self.assertEqual(fit_objective.call_count, 32)

    def test_cartpole_switch_prefilter_caps_tied_candidates_deterministically(self):
        best = BooleanTreeSwitch(ObservationPredicate(2, ">=", 0.0))
        tied = [
            (BooleanTreeSwitch(ObservationPredicate(2, ">=", threshold / 100.0)), 0)
            for threshold in range(80)
        ]
        worse = [
            (BooleanTreeSwitch(ObservationPredicate(3, ">=", threshold / 100.0)), 1)
            for threshold in range(10)
        ]

        selected = _prefilter_switches_by_label_mistakes([(best, 0), *worse, *tied])

        self.assertEqual(len(selected), 32)
        self.assertIn(best, selected)
        self.assertTrue(all(_switch.node_count == 1 for _switch in selected))
        self.assertNotIn(worse[0][0], selected)

    def test_cartpole_depth2_prefilter_mistakes_match_switch_cost(self):
        examples = [
            ([0.0, 0.0, -0.2, -0.1], 0),
            ([0.0, 0.0, -0.1, 0.2], 0),
            ([0.0, 0.0, 0.1, -0.1], 1),
            ([0.0, 0.0, 0.2, 0.2], 1),
        ]
        cache = _switch_example_cache(examples)

        for switch, mistakes in _depth2_switch_candidates_with_mistakes(cache)[:25]:
            self.assertEqual(mistakes, _switch_cost(switch, examples)[0])

    def test_cartpole_boolean_tree_switch_supports_depth_two_conjunction(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 1.0),
        )

        self.assertEqual(switch.decide([0.0, 0.0, 0.1, 0.5]), 1)
        self.assertEqual(switch.decide([0.0, 0.0, -0.1, 0.5]), 0)
        self.assertEqual(switch.decide([0.0, 0.0, 0.1, 2.0]), 0)
        self.assertIn("and", switch.describe())

    def test_cartpole_boolean_tree_switch_supports_depth_two_disjunction(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", -1.0),
            "or",
        )

        self.assertEqual(switch.decide([0.0, 0.0, 0.1, 0.5]), 1)
        self.assertEqual(switch.decide([0.0, 0.0, -0.1, -2.0]), 1)
        self.assertEqual(switch.decide([0.0, 0.0, -0.1, 0.5]), 0)
        self.assertIn("or", switch.describe())

    def test_cartpole_boolean_tree_candidates_include_depth_two(self):
        examples = [
            ([0.0, 0.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 1.0], 1),
        ]

        candidates = _boolean_tree_candidates(examples)

        self.assertTrue(any(candidate.second is not None for candidate in candidates))

    def test_cartpole_boolean_tree_candidates_include_disjunction(self):
        examples = [
            ([0.0, 0.0, 0.0, 0.0], 0),
            ([0.0, 0.0, 1.0, 0.0], 1),
            ([0.0, 0.0, 0.0, 1.0], 1),
            ([0.0, 0.0, 1.0, 1.0], 1),
        ]

        candidates = _greedy_boolean_tree_candidates(examples)

        self.assertTrue(any(candidate.second is not None and candidate.operator == "or" for candidate in candidates))
        self.assertEqual(min(_switch_cost(candidate, examples)[0] for candidate in candidates), 0)

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
            "or",
        )

        sampled = _sample_switch(
            switch,
            [GaussianScalar(0.25, 0.0), GaussianScalar(0.75, 0.0)],
            random.Random(0),
        )

        self.assertIsInstance(sampled, BooleanTreeSwitch)
        self.assertEqual(sampled.first.threshold, 0.25)
        self.assertEqual(sampled.second.threshold, 0.75)
        self.assertEqual(sampled.operator, "or")

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

    def test_cartpole_boolean_tree_cumulative_probability_matches_prefix_union(self):
        first = ObservationPredicate(2, ">=", 0.0)
        second = ObservationPredicate(3, "<=", 0.0)
        first_distribution = GaussianScalar(0.0, 0.2)
        second_distribution = GaussianScalar(0.0, 0.2)
        first_values = (-0.2, 0.1, 0.3, -0.1)
        second_values = (0.2, -0.1, 0.1, -0.3)

        def brute_union(rectangles):
            clamped = [
                (min(max(x_bound, 0.0), 1.0), min(max(y_bound, 0.0), 1.0))
                for x_bound, y_bound in rectangles
                if x_bound > 0.0 and y_bound > 0.0
            ]
            if not clamped:
                return 0.0
            x_edges = sorted({0.0, 1.0, *(x_bound for x_bound, _ in clamped)})
            area = 0.0
            for left, right in zip(x_edges, x_edges[1:]):
                probe = (left + right) / 2.0
                area += (right - left) * max((y for x_bound, y in clamped if probe <= x_bound), default=0.0)
            return min(max(area, 0.0), 1.0)

        cumulative = _predicate_pair_enabled_cumulative_probabilities(
            first,
            second,
            first_distribution,
            second_distribution,
            first_values,
            second_values,
            "or",
        )
        prefix_rectangles = []
        expected = []
        for first_value, second_value in zip(first_values, second_values):
            first_probability = _gaussian_threshold_pass_probability(first_value, first_distribution, first.relation)
            second_probability = _gaussian_threshold_pass_probability(second_value, second_distribution, second.relation)
            prefix_rectangles.extend(_predicate_pair_enabled_rectangles(first_probability, second_probability, "or"))
            expected.append(brute_union(prefix_rectangles))

        for actual, wanted in zip(cumulative, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertAlmostEqual(_anchored_rectangle_union_probability(prefix_rectangles), brute_union(prefix_rectangles))

        disabled_cumulative = _predicate_pair_disabled_cumulative_probabilities(
            first,
            second,
            first_distribution,
            second_distribution,
            first_values,
            second_values,
            "and",
        )
        disabled_rectangles = []
        disabled_expected = []
        for first_value, second_value in zip(first_values, second_values):
            first_probability = _gaussian_threshold_pass_probability(first_value, first_distribution, first.relation)
            second_probability = _gaussian_threshold_pass_probability(second_value, second_distribution, second.relation)
            disabled_rectangles.extend(_predicate_pair_disabled_rectangles(first_probability, second_probability, "and"))
            disabled_expected.append(brute_union(disabled_rectangles))
        for actual, wanted in zip(disabled_cumulative, disabled_expected):
            self.assertAlmostEqual(actual, wanted)

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

    def test_cartpole_deterministic_psm_acts_before_mode_transition(self):
        policy = SynthesizedCartpolePSM(
            -9.0,
            9.0,
            Depth2Switch(1.0, 0.0, 0.0),
        )

        policy.reset()
        first_action = policy.act([0.0, 0.0, 0.1, 0.0])
        second_action = policy.act([0.0, 0.0, -0.1, 0.0])

        self.assertEqual(first_action, -9.0)
        self.assertEqual(second_action, 9.0)
        self.assertEqual(policy.mode, 0)

    def test_cartpole_probabilistic_rollout_resamples_parameters_on_mode_change(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-1.0, 1.0),
                1: GaussianScalar(1.0, 1.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )
        policy = student.sample_segment_resampling_policy(random.Random(0))

        policy.reset()
        initial_right = policy.right_force
        first_action = policy.act([0.0, 0.0, -0.1, 0.0])
        first_left = policy.left_force
        second_action = policy.act([0.0, 0.0, 0.1, 0.0])
        second_right = policy.right_force
        third_action = policy.act([0.0, 0.0, -0.1, 0.0])
        third_left = policy.left_force

        self.assertEqual(policy.mode, 0)
        self.assertEqual(first_action, first_left)
        self.assertEqual(second_action, first_left)
        self.assertEqual(third_action, second_right)
        self.assertEqual(initial_right, 0.0)
        self.assertNotEqual(first_left, third_left)
        self.assertNotEqual(second_right, initial_right)

    def test_cartpole_probabilistic_rollout_acts_before_detected_mode_transition(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-9.0, 0.0),
                1: GaussianScalar(9.0, 0.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(10.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )
        policy = student.sample_segment_resampling_policy(random.Random(0))

        policy.reset()
        policy.switch = Depth2Switch(1.0, 0.0, 0.0)
        action = policy.act([0.0, 0.0, 0.1, 0.0])

        self.assertEqual(policy.mode, 1)
        self.assertEqual(action, -9.0)

    def test_cartpole_student_sampled_trace_labels_action_mode_before_transition(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=1, segments_per_trace=1)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-9.0, 0.0),
                1: GaussianScalar(9.0, 0.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )

        trace = _rollout_student_sampled_trace(
            [0.0, 0.0, 0.1, 0.0],
            env.cfg,
            cfg,
            student,
            random.Random(0),
        )

        self.assertEqual(trace.mode_labels[0], 0)
        self.assertEqual(trace.actions[0], -9.0)

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

    def test_cartpole_combined_switch_probabilities_use_shared_threshold_samples(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 0.5),
        )
        distributions = [GaussianScalar(0.0, 0.2), GaussianScalar(0.5, 0.2)]
        observations = [
            [0.0, 0.0, -0.2, 0.4],
            [0.0, 0.0, 0.1, 0.6],
            [0.0, 0.0, 0.2, 0.2],
        ]

        transition, stay = _switch_transition_and_stay_probabilities(
            switch,
            distributions,
            observations,
            3,
        )
        rectangles = [
            (
                _gaussian_threshold_pass_probability(observation[2], distributions[0], ">="),
                _gaussian_threshold_pass_probability(observation[3], distributions[1], "<="),
            )
            for observation in observations
        ]

        def union_area(prefix):
            xs = sorted({0.0, 1.0, *(x for x, _ in prefix)})
            area = 0.0
            for left, right in zip(xs, xs[1:]):
                probe = (left + right) / 2.0
                area += (right - left) * max((y for x, y in prefix if probe <= x), default=0.0)
            return area

        enabled_before = union_area(rectangles[:2])
        expected_transition = union_area(rectangles) - enabled_before
        expected_stay = 1.0 - enabled_before

        self.assertAlmostEqual(transition, expected_transition)
        self.assertAlmostEqual(stay, expected_stay)
        independent_step_probability = rectangles[2][0] * rectangles[2][1]
        self.assertNotAlmostEqual(transition, independent_step_probability)
        self.assertAlmostEqual(
            transition,
            _switch_transition_probability_at_duration(switch, distributions, observations, 3),
        )
        self.assertAlmostEqual(
            stay,
            _switch_no_transition_probability_before_duration(switch, distributions, observations, 3),
        )

    def test_cartpole_or_switch_probabilities_use_shared_threshold_samples(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 0.5),
            "or",
        )
        distributions = [GaussianScalar(0.0, 0.2), GaussianScalar(0.5, 0.2)]
        observations = [
            [0.0, 0.0, -0.2, 0.7],
            [0.0, 0.0, 0.1, 0.6],
            [0.0, 0.0, -0.1, 0.2],
        ]

        transition, stay = _switch_transition_and_stay_probabilities(
            switch,
            distributions,
            observations,
            3,
        )
        rectangles = []
        for observation in observations:
            first_probability = _gaussian_threshold_pass_probability(observation[2], distributions[0], ">=")
            second_probability = _gaussian_threshold_pass_probability(observation[3], distributions[1], "<=")
            rectangles.extend([(first_probability, 1.0), (1.0, second_probability)])

        def union_area(prefix):
            xs = sorted({0.0, 1.0, *(x for x, _ in prefix)})
            area = 0.0
            for left, right in zip(xs, xs[1:]):
                probe = (left + right) / 2.0
                area += (right - left) * max((y for x, y in prefix if probe <= x), default=0.0)
            return area

        enabled_before = union_area(rectangles[:4])
        expected_transition = union_area(rectangles) - enabled_before
        expected_stay = 1.0 - enabled_before

        self.assertAlmostEqual(transition, expected_transition)
        self.assertAlmostEqual(stay, expected_stay)
        self.assertAlmostEqual(
            transition,
            _switch_transition_probability_at_duration(switch, distributions, observations, 3),
        )
        self.assertAlmostEqual(
            stay,
            _switch_no_transition_probability_before_duration(switch, distributions, observations, 3),
        )

    def test_cartpole_first_step_selector_probability_matches_switch_state_mass(self):
        switch = BooleanTreeSwitch(
            ObservationPredicate(2, ">=", 0.0),
            ObservationPredicate(3, "<=", 0.5),
            "and",
        )
        distributions = [GaussianScalar(0.0, 0.2), GaussianScalar(0.5, 0.2)]
        observation = [0.0, 0.0, 0.1, 0.4]

        off_to_on, on_to_off, stay_off, stay_on = _switch_selector_transition_probabilities(
            switch,
            distributions,
            [observation],
            1,
        )
        expected_on = (
            _gaussian_threshold_pass_probability(observation[2], distributions[0], ">=")
            * _gaussian_threshold_pass_probability(observation[3], distributions[1], "<=")
        )

        self.assertAlmostEqual(off_to_on, expected_on)
        self.assertAlmostEqual(on_to_off, 1.0 - expected_on)
        self.assertEqual(stay_off, 1.0)
        self.assertEqual(stay_on, 1.0)

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
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, 10.0],
            mode_labels=[0, 1],
            reward=1.0,
        )
        mismatched_trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
            ],
            actions=[-10.0, 0.0],
            mode_labels=[0, 1],
            reward=2.0,
        )

        self.assertGreater(
            _teacher_objective(matching_trace, student, cfg),
            _teacher_objective(mismatched_trace, student, cfg),
        )

    def test_cartpole_teacher_objective_recomputes_cached_probability_for_current_student(self):
        cfg = CartpoleSynthesisConfig(teacher_reward_lambda=0.0, teacher_student_regularizer=1.0)
        current_student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 10.0),
            switch_threshold_distribution=GaussianScalar(10.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(10.0, 0.1)],
            responsibilities=[(0.5, 0.5)],
        )
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            actions=[-10.0, -10.0],
            mode_labels=[0, 0],
            reward=1.0,
            segment_actions=(-10.0,),
            segment_durations=(2,),
            student_log_probability=123.0,
        )

        self.assertAlmostEqual(
            _current_student_log_probability(trace, current_student),
            _trace_log_probability(trace, current_student),
        )
        self.assertAlmostEqual(
            _teacher_objective(trace, current_student, cfg),
            _trace_log_probability(trace, current_student),
        )
        self.assertNotEqual(_teacher_objective(trace, current_student, cfg), trace.student_log_probability)

    def test_cartpole_current_student_log_probability_handles_empty_uncached_trace(self):
        student = _bootstrap_probabilistic_student(CartpoleSynthesisConfig())
        trace = CartpoleTrace(observations=[], actions=[], mode_labels=[], reward=0.0)

        self.assertEqual(_current_student_log_probability(trace, student), 0.0)

    def test_cartpole_trace_log_probability_uses_fixed_initial_mode(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-1.0, 1.0),
                1: GaussianScalar(1.0, 1.0),
            },
            switch=Depth2Switch(1.0, 0.0, 10.0),
            switch_threshold_distribution=GaussianScalar(10.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(10.0, 0.1)],
            responsibilities=[(0.5, 0.5)],
        )
        trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[1.0],
            mode_labels=[0],
            reward=1.0,
        )

        self.assertAlmostEqual(
            _trace_log_probability(trace, student),
            GaussianScalar(-1.0, 1.0).log_pdf(1.0),
        )

    def test_cartpole_trace_log_probability_penalizes_final_segment_early_transition(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4, 0.0],
            ],
            actions=[-10.0, -10.0, -10.0],
            mode_labels=[0, 0, 0],
            reward=3.0,
            segment_actions=(-10.0,),
            segment_durations=(3,),
        )
        early_switch_student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 0.001),
            switch_parameter_distributions=[GaussianScalar(0.0, 0.001)],
            responsibilities=[(1.0, 0.0)],
        )
        staying_student = ProbabilisticCartpoleStudent(
            action_distributions=early_switch_student.action_distributions,
            switch=Depth2Switch(1.0, 0.0, 1.0),
            switch_threshold_distribution=GaussianScalar(1.0, 0.001),
            switch_parameter_distributions=[GaussianScalar(1.0, 0.001)],
            responsibilities=[(1.0, 0.0)],
        )

        self.assertGreater(
            _trace_log_probability(trace, staying_student),
            _trace_log_probability(trace, early_switch_student),
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

    def test_cartpole_teacher_elite_distance_treats_missing_time_increments_as_defaults(self):
        implicit_default = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
        )
        explicit_default = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
            segment_time_increments=(0.02,),
        )

        self.assertEqual(_loop_free_trace_distance(implicit_default, explicit_default), 0.0)

    def test_cartpole_teacher_elite_distance_includes_teacher_gains(self):
        reference = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
        )
        same_schedule_different_gains = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            theta_gain=20.0,
            omega_gain=3.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
        )

        self.assertGreater(_loop_free_trace_distance(reference, same_schedule_different_gains), 0.0)

    def test_cartpole_teacher_elite_distance_includes_recorded_modes(self):
        reference = CartpoleTrace(
            observations=[],
            actions=[10.0, 10.0],
            mode_labels=[0, 1],
            reward=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(1, 1),
        )
        same_schedule_different_modes = CartpoleTrace(
            observations=[],
            actions=[10.0, 10.0],
            mode_labels=[1, 1],
            reward=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(1, 1),
        )

        self.assertEqual(
            _loop_free_trace_distance(reference, same_schedule_different_modes),
            1.0,
        )

    def test_cartpole_teacher_elite_distance_normalizes_actions(self):
        left = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=0.0,
            segment_actions=(-10.0,),
            segment_durations=(1,),
        )
        right = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[0],
            reward=0.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
        )

        self.assertAlmostEqual(_loop_free_trace_distance(left, right), 2.0)

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

    def test_cartpole_elite_kernel_recomputes_elite_probability_for_current_student(self):
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 10.0),
            switch_threshold_distribution=GaussianScalar(10.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(10.0, 0.1)],
            responsibilities=[(0.5, 0.5)],
        )
        elite = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            actions=[-10.0, -10.0],
            mode_labels=[0, 0],
            reward=1.0,
            segment_actions=(-10.0,),
            segment_durations=(2,),
            student_log_probability=123.0,
        )
        close = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            segment_actions=(-10.0,),
            segment_durations=(2,),
        )
        far = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
        )

        self.assertAlmostEqual(_elite_kernel_log_probability(close, student, [elite]), 0.0)
        self.assertGreater(
            _elite_kernel_log_probability(close, student, [elite]),
            _elite_kernel_log_probability(far, student, [elite]),
        )

    def test_cartpole_teacher_elite_centroid_recombines_loop_free_schedules(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=3, segments_per_trace=3)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 2.0),
                1: GaussianScalar(10.0, 2.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 3),
            teacher_source="student_sample",
            student_log_probability=-2.0,
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=20.0,
            omega_gain=3.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(3, 1),
            teacher_source="student_sample_refined",
            student_log_probability=-3.0,
        )

        centroid = _elite_centroid_trace(
            [left, right],
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            student,
        )

        self.assertIsNotNone(centroid)
        assert centroid is not None
        self.assertEqual(centroid.segment_actions, (0.0, 10.0))
        self.assertEqual(centroid.segment_durations, (2, 2))
        self.assertEqual(centroid.teacher_source, "student_elite_centroid")
        self.assertEqual(centroid.theta_gain, 15.0)
        self.assertEqual(centroid.omega_gain, 2.0)
        self.assertIsNotNone(centroid.student_log_probability)
        self.assertEqual(len(centroid.segment_actions), len(centroid.segment_durations))

    def test_cartpole_teacher_elite_distribution_sample_uses_top_rho_statistics(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=3)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 2.0),
                1: GaussianScalar(10.0, 2.0),
            },
            switch=Depth2Switch(1.0, 0.0, 0.0),
            switch_threshold_distribution=GaussianScalar(0.0, 1.0),
            switch_parameter_distributions=[GaussianScalar(0.0, 1.0)],
            responsibilities=[(0.5, 0.5)],
        )
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 4),
            segment_time_increments=(env.cfg.dt, env.cfg.dt / 2.0),
            teacher_source="student_sample",
            student_log_probability=-2.0,
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=20.0,
            omega_gain=3.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(4, 1),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt),
            teacher_source="student_sample_refined",
            student_log_probability=-3.0,
        )
        schedules = [
            (left.segment_actions, left.segment_durations, left.segment_time_increments, (0, 1), left),
            (right.segment_actions, right.segment_durations, right.segment_time_increments, (1, 1), right),
        ]

        sample = _elite_distribution_sample_trace(
            schedules,
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            random.Random(3),
            student,
        )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.teacher_source, "student_elite_distribution_sample")
        self.assertEqual(len(sample.segment_actions), len(sample.segment_durations))
        self.assertEqual(len(sample.segment_time_increments), len(sample.segment_durations))
        self.assertLessEqual(len(sample.segment_actions), cfg.segments_per_trace)
        self.assertTrue(all(1 <= duration <= cfg.segment_steps for duration in sample.segment_durations))
        self.assertTrue(all(0.0 < increment <= env.cfg.dt for increment in sample.segment_time_increments))
        self.assertTrue(all(-env.cfg.force_limit <= action <= env.cfg.force_limit for action in sample.segment_actions))
        self.assertNotEqual(sample.theta_gain, 15.0)
        self.assertNotEqual(sample.omega_gain, 2.0)
        self.assertIsNotNone(sample.student_log_probability)

    def test_cartpole_teacher_elite_distribution_sample_uses_gaussian_statistics(self):
        class RecordingRng:
            def __init__(self) -> None:
                self.calls = []

            def gauss(self, mean, std):
                self.calls.append((mean, std))
                return mean + std

        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=5, segments_per_trace=2)
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 5),
            segment_time_increments=(0.01, 0.02),
            teacher_source="student_sample",
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(5, 1),
            segment_time_increments=(0.02, 0.01),
            teacher_source="student_sample",
        )
        rng = RecordingRng()

        sample = _elite_distribution_sample_trace(
            [
                (left.segment_actions, left.segment_durations, left.segment_time_increments, (0, 1), left),
                (right.segment_actions, right.segment_durations, right.segment_time_increments, (1, 1), right),
            ],
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            rng,
        )

        self.assertIsNotNone(sample)
        self.assertEqual(
            rng.calls,
            [
                (0.0, 10.0),
                (3.0, 2.0),
                (0.015, 0.005),
                (10.0, 0.001),
                (3.0, 2.0),
                (0.015, 0.005),
                (0.0, 1e-06),
                (0.0, 1e-06),
            ],
        )
        assert sample is not None
        self.assertEqual(sample.segment_actions, (10.0, 10.0))
        self.assertEqual(sample.segment_durations, (5, 5))
        self.assertEqual(sample.segment_time_increments, (0.02, 0.02))
        self.assertEqual(sample.theta_gain, 1e-06)
        self.assertEqual(sample.omega_gain, 1e-06)

    def test_cartpole_teacher_elite_distribution_sample_fits_gain_statistics(self):
        class RecordingRng:
            def __init__(self) -> None:
                self.calls = []

            def gauss(self, mean, std):
                self.calls.append((mean, std))
                return mean + std

        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=5, segments_per_trace=1)
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(-10.0,),
            segment_durations=(1,),
            segment_time_increments=(0.01,),
            teacher_source="student_sample",
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=20.0,
            omega_gain=3.0,
            segment_actions=(10.0,),
            segment_durations=(5,),
            segment_time_increments=(0.02,),
            teacher_source="student_sample",
        )
        rng = RecordingRng()

        sample = _elite_distribution_sample_trace(
            [
                (left.segment_actions, left.segment_durations, left.segment_time_increments, (0,), left),
                (right.segment_actions, right.segment_durations, right.segment_time_increments, (1,), right),
            ],
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            rng,
        )

        self.assertIsNotNone(sample)
        self.assertEqual(rng.calls[-2:], [(15.0, 5.0), (2.0, 1.0)])
        assert sample is not None
        self.assertEqual(sample.theta_gain, 20.0)
        self.assertEqual(sample.omega_gain, 3.0)

    def test_cartpole_teacher_elite_distribution_mean_uses_fitted_statistics(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=5, segments_per_trace=2)
        student = _bootstrap_probabilistic_student(cfg)
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 5),
            segment_time_increments=(env.cfg.dt, env.cfg.dt / 2.0),
            teacher_source="student_sample",
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=1.0,
            theta_gain=20.0,
            omega_gain=3.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(5, 1),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt),
            teacher_source="student_sample_refined",
        )

        mean_trace = _elite_distribution_mean_trace(
            [
                (left.segment_actions, left.segment_durations, left.segment_time_increments, (0, 1), left),
                (right.segment_actions, right.segment_durations, right.segment_time_increments, (1, 1), right),
            ],
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            student,
        )

        self.assertIsNotNone(mean_trace)
        assert mean_trace is not None
        self.assertEqual(mean_trace.teacher_source, "student_elite_distribution_mean")
        self.assertEqual(mean_trace.segment_actions, (0.0, 10.0))
        self.assertEqual(mean_trace.segment_durations, (3, 3))
        self.assertEqual(mean_trace.segment_time_increments, (0.015, 0.015))
        self.assertEqual(mean_trace.theta_gain, 15.0)
        self.assertEqual(mean_trace.omega_gain, 2.0)
        self.assertIsNotNone(mean_trace.student_log_probability)

    def test_cartpole_teacher_elite_centroid_preserves_recorded_modes(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        left = CartpoleTrace(
            observations=[[0.0, 0.0, 0.05, 0.0], [0.0, 0.0, 0.04, 0.0]],
            actions=[10.0, 10.0],
            mode_labels=[0, 1],
            reward=2.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(1, 1),
            segment_time_increments=(env.cfg.dt, env.cfg.dt),
            teacher_source="student_sample",
        )

        centroid = _elite_centroid_trace(
            [left],
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
        )

        self.assertIsNotNone(centroid)
        assert centroid is not None
        self.assertEqual(centroid.segment_actions, (10.0, 10.0))
        self.assertEqual(centroid.mode_labels, [0, 1])
        self.assertEqual(
            [segment.hard_mode for segment in _teacher_schedule_segments(centroid)],
            [0, 1],
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

    def test_cartpole_student_sampled_teacher_respects_training_horizon(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=8, segments_per_trace=32)
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(0.0, 0.0),
                1: GaussianScalar(0.0, 0.0),
            },
            switch=Depth2Switch(1.0, 0.0, 1.0),
            switch_threshold_distribution=GaussianScalar(1.0, 0.0),
            switch_parameter_distributions=[GaussianScalar(1.0, 0.0)],
            responsibilities=[(0.5, 0.5)],
        )

        trace = _rollout_student_sampled_trace(
            [0.0, 0.0, 0.0, 0.0],
            env.cfg,
            cfg,
            student,
            random.Random(0),
        )

        self.assertEqual(len(trace.actions), env.cfg.max_steps)
        self.assertEqual(trace.reward, float(env.cfg.max_steps))
        self.assertEqual(sum(trace.segment_durations), len(trace.actions))

    def test_cartpole_student_sampled_teacher_respects_loop_free_segment_budget(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=8, segments_per_trace=4)
        raw_trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0] for _ in range(9)],
            actions=[-10.0, 10.0, -8.0, 8.0, -6.0, 6.0, -4.0, 4.0, -2.0],
            mode_labels=[0, 1, 0, 1, 0, 1, 0, 1, 0],
            reward=9.0,
            segment_actions=(-10.0, 10.0, -8.0, 8.0, -6.0, 6.0, -4.0, 4.0, -2.0),
            segment_durations=(1, 1, 1, 1, 1, 1, 1, 1, 1),
            teacher_source="student_sample",
        )

        trace = _limit_loop_free_trace_segment_budget(
            raw_trace,
            [0.0, 0.0, -0.1, -1.0],
            env.cfg,
            cfg,
        )

        self.assertEqual(trace.teacher_source, "student_sample")
        self.assertLessEqual(len(trace.segment_actions), cfg.segments_per_trace)
        self.assertTrue(all(duration <= cfg.segment_steps for duration in trace.segment_durations))
        self.assertEqual(len(trace.segment_actions), len(trace.segment_durations))
        self.assertEqual(sum(trace.segment_durations), len(trace.actions))

    def test_cartpole_student_sample_projection_preserves_action_runs(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=3, segments_per_trace=3)
        raw_trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0] for _ in range(7)],
            actions=(-10.0, -10.0, 10.0, 10.0, 10.0, -10.0, -10.0),
            mode_labels=[0, 0, 1, 1, 1, 0, 0],
            reward=7.0,
            segment_actions=(-10.0, 10.0, -10.0),
            segment_durations=(2, 3, 2),
            teacher_source="student_sample",
        )

        trace = _limit_loop_free_trace_segment_budget(
            raw_trace,
            [0.0, 0.0, -0.1, -1.0],
            env.cfg,
            cfg,
        )

        self.assertEqual(trace.segment_actions, (-10.0, 10.0, -10.0))
        self.assertEqual(trace.segment_durations, (2, 3, 2))
        self.assertTrue(all(action in {-10.0, 10.0} for action in trace.segment_actions))
        self.assertEqual(len(trace.segment_time_increments), len(trace.segment_durations))

    def test_cartpole_student_sample_projection_preserves_mode_runs(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=3, segments_per_trace=3)
        raw_trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0] for _ in range(7)],
            actions=(5.0, 5.0, 5.0, 5.0, 5.0, -5.0, -5.0),
            mode_labels=[0, 0, 1, 1, 1, 0, 0],
            reward=7.0,
            segment_actions=(5.0, 5.0, -5.0),
            segment_durations=(2, 3, 2),
            teacher_source="student_sample",
        )

        trace = _limit_loop_free_trace_segment_budget(
            raw_trace,
            [0.0, 0.0, -0.1, -1.0],
            env.cfg,
            cfg,
        )

        self.assertEqual(trace.segment_actions, (5.0, 5.0, -5.0))
        self.assertEqual(trace.segment_durations, (2, 3, 2))
        self.assertEqual(_mode_run_lengths(trace.mode_labels), (2, 3, 2))
        self.assertEqual(_mode_run_actions(trace.actions, trace.mode_labels), trace.segment_actions)

    def test_cartpole_projected_student_sample_recomputes_student_log_probability(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
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
        raw_trace = CartpoleTrace(
            observations=[],
            actions=[-10.0, 10.0, -10.0, 10.0, -10.0],
            mode_labels=[0, 1, 0, 1, 0],
            reward=5.0,
            segment_actions=(-10.0, 10.0, -10.0, 10.0, -10.0),
            segment_durations=(1, 1, 1, 1, 1),
            segment_time_increments=tuple(env.cfg.dt for _ in range(5)),
            teacher_source="student_sample",
            student_log_probability=123.0,
        )

        projected = _limit_loop_free_trace_segment_budget(
            raw_trace,
            [0.0, 0.0, -0.1, -1.0],
            env.cfg,
            cfg,
            student,
        )

        self.assertLessEqual(len(projected.segment_actions), cfg.segments_per_trace)
        self.assertIsNotNone(projected.student_log_probability)
        self.assertNotEqual(projected.student_log_probability, raw_trace.student_log_probability)
        self.assertAlmostEqual(
            projected.student_log_probability,
            _trace_log_probability(projected, student),
        )
        self.assertLess(
            _teacher_objective(projected, student, cfg),
            cfg.teacher_reward_lambda * projected.reward + 123.0,
        )

    def test_cartpole_student_sample_projection_uses_single_likelihood_recompute(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=1, segments_per_trace=1)
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

        with patch("cartpole_synthesis._trace_log_probability", return_value=-7.0) as log_probability:
            trace = _rollout_student_sampled_trace(
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                student,
                random.Random(0),
            )

        self.assertLessEqual(len(trace.segment_actions), cfg.segments_per_trace)
        self.assertEqual(trace.student_log_probability, -7.0)
        self.assertEqual(log_probability.call_count, 1)

    def test_cartpole_teacher_bootstrap_uses_probabilistic_student_prior(self):
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=4,
        )

        bootstrap = _bootstrap_probabilistic_student(cfg)
        candidates = _teacher_candidate_traces(
            [0.0, 0.0, 0.05, 0.0],
            CartpoleEnv.train_env(seed=0).cfg,
            cfg,
            random.Random(1),
            None,
        )

        self.assertEqual(bootstrap.switch.describe(), "mode=1 if 1.000*theta + 0.250*omega >= 0.000, else mode=0")
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(trace.teacher_source == "bootstrap_student_sample" for trace in candidates))
        self.assertTrue(all(trace.student_log_probability is not None for trace in candidates))
        self.assertTrue(all(len(trace.segment_actions) <= cfg.segments_per_trace for trace in candidates))

    def test_cartpole_teacher_candidate_pool_uses_student_samples_after_first_iteration(self):
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
        self.assertTrue(all(trace.teacher_source == "student_sample" for trace in candidates))
        self.assertTrue(all(len(trace.segment_actions) <= cfg.segments_per_trace for trace in candidates))

    def test_cartpole_teacher_optimization_bootstrap_returns_prior_sample(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=4,
            segment_steps=2,
            segments_per_trace=3,
            teacher_top_rho=4,
            teacher_refinement_steps=1,
        )

        trace = _optimize_loop_free_trace(
            [0.0, 0.0, 0.05, 0.0],
            env.cfg,
            cfg,
            random.Random(0),
            None,
        )

        self.assertIn(
            trace.teacher_source,
            {
                "bootstrap_student_sample",
                "bootstrap_student_sample_refined",
                "bootstrap_elite_centroid",
                "bootstrap_elite_centroid_refined",
                "bootstrap_elite_distribution_mean",
                "bootstrap_elite_distribution_mean_refined",
                "bootstrap_elite_distribution_sample",
                "bootstrap_elite_distribution_sample_refined",
            },
        )
        self.assertIsNotNone(trace.student_log_probability)
        self.assertIsNotNone(trace.teacher_objective)
        self.assertIsNotNone(trace.teacher_refinement_objective)
        self.assertAlmostEqual(
            trace.teacher_objective,
            _teacher_objective(trace, _bootstrap_probabilistic_student(cfg), cfg),
        )

    def test_cartpole_teacher_optimization_records_selected_refinement_objective(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=1,
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=1,
            teacher_refinement_steps=0,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        student = _bootstrap_probabilistic_student(cfg)
        candidate = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=3.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )

        with patch(
            "cartpole_synthesis._teacher_candidate_traces",
            return_value=[candidate],
        ), patch(
            "cartpole_synthesis._elite_centroid_trace",
            return_value=None,
        ), patch(
            "cartpole_synthesis._refresh_teacher_elites_with_distribution",
            return_value=([candidate], []),
        ):
            trace = _optimize_loop_free_trace(
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
                student,
            )

        self.assertEqual(trace, candidate)
        self.assertEqual(trace.teacher_objective, _teacher_objective(trace, student, cfg))
        self.assertEqual(
            trace.teacher_refinement_objective,
            _teacher_refinement_objective(trace, student, cfg, [candidate]),
        )

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
            {
                "student_sample",
                "student_sample_refined",
                "student_elite_centroid",
                "student_elite_centroid_refined",
                "student_elite_distribution_mean",
                "student_elite_distribution_mean_refined",
                "student_elite_distribution_sample",
                "student_elite_distribution_sample_refined",
            },
        )
        self.assertGreaterEqual(trace.reward, 1.0)

    def test_cartpole_teacher_optimization_can_select_elite_centroid(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=2,
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=2,
            teacher_refinement_steps=0,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        student = _bootstrap_probabilistic_student(cfg)
        poor_left = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 1),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )
        poor_right = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(1, 1),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )

        with patch(
            "cartpole_synthesis._teacher_candidate_traces",
            return_value=[poor_left, poor_right],
        ), patch(
            "cartpole_synthesis._refresh_teacher_elites_with_distribution",
            return_value=([poor_left, poor_right], []),
        ), patch(
            "cartpole_synthesis._rollout_with_teacher_gains",
            side_effect=[
                CartpoleTrace(
                    observations=[],
                    actions=[0.0, 10.0],
                    mode_labels=[0, 1],
                    reward=5.0,
                    theta_gain=0.0,
                    omega_gain=0.0,
                    segment_actions=(0.0, 10.0),
                    segment_durations=(1, 1),
                    teacher_source="gain_sample",
                )
            ],
        ):
            trace = _optimize_loop_free_trace(
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
                student,
            )

        self.assertEqual(trace.teacher_source, "student_elite_centroid")
        self.assertEqual(trace.segment_actions, (0.0, 10.0))
        self.assertEqual(trace.reward, 5.0)

    def test_cartpole_teacher_optimization_can_select_elite_distribution_sample(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=2,
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=2,
            teacher_refinement_steps=0,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        student = _bootstrap_probabilistic_student(cfg)
        poor_left = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, 10.0),
            segment_durations=(1, 1),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )
        poor_right = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_durations=(1, 1),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )
        distribution_sample = CartpoleTrace(
            observations=[],
            actions=[1.0, 10.0],
            mode_labels=[1, 1],
            reward=6.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(1.0, 10.0),
            segment_durations=(1, 1),
            teacher_source="student_elite_distribution_sample",
            student_log_probability=0.0,
        )

        with patch(
            "cartpole_synthesis._teacher_candidate_traces",
            return_value=[poor_left, poor_right],
        ), patch(
            "cartpole_synthesis._refresh_teacher_elites_with_distribution",
            return_value=([distribution_sample, poor_left], [distribution_sample]),
        ), patch(
            "cartpole_synthesis._rollout_with_teacher_gains",
            return_value=CartpoleTrace(
                observations=[],
                actions=[0.0, 10.0],
                mode_labels=[0, 1],
                reward=5.0,
                theta_gain=0.0,
                omega_gain=0.0,
                segment_actions=(0.0, 10.0),
                segment_durations=(1, 1),
                teacher_source="gain_sample",
            ),
        ):
            trace = _optimize_loop_free_trace(
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
                student,
            )

        self.assertEqual(trace.teacher_source, "student_elite_distribution_sample")
        self.assertEqual(trace.segment_actions, (1.0, 10.0))
        self.assertEqual(trace.reward, 6.0)

    def test_cartpole_teacher_refinement_uses_refreshed_distribution_elites(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            candidate_rollouts=1,
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=1,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=10.0,
        )
        student = _bootstrap_probabilistic_student(cfg)
        initial_elite = CartpoleTrace(
            observations=[],
            actions=[0.0],
            mode_labels=[1],
            reward=5.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
            teacher_source="student_sample",
            student_log_probability=0.0,
        )
        refreshed_elite = CartpoleTrace(
            observations=[],
            actions=[5.0],
            mode_labels=[1],
            reward=5.0,
            segment_actions=(5.0,),
            segment_durations=(1,),
            teacher_source="student_elite_distribution_sample",
            student_log_probability=0.0,
        )
        refinement_elite_rewards: list[float] = []

        def fake_refine(candidate, _initial_state, _env_cfg, _cfg, _student, elites):
            refinement_elite_rewards.append(elites[0].reward)
            return candidate

        with patch(
            "cartpole_synthesis._teacher_candidate_traces",
            return_value=[initial_elite],
        ), patch(
            "cartpole_synthesis._refresh_teacher_elites_with_distribution",
            return_value=([refreshed_elite], [refreshed_elite]),
        ), patch(
            "cartpole_synthesis._elite_centroid_trace",
            return_value=None,
        ), patch(
            "cartpole_synthesis._refine_loop_free_trace",
            side_effect=fake_refine,
        ):
            trace = _optimize_loop_free_trace(
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
                student,
            )

        self.assertTrue(refinement_elite_rewards)
        self.assertEqual(set(refinement_elite_rewards), {5.0})
        self.assertEqual(trace.teacher_source, "student_elite_distribution_sample")

    def test_cartpole_teacher_elite_distribution_resample_count_is_configurable(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=2,
            teacher_elite_distribution_resamples=3,
        )
        elite = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=1.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
            teacher_source="student_sample",
        )
        samples = [
            CartpoleTrace(
                observations=[],
                actions=[float(index)],
                mode_labels=[1],
                reward=float(index),
                segment_actions=(float(index),),
                segment_durations=(1,),
                teacher_source="student_elite_distribution_sample",
            )
            for index in range(3)
        ]

        with patch(
            "cartpole_synthesis._elite_distribution_sample_trace_from_distribution",
            side_effect=samples,
        ) as sample_trace:
            candidates = _elite_distribution_sample_traces(
                [elite],
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
            )

        self.assertEqual(candidates[1:], samples)
        self.assertEqual(candidates[0].teacher_source, "student_elite_distribution_mean")
        self.assertEqual(sample_trace.call_count, 3)

    def test_cartpole_teacher_elite_distribution_fits_schedule_parameters(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=2)
        left = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=1.0,
            segment_actions=(-10.0, 0.0),
            segment_durations=(1, 2),
            segment_time_increments=(env.cfg.dt, env.cfg.dt / 2.0),
            teacher_source="student_sample",
        )
        right = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=2.0,
            theta_gain=14.0,
            omega_gain=3.0,
            segment_actions=(10.0, 4.0),
            segment_durations=(3, 4),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt),
            teacher_source="student_sample",
        )

        distribution = _fit_elite_schedule_distribution(
            _elite_loop_free_schedules([left, right], env.cfg.dt),
            env.cfg,
            cfg,
        )

        self.assertIsNotNone(distribution)
        assert distribution is not None
        self.assertEqual(len(distribution.segments), 2)
        self.assertAlmostEqual(distribution.theta_gain.mean, 12.0)
        self.assertAlmostEqual(distribution.theta_gain.std, 2.0)
        self.assertAlmostEqual(distribution.omega_gain.mean, 2.0)
        self.assertAlmostEqual(distribution.segments[0].action.mean, 0.0)
        self.assertAlmostEqual(distribution.segments[0].action.std, 10.0)
        self.assertAlmostEqual(distribution.segments[0].duration.mean, 2.0)
        self.assertEqual(distribution.segments[0].mode, 0)
        self.assertAlmostEqual(
            distribution.segments[1].time_increment.mean,
            0.75 * env.cfg.dt,
        )
        self.assertEqual(distribution.source_elites, (left, right))

    def test_cartpole_teacher_elite_schedule_weights_follow_student_objective(self):
        cfg = CartpoleSynthesisConfig(teacher_reward_lambda=1.0, teacher_student_regularizer=0.0)
        low = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
        )
        high = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=2.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
        )
        student = _bootstrap_probabilistic_student(cfg)
        schedules = _elite_loop_free_schedules([low, high], CartpoleEnv.train_env(seed=0).cfg.dt)

        weights = _elite_schedule_weights(schedules, student, cfg)

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[1], weights[0])

    def test_cartpole_teacher_elite_schedule_weights_include_student_likelihood(self):
        cfg = CartpoleSynthesisConfig(teacher_reward_lambda=0.0, teacher_student_regularizer=1.0)
        likely = CartpoleTrace(
            observations=[
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            actions=[-10.0, -10.0],
            mode_labels=[0, 0],
            reward=1.0,
            segment_actions=(-10.0,),
            segment_durations=(2,),
        )
        unlikely = CartpoleTrace(
            observations=[
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            actions=[0.0, 0.0],
            mode_labels=[0, 0],
            reward=1.0,
            segment_actions=(0.0,),
            segment_durations=(2,),
        )
        student = ProbabilisticCartpoleStudent(
            action_distributions={
                0: GaussianScalar(-10.0, 0.1),
                1: GaussianScalar(10.0, 0.1),
            },
            switch=Depth2Switch(1.0, 0.0, 1.0),
            switch_threshold_distribution=GaussianScalar(1.0, 0.1),
            switch_parameter_distributions=[GaussianScalar(1.0, 0.1)],
            responsibilities=[(0.5, 0.5)],
        )
        schedules = _elite_loop_free_schedules([unlikely, likely], CartpoleEnv.train_env(seed=0).cfg.dt)

        weights = _elite_schedule_weights(schedules, student, cfg)

        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[1], weights[0])

    def test_cartpole_teacher_elite_schedule_weights_are_uniform_without_student(self):
        cfg = CartpoleSynthesisConfig()
        left = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
        )
        right = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=100.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
        )
        schedules = _elite_loop_free_schedules([left, right], CartpoleEnv.train_env(seed=0).cfg.dt)

        self.assertEqual(_elite_schedule_weights(schedules, None, cfg), [0.5, 0.5])

    def test_cartpole_teacher_elite_distribution_weights_top_rho_statistics_by_objective(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=4,
            segments_per_trace=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        low = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=0.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
            segment_time_increments=(env.cfg.dt / 2.0,),
            teacher_source="student_sample",
        )
        high = CartpoleTrace(
            observations=[],
            actions=[],
            mode_labels=[],
            reward=2.0,
            theta_gain=10.0,
            omega_gain=2.0,
            segment_actions=(10.0,),
            segment_durations=(3,),
            segment_time_increments=(env.cfg.dt,),
            teacher_source="student_sample",
        )
        schedules = _elite_loop_free_schedules([low, high], env.cfg.dt)
        student = _bootstrap_probabilistic_student(cfg)

        uniform_distribution = _fit_elite_schedule_distribution(schedules, env.cfg, cfg)
        weighted_distribution = _fit_elite_schedule_distribution(schedules, env.cfg, cfg, student)

        self.assertIsNotNone(uniform_distribution)
        self.assertIsNotNone(weighted_distribution)
        assert uniform_distribution is not None
        assert weighted_distribution is not None
        self.assertAlmostEqual(uniform_distribution.segments[0].action.mean, 5.0)
        self.assertGreater(weighted_distribution.segments[0].action.mean, uniform_distribution.segments[0].action.mean)
        self.assertGreater(weighted_distribution.theta_gain.mean, uniform_distribution.theta_gain.mean)
        self.assertEqual(weighted_distribution.segments[0].mode, 1)

    def test_cartpole_teacher_elite_distribution_rounds_refresh_elites(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
            teacher_elite_distribution_resamples=1,
            teacher_elite_distribution_rounds=2,
        )
        initial_elite = CartpoleTrace(
            observations=[],
            actions=[0.0],
            mode_labels=[1],
            reward=1.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
            teacher_source="student_sample",
        )
        improved_sample = CartpoleTrace(
            observations=[],
            actions=[5.0],
            mode_labels=[1],
            reward=5.0,
            segment_actions=(5.0,),
            segment_durations=(1,),
            teacher_source="student_elite_distribution_sample",
        )
        second_round_sample = CartpoleTrace(
            observations=[],
            actions=[6.0],
            mode_labels=[1],
            reward=6.0,
            segment_actions=(6.0,),
            segment_durations=(1,),
            teacher_source="student_elite_distribution_sample",
        )
        sampled_from: list[float] = []

        def fake_sample(distribution, *_args):
            sampled_from.append(distribution.segments[0].action.mean)
            return improved_sample if len(sampled_from) == 1 else second_round_sample

        with patch("cartpole_synthesis._elite_distribution_sample_trace_from_distribution", side_effect=fake_sample):
            refreshed_elites, candidates = _refresh_teacher_elites_with_distribution(
                [initial_elite],
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
            )

        self.assertEqual(
            [candidate.teacher_source for candidate in candidates],
            [
                "student_elite_distribution_mean",
                "student_elite_distribution_sample",
                "student_elite_distribution_mean",
                "student_elite_distribution_sample",
            ],
        )
        self.assertEqual([candidates[1].reward, candidates[3].reward], [5.0, 6.0])
        self.assertEqual(sampled_from, [0.0, 5.0])
        self.assertEqual(refreshed_elites, [second_round_sample])
        self.assertEqual(_top_teacher_elites([initial_elite, improved_sample], None, cfg), [improved_sample])

    def test_cartpole_teacher_elite_distribution_rounds_refit_distribution(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=2,
            teacher_top_rho=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
            teacher_elite_distribution_resamples=1,
            teacher_elite_distribution_rounds=2,
        )
        initial_elite = CartpoleTrace(
            observations=[],
            actions=[0.0],
            mode_labels=[0],
            reward=1.0,
            segment_actions=(0.0,),
            segment_durations=(1,),
            teacher_source="student_sample",
        )
        improved_sample = CartpoleTrace(
            observations=[],
            actions=[5.0],
            mode_labels=[1],
            reward=5.0,
            segment_actions=(5.0,),
            segment_durations=(1,),
            teacher_source="student_elite_distribution_sample",
        )
        second_sample = CartpoleTrace(
            observations=[],
            actions=[6.0],
            mode_labels=[1],
            reward=6.0,
            segment_actions=(6.0,),
            segment_durations=(1,),
            teacher_source="student_elite_distribution_sample",
        )
        sampled_means: list[float] = []

        def fake_sample(distribution, *_args):
            sampled_means.append(distribution.segments[0].action.mean)
            return improved_sample if len(sampled_means) == 1 else second_sample

        with patch(
            "cartpole_synthesis._elite_distribution_sample_trace_from_distribution",
            side_effect=fake_sample,
        ):
            refreshed_elites, _ = _refresh_teacher_elites_with_distribution(
                [initial_elite],
                [0.0, 0.0, 0.05, 0.0],
                env.cfg,
                cfg,
                random.Random(0),
            )

        self.assertEqual(sampled_means, [0.0, 5.0])
        self.assertEqual(refreshed_elites, [second_sample])

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
        self.assertEqual(trace.segment_time_increments, (env.cfg.dt, env.cfg.dt, env.cfg.dt))
        self.assertEqual(trace.segment_actions, (10.0, 10.0, 10.0))
        self.assertEqual(len(trace.segment_actions), len(trace.segment_durations))
        self.assertEqual(len(trace.segment_time_increments), len(trace.segment_durations))

    def test_cartpole_teacher_schedule_segments_use_recorded_modes(self):
        trace = CartpoleTrace(
            observations=[
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2, 0.0],
            ],
            actions=[5.0, 5.0],
            mode_labels=[0, 0],
            reward=2.0,
            segment_actions=(5.0,),
            segment_durations=(2,),
            segment_time_increments=(0.02,),
        )

        segments = _teacher_schedule_segments(trace)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].action_parameter, 5.0)
        self.assertEqual(segments[0].hard_mode, 0)

    def test_cartpole_teacher_rollout_uses_segment_time_increments(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=1)
        initial_state = [0.0, 1.0, 0.05, 0.0]

        default_trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0,),
        )
        half_dt_trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_durations=(2,),
            segment_actions=(10.0,),
            segment_time_increments=(env.cfg.dt / 2.0,),
        )

        self.assertEqual(half_dt_trace.segment_time_increments, (env.cfg.dt / 2.0,))
        self.assertEqual(len(default_trace.actions), len(half_dt_trace.actions))
        self.assertAlmostEqual(half_dt_trace.reward, default_trace.reward / 2.0)
        self.assertNotEqual(default_trace.observations[-1], half_dt_trace.observations[-1])

    def test_cartpole_teacher_rollout_respects_training_horizon(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=8, segments_per_trace=32)

        trace = _rollout_with_teacher_gains(
            [0.0, 0.0, 0.0, 0.0],
            env.cfg,
            cfg,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_durations=tuple(300 for _ in range(40)),
            segment_actions=tuple(0.0 for _ in range(40)),
        )

        self.assertEqual(len(trace.actions), env.cfg.max_steps)
        self.assertEqual(trace.reward, float(env.cfg.max_steps))
        self.assertEqual(sum(trace.segment_durations), len(trace.actions))
        self.assertLessEqual(len(trace.segment_durations), cfg.segments_per_trace)
        self.assertTrue(all(duration <= cfg.segment_steps for duration in trace.segment_durations))

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

    def test_cartpole_teacher_duration_refinement_preserves_recorded_modes(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_modes=(0, 1),
        )

        candidates = _duration_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.mode_labels[0] == 0 for candidate in candidates))
        self.assertTrue(all(1 in candidate.mode_labels for candidate in candidates))

    def test_cartpole_teacher_time_increment_refinement_preserves_action_and_duration(self):
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
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt / 2.0, env.cfg.dt / 2.0),
        )

        candidates = _time_increment_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.segment_actions, trace.segment_actions)
            self.assertEqual(candidate.segment_durations, trace.segment_durations)
            changed = sum(
                int(abs(left - right) > 1e-12)
                for left, right in zip(candidate.segment_time_increments, trace.segment_time_increments)
            )
            self.assertEqual(changed, 1)

    def test_cartpole_teacher_time_increment_refinement_preserves_recorded_modes(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt / 2.0),
            segment_modes=(0, 1),
        )

        candidates = _time_increment_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.mode_labels == trace.mode_labels for candidate in candidates))

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
            self.assertTrue(
                all(
                    -env.cfg.force_limit <= action <= env.cfg.force_limit
                    for action in candidate.segment_actions
                )
            )

    def test_cartpole_teacher_action_refinement_uses_continuous_local_steps(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(0.0, 10.0),
        )

        candidates = _action_refinement_candidates(trace, initial_state, env.cfg, cfg)
        actions = {candidate.segment_actions for candidate in candidates}

        self.assertIn((-5.0, 10.0), actions)
        self.assertIn((5.0, 10.0), actions)
        self.assertIn((0.0, 5.0), actions)
        self.assertNotIn((0.0, -10.0), actions)

    def test_cartpole_teacher_action_refinement_preserves_recorded_modes(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, 10.0),
            segment_modes=(0, 1),
        )

        candidates = _action_refinement_candidates(trace, initial_state, env.cfg, cfg)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.mode_labels == trace.mode_labels for candidate in candidates))

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

    def test_cartpole_teacher_gain_gradient_uses_central_differences(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.01, -0.049]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=5.0,
            omega_gain=2.0,
        )
        seen_gains = []

        def objective(candidate):
            seen_gains.append((candidate.theta_gain, candidate.omega_gain))
            return candidate.theta_gain

        candidate = _gain_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((4.875, 2.0), seen_gains)
        self.assertIn((5.125, 2.0), seen_gains)
        self.assertIn((5.0, 1.95), seen_gains)
        self.assertIn((5.0, 2.05), seen_gains)
        assert candidate is not None
        self.assertGreater(candidate.theta_gain, trace.theta_gain)
        self.assertEqual(candidate.omega_gain, trace.omega_gain)

    def test_cartpole_teacher_gain_gradient_backtracks_to_improving_step(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.01, -0.049]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=20.0,
            omega_gain=2.0,
        )
        evaluated_theta = []

        def objective(candidate):
            evaluated_theta.append(candidate.theta_gain)
            if abs(candidate.theta_gain - 21.0) < 1e-9:
                return -1.0
            return 1.0 - abs(candidate.theta_gain - 20.5)

        candidate = _gain_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.theta_gain, 20.5)
        self.assertEqual(candidate.omega_gain, trace.omega_gain)
        self.assertIn(21.0, evaluated_theta)
        self.assertIn(20.5, evaluated_theta)

    def test_cartpole_teacher_gain_gradient_refinement_can_be_accepted(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=3,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=10.0,
            omega_gain=2.0,
            segment_actions=(-10.0,),
            segment_durations=(1,),
            teacher_source="gain_sample",
        )
        gradient_candidate = CartpoleTrace(
            observations=[],
            actions=[10.0],
            mode_labels=[1],
            reward=5.0,
            theta_gain=11.0,
            omega_gain=2.0,
            segment_actions=(10.0,),
            segment_durations=(1,),
            teacher_source="gain_sample",
        )

        with patch(
            "cartpole_synthesis._duration_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._time_increment_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._gain_gradient_refinement_candidate",
            return_value=gradient_candidate,
        ), patch(
            "cartpole_synthesis._action_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._duration_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._time_increment_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._schedule_gradient_refinement_candidate",
            return_value=None,
        ):
            refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertEqual(refined.teacher_source, "gain_refined")
        self.assertEqual(refined.theta_gain, gradient_candidate.theta_gain)
        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_action_gradient_uses_central_differences(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(0.0, 10.0),
        )
        seen_actions = []

        def objective(candidate):
            seen_actions.append(candidate.segment_actions)
            first_action = candidate.segment_actions[0]
            return first_action

        candidate = _action_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((-1.0, 10.0), seen_actions)
        self.assertIn((1.0, 10.0), seen_actions)
        assert candidate is not None
        self.assertEqual(candidate.segment_durations, trace.segment_durations)
        self.assertGreater(candidate.segment_actions[0], trace.segment_actions[0])
        self.assertEqual(candidate.segment_actions[1], trace.segment_actions[1])

    def test_cartpole_teacher_action_gradient_backtracks_to_improving_step(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(0.0, 10.0),
        )
        evaluated_actions = []

        def objective(candidate):
            first_action = candidate.segment_actions[0]
            evaluated_actions.append(first_action)
            if abs(first_action - 2.0) < 1e-9:
                return -1.0
            return 1.0 - abs(first_action - 1.0)

        candidate = _action_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.segment_actions[0], 1.0)
        self.assertIn(2.0, evaluated_actions)
        self.assertIn(1.0, evaluated_actions)

    def test_cartpole_teacher_action_gradient_refinement_can_be_accepted(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=2,
            segments_per_trace=3,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(1, 1, 1),
            teacher_source="student_sample",
        )
        gradient_candidate = CartpoleTrace(
            observations=[],
            actions=[0.0],
            mode_labels=[0],
            reward=5.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0, -10.0, -10.0),
            segment_durations=(1, 1, 1),
            teacher_source="gain_sample",
        )

        with patch(
            "cartpole_synthesis._duration_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_gradient_refinement_candidate",
            return_value=gradient_candidate,
        ), patch(
            "cartpole_synthesis._duration_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._time_increment_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._schedule_gradient_refinement_candidate",
            return_value=None,
        ):
            refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertEqual(refined.teacher_source, "student_sample_refined")
        self.assertEqual(refined.segment_actions, gradient_candidate.segment_actions)
        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_duration_gradient_uses_central_differences(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, -10.0),
            segment_durations=(2, 2),
        )
        seen_durations = []

        def objective(candidate):
            seen_durations.append(candidate.segment_durations)
            return candidate.segment_durations[0]

        candidate = _duration_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((1, 2), seen_durations)
        self.assertIn((3, 2), seen_durations)
        assert candidate is not None
        self.assertEqual(candidate.segment_actions, trace.segment_actions)
        self.assertGreater(candidate.segment_durations[0], trace.segment_durations[0])
        self.assertEqual(candidate.segment_durations[1], trace.segment_durations[1])

    def test_cartpole_teacher_duration_gradient_backtracking_rejects_worse_integer_step(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, -10.0),
            segment_durations=(2, 2),
        )
        evaluated_durations = []

        def objective(candidate):
            evaluated_durations.append(candidate.segment_durations)
            first_duration = candidate.segment_durations[0]
            if first_duration == 1:
                return -3.0
            if first_duration == 3:
                return -1.0
            return 0.0

        candidate = _duration_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNone(candidate)
        self.assertGreaterEqual(evaluated_durations.count((3, 2)), 3)

    def test_cartpole_teacher_duration_gradient_refinement_can_be_accepted(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=4,
            segments_per_trace=3,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(1, 1, 1),
            teacher_source="student_sample",
        )
        gradient_candidate = CartpoleTrace(
            observations=[],
            actions=[-10.0, -10.0],
            mode_labels=[0, 0],
            reward=5.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(2, 1, 1),
            teacher_source="gain_sample",
        )

        with patch(
            "cartpole_synthesis._duration_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._duration_gradient_refinement_candidate",
            return_value=gradient_candidate,
        ), patch(
            "cartpole_synthesis._time_increment_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._schedule_gradient_refinement_candidate",
            return_value=None,
        ):
            refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertEqual(refined.teacher_source, "student_sample_refined")
        self.assertEqual(refined.segment_durations, gradient_candidate.segment_durations)
        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_time_increment_gradient_uses_central_differences(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, -10.0),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt / 2.0),
        )
        seen_increments = []

        def objective(candidate):
            seen_increments.append(candidate.segment_time_increments)
            return candidate.segment_time_increments[0]

        candidate = _time_increment_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((0.009000000000000001, 0.01), seen_increments)
        self.assertIn((0.011, 0.01), seen_increments)
        assert candidate is not None
        self.assertEqual(candidate.segment_actions, trace.segment_actions)
        self.assertEqual(candidate.segment_durations, trace.segment_durations)
        self.assertGreater(candidate.segment_time_increments[0], trace.segment_time_increments[0])
        self.assertEqual(candidate.segment_time_increments[1], trace.segment_time_increments[1])

    def test_cartpole_teacher_time_increment_gradient_backtracks_to_improving_step(self):
        env = CartpoleEnv(CartpoleConfig(pole_length=0.5, horizon_seconds=5.0, dt=0.04), seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=2)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=1.0,
            omega_gain=0.0,
            segment_actions=(10.0, -10.0),
            segment_time_increments=(env.cfg.dt / 2.0, env.cfg.dt / 2.0),
        )
        evaluated_increments = []

        def objective(candidate):
            first_increment = candidate.segment_time_increments[0]
            evaluated_increments.append(first_increment)
            if abs(first_increment - 0.024) < 1e-12:
                return -1.0
            return 1.0 - abs(first_increment - 0.022)

        candidate = _time_increment_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.segment_time_increments[0], 0.022)
        self.assertTrue(any(abs(value - 0.024) < 1e-12 for value in evaluated_increments))
        self.assertTrue(any(abs(value - 0.022) < 1e-12 for value in evaluated_increments))

    def test_cartpole_teacher_schedule_gradient_uses_joint_central_differences(self):
        env = CartpoleEnv(CartpoleConfig(pole_length=0.5, horizon_seconds=5.0, dt=0.04), seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=1)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0,),
            segment_durations=(2,),
            segment_time_increments=(0.02,),
            segment_modes=(0,),
        )
        seen_actions = []
        seen_durations = []
        seen_increments = []

        def objective(candidate):
            seen_actions.append(candidate.segment_actions)
            seen_durations.append(candidate.segment_durations)
            seen_increments.append(candidate.segment_time_increments)
            return (
                candidate.segment_actions[0]
                + 5.0 * candidate.segment_durations[0]
                + 100.0 * candidate.segment_time_increments[0]
            )

        candidate = _schedule_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((-1.0,), seen_actions)
        self.assertIn((1.0,), seen_actions)
        self.assertIn((1,), seen_durations)
        self.assertIn((3,), seen_durations)
        self.assertTrue(any(abs(increments[0] - 0.018) < 1e-12 for increments in seen_increments))
        self.assertTrue(any(abs(increments[0] - 0.022) < 1e-12 for increments in seen_increments))
        assert candidate is not None
        self.assertGreater(candidate.segment_actions[0], trace.segment_actions[0])
        self.assertGreater(candidate.segment_durations[0], trace.segment_durations[0])
        self.assertGreater(candidate.segment_time_increments[0], trace.segment_time_increments[0])

    def test_cartpole_teacher_schedule_gradient_includes_teacher_gains(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=2, segments_per_trace=1)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=10.0,
            omega_gain=2.0,
            segment_actions=(10.0,),
            segment_durations=(2,),
            segment_modes=(1,),
        )
        seen_gains = []

        def objective(candidate):
            seen_gains.append((candidate.theta_gain, candidate.omega_gain))
            return candidate.theta_gain + 2.0 * candidate.omega_gain

        candidate = _schedule_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        self.assertIn((9.75, 2.0), seen_gains)
        self.assertIn((10.25, 2.0), seen_gains)
        self.assertIn((10.0, 1.95), seen_gains)
        self.assertIn((10.0, 2.05), seen_gains)
        assert candidate is not None
        self.assertGreater(candidate.theta_gain, trace.theta_gain)
        self.assertGreater(candidate.omega_gain, trace.omega_gain)
        self.assertEqual(candidate.segment_actions, trace.segment_actions)
        self.assertEqual(candidate.segment_durations, trace.segment_durations)

    def test_cartpole_teacher_schedule_gradient_backtracks_to_improving_step(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(segment_steps=4, segments_per_trace=1)
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = _rollout_with_teacher_gains(
            initial_state,
            env.cfg,
            cfg,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0,),
            segment_durations=(2,),
            segment_time_increments=(env.cfg.dt,),
            segment_modes=(0,),
        )
        evaluated_actions = []

        def objective(candidate):
            action = candidate.segment_actions[0]
            evaluated_actions.append(action)
            return 1.0 - abs(action - 1.0)

        candidate = _schedule_gradient_refinement_candidate(
            trace,
            initial_state,
            env.cfg,
            cfg,
            objective,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.segment_actions[0], 1.0)
        self.assertIn(2.0, evaluated_actions)
        self.assertIn(1.0, evaluated_actions)

    def test_cartpole_teacher_schedule_gradient_refinement_can_be_accepted(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=4,
            segments_per_trace=3,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(1, 1, 1),
            segment_time_increments=(env.cfg.dt, env.cfg.dt, env.cfg.dt),
            teacher_source="student_sample",
        )
        gradient_candidate = CartpoleTrace(
            observations=[],
            actions=[0.0, -10.0],
            mode_labels=[0, 0],
            reward=5.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0, -10.0, -10.0),
            segment_durations=(2, 1, 1),
            segment_time_increments=(env.cfg.dt, env.cfg.dt, env.cfg.dt),
            teacher_source="gain_sample",
        )

        with patch(
            "cartpole_synthesis._duration_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._time_increment_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._duration_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._time_increment_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._schedule_gradient_refinement_candidate",
            return_value=gradient_candidate,
        ):
            refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertEqual(refined.teacher_source, "student_sample_refined")
        self.assertEqual(refined.segment_actions, gradient_candidate.segment_actions)
        self.assertEqual(refined.segment_durations, gradient_candidate.segment_durations)
        self.assertGreaterEqual(
            _teacher_objective(refined, None, cfg),
            _teacher_objective(trace, None, cfg),
        )

    def test_cartpole_teacher_finite_difference_refinement_rejects_worse_candidates(self):
        env = CartpoleEnv.train_env(seed=0)
        cfg = CartpoleSynthesisConfig(
            segment_steps=4,
            segments_per_trace=3,
            teacher_refinement_steps=1,
            teacher_reward_lambda=1.0,
            teacher_student_regularizer=0.0,
        )
        initial_state = [0.0, 0.0, 0.05, 0.0]
        trace = CartpoleTrace(
            observations=[],
            actions=[-10.0] * 5,
            mode_labels=[0] * 5,
            reward=5.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(2, 2, 1),
            teacher_source="student_sample",
        )
        worse_action = CartpoleTrace(
            observations=[],
            actions=[0.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(0.0, -10.0, -10.0),
            segment_durations=(2, 2, 1),
            teacher_source="gain_sample",
        )
        worse_duration = CartpoleTrace(
            observations=[],
            actions=[-10.0],
            mode_labels=[0],
            reward=1.0,
            theta_gain=0.0,
            omega_gain=0.0,
            segment_actions=(-10.0, -10.0, -10.0),
            segment_durations=(1, 2, 1),
            teacher_source="gain_sample",
        )

        with patch(
            "cartpole_synthesis._duration_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_refinement_candidates",
            return_value=[],
        ), patch(
            "cartpole_synthesis._action_gradient_refinement_candidate",
            return_value=worse_action,
        ), patch(
            "cartpole_synthesis._duration_gradient_refinement_candidate",
            return_value=worse_duration,
        ), patch(
            "cartpole_synthesis._time_increment_gradient_refinement_candidate",
            return_value=None,
        ), patch(
            "cartpole_synthesis._schedule_gradient_refinement_candidate",
            return_value=None,
        ):
            refined = _refine_loop_free_trace(trace, initial_state, env.cfg, cfg, None)

        self.assertIs(refined, trace)
        self.assertEqual(refined.teacher_source, "student_sample")
        self.assertEqual(refined.segment_actions, trace.segment_actions)
        self.assertEqual(refined.segment_durations, trace.segment_durations)

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
    def test_ppo_config_defaults_to_paper_timestep_budget(self):
        self.assertEqual(PPOConfig().total_timesteps, PAPER_PPO_TIMESTEPS)
        self.assertEqual(PAPER_PPO_TIMESTEPS, 10_000_000)
        self.assertEqual(
            PPOConfig().pretrain_teacher_mode_update_order,
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertEqual(PPOConfig().pretrain_teacher_policy, "BangBangCartpolePSM")

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_protocol_status_distinguishes_single_run_from_full_baseline(self):
        status = ppo_paper_protocol_status(
            PPOConfig(
                policy_type="lstm",
                total_timesteps=PAPER_PPO_TIMESTEPS,
                eval_test_max_steps=CartpoleEnv.test_env().cfg.max_steps,
                eval_rollouts=PAPER_EVAL_ROLLOUTS,
                minibatches=1,
            )
        )

        self.assertEqual(status["train_horizon_seconds"], 5.0)
        self.assertEqual(status["train_pole_length"], 0.5)
        self.assertEqual(status["test_horizon_seconds"], 300.0)
        self.assertEqual(status["test_pole_length"], 1.0)
        self.assertTrue(status["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(status["space_spec"]["action_dimension"], 1)
        self.assertEqual(status["space_spec"]["observation_dimension"], 4)
        self.assertEqual(status["space_spec"]["initial_state_distribution"]["low"], -0.05)
        self.assertEqual(status["space_spec"]["initial_state_distribution"]["high"], 0.05)
        self.assertTrue(status["paper_timestep_budget"])
        self.assertTrue(status["paper_test_horizon"])
        self.assertEqual(status["paper_eval_rollouts"], 1000)
        self.assertEqual(status["selected_eval_rollouts"], 1000)
        self.assertTrue(status["uses_paper_eval_rollouts"])
        self.assertEqual(status["pretrain_steps"], 0)
        self.assertIsNone(status["pretrain_teacher_policy"])
        self.assertTrue(status["pretrain_teacher_mode_order_recorded"])
        self.assertTrue(status["pretrain_teacher_policy_matches_implementation"])
        self.assertTrue(status["pretrain_teacher_mode_order_matches_implementation"])
        self.assertFalse(status["local_supervised_warm_start"])
        self.assertTrue(status["no_local_supervised_warm_start"])
        self.assertTrue(status["ppo_lstm_minibatches_fixed_to_one"])
        self.assertTrue(status["single_run_matches_paper_budget"])
        self.assertFalse(status["five_seed_hyperparameter_search"])
        self.assertFalse(status["paper_scale_baseline_protocol"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_protocol_status_records_warm_start_teacher_order(self):
        status = ppo_paper_protocol_status(
            PPOConfig(
                policy_type="lstm",
                pretrain_steps=1,
                pretrain_teacher_mode_update_order="act_with_current_mode_then_update_next_mode",
            )
        )

        self.assertEqual(status["pretrain_steps"], 1)
        self.assertEqual(status["pretrain_teacher_policy"], "BangBangCartpolePSM")
        self.assertEqual(
            status["pretrain_teacher_mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertEqual(status["implemented_pretrain_teacher_policy"], "BangBangCartpolePSM")
        self.assertEqual(
            status["implemented_pretrain_teacher_mode_update_order"],
            "act_with_current_mode_then_update_next_mode",
        )
        self.assertTrue(status["pretrain_teacher_policy_matches_implementation"])
        self.assertTrue(status["pretrain_teacher_mode_order_matches_implementation"])
        self.assertTrue(status["pretrain_teacher_mode_order_recorded"])
        self.assertTrue(status["local_supervised_warm_start"])
        self.assertFalse(status["no_local_supervised_warm_start"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_protocol_status_excludes_warm_start_from_paper_budget_match(self):
        status = ppo_paper_protocol_status(
            PPOConfig(
                policy_type="lstm",
                total_timesteps=PAPER_PPO_TIMESTEPS,
                eval_test_max_steps=CartpoleEnv.test_env().cfg.max_steps,
                eval_rollouts=PAPER_EVAL_ROLLOUTS,
                minibatches=1,
                pretrain_steps=1,
            )
        )

        self.assertTrue(status["paper_timestep_budget"])
        self.assertTrue(status["paper_test_horizon"])
        self.assertTrue(status["uses_paper_eval_rollouts"])
        self.assertTrue(status["local_supervised_warm_start"])
        self.assertFalse(status["no_local_supervised_warm_start"])
        self.assertFalse(status["single_run_matches_paper_budget"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_protocol_status_rejects_mismatched_warm_start_teacher_claim(self):
        status = ppo_paper_protocol_status(
            PPOConfig(
                policy_type="lstm",
                total_timesteps=PAPER_PPO_TIMESTEPS,
                eval_test_max_steps=CartpoleEnv.test_env().cfg.max_steps,
                eval_rollouts=PAPER_EVAL_ROLLOUTS,
                minibatches=1,
                pretrain_steps=1,
                pretrain_teacher_policy="OtherTeacher",
                pretrain_teacher_mode_update_order="update_mode_before_acting",
            )
        )

        self.assertFalse(status["pretrain_teacher_policy_matches_implementation"])
        self.assertFalse(status["pretrain_teacher_mode_order_matches_implementation"])
        self.assertFalse(status["single_run_matches_paper_budget"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_training_rejects_mismatched_warm_start_teacher_claim(self):
        with self.assertRaises(ValueError):
            train_ppo_cartpole(
                PPOConfig(
                    policy_type="lstm",
                    total_timesteps=1,
                    pretrain_steps=1,
                    pretrain_teacher_policy="OtherTeacher",
                )
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
    def test_ppo_training_does_not_exceed_configured_timestep_budget(self):
        _, result = train_ppo_cartpole(
            PPOConfig(
                policy_type="mlp",
                total_timesteps=10,
                rollout_steps=4,
                update_epochs=1,
                minibatches=1,
                hidden_size=8,
                num_envs=3,
                seed=7,
            )
        )

        self.assertEqual(result.timesteps, 10)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_ppo_metrics_record_partial_final_rollout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "ppo_metrics.json")
            _, result = train_ppo_cartpole(
                PPOConfig(
                    policy_type="mlp",
                    total_timesteps=10,
                    rollout_steps=4,
                    update_epochs=1,
                    minibatches=1,
                    hidden_size=8,
                    num_envs=3,
                    seed=8,
                    eval_interval=0,
                    eval_rollouts=1,
                    eval_test_max_steps=20,
                    metrics_output=metrics_path,
                )
            )

            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)

        self.assertEqual(result.timesteps, 10)
        self.assertEqual([row["rollout_steps"] for row in metrics["update_history"]], [9, 1])
        self.assertEqual([row["timesteps"] for row in metrics["update_history"]], [9, 10])

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
        self.assertIn("test_cartpole_paper", metrics["command"])
        self.assertGreaterEqual(len(metrics["eval_history"]), 1)
        self.assertEqual(len(metrics["update_history"]), 2)
        self.assertEqual(metrics["update_history"][0]["update"], 1)
        self.assertEqual(metrics["update_history"][0]["timesteps"], 32)
        self.assertEqual(metrics["update_history"][0]["rollout_steps"], 32)
        self.assertIn("reward_mean", metrics["update_history"][0])
        self.assertIn("horizon_truncations", metrics["update_history"][0])
        self.assertIn("failure_terminations", metrics["update_history"][0])
        self.assertEqual(metrics["selected_result"]["timesteps"], result.timesteps)
        self.assertEqual(metrics["paper_protocol_status"]["selected_test_max_steps"], 20)
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertTrue(metrics["paper_protocol_status"]["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["paper_protocol_status"]["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["paper_protocol_status"]["selected_eval_rollouts"], 1)
        self.assertFalse(metrics["paper_protocol_status"]["uses_paper_eval_rollouts"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_timestep_budget"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_test_horizon"])
        self.assertFalse(metrics["paper_protocol_status"]["paper_scale_baseline_protocol"])
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
