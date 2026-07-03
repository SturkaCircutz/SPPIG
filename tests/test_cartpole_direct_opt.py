import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
SCRIPT = os.path.join(ROOT, "src", "train_cartpole_direct_opt.py")

from cartpole_direct_opt import (  # noqa: E402
    DirectOptConfig,
    DirectOptCandidate,
    DirectOptContinuousOneHotSwitch,
    _boolean_local_neighbor_candidates,
    _boolean_tree_candidates,
    _candidate_policy,
    _candidate_rank_key,
    _candidate_switch,
    _continuous_one_hot_candidates,
    _continuous_one_hot_local_neighbor_candidates,
    cartpole_direct_opt_protocol_status,
    direct_opt_metrics,
    run_cartpole_direct_opt,
)
from cartpole_synthesis import BooleanTreeSwitch, ObservationPredicate  # noqa: E402
from cartpole_env import CartpoleEnv  # noqa: E402


class CartpoleDirectOptTest(unittest.TestCase):
    def test_direct_opt_returns_policy_and_provenance(self):
        result = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=0,
                num_train_states=2,
                random_candidates=4,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
            )
        )
        metrics = direct_opt_metrics(result)

        diagnostics = metrics["search_diagnostics"]
        expected_evaluations = (
            diagnostics["grid_candidates"]
            + diagnostics["random_candidates"]
            + diagnostics["boolean_stump_candidates"]
            + diagnostics["boolean_depth2_candidates"]
            + diagnostics["continuous_one_hot_candidates"]
            + diagnostics["batch_refinement_candidates"]
            + diagnostics["batch_seed_evaluations"]
            + diagnostics["batch_local_evaluations"]
            + diagnostics["restart_evaluations"]
        )
        self.assertEqual(result.searched_candidates, expected_evaluations)
        self.assertEqual(diagnostics["candidate_evaluation_calls"], expected_evaluations)
        self.assertEqual(diagnostics["evaluated_candidates"], expected_evaluations)
        self.assertEqual(diagnostics["evaluated_candidates_units"], "candidate_evaluation_calls")
        expected_full_train_rollouts = (
            diagnostics["grid_candidates"]
            + diagnostics["random_candidates"]
            + diagnostics["boolean_stump_candidates"]
            + diagnostics["boolean_depth2_candidates"]
            + diagnostics["continuous_one_hot_candidates"]
            + diagnostics["batch_refinement_candidates"]
        ) * metrics["config"]["num_train_states"]
        expected_batch_rollouts = (
            diagnostics["batch_seed_rollout_evaluations"]
            + diagnostics["batch_local_rollout_evaluations"]
            + diagnostics["restart_rollout_evaluations"]
        )
        self.assertEqual(diagnostics["full_train_rollout_evaluations"], expected_full_train_rollouts)
        self.assertEqual(diagnostics["batch_rollout_evaluations"], expected_batch_rollouts)
        self.assertEqual(
            diagnostics["train_rollout_evaluations"],
            expected_full_train_rollouts + expected_batch_rollouts,
        )
        self.assertGreater(diagnostics["train_rollout_evaluations"], diagnostics["candidate_evaluation_calls"])
        self.assertIn("mode=1 if", metrics["policy_description"])
        self.assertEqual(metrics["algorithm_provenance"]["paper_baseline"], "Direct-Opt")
        self.assertTrue(metrics["algorithm_provenance"]["not_paper_scale"])
        self.assertIn("candidate evaluation calls", metrics["algorithm_provenance"]["candidate_accounting"])
        self.assertEqual(metrics["algorithm_provenance"]["paper_batch_size"], 10)
        self.assertEqual(metrics["algorithm_provenance"]["paper_parallel_threads"], 10)
        self.assertEqual(metrics["algorithm_provenance"]["paper_time_limit_seconds"], 7200)
        self.assertEqual(metrics["algorithm_provenance"]["local_parallel_threads"], "configurable_via_parallel_threads")
        self.assertEqual(
            metrics["algorithm_provenance"]["policy_class"],
            "two_mode_constant_action_linear_depth2_boolean_or_continuous_one_hot_switch",
        )
        self.assertEqual(
            metrics["algorithm_provenance"]["selection_objective"],
            "mean_combined_reward_over_selected_initial_states_then_success",
        )
        self.assertEqual(
            metrics["algorithm_provenance"]["switch_search_space"],
            "linear_theta_omega_grid_plus_bounded_boolean_tree_predicates_plus_bounded_continuous_one_hot_leaf_depth2_mixtures",
        )
        self.assertEqual(metrics["algorithm_provenance"]["boolean_tree_depth"], 2)
        self.assertEqual(
            metrics["algorithm_provenance"]["boolean_tree_features"],
            ["x", "cart_velocity", "theta", "omega"],
        )
        self.assertEqual(metrics["algorithm_provenance"]["boolean_tree_relations"], [">=", "<="])
        self.assertEqual(metrics["algorithm_provenance"]["boolean_tree_operator_choices"], ["leaf", "and", "or"])
        self.assertEqual(
            metrics["algorithm_provenance"]["continuous_one_hot_candidate_family"],
            "bounded_appendix_b3_alpha_s_feature_mix_leaf_and_depth2_predicates",
        )
        self.assertEqual(metrics["algorithm_provenance"]["continuous_one_hot_top_leaves_for_depth2"], 4)
        self.assertIn("depth2", metrics["algorithm_provenance"]["continuous_one_hot_expansion"])
        self.assertIn("bounded Appendix B.3", metrics["algorithm_provenance"]["one_hot_switch_encoding"])
        self.assertIn("bounded continuous one-hot", metrics["algorithm_provenance"]["limitations"])
        self.assertEqual(diagnostics["grid_candidates"], 156)
        self.assertEqual(diagnostics["random_candidates"], 4)
        self.assertEqual(diagnostics["boolean_stump_candidates"], 24)
        self.assertGreater(diagnostics["boolean_depth2_candidates"], 0)
        self.assertEqual(diagnostics["boolean_top_stumps_for_depth2"], 4)
        self.assertEqual(
            diagnostics["boolean_candidates_with_one_hot_metadata"],
            diagnostics["boolean_stump_candidates"] + diagnostics["boolean_depth2_candidates"],
        )
        self.assertEqual(
            diagnostics["boolean_candidates_with_appendix_b3_vertex_metadata"],
            diagnostics["boolean_candidates_with_one_hot_metadata"],
        )
        self.assertEqual(diagnostics["continuous_one_hot_leaf_candidates"], 24)
        self.assertEqual(diagnostics["continuous_one_hot_depth2_candidates"], 192)
        self.assertEqual(diagnostics["continuous_one_hot_top_leaves_for_depth2"], 4)
        self.assertEqual(diagnostics["continuous_one_hot_candidates"], 216)
        self.assertEqual(
            diagnostics["continuous_one_hot_candidates_with_appendix_b3_metadata"],
            diagnostics["continuous_one_hot_candidates"],
        )
        self.assertEqual(diagnostics["batch_count"], 1)
        self.assertEqual(diagnostics["batch_rounds"], 1)
        self.assertEqual(diagnostics["batch_refinement_candidates"], 1)
        self.assertEqual(diagnostics["parallel_threads"], 1)
        self.assertFalse(diagnostics["uses_parallel_candidate_evaluation"])
        self.assertEqual(diagnostics["batch_seed_evaluations"], 1)
        self.assertEqual(diagnostics["batch_seed_rollout_evaluations"], 2)
        self.assertGreater(diagnostics["batch_local_evaluations"], 0)
        self.assertGreater(diagnostics["batch_local_rollout_evaluations"], 0)
        self.assertGreaterEqual(diagnostics["restart_evaluations"], 0)
        self.assertGreaterEqual(diagnostics["restart_rollout_evaluations"], 0)
        self.assertEqual(metrics["best_candidate"]["source"], result.candidate.source)
        status = metrics["paper_protocol_status"]
        self.assertEqual(status["paper_baseline"], "Direct-Opt")
        self.assertEqual(status["paper_batch_size"], 10)
        self.assertEqual(status["selected_batch_size"], 10)
        self.assertTrue(status["configured_paper_batch_size"])
        self.assertFalse(status["uses_paper_batch_size"])
        self.assertEqual(status["selected_train_initial_states"], 2)
        self.assertEqual(status["paper_parallel_threads"], 10)
        self.assertEqual(status["selected_parallel_threads"], 1)
        self.assertFalse(status["uses_paper_parallel_threads"])
        self.assertEqual(status["paper_time_limit_seconds"], 7200)
        self.assertIsNone(status["selected_time_limit_seconds"])
        self.assertFalse(status["uses_paper_time_limit"])
        self.assertFalse(status["full_continuous_one_hot_switch_grammar"])
        self.assertTrue(status["bounded_one_hot_switch_metadata"])
        self.assertTrue(status["bounded_continuous_one_hot_switch_relaxation"])
        self.assertTrue(status["appendix_b3_one_hot_vertex_metadata"])
        self.assertTrue(status["optimizes_combined_reward_over_selected_initial_states"])
        self.assertTrue(status["optimizes_combined_reward_over_all_selected_initial_states"])
        self.assertFalse(status["optimizes_full_initial_state_distribution"])
        self.assertEqual(
            status["combined_reward_aggregation"],
            "mean_train_horizon_reward_over_selected_initial_states",
        )
        self.assertFalse(status["paper_scale_direct_opt_protocol"])

        default_status = cartpole_direct_opt_protocol_status(DirectOptConfig())
        self.assertTrue(default_status["configured_paper_batch_size"])
        self.assertTrue(default_status["uses_paper_batch_size"])
        self.assertEqual(default_status["selected_train_initial_states"], 10)
        parallel_status = cartpole_direct_opt_protocol_status(DirectOptConfig(parallel_threads=10))
        self.assertEqual(parallel_status["selected_parallel_threads"], 10)
        self.assertTrue(parallel_status["uses_paper_parallel_threads"])
        self.assertFalse(parallel_status["paper_scale_direct_opt_protocol"])
        timed_status = cartpole_direct_opt_protocol_status(DirectOptConfig(time_limit_seconds=7200))
        self.assertEqual(timed_status["selected_time_limit_seconds"], 7200)
        self.assertTrue(timed_status["uses_paper_time_limit"])
        self.assertFalse(timed_status["paper_scale_direct_opt_protocol"])
        self.assertEqual(metrics["paper_eval_rollouts"], 1000)
        self.assertFalse(metrics["uses_paper_eval_rollouts"])
        self.assertTrue(metrics["reward_spec"]["reward_equals_survived_steps"])
        self.assertEqual(metrics["space_spec"]["action_dimension"], 1)
        self.assertEqual(metrics["space_spec"]["observation_dimension"], 4)
        self.assertEqual(metrics["space_spec"]["initial_state_distribution"]["type"], "independent_uniform")
        self.assertEqual(metrics["paper_test_horizon_steps"], 15000)
        self.assertIn("train", metrics)
        self.assertIn("test", metrics)
        self.assertIn("steps_mean", metrics["train"])
        self.assertIn("survival_seconds_mean", metrics["train"])
        self.assertIn("steps_mean", metrics["test"])
        self.assertIn("survival_seconds_mean", metrics["test"])

    def test_direct_opt_protocol_status_marks_quick_diagnostic_limits(self):
        status = cartpole_direct_opt_protocol_status(
            DirectOptConfig(
                batch_size=2,
                batch_refinement_rounds=0,
                restart_candidates_on_stall=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
            )
        )

        self.assertFalse(status["uses_paper_batch_size"])
        self.assertFalse(status["batch_optimization_seeded_from_best_so_far"])
        self.assertFalse(status["random_restart_on_stall"])
        self.assertFalse(status["uses_full_test_horizon"])
        self.assertFalse(status["uses_paper_eval_rollouts"])
        self.assertTrue(status["optimizes_combined_reward_over_selected_initial_states"])
        self.assertTrue(status["quick_diagnostic"])
        self.assertFalse(status["paper_scale_direct_opt_protocol"])

    def test_direct_opt_parallel_candidate_evaluation_preserves_counts(self):
        serial = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=2,
                num_train_states=2,
                random_candidates=4,
                batch_refinement_rounds=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
                parallel_threads=1,
            )
        )
        parallel = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=2,
                num_train_states=2,
                random_candidates=4,
                batch_refinement_rounds=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
                parallel_threads=2,
            )
        )

        self.assertEqual(serial.search_diagnostics["candidate_evaluation_calls"], parallel.search_diagnostics["candidate_evaluation_calls"])
        self.assertEqual(serial.candidate.train_reward_mean, parallel.candidate.train_reward_mean)
        self.assertEqual(serial.policy.describe(), parallel.policy.describe())
        self.assertEqual(parallel.search_diagnostics["parallel_threads"], 2)
        self.assertTrue(parallel.search_diagnostics["uses_parallel_candidate_evaluation"])

    def test_direct_opt_time_limit_records_early_stop(self):
        result = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=2,
                num_train_states=2,
                random_candidates=4,
                batch_refinement_rounds=1,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
                time_limit_seconds=0.0,
            )
        )

        diagnostics = result.search_diagnostics
        self.assertEqual(diagnostics["time_limit_seconds"], 0.0)
        self.assertTrue(diagnostics["time_limit_reached"])
        self.assertEqual(diagnostics["grid_candidates"], 1)
        self.assertEqual(diagnostics["random_candidates"], 0)
        self.assertEqual(diagnostics["boolean_stump_candidates"], 0)
        self.assertEqual(diagnostics["boolean_depth2_candidates"], 0)
        self.assertEqual(diagnostics["continuous_one_hot_leaf_candidates"], 0)
        self.assertEqual(diagnostics["continuous_one_hot_depth2_candidates"], 0)
        self.assertEqual(diagnostics["continuous_one_hot_candidates"], 0)
        self.assertEqual(diagnostics["batch_count"], 1)
        self.assertEqual(diagnostics["batch_refinement_candidates"], 0)
        self.assertTrue(diagnostics["batch_time_limit_reached"])

    def test_direct_opt_can_disable_batch_refinement_for_grid_random_diagnostic(self):
        result = run_cartpole_direct_opt(
            DirectOptConfig(
                seed=0,
                num_train_states=2,
                random_candidates=4,
                batch_refinement_rounds=0,
                eval_rollouts=1,
                test_max_steps=20,
                quick=True,
            )
        )

        diagnostics = result.search_diagnostics
        expected_evaluations = (
            diagnostics["grid_candidates"]
            + diagnostics["random_candidates"]
            + diagnostics["boolean_stump_candidates"]
            + diagnostics["boolean_depth2_candidates"]
            + diagnostics["continuous_one_hot_candidates"]
        )
        self.assertEqual(result.searched_candidates, expected_evaluations)
        self.assertEqual(result.search_diagnostics["batch_refinement_candidates"], 0)
        self.assertEqual(result.search_diagnostics["batch_seed_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["batch_seed_rollout_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["batch_local_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["batch_local_rollout_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["restart_evaluations"], 0)
        self.assertEqual(result.search_diagnostics["restart_rollout_evaluations"], 0)

    def test_direct_opt_boolean_tree_candidates_use_cartpole_switch_grammar(self):
        env = CartpoleEnv.train_env(seed=0)
        train_states = [env.reset() for _ in range(2)]

        candidates, diagnostics = _boolean_tree_candidates(train_states)
        depth2 = next(candidate for candidate in candidates if candidate.source == "boolean_depth2")
        policy = _candidate_policy(depth2)

        self.assertEqual(diagnostics["boolean_stump_candidates"], 24)
        self.assertGreater(diagnostics["boolean_depth2_candidates"], 0)
        self.assertEqual(depth2.switch_kind, "boolean_tree")
        self.assertIsNotNone(depth2.first_feature)
        self.assertIsNotNone(depth2.second_feature)
        self.assertIn(depth2.operator, {"and", "or"})
        self.assertEqual(sum(depth2.first_feature_one_hot), 1)
        self.assertEqual(len(depth2.first_feature_one_hot), 4)
        self.assertEqual(sum(depth2.first_relation_one_hot), 1)
        self.assertEqual(len(depth2.first_relation_one_hot), 2)
        self.assertEqual(sum(depth2.second_feature_one_hot), 1)
        self.assertEqual(sum(depth2.second_relation_one_hot), 1)
        self.assertEqual(sum(depth2.operator_one_hot), 1)
        self.assertEqual(len(depth2.operator_one_hot), 3)
        self.assertIsNotNone(depth2.first_appendix_b3_alpha_s)
        self.assertIsNotNone(depth2.first_appendix_b3_alpha_0)
        self.assertEqual(sum(depth2.first_appendix_b3_feature_weights), 1.0)
        self.assertEqual(len(depth2.first_appendix_b3_feature_weights), 4)
        self.assertIsNotNone(depth2.second_appendix_b3_alpha_s)
        self.assertIsNotNone(depth2.second_appendix_b3_alpha_0)
        self.assertEqual(sum(depth2.second_appendix_b3_feature_weights), 1.0)
        self.assertIn(" o[", policy.describe())

    def test_direct_opt_continuous_one_hot_candidates_use_appendix_b3_mixture(self):
        env = CartpoleEnv.train_env(seed=0)
        train_states = [env.reset() for _ in range(2)]

        candidates, diagnostics = _continuous_one_hot_candidates(train_states, DirectOptConfig())
        candidate = next(
            candidate
            for candidate in candidates
            if candidate.continuous_one_hot_alpha_s == 1.0
            and candidate.continuous_one_hot_feature_weights == (0.0, 0.0, 0.5, 0.5)
            and candidate.continuous_one_hot_alpha_0 == 0.0
        )
        switch = _candidate_switch(candidate)
        policy = _candidate_policy(candidate)

        self.assertEqual(diagnostics["continuous_one_hot_leaf_candidates"], 24)
        self.assertEqual(diagnostics["continuous_one_hot_depth2_candidates"], 192)
        self.assertEqual(diagnostics["continuous_one_hot_top_leaves_for_depth2"], 4)
        self.assertEqual(diagnostics["continuous_one_hot_candidates"], 216)
        self.assertEqual(diagnostics["continuous_one_hot_candidates_with_appendix_b3_metadata"], 216)
        self.assertEqual(candidate.switch_kind, "continuous_one_hot")
        self.assertEqual(sum(candidate.continuous_one_hot_feature_weights), 1.0)
        self.assertEqual(
            candidate.first_appendix_b3_feature_weights,
            candidate.continuous_one_hot_feature_weights,
        )
        self.assertEqual(candidate.first_appendix_b3_alpha_s, 1.0)
        self.assertEqual(candidate.first_appendix_b3_alpha_0, 0.0)
        self.assertEqual(candidate.operator_one_hot, (1, 0, 0))
        self.assertIsInstance(switch, DirectOptContinuousOneHotSwitch)
        self.assertEqual(switch.decide([0.0, 0.0, -0.2, 0.1]), 1)
        self.assertEqual(switch.decide([0.0, 0.0, 0.2, -0.1]), 0)
        self.assertIn("0.500*o[2] + 0.500*o[3]", policy.describe())

    def test_direct_opt_continuous_one_hot_depth2_candidates_preserve_second_predicate(self):
        env = CartpoleEnv.train_env(seed=0)
        train_states = [env.reset() for _ in range(2)]

        candidates, _ = _continuous_one_hot_candidates(train_states, DirectOptConfig())
        candidate = next(candidate for candidate in candidates if candidate.source == "continuous_one_hot_depth2")
        switch = _candidate_switch(candidate)

        self.assertEqual(candidate.switch_kind, "continuous_one_hot")
        self.assertIn(candidate.continuous_one_hot_operator, {"and", "or"})
        self.assertIsNotNone(candidate.second_appendix_b3_alpha_s)
        self.assertIsNotNone(candidate.second_appendix_b3_alpha_0)
        self.assertEqual(sum(candidate.second_appendix_b3_feature_weights), 1.0)
        self.assertIsInstance(switch, DirectOptContinuousOneHotSwitch)
        self.assertIsNotNone(switch.second)
        self.assertEqual(candidate.second_appendix_b3_feature_weights, switch.second.feature_weights)
        self.assertIn(candidate.continuous_one_hot_operator, switch.describe())

    def test_direct_opt_continuous_one_hot_rank_key_counts_depth2_thresholds(self):
        simple = DirectOptCandidate(
            theta_weight=0.0,
            omega_weight=0.0,
            threshold=0.0,
            left_force=-10.0,
            right_force=10.0,
            train_reward_mean=1.0,
            train_success_rate=0.0,
            switch_kind="continuous_one_hot",
            continuous_one_hot_alpha_s=1.0,
            continuous_one_hot_feature_weights=(0.0, 0.0, 0.5, 0.5),
            continuous_one_hot_alpha_0=0.1,
            continuous_one_hot_operator="leaf",
        )
        depth2 = DirectOptCandidate(
            theta_weight=0.0,
            omega_weight=0.0,
            threshold=0.0,
            left_force=-10.0,
            right_force=10.0,
            train_reward_mean=1.0,
            train_success_rate=0.0,
            switch_kind="continuous_one_hot",
            continuous_one_hot_alpha_s=1.0,
            continuous_one_hot_feature_weights=(0.0, 0.0, 0.5, 0.5),
            continuous_one_hot_alpha_0=0.1,
            continuous_one_hot_operator="and",
            second_appendix_b3_alpha_s=1.0,
            second_appendix_b3_feature_weights=(0.0, 0.0, 0.75, 0.25),
            second_appendix_b3_alpha_0=0.2,
        )

        self.assertGreater(_candidate_rank_key(simple), _candidate_rank_key(depth2))

    def test_direct_opt_appendix_b3_vertex_encoding_matches_predicate(self):
        env = CartpoleEnv.train_env(seed=1)
        train_states = [env.reset() for _ in range(2)]

        candidates, _ = _boolean_tree_candidates(train_states)
        stump = next(candidate for candidate in candidates if candidate.source == "boolean_stump")
        switch = _candidate_switch(stump)
        self.assertIsInstance(switch, BooleanTreeSwitch)

        for observation in (
            [-0.3, 0.1, -0.02, 0.4],
            [0.2, -0.1, 0.04, -0.3],
        ):
            weighted_observation = sum(
                weight * value
                for weight, value in zip(stump.first_appendix_b3_feature_weights, observation)
            )
            encoded_enabled = (
                float(stump.first_appendix_b3_alpha_s) * weighted_observation
                <= float(stump.first_appendix_b3_alpha_0)
            )
            self.assertEqual(encoded_enabled, switch.first.evaluate(observation))

    def test_direct_opt_boolean_local_refinement_dedupes_before_evaluation(self):
        candidate = DirectOptCandidate(
            theta_weight=0.0,
            omega_weight=0.0,
            threshold=0.0,
            left_force=-10.0,
            right_force=10.0,
            train_reward_mean=1.0,
            train_success_rate=0.0,
            switch_kind="boolean_tree",
            first_feature=2,
            first_relation=">=",
            first_threshold=0.0,
        )
        evaluated: list[tuple[str, float, float]] = []

        def fake_evaluate(switch, left_force, right_force, *_args):
            evaluated.append((switch.describe(), left_force, right_force))
            return DirectOptCandidate(
                theta_weight=0.0,
                omega_weight=0.0,
                threshold=0.0,
                left_force=left_force,
                right_force=right_force,
                train_reward_mean=1.0,
                train_success_rate=0.0,
                source="batch_local_refinement",
                switch_kind="boolean_tree",
                first_feature=switch.first.feature_index,
                first_relation=switch.first.relation,
                first_threshold=switch.first.threshold,
            )

        with patch("cartpole_direct_opt._evaluate_boolean_candidate", side_effect=fake_evaluate):
            neighbors = _boolean_local_neighbor_candidates(
                candidate,
                [[0.0, 0.0, 0.0, 0.0]],
                DirectOptConfig(local_step_fraction=0.25),
            )

        self.assertEqual(len(evaluated), len(neighbors))
        self.assertEqual(len(evaluated), len(set(evaluated)))
        raw_switches = [
            BooleanTreeSwitch(ObservationPredicate(2, ">=", 0.0)),
            BooleanTreeSwitch(ObservationPredicate(2, ">=", -0.025)),
            BooleanTreeSwitch(ObservationPredicate(2, ">=", 0.025)),
        ]
        self.assertLess(
            len(neighbors),
            len(raw_switches) * 5,
        )

    def test_direct_opt_continuous_one_hot_local_refinement_preserves_metadata(self):
        candidate = DirectOptCandidate(
            theta_weight=0.0,
            omega_weight=0.0,
            threshold=0.0,
            left_force=-10.0,
            right_force=10.0,
            train_reward_mean=1.0,
            train_success_rate=0.0,
            switch_kind="continuous_one_hot",
            continuous_one_hot_alpha_s=1.0,
            continuous_one_hot_feature_weights=(0.0, 0.0, 0.5, 0.5),
            continuous_one_hot_alpha_0=0.0,
            continuous_one_hot_operator="leaf",
        )
        evaluated: list[tuple[str, float, float]] = []

        def fake_evaluate(switch, left_force, right_force, *_args):
            evaluated.append((switch.describe(), left_force, right_force))
            return DirectOptCandidate(
                theta_weight=0.0,
                omega_weight=0.0,
                threshold=0.0,
                left_force=left_force,
                right_force=right_force,
                train_reward_mean=1.0,
                train_success_rate=0.0,
                source="batch_local_refinement",
                switch_kind="continuous_one_hot",
                continuous_one_hot_alpha_s=switch.first.alpha_s,
                continuous_one_hot_feature_weights=switch.first.feature_weights,
                continuous_one_hot_alpha_0=switch.first.alpha_0,
                continuous_one_hot_operator=switch.operator,
                first_appendix_b3_alpha_s=switch.first.alpha_s,
                first_appendix_b3_feature_weights=switch.first.feature_weights,
                first_appendix_b3_alpha_0=switch.first.alpha_0,
            )

        with patch("cartpole_direct_opt._evaluate_continuous_one_hot_candidate", side_effect=fake_evaluate):
            neighbors = _continuous_one_hot_local_neighbor_candidates(
                candidate,
                [[0.0, 0.0, 0.0, 0.0]],
                DirectOptConfig(local_step_fraction=0.25),
            )

        thresholds = {neighbor.continuous_one_hot_alpha_0 for neighbor in neighbors}
        self.assertEqual(len(evaluated), len(neighbors))
        self.assertEqual(len(evaluated), len(set(evaluated)))
        self.assertTrue(all(neighbor.switch_kind == "continuous_one_hot" for neighbor in neighbors))
        self.assertIn(-0.0625, thresholds)
        self.assertIn(0.0625, thresholds)
        self.assertTrue(
            all(
                neighbor.first_appendix_b3_feature_weights
                == neighbor.continuous_one_hot_feature_weights
                for neighbor in neighbors
            )
        )

    def test_direct_opt_batch_refinement_preserves_full_train_best_so_far(self):
        base_cfg = DirectOptConfig(
            seed=1,
            num_train_states=3,
            random_candidates=4,
            batch_size=1,
            batch_refinement_rounds=0,
            eval_rollouts=1,
            test_max_steps=20,
            quick=True,
        )
        refined_cfg = DirectOptConfig(
            seed=1,
            num_train_states=3,
            random_candidates=4,
            batch_size=1,
            batch_refinement_rounds=2,
            local_refinement_steps=1,
            restart_candidates_on_stall=1,
            eval_rollouts=1,
            test_max_steps=20,
            quick=True,
        )

        base = run_cartpole_direct_opt(base_cfg)
        refined = run_cartpole_direct_opt(refined_cfg)

        self.assertGreaterEqual(
            refined.candidate.train_reward_mean,
            base.candidate.train_reward_mean,
        )
        self.assertGreater(refined.search_diagnostics["batch_refinement_candidates"], 0)

    def test_direct_opt_cli_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "direct_opt_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
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

        self.assertEqual(metrics["config"]["quick"], True)
        self.assertEqual(metrics["config"]["batch_size"], 2)
        self.assertEqual(metrics["config"]["batch_refinement_rounds"], 1)
        self.assertEqual(metrics["config"]["local_refinement_steps"], 1)
        self.assertEqual(metrics["config"]["restart_candidates_on_stall"], 1)
        self.assertEqual(metrics["config"]["parallel_threads"], 1)
        self.assertIsNone(metrics["config"]["time_limit_seconds"])
        status = metrics["paper_protocol_status"]
        self.assertFalse(status["uses_paper_batch_size"])
        self.assertEqual(status["selected_test_max_steps"], 20)
        self.assertFalse(status["uses_full_test_horizon"])
        self.assertFalse(status["uses_paper_eval_rollouts"])
        self.assertFalse(status["paper_scale_direct_opt_protocol"])
        self.assertEqual(metrics["eval_rollouts"], 1)
        self.assertEqual(metrics["paper_eval_rollouts"], 1000)
        self.assertFalse(metrics["uses_paper_eval_rollouts"])
        self.assertEqual(metrics["reward_spec"]["reward_per_alive_step"], 1.0)
        self.assertEqual(metrics["space_spec"]["action_space"]["high"], 10.0)
        self.assertEqual(metrics["test_max_steps"], 20)
        self.assertIn("steps_mean", metrics["train"])
        self.assertIn("survival_seconds_mean", metrics["test"])
        self.assertEqual(metrics["algorithm_provenance"]["baseline"], "direct_opt")
        self.assertEqual(metrics["search_diagnostics"]["boolean_stump_candidates"], 24)
        self.assertGreater(metrics["search_diagnostics"]["boolean_depth2_candidates"], 0)
        self.assertEqual(metrics["search_diagnostics"]["continuous_one_hot_leaf_candidates"], 24)
        self.assertEqual(metrics["search_diagnostics"]["continuous_one_hot_depth2_candidates"], 192)
        self.assertEqual(metrics["search_diagnostics"]["continuous_one_hot_candidates"], 216)
        self.assertEqual(metrics["search_diagnostics"]["evaluated_candidates_units"], "candidate_evaluation_calls")
        self.assertEqual(metrics["search_diagnostics"]["parallel_threads"], 1)
        self.assertFalse(metrics["search_diagnostics"]["uses_parallel_candidate_evaluation"])
        self.assertIsNone(metrics["search_diagnostics"]["time_limit_seconds"])
        self.assertFalse(metrics["search_diagnostics"]["time_limit_reached"])
        self.assertGreater(
            metrics["search_diagnostics"]["train_rollout_evaluations"],
            metrics["search_diagnostics"]["candidate_evaluation_calls"],
        )
        self.assertIn("search_diagnostics", metrics)
        self.assertIn("best_candidate", metrics)

    def test_direct_opt_cli_quick_honors_disabled_batch_refinement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "direct_opt_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--batch-refinement-rounds",
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

        self.assertEqual(metrics["config"]["quick"], True)
        self.assertEqual(metrics["config"]["batch_refinement_rounds"], 0)
        self.assertFalse(metrics["paper_protocol_status"]["batch_optimization_seeded_from_best_so_far"])
        self.assertEqual(metrics["search_diagnostics"]["batch_refinement_candidates"], 0)
        self.assertEqual(metrics["search_diagnostics"]["batch_seed_evaluations"], 0)
        self.assertEqual(metrics["search_diagnostics"]["batch_seed_rollout_evaluations"], 0)

    def test_direct_opt_cli_records_parallel_thread_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "direct_opt_metrics.json")
            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--quick",
                    "--parallel-threads",
                    "2",
                    "--time-limit-seconds",
                    "7200",
                    "--batch-refinement-rounds",
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

        self.assertEqual(metrics["config"]["parallel_threads"], 2)
        self.assertEqual(metrics["config"]["time_limit_seconds"], 7200)
        self.assertEqual(metrics["search_diagnostics"]["parallel_threads"], 2)
        self.assertTrue(metrics["search_diagnostics"]["uses_parallel_candidate_evaluation"])
        self.assertEqual(metrics["paper_protocol_status"]["selected_parallel_threads"], 2)
        self.assertFalse(metrics["paper_protocol_status"]["uses_paper_parallel_threads"])
        self.assertEqual(metrics["paper_protocol_status"]["selected_time_limit_seconds"], 7200)
        self.assertTrue(metrics["paper_protocol_status"]["uses_paper_time_limit"])


if __name__ == "__main__":
    unittest.main()
