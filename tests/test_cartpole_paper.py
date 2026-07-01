import json
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
    fit_probabilistic_cartpole_student,
    synthesize_cartpole_policy,
    _eq12_switch_log_likelihood,
    _boolean_tree_candidates,
    _fit_switch_parameter_distributions,
    _greedy_boolean_tree_candidates,
    _refine_loop_free_trace,
    _sample_switch,
    _switch_cost,
    _switch_timing_loss,
    _teacher_objective,
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

    def test_cartpole_teacher_objective_defaults_to_reward(self):
        cfg = CartpoleSynthesisConfig()
        trace = CartpoleTrace(
            observations=[[0.0, 0.0, 0.0, 0.0]],
            actions=[10.0],
            mode_labels=[1],
            reward=7.0,
        )

        self.assertEqual(_teacher_objective(trace, None, cfg), 7.0)

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
        self.assertEqual(rollout.next_episode_steps[0].item(), 0)

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
