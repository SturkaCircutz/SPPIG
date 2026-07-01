import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from shuttle_env import ShuttleLineEnv
from student import StudentConfig
from teacher import TeacherConfig, optimize_teacher
from train import alternating_train


class ShuttleTrainingTest(unittest.TestCase):
    def test_teacher_produces_loop_free_traces(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2])
        traces = optimize_teacher(env, None, TeacherConfig(num_traces=2, random_seed=0))

        self.assertEqual(len(traces), 2)
        self.assertTrue(all(trace.reward == 1.0 for trace in traces))
        self.assertTrue(all(len(trace.actions) == len(trace.mode_hints) for trace in traces))

    def test_student_generalizes_to_more_crossings_than_training(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6])
        policy, history = alternating_train(
            env=env,
            initial_student=None,
            grammar=None,
            teacher_cfg=TeacherConfig(num_traces=8, random_seed=13, max_steps=200),
            student_cfg=StudentConfig(num_modes=3),
            num_outer_iters=3,
        )

        self.assertEqual(len(history), 3)
        for crossings in (2, 3, 6):
            result = env.evaluate_policy(policy, crossings)
            self.assertTrue(result.success)
            self.assertEqual(result.final_mode, policy.end_mode)
        self.assertIn(policy.start_mode, policy.modes)

        right_mode = next(
            name for name, mode in policy.modes.items() if mode.action_fn([0.0, 1.0])[0] > 0.0
        )
        left_mode = next(
            name for name, mode in policy.modes.items() if mode.action_fn([0.0, 1.0])[0] < 0.0
        )
        self.assertIn(left_mode, policy.modes[right_mode].switches)
        self.assertIn(right_mode, policy.modes[left_mode].switches)
        self.assertTrue(
            policy.end_mode in policy.modes[right_mode].switches
            or policy.end_mode in policy.modes[left_mode].switches
        )

    def test_unsupported_grammar_fails_fast(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2])
        with self.assertRaises(ValueError):
            alternating_train(
                env=env,
                initial_student=None,
                grammar={"action_grammar": "proportional", "switch_grammar": "axis_threshold"},
                teacher_cfg=TeacherConfig(num_traces=2, random_seed=0, max_steps=100),
                student_cfg=StudentConfig(num_modes=3),
                num_outer_iters=1,
            )


if __name__ == "__main__":
    unittest.main()
