import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from neural_policy import (
    NeuralTrainConfig,
    collect_expert_dataset,
    collect_expert_sequences,
    evaluate_neural_policy,
    evaluate_recurrent_policy,
    train_behavior_cloning,
    train_recurrent_behavior_cloning,
)
from shuttle_env import ShuttleLineEnv


class NeuralPolicyTest(unittest.TestCase):
    def test_collect_expert_dataset(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3])
        observations, labels = collect_expert_dataset(env, num_traces=4, seed=0, max_steps=100)
        sequences = collect_expert_sequences(env, num_traces=4, seed=0, max_steps=100)

        self.assertEqual(observations.shape[1], 2)
        self.assertEqual(len(observations), len(labels))
        self.assertTrue(set(labels).issubset({0, 1, 2}))
        self.assertEqual(len(sequences), 4)

    def test_training_reduces_behavior_cloning_loss(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6])
        policy, result = train_behavior_cloning(
            env,
            NeuralTrainConfig(
                hidden_sizes=(16,),
                epochs=25,
                num_traces=16,
                batch_size=16,
                learning_rate=0.05,
                random_seed=3,
            ),
        )

        self.assertLess(result.final_loss, result.initial_loss)
        evaluation = evaluate_neural_policy(env, policy, 2, max_steps=100)
        self.assertGreaterEqual(evaluation.crossings, 0)

    def test_recurrent_training_reduces_loss(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6])
        policy, result = train_recurrent_behavior_cloning(
            env,
            NeuralTrainConfig(
                recurrent_hidden_size=8,
                epochs=20,
                num_traces=16,
                learning_rate=0.05,
                random_seed=5,
            ),
        )

        self.assertLess(result.final_loss, result.initial_loss)
        evaluation = evaluate_recurrent_policy(env, policy, 2, max_steps=100)
        self.assertGreaterEqual(evaluation.crossings, 0)


if __name__ == "__main__":
    unittest.main()
