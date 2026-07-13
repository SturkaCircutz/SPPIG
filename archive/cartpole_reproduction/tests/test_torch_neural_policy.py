import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    from shuttle_env import ShuttleLineEnv
    from torch_neural_policy import (
        TorchTrainConfig,
        collect_expert_sequences,
        evaluate_torch_policy,
        train_torch_behavior_cloning,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchNeuralPolicyTest(unittest.TestCase):
    def test_collect_expert_sequences(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3])
        sequences = collect_expert_sequences(env, num_traces=3, seed=0, max_steps=100)

        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0][0].shape[1], 2)
        self.assertEqual(len(sequences[0][0]), len(sequences[0][1]))

    def test_lstm_training_reduces_loss(self):
        env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6])
        model, result = train_torch_behavior_cloning(
            env,
            TorchTrainConfig(
                model="lstm",
                hidden_size=8,
                epochs=5,
                num_traces=8,
                batch_size=4,
                learning_rate=0.01,
                random_seed=11,
            ),
        )

        self.assertLess(result.final_loss, result.initial_loss)
        evaluation = evaluate_torch_policy(env, model, 2, max_steps=100)
        self.assertGreaterEqual(evaluation.crossings, 0)


if __name__ == "__main__":
    unittest.main()
