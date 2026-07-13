from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from shuttle_env import ShuttleLineEnv, ShuttleResult


ACTION_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class NeuralTrainConfig:
    hidden_sizes: Tuple[int, ...] = (32, 32)
    recurrent_hidden_size: int = 16
    learning_rate: float = 0.05
    epochs: int = 600
    batch_size: int = 32
    num_traces: int = 128
    random_seed: int = 7
    max_steps: int = 200


@dataclass
class NeuralTrainingResult:
    initial_loss: float
    final_loss: float
    num_examples: int


class FeedForwardNeuralPolicy:
    """Small MLP policy trained as a neural baseline.

    The paper's neural baselines are RL policies.  This repository has no RL
    stack, so this model is a supervised neural controller trained from expert
    traces on the same train/test split used by the programmatic learner.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        length_scale: float,
        crossing_scale: float,
    ) -> None:
        self.weights = weights
        self.biases = biases
        self.length_scale = length_scale
        self.crossing_scale = crossing_scale

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        length_scale: float,
        crossing_scale: float,
        rng: np.random.Generator,
    ) -> "FeedForwardNeuralPolicy":
        layer_sizes = [input_dim, *hidden_sizes, output_dim]
        weights: List[np.ndarray] = []
        biases: List[np.ndarray] = []
        for left, right in zip(layer_sizes, layer_sizes[1:]):
            # Xavier-style scaling keeps the pure NumPy trainer stable.
            scale = np.sqrt(2.0 / (left + right))
            weights.append(rng.normal(0.0, scale, size=(left, right)))
            biases.append(np.zeros(right, dtype=np.float64))
        return cls(weights, biases, length_scale, crossing_scale)

    def logits(self, observations: np.ndarray) -> np.ndarray:
        activations = self._normalize(observations)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activations = np.tanh(activations @ weight + bias)
        return activations @ self.weights[-1] + self.biases[-1]

    def predict_class(self, observation: Sequence[float]) -> int:
        batch = np.asarray([observation], dtype=np.float64)
        return int(np.argmax(self.logits(batch), axis=1)[0])

    def act(self, observation: Sequence[float]) -> float:
        return float(ACTION_VALUES[self.predict_class(observation)])

    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {
            "num_layers": len(self.weights),
            "length_scale": self.length_scale,
            "crossing_scale": self.crossing_scale,
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{index}"] = weight
            payload[f"b{index}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "FeedForwardNeuralPolicy":
        data = np.load(path)
        num_layers = int(data["num_layers"])
        weights = [data[f"W{index}"] for index in range(num_layers)]
        biases = [data[f"b{index}"] for index in range(num_layers)]
        return cls(
            weights=weights,
            biases=biases,
            length_scale=float(data["length_scale"]),
            crossing_scale=float(data["crossing_scale"]),
        )

    def _normalize(self, observations: np.ndarray) -> np.ndarray:
        scale = np.array([self.length_scale, self.crossing_scale], dtype=np.float64)
        return observations / scale


class RecurrentNeuralPolicy:
    """A tanh RNN baseline with one hidden state per rollout."""

    def __init__(
        self,
        w_xh: np.ndarray,
        w_hh: np.ndarray,
        b_h: np.ndarray,
        w_hy: np.ndarray,
        b_y: np.ndarray,
        length_scale: float,
        crossing_scale: float,
    ) -> None:
        self.w_xh = w_xh
        self.w_hh = w_hh
        self.b_h = b_h
        self.w_hy = w_hy
        self.b_y = b_y
        self.length_scale = length_scale
        self.crossing_scale = crossing_scale

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        length_scale: float,
        crossing_scale: float,
        rng: np.random.Generator,
    ) -> "RecurrentNeuralPolicy":
        w_xh = rng.normal(0.0, np.sqrt(2.0 / (input_dim + hidden_size)), size=(input_dim, hidden_size))
        w_hh = rng.normal(0.0, np.sqrt(1.0 / hidden_size), size=(hidden_size, hidden_size))
        b_h = np.zeros(hidden_size, dtype=np.float64)
        w_hy = rng.normal(0.0, np.sqrt(2.0 / (hidden_size + output_dim)), size=(hidden_size, output_dim))
        b_y = np.zeros(output_dim, dtype=np.float64)
        return cls(w_xh, w_hh, b_h, w_hy, b_y, length_scale, crossing_scale)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.b_h.shape[0], dtype=np.float64)

    def step_logits(self, observation: Sequence[float], hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        normalized = self._normalize(np.asarray(observation, dtype=np.float64))
        next_hidden = np.tanh(normalized @ self.w_xh + hidden @ self.w_hh + self.b_h)
        logits = next_hidden @ self.w_hy + self.b_y
        return logits, next_hidden

    def act(self, observation: Sequence[float], hidden: np.ndarray) -> Tuple[float, np.ndarray]:
        logits, next_hidden = self.step_logits(observation, hidden)
        label = int(np.argmax(logits))
        return float(ACTION_VALUES[label]), next_hidden

    def save(self, path: str) -> None:
        np.savez(
            path,
            policy_type="rnn",
            w_xh=self.w_xh,
            w_hh=self.w_hh,
            b_h=self.b_h,
            w_hy=self.w_hy,
            b_y=self.b_y,
            length_scale=self.length_scale,
            crossing_scale=self.crossing_scale,
        )

    @classmethod
    def load(cls, path: str) -> "RecurrentNeuralPolicy":
        data = np.load(path)
        return cls(
            w_xh=data["w_xh"],
            w_hh=data["w_hh"],
            b_h=data["b_h"],
            w_hy=data["w_hy"],
            b_y=data["b_y"],
            length_scale=float(data["length_scale"]),
            crossing_scale=float(data["crossing_scale"]),
        )

    def _normalize(self, observation: np.ndarray) -> np.ndarray:
        scale = np.array([self.length_scale, self.crossing_scale], dtype=np.float64)
        return observation / scale


def collect_expert_dataset(
    env: ShuttleLineEnv,
    num_traces: int,
    seed: int,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    observations: List[List[float]] = []
    labels: List[int] = []
    for _ in range(num_traces):
        required_crossings = rng.choice(env.train_crossings)
        sequence_obs, sequence_labels = _expert_sequence(env, required_crossings, max_steps)
        observations.extend(sequence_obs)
        labels.extend(sequence_labels)
    return np.asarray(observations, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def collect_expert_sequences(
    env: ShuttleLineEnv,
    num_traces: int,
    seed: int,
    max_steps: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = random.Random(seed)
    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_traces):
        required_crossings = rng.choice(env.train_crossings)
        observations, labels = _expert_sequence(env, required_crossings, max_steps)
        sequences.append(
            (
                np.asarray(observations, dtype=np.float64),
                np.asarray(labels, dtype=np.int64),
            )
        )
    return sequences


def train_behavior_cloning(
    env: ShuttleLineEnv,
    cfg: NeuralTrainConfig,
) -> Tuple[FeedForwardNeuralPolicy, NeuralTrainingResult]:
    x_train, y_train = collect_expert_dataset(
        env,
        num_traces=cfg.num_traces,
        seed=cfg.random_seed,
        max_steps=cfg.max_steps,
    )
    rng = np.random.default_rng(cfg.random_seed)
    policy = FeedForwardNeuralPolicy.initialize(
        input_dim=x_train.shape[1],
        hidden_sizes=cfg.hidden_sizes,
        output_dim=len(ACTION_VALUES),
        length_scale=env.length,
        crossing_scale=float(max(env.train_crossings)),
        rng=rng,
    )

    initial_loss = _loss(policy, x_train, y_train)
    for _ in range(cfg.epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), cfg.batch_size):
            batch_idx = order[start : start + cfg.batch_size]
            _train_batch(policy, x_train[batch_idx], y_train[batch_idx], cfg.learning_rate)
    final_loss = _loss(policy, x_train, y_train)
    return policy, NeuralTrainingResult(initial_loss, final_loss, len(x_train))


def train_recurrent_behavior_cloning(
    env: ShuttleLineEnv,
    cfg: NeuralTrainConfig,
) -> Tuple[RecurrentNeuralPolicy, NeuralTrainingResult]:
    sequences = collect_expert_sequences(
        env,
        num_traces=cfg.num_traces,
        seed=cfg.random_seed,
        max_steps=cfg.max_steps,
    )
    rng = np.random.default_rng(cfg.random_seed)
    policy = RecurrentNeuralPolicy.initialize(
        input_dim=sequences[0][0].shape[1],
        hidden_size=cfg.recurrent_hidden_size,
        output_dim=len(ACTION_VALUES),
        length_scale=env.length,
        crossing_scale=float(max(env.train_crossings)),
        rng=rng,
    )

    initial_loss = _recurrent_loss(policy, sequences)
    for _ in range(cfg.epochs):
        order = rng.permutation(len(sequences))
        for sequence_index in order:
            _train_recurrent_sequence(policy, *sequences[sequence_index], cfg.learning_rate)
    final_loss = _recurrent_loss(policy, sequences)
    num_examples = sum(len(labels) for _, labels in sequences)
    return policy, NeuralTrainingResult(initial_loss, final_loss, num_examples)


def evaluate_neural_policy(
    env: ShuttleLineEnv,
    policy: FeedForwardNeuralPolicy,
    required_crossings: int,
    max_steps: int = 200,
) -> ShuttleResult:
    x = 0.0
    crossings = 0
    remaining = required_crossings
    expected_direction = 1.0

    for step in range(1, max_steps + 1):
        action = policy.act([x, float(remaining)])
        if action == 0.0:
            success = crossings >= required_crossings
            return ShuttleResult(
                required_crossings,
                success,
                crossings,
                step - 1,
                1.0 if success else 0.0,
                "end",
            )

        old_x = x
        x = env._advance(x, action)
        target = env.length if expected_direction > 0 else 0.0
        if old_x != target and x == target:
            crossings += 1
            remaining = max(0, required_crossings - crossings)
            expected_direction *= -1.0

    return ShuttleResult(required_crossings, False, crossings, max_steps, 0.0, "running")


def evaluate_recurrent_policy(
    env: ShuttleLineEnv,
    policy: RecurrentNeuralPolicy,
    required_crossings: int,
    max_steps: int = 200,
) -> ShuttleResult:
    x = 0.0
    crossings = 0
    remaining = required_crossings
    expected_direction = 1.0
    hidden = policy.initial_state()

    for step in range(1, max_steps + 1):
        action, hidden = policy.act([x, float(remaining)], hidden)
        if action == 0.0:
            success = crossings >= required_crossings
            return ShuttleResult(
                required_crossings,
                success,
                crossings,
                step - 1,
                1.0 if success else 0.0,
                "end",
            )

        old_x = x
        x = env._advance(x, action)
        target = env.length if expected_direction > 0 else 0.0
        if old_x != target and x == target:
            crossings += 1
            remaining = max(0, required_crossings - crossings)
            expected_direction *= -1.0

    return ShuttleResult(required_crossings, False, crossings, max_steps, 0.0, "running")


def _expert_sequence(
    env: ShuttleLineEnv,
    required_crossings: int,
    max_steps: int,
) -> Tuple[List[List[float]], List[int]]:
    observations: List[List[float]] = []
    labels: List[int] = []
    x = 0.0
    remaining = required_crossings
    expected_direction = 1.0

    while remaining > 0 and len(labels) < max_steps:
        observations.append([x, float(remaining)])
        labels.append(_action_to_label(expected_direction))
        old_x = x
        x = env._advance(x, expected_direction)
        target = env.length if expected_direction > 0 else 0.0
        if old_x != target and x == target:
            remaining -= 1
            expected_direction *= -1.0

    if len(labels) < max_steps:
        observations.append([x, 0.0])
        labels.append(_action_to_label(0.0))
    return observations, labels


def _action_to_label(action: float) -> int:
    if action < 0.0:
        return 0
    if action > 0.0:
        return 2
    return 1


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _loss(policy: FeedForwardNeuralPolicy, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
    probabilities = _softmax(policy.logits(x_batch))
    selected = probabilities[np.arange(len(y_batch)), y_batch]
    return float(-np.mean(np.log(selected + 1e-12)))


def _forward_with_cache(
    policy: FeedForwardNeuralPolicy,
    x_batch: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    activations = [policy._normalize(x_batch)]
    current = activations[0]
    for weight, bias in zip(policy.weights[:-1], policy.biases[:-1]):
        current = np.tanh(current @ weight + bias)
        activations.append(current)
    logits = current @ policy.weights[-1] + policy.biases[-1]
    return activations, logits


def _train_batch(
    policy: FeedForwardNeuralPolicy,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    learning_rate: float,
) -> None:
    activations, logits = _forward_with_cache(policy, x_batch)
    grad = _softmax(logits)
    grad[np.arange(len(y_batch)), y_batch] -= 1.0
    grad /= len(y_batch)

    for layer_index in reversed(range(len(policy.weights))):
        previous_activation = activations[layer_index]
        weight = policy.weights[layer_index]
        grad_weight = previous_activation.T @ grad
        grad_bias = np.sum(grad, axis=0)

        if layer_index > 0:
            grad_activation = grad @ weight.T
            grad = grad_activation * (1.0 - activations[layer_index] ** 2)

        policy.weights[layer_index] -= learning_rate * grad_weight
        policy.biases[layer_index] -= learning_rate * grad_bias


def _recurrent_forward(
    policy: RecurrentNeuralPolicy,
    observations: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    inputs = [policy._normalize(observation) for observation in observations]
    hidden_states = [policy.initial_state()]
    logits_list = []
    for normalized in inputs:
        hidden = np.tanh(normalized @ policy.w_xh + hidden_states[-1] @ policy.w_hh + policy.b_h)
        hidden_states.append(hidden)
        logits_list.append(hidden @ policy.w_hy + policy.b_y)
    return inputs, hidden_states, np.vstack(logits_list)


def _recurrent_loss(
    policy: RecurrentNeuralPolicy,
    sequences: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    losses = []
    for observations, labels in sequences:
        _, _, logits = _recurrent_forward(policy, observations)
        probabilities = _softmax(logits)
        selected = probabilities[np.arange(len(labels)), labels]
        losses.append(-np.mean(np.log(selected + 1e-12)))
    return float(np.mean(losses))


def _train_recurrent_sequence(
    policy: RecurrentNeuralPolicy,
    observations: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> None:
    inputs, hidden_states, logits = _recurrent_forward(policy, observations)
    probabilities = _softmax(logits)
    probabilities[np.arange(len(labels)), labels] -= 1.0
    probabilities /= len(labels)

    grad_w_xh = np.zeros_like(policy.w_xh)
    grad_w_hh = np.zeros_like(policy.w_hh)
    grad_b_h = np.zeros_like(policy.b_h)
    grad_w_hy = np.zeros_like(policy.w_hy)
    grad_b_y = np.zeros_like(policy.b_y)
    grad_next_hidden = np.zeros_like(policy.b_h)

    for index in reversed(range(len(labels))):
        grad_logits = probabilities[index]
        hidden = hidden_states[index + 1]
        previous_hidden = hidden_states[index]
        grad_w_hy += np.outer(hidden, grad_logits)
        grad_b_y += grad_logits

        grad_hidden = grad_logits @ policy.w_hy.T + grad_next_hidden
        grad_raw_hidden = grad_hidden * (1.0 - hidden**2)
        grad_w_xh += np.outer(inputs[index], grad_raw_hidden)
        grad_w_hh += np.outer(previous_hidden, grad_raw_hidden)
        grad_b_h += grad_raw_hidden
        grad_next_hidden = grad_raw_hidden @ policy.w_hh.T

    for grad in (grad_w_xh, grad_w_hh, grad_b_h, grad_w_hy, grad_b_y):
        np.clip(grad, -5.0, 5.0, out=grad)

    policy.w_xh -= learning_rate * grad_w_xh
    policy.w_hh -= learning_rate * grad_w_hh
    policy.b_h -= learning_rate * grad_b_h
    policy.w_hy -= learning_rate * grad_w_hy
    policy.b_y -= learning_rate * grad_b_y
