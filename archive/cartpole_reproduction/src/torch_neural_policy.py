from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from shuttle_env import ShuttleLineEnv, ShuttleResult


ACTION_VALUES = [-1.0, 0.0, 1.0]


@dataclass
class TorchTrainConfig:
    model: str = "lstm"
    hidden_size: int = 64
    epochs: int = 80
    learning_rate: float = 0.003
    num_traces: int = 128
    batch_size: int = 32
    random_seed: int = 7
    max_steps: int = 200


@dataclass
class TorchTrainingResult:
    initial_loss: float
    final_loss: float
    num_examples: int


class Normalizer(nn.Module):
    def __init__(self, length_scale: float, crossing_scale: float) -> None:
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor([length_scale, crossing_scale], dtype=torch.float32),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation / self.scale


class MLPPolicy(nn.Module):
    def __init__(self, hidden_size: int, length_scale: float, crossing_scale: float) -> None:
        super().__init__()
        self.normalizer = Normalizer(length_scale, crossing_scale)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(ACTION_VALUES)),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(observations))


class LSTMPolicy(nn.Module):
    def __init__(self, hidden_size: int, length_scale: float, crossing_scale: float) -> None:
        super().__init__()
        self.normalizer = Normalizer(length_scale, crossing_scale)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, len(ACTION_VALUES))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer(observations)
        output, _ = self.lstm(normalized)
        return self.head(output)

    def initial_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_size = self.lstm.hidden_size
        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
        return h0, c0

    def step(
        self,
        observation: Sequence[float],
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            obs = torch.tensor([[observation]], dtype=torch.float32)
            normalized = self.normalizer(obs)
            output, next_state = self.lstm(normalized, state)
            logits = self.head(output[:, -1, :])
            label = int(torch.argmax(logits, dim=-1).item())
        return label, next_state


def make_model(env: ShuttleLineEnv, cfg: TorchTrainConfig) -> nn.Module:
    crossing_scale = float(max(env.train_crossings))
    if cfg.model == "mlp":
        return MLPPolicy(cfg.hidden_size, env.length, crossing_scale)
    if cfg.model == "lstm":
        return LSTMPolicy(cfg.hidden_size, env.length, crossing_scale)
    raise ValueError("model must be 'mlp' or 'lstm'")


def train_torch_behavior_cloning(
    env: ShuttleLineEnv,
    cfg: TorchTrainConfig,
) -> Tuple[nn.Module, TorchTrainingResult]:
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    sequences = collect_expert_sequences(env, cfg.num_traces, cfg.random_seed, cfg.max_steps)
    model = make_model(env, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    initial_loss = _dataset_loss(model, sequences)
    for _ in range(cfg.epochs):
        random.shuffle(sequences)
        for start in range(0, len(sequences), cfg.batch_size):
            batch = sequences[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = _batch_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    final_loss = _dataset_loss(model, sequences)
    num_examples = sum(len(labels) for _, labels in sequences)
    return model, TorchTrainingResult(initial_loss, final_loss, num_examples)


def collect_expert_sequences(
    env: ShuttleLineEnv,
    num_traces: int,
    seed: int,
    max_steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_traces):
        observations, labels = _expert_sequence(
            env,
            required_crossings=rng.choice(env.train_crossings),
            max_steps=max_steps,
        )
        sequences.append(
            (
                torch.tensor(observations, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        )
    return sequences


def evaluate_torch_policy(
    env: ShuttleLineEnv,
    model: nn.Module,
    required_crossings: int,
    max_steps: int = 200,
) -> ShuttleResult:
    if isinstance(model, LSTMPolicy):
        return _evaluate_lstm(env, model, required_crossings, max_steps)
    return _evaluate_mlp(env, model, required_crossings, max_steps)


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: TorchTrainConfig,
    result: TorchTrainingResult,
) -> None:
    torch.save(
        {
            "model_type": cfg.model,
            "hidden_size": cfg.hidden_size,
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "num_examples": result.num_examples,
        },
        path,
    )


def _sequence_loss(model: nn.Module, observations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(model, LSTMPolicy):
        logits = model(observations.unsqueeze(0)).squeeze(0)
    else:
        logits = model(observations)
    return F.cross_entropy(logits, labels)


def _batch_loss(model: nn.Module, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    observations = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    if isinstance(model, LSTMPolicy):
        padded_observations = pad_sequence(observations, batch_first=True)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        logits = model(padded_observations)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            padded_labels.reshape(-1),
            ignore_index=-100,
        )

    flat_observations = torch.cat(observations, dim=0)
    flat_labels = torch.cat(labels, dim=0)
    return F.cross_entropy(model(flat_observations), flat_labels)


def _dataset_loss(model: nn.Module, sequences: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    with torch.no_grad():
        return _batch_loss(model, sequences).item()


def _evaluate_mlp(
    env: ShuttleLineEnv,
    model: nn.Module,
    required_crossings: int,
    max_steps: int,
) -> ShuttleResult:
    x = 0.0
    crossings = 0
    remaining = required_crossings
    expected_direction = 1.0

    for step in range(1, max_steps + 1):
        observation = torch.tensor([[x, float(remaining)]], dtype=torch.float32)
        with torch.no_grad():
            label = int(torch.argmax(model(observation), dim=-1).item())
        action = ACTION_VALUES[label]
        if action == 0.0:
            success = crossings >= required_crossings
            return ShuttleResult(required_crossings, success, crossings, step - 1, float(success), "end")

        old_x = x
        x = env._advance(x, action)
        target = env.length if expected_direction > 0 else 0.0
        if old_x != target and x == target:
            crossings += 1
            remaining = max(0, required_crossings - crossings)
            expected_direction *= -1.0
    return ShuttleResult(required_crossings, False, crossings, max_steps, 0.0, "running")


def _evaluate_lstm(
    env: ShuttleLineEnv,
    model: LSTMPolicy,
    required_crossings: int,
    max_steps: int,
) -> ShuttleResult:
    x = 0.0
    crossings = 0
    remaining = required_crossings
    expected_direction = 1.0
    state = model.initial_state()

    for step in range(1, max_steps + 1):
        label, state = model.step([x, float(remaining)], state)
        action = ACTION_VALUES[label]
        if action == 0.0:
            success = crossings >= required_crossings
            return ShuttleResult(required_crossings, success, crossings, step - 1, float(success), "end")

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
