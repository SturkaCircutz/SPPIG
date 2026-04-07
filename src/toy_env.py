"""Toy repeated-structure environment used by the runnable prototype."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from psm import Action, Observation, ProgrammaticStateMachine
from teacher import StepResult


@dataclass
class ToyState:
    """Environment state for the arm-then-advance repetition task."""

    remaining_steps: int
    armed: bool = False


@dataclass
class RolloutStep:
    """One logged transition from a policy rollout."""

    step_index: int
    mode_name: str
    next_mode_name: str
    observation: Observation
    action: Action
    reward: float
    next_state: ToyState
    done: bool


@dataclass
class RolloutSummary:
    """A full rollout record together with aggregate metrics."""

    initial_state: ToyState
    steps: List[RolloutStep]
    total_reward: float
    success: bool


class RepetitionToyEnv:
    """
    A tiny benchmark with an explicit repeated structure.

    The agent must repeat an "arm -> advance" pattern until `remaining_steps`
    reaches zero. Training episodes are shorter than test episodes, so a compact
    state machine can generalize by repeating the same two-mode logic.
    """

    def __init__(
        self,
        train_lengths: Iterable[int] = (1, 2, 3, 4),
        test_lengths: Iterable[int] = (5, 6, 7, 8),
        seed: Optional[int] = 0,
    ):
        """Create a deterministic toy benchmark with separate train/test lengths."""

        self.train_lengths = tuple(int(length) for length in train_lengths)
        self.test_lengths = tuple(int(length) for length in test_lengths)
        self._rng = random.Random(seed)

    def sample_initial_state(self) -> ToyState:
        """Sample an initial training state for teacher-trace collection."""

        return ToyState(remaining_steps=self._rng.choice(self.train_lengths), armed=False)

    def observe(self, state: ToyState) -> Observation:
        """Convert the latent state into the observation seen by the policy."""

        return [float(state.remaining_steps), 1.0 if state.armed else 0.0]

    def teacher_action(self, observation: Observation, student_action: Optional[Action], step_index: int) -> Action:
        """Return the oracle action for the current observation."""

        remaining_steps = int(observation[0])
        armed = observation[1] >= 0.5
        if remaining_steps <= 0:
            return [0.0]
        return [1.0] if not armed else [0.0]

    def step(self, state: ToyState, action: Action) -> StepResult:
        """Apply one action and return the resulting transition."""

        if state.remaining_steps <= 0:
            return StepResult(next_state=ToyState(remaining_steps=0, armed=False), reward=0.0, done=True)

        next_state = ToyState(remaining_steps=state.remaining_steps, armed=state.armed)
        scalar_action = float(action[0])

        if not next_state.armed:
            if scalar_action >= 0.5:
                next_state.armed = True
                reward = 0.1
            else:
                reward = -0.2
        else:
            if scalar_action < 0.5:
                next_state.armed = False
                next_state.remaining_steps -= 1
                reward = 1.0 if next_state.remaining_steps == 0 else 0.4
            else:
                reward = -0.2

        done = next_state.remaining_steps == 0
        return StepResult(next_state=next_state, reward=reward, done=done)

    def label_mode(self, observation: Observation, action: Action, next_state: ToyState, done: bool) -> int:
        """Label the teacher mode used for the simplified supervised student fit."""

        remaining_steps = int(observation[0])
        armed = observation[1] >= 0.5
        if remaining_steps <= 0 or done:
            return 2
        return 0 if not armed else 1

    def rollout_policy(self, policy: ProgrammaticStateMachine, initial_state: ToyState) -> RolloutSummary:
        """Roll out one policy episode and keep a detailed transition log."""

        state = initial_state
        current_mode = policy.start_mode
        total_reward = 0.0
        max_steps = max(2 * initial_state.remaining_steps + 2, 2)
        steps: List[RolloutStep] = []

        for step_index in range(max_steps):
            observation = self.observe(state)
            mode_name = current_mode
            action, current_mode = policy.step(observation, current_mode)
            result = self.step(state, action)
            total_reward += result.reward

            steps.append(
                RolloutStep(
                    step_index=step_index,
                    mode_name=mode_name,
                    next_mode_name=current_mode,
                    observation=list(observation),
                    action=list(action),
                    reward=float(result.reward),
                    next_state=result.next_state,
                    done=result.done,
                )
            )

            state = result.next_state
            if result.done:
                return RolloutSummary(
                    initial_state=initial_state,
                    steps=steps,
                    total_reward=total_reward,
                    success=True,
                )

        return RolloutSummary(
            initial_state=initial_state,
            steps=steps,
            total_reward=total_reward,
            success=False,
        )

    def evaluate_policy(self, policy: ProgrammaticStateMachine, split: str) -> Dict[str, float]:
        """Evaluate one policy on every length from the chosen split."""

        lengths = self.train_lengths if split == "train" else self.test_lengths
        rollouts = [self.rollout_policy(policy, ToyState(remaining_steps=length, armed=False)) for length in lengths]
        return {
            f"{split}_success_rate": sum(1.0 if item.success else 0.0 for item in rollouts) / len(rollouts),
            f"{split}_avg_return": sum(item.total_reward for item in rollouts) / len(rollouts),
            f"{split}_avg_steps": sum(float(len(item.steps)) for item in rollouts) / len(rollouts),
        }
