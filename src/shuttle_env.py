from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from teacher import TeacherTrace


Observation = List[float]
Action = List[float]


@dataclass
class ShuttleResult:
    required_crossings: int
    success: bool
    crossings: int
    steps: int
    reward: float
    final_mode: str


class ShuttleLineEnv:
    """A tiny inductive-generalization benchmark.

    The observation is position plus the remaining number of required crossings.
    The mode still matters: away from a boundary, position alone does not say
    whether the agent is currently moving right or left.
    """

    def __init__(
        self,
        length: int = 5,
        train_crossings: Optional[List[int]] = None,
        test_crossings: Optional[List[int]] = None,
    ) -> None:
        self.length = float(length)
        self.train_crossings = train_crossings or [2, 3]
        self.test_crossings = test_crossings or [6]

    def sample_initial_state(self, rng: Any) -> Dict[str, int]:
        return {"required_crossings": rng.choice(self.train_crossings)}

    def teacher_trace(
        self,
        initial_state: Dict[str, int],
        student_policy: Any,
        cfg: Any,
    ) -> TeacherTrace:
        if getattr(cfg, "prefer_student_rollouts", True) and student_policy is not None:
            student_trace = self._trace_student(
                initial_state["required_crossings"],
                student_policy,
                cfg.max_steps,
            )
            if student_trace is not None and student_trace.reward > 0.0:
                return student_trace
        return self._expert_trace(initial_state["required_crossings"], cfg.max_steps)

    def _expert_trace(self, required_crossings: int, max_steps: int) -> TeacherTrace:
        observations: List[Observation] = []
        actions: List[Action] = []
        mode_hints: List[Any] = []
        x = 0.0
        expected_direction = 1.0
        crossings = 0
        remaining = required_crossings
        segment = 0

        while crossings < required_crossings and len(actions) < max_steps:
            target = self.length if expected_direction > 0 else 0.0
            while x != target and len(actions) < max_steps:
                observations.append([x, float(remaining)])
                actions.append([expected_direction])
                mode_hints.append(segment)
                x = self._advance(x, expected_direction)
                if x == target:
                    crossings += 1
                    remaining = required_crossings - crossings

            # Add the boundary observation to the ending segment.  The student
            # can then learn a switch that fires at the boundary under the
            # act-then-switch semantics used by ProgrammaticStateMachine.step.
            if len(actions) < max_steps:
                observations.append([x, float(remaining)])
                actions.append([expected_direction])
                mode_hints.append(segment)
            if crossings >= required_crossings:
                break
            if len(actions) < max_steps:
                segment += 1
                expected_direction *= -1.0

        reward = 1.0 if crossings >= required_crossings else 0.0
        if reward > 0.0 and len(actions) < max_steps:
            observations.append([x, 0.0])
            actions.append([0.0])
            mode_hints.append("end")

        return TeacherTrace(
            initial_state={"required_crossings": required_crossings},
            observations=observations,
            actions=actions,
            mode_hints=mode_hints,
            reward=reward,
        )

    def _trace_student(
        self,
        required_crossings: int,
        policy: Any,
        max_steps: int,
    ) -> Optional[TeacherTrace]:
        x = 0.0
        mode = policy.start_mode
        crossings = 0
        remaining = required_crossings
        expected_direction = 1.0
        observations: List[Observation] = []
        actions: List[Action] = []
        mode_hints: List[int] = []
        segment = 0

        for _ in range(max_steps):
            if mode == policy.end_mode:
                observations.append([x, float(remaining)])
                actions.append([0.0])
                mode_hints.append(segment)
                break

            observation = [x, float(remaining)]
            action, next_mode = policy.step(observation, mode)
            clipped_action = 1.0 if action[0] >= 0.0 else -1.0
            observations.append(observation)
            actions.append([clipped_action])
            mode_hints.append(segment)
            x = self._advance(x, clipped_action)
            target = self.length if expected_direction > 0 else 0.0
            if x == target:
                crossings += 1
                remaining = max(0, required_crossings - crossings)
                expected_direction *= -1.0
            if next_mode != mode:
                segment += 1
            mode = next_mode

        if crossings < required_crossings or mode != policy.end_mode:
            return None
        return TeacherTrace(
            initial_state={"required_crossings": required_crossings},
            observations=observations,
            actions=actions,
            mode_hints=mode_hints,
            reward=1.0,
        )

    def evaluate_policy(
        self,
        policy: Any,
        required_crossings: int,
        max_steps: int = 200,
    ) -> ShuttleResult:
        x = 0.0
        mode = policy.start_mode
        crossings = 0
        remaining = required_crossings
        expected_direction = 1.0

        for step in range(1, max_steps + 1):
            if mode == policy.end_mode:
                success = crossings >= required_crossings
                return ShuttleResult(
                    required_crossings,
                    success,
                    crossings,
                    step - 1,
                    1.0 if success else 0.0,
                    mode,
                )

            action, mode = policy.step([x, float(remaining)], mode)
            clipped_action = 1.0 if action[0] >= 0.0 else -1.0
            x = self._advance(x, clipped_action)
            target = self.length if expected_direction > 0 else 0.0
            if x == target:
                crossings += 1
                remaining = max(0, required_crossings - crossings)
                expected_direction *= -1.0
        return ShuttleResult(required_crossings, False, crossings, max_steps, 0.0, mode)

    def _advance(self, x: float, action: float) -> float:
        if action > 0.0:
            return min(self.length, x + 1.0)
        return max(0.0, x - 1.0)
