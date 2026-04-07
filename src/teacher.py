"""Teacher-side rollout utilities for the simplified adaptive-teaching prototype."""

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

from psm import Action, Observation, ProgrammaticStateMachine


@dataclass
class StepResult:
    """Environment transition returned after applying one action."""

    next_state: Any
    reward: float
    done: bool


@dataclass
class TeacherTrace:
    """A loop-free teacher trace used as supervision for the student fitter."""

    initial_state: Any
    observations: List[Observation]
    actions: List[Action]
    mode_hints: List[int]
    reward: float


@dataclass
class TeacherConfig:
    """Configuration for collecting a batch of teacher traces."""

    num_traces: int = 16
    max_steps: int = 32
    student_action_weight: float = 0.0


class TeacherEnvironment(Protocol):
    """Minimal environment interface required by the simplified teacher step."""

    def sample_initial_state(self) -> Any:
        ...

    def observe(self, state: Any) -> Observation:
        ...

    def teacher_action(
        self,
        observation: Observation,
        student_action: Optional[Action],
        step_index: int,
    ) -> Action:
        ...

    def step(self, state: Any, action: Action) -> StepResult:
        ...

    def label_mode(self, observation: Observation, action: Action, next_state: Any, done: bool) -> int:
        ...


def _blend_actions(teacher_action: Action, student_action: Optional[Action], weight: float) -> Action:
    """Interpolate between teacher and student actions for lightweight regularization."""

    if student_action is None or weight <= 0.0:
        return list(teacher_action)
    if weight >= 1.0:
        return list(student_action)

    blended: Action = []
    for teacher_component, student_component in zip(teacher_action, student_action):
        blended.append((1.0 - weight) * float(teacher_component) + weight * float(student_component))
    return blended


def optimize_teacher(
    env: TeacherEnvironment,
    student_policy: Optional[ProgrammaticStateMachine],
    teacher_cfg: TeacherConfig,
) -> List[TeacherTrace]:
    """
    Collect loop-free teacher traces from an environment-specific teacher policy.

    This is a simplified prototype of the teacher step: the environment supplies
    the teacher action and the mode labels, while this function handles rollout
    bookkeeping and optional regularization toward the current student action.
    """

    traces: List[TeacherTrace] = []
    for _ in range(teacher_cfg.num_traces):
        initial_state = env.sample_initial_state()
        state = initial_state
        current_mode = student_policy.start_mode if student_policy is not None else None

        observations: List[Observation] = []
        actions: List[Action] = []
        mode_hints: List[int] = []
        total_reward = 0.0

        for step_index in range(teacher_cfg.max_steps):
            observation = [float(value) for value in env.observe(state)]
            student_action: Optional[Action] = None
            next_student_mode = current_mode
            if student_policy is not None and current_mode is not None:
                student_action, next_student_mode = student_policy.step(observation, current_mode)

            teacher_action = env.teacher_action(observation, student_action, step_index)
            action = _blend_actions(teacher_action, student_action, teacher_cfg.student_action_weight)
            result = env.step(state, action)
            mode_hint = int(env.label_mode(observation, action, result.next_state, result.done))

            # Keep the full supervised trace so that the student can fit both
            # per-mode actions and cross-mode switching guards from the same data.
            observations.append(observation)
            actions.append([float(value) for value in action])
            mode_hints.append(mode_hint)
            total_reward += float(result.reward)

            state = result.next_state
            current_mode = next_student_mode
            if result.done:
                break

        traces.append(
            TeacherTrace(
                initial_state=initial_state,
                observations=observations,
                actions=actions,
                mode_hints=mode_hints,
                reward=total_reward,
            )
        )

    return traces
