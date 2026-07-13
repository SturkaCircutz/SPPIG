from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, List, Optional


@dataclass
class TeacherTrace:
    initial_state: Any
    observations: List[Any]
    actions: List[Any]
    mode_hints: List[Any]
    reward: float


@dataclass
class TeacherConfig:
    """Configuration for the teacher side of adaptive teaching.

    This generic interface mirrors the paper's teacher/student split while
    letting each benchmark provide its own trajectory optimizer.
    """

    num_traces: int = 8
    random_seed: int = 0
    max_steps: int = 200
    prefer_student_rollouts: bool = True


def _validate_trace(trace: TeacherTrace) -> None:
    if len(trace.observations) != len(trace.actions):
        raise ValueError("teacher trace must have one observation per action")
    if len(trace.mode_hints) != len(trace.actions):
        raise ValueError("teacher trace must have one mode hint per action")


def optimize_teacher(
    env: Any,
    student_policy: Optional[Any],
    teacher_cfg: TeacherConfig,
) -> List[TeacherTrace]:
    """Sample train states and produce benchmark-specific teacher traces."""

    if not hasattr(env, "sample_initial_state"):
        raise TypeError("env must define sample_initial_state(rng)")
    if not hasattr(env, "teacher_trace"):
        raise TypeError("env must define teacher_trace(initial_state, student_policy, cfg)")

    rng = random.Random(teacher_cfg.random_seed)
    traces: List[TeacherTrace] = []
    for _ in range(teacher_cfg.num_traces):
        initial_state = env.sample_initial_state(rng)
        trace = env.teacher_trace(initial_state, student_policy, teacher_cfg)
        _validate_trace(trace)
        traces.append(trace)
    return traces
