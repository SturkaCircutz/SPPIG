from dataclasses import dataclass
from typing import Any, List, Optional

from adaptive_teaching_sim import teacher_optimize_task


@dataclass
class TeacherTrace:
    initial_state: Any
    observations: List[Any]
    actions: List[Any]
    mode_hints: List[int]
    reward: float


def optimize_teacher(tasks, reuse, rng, teacher_cfg, student_params: Optional[Any] = None):
    """Optimize loop-free teacher traces for the simplified parking benchmark.

    This is the implementation counterpart of the paper's teacher step: each
    training initial state gets an over-parameterized, task-specific policy found
    by random-search/CEM-style optimization, with optional regularization toward
    the current student parameters.
    """

    if isinstance(teacher_cfg, dict):
        teacher_iters = int(teacher_cfg.get("teacher_iters", 3))
    else:
        teacher_iters = int(getattr(teacher_cfg, "teacher_iters", 3))
    return [
        teacher_optimize_task(task, reuse, rng, teacher_iters, student_params)
        for task in tasks
    ]
