from dataclasses import dataclass
from typing import Any, Iterable

from adaptive_teaching_sim import distill_student


@dataclass
class StudentConfig:
    num_modes: int
    action_grammar: Any
    switch_grammar: Any
    max_em_iters: int = 20


def fit_student_from_traces(traces: Iterable[Any], grammar, student_cfg: StudentConfig):
    """Fit a compact student state machine from teacher traces.

    The backend uses a lightweight EM-style loop: assign loop-free teacher
    segments to student modes, then update action-program parameters and
    switching thresholds from the weighted segment data.
    """

    return distill_student(list(traces))
