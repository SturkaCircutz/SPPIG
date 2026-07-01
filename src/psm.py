from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional


Observation = List[float]
Action = List[float]
ActionFn = Callable[[Observation], Action]
SwitchFn = Callable[[Observation], float]


def constant_action(action: Action) -> ActionFn:
    """Build a controller from the constant-action grammar in the paper."""

    def _action(_: Observation) -> Action:
        return list(action)

    return _action


@dataclass(frozen=True)
class ThresholdSwitch:
    """Axis-aligned Boolean switch: observation[i] >= c or observation[i] <= c.

    The paper searches a Boolean grammar for switching conditions.  This class is
    the depth-1 case of that grammar, which is enough for the compact benchmark
    in this repository and keeps the implementation inspectable.
    """

    feature_index: int
    relation: str
    threshold: float

    def __call__(self, observation: Observation) -> float:
        value = observation[self.feature_index]
        if self.relation == ">=":
            return value - self.threshold
        if self.relation == "<=":
            return self.threshold - value
        raise ValueError(f"unknown switch relation: {self.relation}")

    def describe(self) -> str:
        return f"o[{self.feature_index}] {self.relation} {self.threshold:.3f}"


@dataclass
class Mode:
    name: str
    action_fn: ActionFn
    switches: Dict[str, SwitchFn] = field(default_factory=dict)
    action_description: str = ""


@dataclass
class ProgrammaticStateMachine:
    modes: Mapping[str, Mode]
    start_mode: str
    end_mode: str

    def step(self, observation: Observation, mode_name: str) -> tuple[Action, str]:
        mode = self.modes[mode_name]
        action = mode.action_fn(observation)

        # The state-machine semantics match Eq. (2) in the paper: act with the
        # current mode, then choose the strongest enabled outgoing switch.
        best_target: Optional[str] = None
        best_score = 0.0
        for target, switch_fn in mode.switches.items():
            score = switch_fn(observation)
            if score >= 0.0 and (best_target is None or score > best_score):
                best_target = target
                best_score = score

        next_mode = best_target if best_target is not None else mode_name
        return action, next_mode

    def describe(self) -> str:
        lines = [
            f"ProgrammaticStateMachine(start={self.start_mode}, end={self.end_mode})"
        ]
        for name in self.modes:
            mode = self.modes[name]
            action = mode.action_description or "<callable action>"
            lines.append(f"- {name}: action {action}")
            for target, switch_fn in mode.switches.items():
                if hasattr(switch_fn, "describe"):
                    switch = switch_fn.describe()
                else:
                    switch = "<callable switch>"
                lines.append(f"  -> {target} when {switch}")
        return "\n".join(lines)
