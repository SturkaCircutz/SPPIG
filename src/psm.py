"""Programmatic state-machine policy primitives used by the toy prototype."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence


Observation = List[float]
Action = List[float]
ActionFn = Callable[[Observation], Action]
SwitchFn = Callable[[Observation], float]


@dataclass(frozen=True)
class ConstantAction:
    """A grammar leaf that always emits the same action vector."""

    value: Sequence[float]

    def __call__(self, observation: Observation) -> Action:
        return [float(component) for component in self.value]

    def describe(self) -> str:
        return f"constant({', '.join(f'{value:.3f}' for value in self.value)})"


@dataclass(frozen=True)
class ThresholdSwitch:
    """A grammar leaf that activates when one observation dimension crosses a threshold."""

    index: int
    threshold: float
    direction: str

    def __call__(self, observation: Observation) -> float:
        if self.index >= len(observation):
            return -1.0

        value = float(observation[self.index])
        if self.direction == ">=":
            return value - self.threshold
        if self.direction == "<=":
            return self.threshold - value
        raise ValueError(f"Unsupported switch direction: {self.direction}")

    def describe(self) -> str:
        return f"obs[{self.index}] {self.direction} {self.threshold:.3f}"


@dataclass
class Mode:
    """One controller mode inside the programmatic state machine."""

    name: str
    action_fn: ActionFn
    switches: Dict[str, SwitchFn] = field(default_factory=dict)


@dataclass
class ProgrammaticStateMachine:
    """A compact policy made of reusable controller modes and symbolic switches."""

    modes: Mapping[str, Mode]
    start_mode: str
    end_mode: str

    def step(self, observation: Observation, mode_name: str) -> tuple[Action, str]:
        """Emit an action and choose the next mode from the current observation."""

        mode = self.modes[mode_name]
        action = mode.action_fn(observation)

        best_target: Optional[str] = None
        best_score = 0.0
        for target, switch_fn in mode.switches.items():
            score = switch_fn(observation)
            if score >= 0.0 and (best_target is None or score > best_score):
                best_target = target
                best_score = score

        next_mode = best_target if best_target is not None else mode_name
        return action, next_mode


def describe_policy(policy: ProgrammaticStateMachine) -> str:
    """Return a readable multi-line summary of the learned state machine."""

    lines = [f"start={policy.start_mode}, end={policy.end_mode}"]
    for mode_name, mode in policy.modes.items():
        if hasattr(mode.action_fn, "describe"):
            action_desc = mode.action_fn.describe()
        else:
            action_desc = repr(mode.action_fn)

        if mode.switches:
            switch_desc = []
            for target, switch_fn in mode.switches.items():
                if hasattr(switch_fn, "describe"):
                    switch_desc.append(f"{target} if {switch_fn.describe()}")
                else:
                    switch_desc.append(f"{target} if {switch_fn!r}")
            switch_summary = "; ".join(switch_desc)
        else:
            switch_summary = "stay"

        lines.append(f"{mode_name}: {action_desc}; {switch_summary}")
    return "\n".join(lines)
