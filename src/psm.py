from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional


Observation = List[float]
Action = List[float]
ActionFn = Callable[[Observation], Action]
SwitchFn = Callable[[Observation], float]


@dataclass
class Mode:
    name: str
    action_fn: ActionFn
    switches: Dict[str, SwitchFn] = field(default_factory=dict)


@dataclass
class ProgrammaticStateMachine:
    modes: Mapping[str, Mode]
    start_mode: str
    end_mode: str

    def step(self, observation: Observation, mode_name: str) -> tuple[Action, str]:
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
