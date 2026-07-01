"""Programmatic state-machine policies for the parking simulation."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple


Observation = List[float]
Action = List[float]
ActionFn = Callable[[Observation], Action]
SwitchFn = Callable[[Observation], float]


@dataclass
class LocalMode:
    name: str
    action_fn: ActionFn
    switches: Dict[str, SwitchFn]


class LocalProgrammaticStateMachine:
    def __init__(self, modes: Mapping[str, LocalMode], start_mode: str, end_mode: str):
        self.modes = dict(modes)
        self.start_mode = start_mode
        self.end_mode = end_mode

    def step(self, observation: Observation, mode_name: str) -> Tuple[Action, str]:
        mode = self.modes[mode_name]
        action = mode.action_fn(observation)
        best_target: Optional[str] = None
        best_score = -float("inf")
        for target, switch_fn in mode.switches.items():
            score = float(switch_fn(observation))
            if score >= 0.0 and score > best_score:
                best_target = target
                best_score = score
        return action, best_target if best_target is not None else mode_name


@dataclass
class ReuseResult:
    status: str
    reason: str
    module_path: Optional[str]
    psm_cls: type
    mode_cls: type


@dataclass
class PolicyParams:
    front_threshold: float
    arc_y_threshold: float
    repeat_back_threshold: float
    theta_threshold: float
    center_y_threshold: float
    done_radius: float
    approach_speed: float
    approach_heading_gain: float
    approach_lane_y: float
    approach_lateral_gain: float
    arc_speed: float
    arc_steer: float
    arc_lateral: float
    counter_speed: float
    counter_steer: float
    counter_lateral: float
    center_speed_gain: float
    center_theta_gain: float
    center_goal_y_gain: float
    center_lateral_gain: float


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def default_policy_params() -> PolicyParams:
    return PolicyParams(
        front_threshold=2.10,
        arc_y_threshold=1.30,
        repeat_back_threshold=0.62,
        theta_threshold=0.18,
        center_y_threshold=1.00,
        done_radius=0.65,
        approach_speed=1.25,
        approach_heading_gain=-1.10,
        approach_lane_y=3.00,
        approach_lateral_gain=0.65,
        arc_speed=-0.92,
        arc_steer=-0.58,
        arc_lateral=-0.18,
        counter_speed=-0.70,
        counter_steer=0.60,
        counter_lateral=-0.07,
        center_speed_gain=0.72,
        center_theta_gain=-1.35,
        center_goal_y_gain=0.55,
        center_lateral_gain=0.85,
    )


def mode_action_params(params: PolicyParams, mode_name: str) -> List[float]:
    if mode_name == "approach":
        return [
            params.approach_speed,
            params.approach_heading_gain,
            params.approach_lane_y,
            params.approach_lateral_gain,
        ]
    if mode_name == "arc_back":
        return [params.arc_speed, params.arc_steer, params.arc_lateral]
    if mode_name == "counter":
        return [params.counter_speed, params.counter_steer, params.counter_lateral]
    if mode_name == "center":
        return [
            params.center_speed_gain,
            params.center_theta_gain,
            params.center_goal_y_gain,
            params.center_lateral_gain,
        ]
    return [0.0, 0.0, 0.0]


def action_from_mode_params(mode_name: str, action_params: List[float], obs: Observation) -> Action:
    if mode_name == "approach":
        speed, heading_gain, lane_y, lateral_gain = action_params
        lane_correction = clip(lateral_gain * (lane_y - obs[1]), -0.35, 0.35)
        return [speed, clip(heading_gain * obs[2], -0.35, 0.35), lane_correction]
    if mode_name in {"arc_back", "counter"}:
        return [action_params[0], action_params[1], action_params[2]]
    if mode_name == "center":
        speed_gain, theta_gain, goal_y_gain, lateral_gain = action_params
        speed = clip(speed_gain * obs[5], -0.65, 0.65)
        if abs(obs[5]) < 0.18 and abs(obs[6]) > 0.12:
            speed = 0.38 if obs[6] < 0.0 else -0.34
        steer = clip(theta_gain * obs[2] + goal_y_gain * obs[6], -0.58, 0.58)
        lateral_rate = clip(lateral_gain * obs[6], -0.42, 0.42)
        return [speed, steer, lateral_rate]
    return [0.0, 0.0, 0.0]


def try_reuse_state_machine(repo_root: Path, manifest: dict) -> ReuseResult:
    """Reuse the existing repo PSM API if it is safe to import."""

    failures: List[str] = []
    py_files = [item for item in manifest.get("files", []) if item.get("suffix") == ".py"]
    likely = sorted(
        py_files,
        key=lambda item: (
            "psm" not in item.get("path", "").lower(),
            "policy" not in item.get("path", "").lower(),
            item.get("path", ""),
        ),
    )
    for item in likely:
        rel = item.get("path", "")
        if not any(hint in rel.lower() for hint in ("psm", "policy", "controller")):
            continue
        path = repo_root / rel
        try:
            spec = importlib.util.spec_from_file_location("repo_psm_adapter", path)
            if spec is None or spec.loader is None:
                failures.append(f"{rel}: no import loader")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["repo_psm_adapter"] = module
            spec.loader.exec_module(module)
            psm_cls = getattr(module, "ProgrammaticStateMachine", None)
            mode_cls = getattr(module, "Mode", None)
            if psm_cls is None or mode_cls is None:
                failures.append(f"{rel}: missing ProgrammaticStateMachine or Mode")
                continue
            if "step" not in dir(psm_cls) or len(inspect.signature(psm_cls.step).parameters) < 3:
                failures.append(f"{rel}: incompatible step signature")
                continue
            return ReuseResult("reused", "compatible ProgrammaticStateMachine and Mode found", rel, psm_cls, mode_cls)
        except Exception as exc:  # noqa: BLE001 - keep fallback path explicit.
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")

    reason = "no compatible repo state-machine class found"
    if failures:
        reason += "; " + " | ".join(failures[:5])
    return ReuseResult("fallback", reason, None, LocalProgrammaticStateMachine, LocalMode)


def build_policy(params: PolicyParams, reuse: ReuseResult) -> object:
    """Build the compact student state machine.

    The modes intentionally form repeated forward/backward maneuvers before
    exiting, matching the inductive-generalization mechanism from the paper.
    """

    ModeCls = reuse.mode_cls
    PsmCls = reuse.psm_cls

    def approach_action(obs: Observation) -> Action:
        return action_from_mode_params("approach", mode_action_params(params, "approach"), obs)

    def arc_back_action(obs: Observation) -> Action:
        return action_from_mode_params("arc_back", mode_action_params(params, "arc_back"), obs)

    def counter_action(obs: Observation) -> Action:
        return action_from_mode_params("counter", mode_action_params(params, "counter"), obs)

    def center_action(obs: Observation) -> Action:
        return action_from_mode_params("center", mode_action_params(params, "center"), obs)

    modes = {
        "approach": ModeCls("approach", approach_action, {"arc_back": lambda obs: params.front_threshold - obs[3]}),
        "arc_back": ModeCls(
            "arc_back",
            arc_back_action,
            {
                "counter": lambda obs: params.arc_y_threshold - obs[1],
                "approach": lambda obs: params.repeat_back_threshold - obs[4],
            },
        ),
        "counter": ModeCls(
            "counter",
            counter_action,
            {"center": lambda obs: min(params.theta_threshold - abs(obs[2]), params.center_y_threshold - obs[1])},
        ),
        "center": ModeCls(
            "center",
            center_action,
            {"done": lambda obs: min(params.done_radius - obs[7], 0.22 - abs(obs[2]))},
        ),
        "done": ModeCls("done", lambda obs: [0.0, 0.0, 0.0], {}),
    }
    return PsmCls(modes=modes, start_mode="approach", end_mode="done")


def build_baseline_policy(reuse: ReuseResult) -> object:
    """A non-adaptive baseline with fixed, hand-picked thresholds."""

    return build_policy(default_policy_params(), reuse)
