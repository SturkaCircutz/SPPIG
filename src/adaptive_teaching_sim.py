"""Adaptive-teaching approximation for programmatic parking policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments.
    plt = None

from parking_env import ParkingTask, Trajectory, collision_or_bounds, is_success, make_tasks, observe, step_dynamics
from programmatic_policy import (
    PolicyParams,
    ReuseResult,
    action_from_mode_params,
    build_baseline_policy,
    build_policy,
    clip,
    default_policy_params,
    mode_action_params,
    try_reuse_state_machine,
)
from repo_scan import scan_repo


MODE_SEQUENCE = ("approach", "arc_back", "approach", "arc_back", "counter", "center")
STUDENT_MODES = ("approach", "arc_back", "counter", "center")
ACTION_SLICES = {
    "approach": slice(0, 4),
    "arc_back": slice(4, 7),
    "counter": slice(7, 10),
    "center": slice(10, 14),
}
ACTION_LOW = np.array([0.70, -2.00, 2.30, 0.20, -1.35, -0.78, -0.34, -1.05, 0.28, -0.20, 0.35, -2.20, 0.00, 0.35], dtype=float)
ACTION_HIGH = np.array([1.55, -0.15, 3.55, 1.10, -0.45, -0.22, 0.02, -0.35, 0.85, 0.08, 1.20, -0.55, 1.15, 1.20], dtype=float)
DEFAULT_SEGMENT_DURATIONS = np.array([105, 28, 20, 20, 16, 14], dtype=float)
DURATION_LOW = np.array([55, 14, 8, 0, 0, 5], dtype=float)
DURATION_HIGH = np.array([135, 44, 36, 38, 30, 30], dtype=float)


@dataclass
class LoopFreeProgram:
    mode_sequence: List[str]
    action_params: List[List[float]]
    durations: List[int]


def simulate_policy(policy: object, params: dict, task: ParkingTask) -> Trajectory:
    state = task.start.copy()
    mode = str(getattr(policy, "start_mode", "approach"))
    states = [state.tolist()]
    observations = []
    actions = []
    modes = []
    cycle_count = 0
    collision = False

    for _ in range(task.max_steps):
        obs = observe(state, task)
        action, next_mode = policy.step(obs, mode)
        observations.append(obs)
        actions.append([float(value) for value in action])
        modes.append(mode)
        if mode == "arc_back" and str(next_mode) == "approach":
            cycle_count += 1
        mode = str(next_mode)
        if mode == "done":
            states.append(state.tolist())
            break
        state = step_dynamics(state, action)
        states.append(state.tolist())
        if collision_or_bounds(state, task):
            collision = True
            break

    success = is_success(state, mode, task) and not collision
    goal_error = float(np.linalg.norm(state[:2] - task.goal[:2]))
    heading_error = float(abs(state[2]))
    score = 120.0 * float(success) - 35.0 * float(collision) - 8.0 * goal_error - 4.0 * heading_error - 0.20 * len(actions)
    return Trajectory(task.task_id, states, observations, actions, modes, success, collision, float(score), int(cycle_count), dict(params))


def _base_action_vector(params: Optional[PolicyParams]) -> np.ndarray:
    source = params if params is not None else default_policy_params()
    return np.array(
        mode_action_params(source, "approach")
        + mode_action_params(source, "arc_back")
        + mode_action_params(source, "counter")
        + mode_action_params(source, "center"),
        dtype=float,
    )


def _clip_actions(vector: Sequence[float]) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(vector, dtype=float), ACTION_LOW), ACTION_HIGH)


def _clip_durations(vector: Sequence[float]) -> np.ndarray:
    clipped = np.minimum(np.maximum(np.asarray(vector, dtype=float), DURATION_LOW), DURATION_HIGH)
    return np.rint(clipped).astype(int)


def _decode_loop_free(vector: Sequence[float]) -> LoopFreeProgram:
    array = np.asarray(vector, dtype=float)
    actions = _clip_actions(array[: len(ACTION_LOW)])
    durations = _clip_durations(array[len(ACTION_LOW) :])
    action_params = []
    for mode_name in MODE_SEQUENCE:
        action_params.append(actions[ACTION_SLICES[mode_name]].astype(float).tolist())
    return LoopFreeProgram(list(MODE_SEQUENCE), action_params, durations.astype(int).tolist())


def _loop_free_vector(program: LoopFreeProgram) -> np.ndarray:
    action_by_mode: Dict[str, List[float]] = {}
    for mode_name, action_params in zip(program.mode_sequence, program.action_params):
        action_by_mode.setdefault(mode_name, action_params)
    action_vector = np.array(
        action_by_mode["approach"] + action_by_mode["arc_back"] + action_by_mode["counter"] + action_by_mode["center"],
        dtype=float,
    )
    return np.concatenate([action_vector, np.array(program.durations, dtype=float)])


def _segments_from_program(program: LoopFreeProgram, segment_starts: List[List[float]], segment_ends: List[List[float]]) -> List[dict]:
    segments = []
    for mode_name, action_params, duration, start_obs, end_obs in zip(
        program.mode_sequence,
        program.action_params,
        program.durations,
        segment_starts,
        segment_ends,
    ):
        if duration <= 0:
            continue
        segments.append(
            {
                "mode_hint": mode_name,
                "action_params": [float(value) for value in action_params],
                "duration": int(duration),
                "start_observation": [float(value) for value in start_obs],
                "end_observation": [float(value) for value in end_obs],
            }
        )
    return segments


def simulate_loop_free(program: LoopFreeProgram, task: ParkingTask, params: Optional[dict] = None) -> Trajectory:
    state = task.start.copy()
    states = [state.tolist()]
    observations: List[List[float]] = []
    actions: List[List[float]] = []
    modes: List[str] = []
    segment_starts: List[List[float]] = []
    segment_ends: List[List[float]] = []
    collision = False

    for mode_name, action_params, duration in zip(program.mode_sequence, program.action_params, program.durations):
        if duration <= 0:
            segment_starts.append(observe(state, task))
            segment_ends.append(observe(state, task))
            continue
        segment_starts.append(observe(state, task))
        for _ in range(duration):
            if len(actions) >= task.max_steps:
                break
            obs = observe(state, task)
            action = action_from_mode_params(mode_name, action_params, obs)
            observations.append(obs)
            actions.append([float(value) for value in action])
            modes.append(mode_name)
            state = step_dynamics(state, action)
            states.append(state.tolist())
            if collision_or_bounds(state, task):
                collision = True
                break
        segment_ends.append(observe(state, task))
        if collision or len(actions) >= task.max_steps:
            break

    mode = "done" if not collision else (modes[-1] if modes else "approach")
    success = is_success(state, mode, task) and not collision
    goal_error = float(np.linalg.norm(state[:2] - task.goal[:2]))
    heading_error = float(abs(state[2]))
    score = 120.0 * float(success) - 35.0 * float(collision) - 8.0 * goal_error - 4.0 * heading_error - 0.20 * len(actions)
    metadata = dict(params or {})
    metadata["teacher_type"] = "loop_free"
    metadata["loop_free_segments"] = _segments_from_program(program, segment_starts, segment_ends)
    return Trajectory(task.task_id, states, observations, actions, modes, success, collision, float(score), 0, metadata)


def teacher_optimize_task(task: ParkingTask, reuse: ReuseResult, rng: np.random.Generator, teacher_iters: int, student_params: Optional[PolicyParams]) -> Trajectory:
    """Optimize a task-specific loop-free teacher policy.

    This mirrors Section 4.2 of the paper: each teacher trace is a fixed sequence
    of action-program segments and durations.  The adaptive part is the
    regularizer toward the current compact student, which biases future traces
    toward structures that the student can imitate.
    """

    del reuse
    action_mean = _base_action_vector(student_params)
    duration_mean = DEFAULT_SEGMENT_DURATIONS.copy()
    mean = np.concatenate([action_mean, duration_mean])
    std = np.concatenate(
        [
            np.array([0.18, 0.35, 0.22, 0.20, 0.18, 0.14, 0.08, 0.16, 0.12, 0.06, 0.14, 0.28, 0.20, 0.18], dtype=float),
            np.array([18, 8, 7, 10, 8, 6], dtype=float),
        ]
    )
    best: Optional[Trajectory] = None

    for _ in range(max(1, teacher_iters)):
        population: List[Tuple[float, np.ndarray, Trajectory]] = []
        for _candidate in range(64):
            vector = rng.normal(mean, std)
            program = _decode_loop_free(vector)
            clipped = _loop_free_vector(program)
            trajectory = simulate_loop_free(program, task, {"loop_free_vector": clipped.tolist()})
            objective = trajectory.score
            if student_params is not None:
                action_delta = np.linalg.norm(clipped[: len(ACTION_LOW)] - _base_action_vector(student_params))
                duration_delta = np.linalg.norm((clipped[len(ACTION_LOW) :] - DEFAULT_SEGMENT_DURATIONS) / 20.0)
                objective -= float(0.45 * action_delta + 0.15 * duration_delta)
                trajectory.score = float(objective)
            population.append((objective, clipped, trajectory))
            if best is None or objective > best.score:
                best = trajectory
        population.sort(key=lambda item: item[0], reverse=True)
        elites = np.array([item[1] for item in population[:10]], dtype=float)
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), np.concatenate([np.full(len(ACTION_LOW), 0.035), np.full(len(DEFAULT_SEGMENT_DURATIONS), 2.0)]))

    if best is None:
        raise RuntimeError("teacher optimization produced no trajectory")
    return best


def _transition_values(trace: Trajectory) -> Dict[str, List[float]]:
    values = {key: [] for key in ("front_threshold", "arc_y_threshold", "repeat_back_threshold", "theta_threshold", "center_y_threshold", "done_radius")}
    segments = trace.params.get("loop_free_segments", [])
    if segments:
        for idx, segment in enumerate(segments):
            mode = str(segment.get("mode_hint", ""))
            end_obs = segment.get("end_observation", [])
            next_mode = str(segments[idx + 1].get("mode_hint", "done")) if idx + 1 < len(segments) else "done"
            if len(end_obs) < 8:
                continue
            if mode == "approach" and next_mode == "arc_back":
                values["front_threshold"].append(float(end_obs[3]))
            elif mode == "arc_back" and next_mode == "counter":
                values["arc_y_threshold"].append(float(end_obs[1]))
            elif mode == "arc_back" and next_mode == "approach":
                values["repeat_back_threshold"].append(float(end_obs[4]))
            elif mode == "counter" and next_mode == "center":
                values["theta_threshold"].append(abs(float(end_obs[2])))
                values["center_y_threshold"].append(float(end_obs[1]))
            elif mode == "center" and next_mode == "done":
                values["done_radius"].append(float(end_obs[7]))
        return values

    for idx in range(1, len(trace.modes)):
        prev_mode = trace.modes[idx - 1]
        mode = trace.modes[idx]
        obs = trace.observations[idx]
        if prev_mode == "approach" and mode == "arc_back":
            values["front_threshold"].append(obs[3])
        elif prev_mode == "arc_back" and mode == "counter":
            values["arc_y_threshold"].append(obs[1])
        elif prev_mode == "arc_back" and mode == "approach":
            values["repeat_back_threshold"].append(obs[4])
        elif prev_mode == "counter" and mode == "center":
            values["theta_threshold"].append(abs(obs[2]))
            values["center_y_threshold"].append(obs[1])
        elif prev_mode == "center" and mode == "done":
            values["done_radius"].append(obs[7])
    return values


def _median_or_default(values: Iterable[float], default: float) -> float:
    array = np.array(list(values), dtype=float)
    return default if array.size == 0 else float(np.median(array))


def _segment_records(traces: List[Trajectory]) -> List[dict]:
    records = []
    for trace in traces:
        for segment in trace.params.get("loop_free_segments", []):
            mode_name = str(segment.get("mode_hint", ""))
            action_params = segment.get("action_params", [])
            end_obs = segment.get("end_observation", [])
            if mode_name in STUDENT_MODES and action_params and len(end_obs) >= 8:
                records.append(
                    {
                        "mode_hint": mode_name,
                        "action_params": [float(value) for value in action_params],
                        "end_observation": [float(value) for value in end_obs],
                    }
                )
    return records


def _weighted_mean_params(samples: List[Tuple[List[float], float]], default: List[float], low: Sequence[float], high: Sequence[float]) -> List[float]:
    arity = len(default)
    compatible = [(sample, weight) for sample, weight in samples if len(sample) == arity]
    if not compatible or sum(weight for _, weight in compatible) <= 1e-9:
        return list(default)
    array = np.array([sample for sample, _ in compatible], dtype=float)
    weights = np.array([weight for _, weight in compatible], dtype=float)
    mean = np.average(array, axis=0, weights=weights)
    return np.minimum(np.maximum(mean, np.array(low, dtype=float)), np.array(high, dtype=float)).astype(float).tolist()


def _action_distance(mode_name: str, action_params: List[float], params: PolicyParams) -> float:
    expected = np.array(mode_action_params(params, mode_name), dtype=float)
    actual = np.array(action_params, dtype=float)
    if expected.shape != actual.shape:
        return float("inf")
    return float(np.linalg.norm(expected - actual))


def _mode_responsibilities(records: List[dict], params: PolicyParams) -> List[Dict[str, float]]:
    responsibilities = []
    for record in records:
        scores = []
        for mode_name in STUDENT_MODES:
            if len(record["action_params"]) != len(mode_action_params(params, mode_name)):
                scores.append(-80.0)
                continue
            score = -4.0 * _action_distance(mode_name, record["action_params"], params)
            if mode_name == record["mode_hint"]:
                score += 1.5
            scores.append(score)
        scores_array = np.array(scores, dtype=float)
        scores_array -= np.max(scores_array)
        weights = np.exp(scores_array)
        weights /= weights.sum()
        responsibilities.append({mode_name: float(weight) for mode_name, weight in zip(STUDENT_MODES, weights)})
    return responsibilities


def _params_from_components(
    switch_values: Dict[str, List[float]],
    action_by_mode: Dict[str, List[Tuple[List[float], float]]],
) -> PolicyParams:
    defaults = default_policy_params()
    approach = _weighted_mean_params(action_by_mode["approach"], mode_action_params(defaults, "approach"), ACTION_LOW[ACTION_SLICES["approach"]], ACTION_HIGH[ACTION_SLICES["approach"]])
    arc_back = _weighted_mean_params(action_by_mode["arc_back"], mode_action_params(defaults, "arc_back"), ACTION_LOW[ACTION_SLICES["arc_back"]], ACTION_HIGH[ACTION_SLICES["arc_back"]])
    counter = _weighted_mean_params(action_by_mode["counter"], mode_action_params(defaults, "counter"), ACTION_LOW[ACTION_SLICES["counter"]], ACTION_HIGH[ACTION_SLICES["counter"]])
    center = _weighted_mean_params(action_by_mode["center"], mode_action_params(defaults, "center"), ACTION_LOW[ACTION_SLICES["center"]], ACTION_HIGH[ACTION_SLICES["center"]])
    return PolicyParams(
        front_threshold=clip(_median_or_default(switch_values["front_threshold"], defaults.front_threshold), 1.15, 3.20),
        arc_y_threshold=clip(_median_or_default(switch_values["arc_y_threshold"], defaults.arc_y_threshold), 0.75, 2.15),
        repeat_back_threshold=clip(_median_or_default(switch_values["repeat_back_threshold"], defaults.repeat_back_threshold), 0.25, 1.65),
        theta_threshold=clip(_median_or_default(switch_values["theta_threshold"], defaults.theta_threshold) + 0.04, 0.05, 0.42),
        center_y_threshold=clip(_median_or_default(switch_values["center_y_threshold"], defaults.center_y_threshold) + 0.05, 0.58, 1.55),
        done_radius=clip(_median_or_default(switch_values["done_radius"], defaults.done_radius) + 0.10, 0.45, 1.10),
        approach_speed=approach[0],
        approach_heading_gain=approach[1],
        approach_lane_y=approach[2],
        approach_lateral_gain=approach[3],
        arc_speed=arc_back[0],
        arc_steer=arc_back[1],
        arc_lateral=arc_back[2],
        counter_speed=counter[0],
        counter_steer=counter[1],
        counter_lateral=counter[2],
        center_speed_gain=center[0],
        center_theta_gain=center[1],
        center_goal_y_gain=center[2],
        center_lateral_gain=center[3],
    )


def distill_student(teacher_traces: List[Trajectory]) -> PolicyParams:
    collected = {key: [] for key in ("front_threshold", "arc_y_threshold", "repeat_back_threshold", "theta_threshold", "center_y_threshold", "done_radius")}
    source = [trace for trace in teacher_traces if trace.success] or teacher_traces
    for trace in source:
        for key, values in _transition_values(trace).items():
            collected[key].extend(values)
    records = _segment_records(source)
    params = default_policy_params()
    for _ in range(4):
        responsibilities = _mode_responsibilities(records, params)
        action_by_mode: Dict[str, List[Tuple[List[float], float]]] = {mode_name: [] for mode_name in STUDENT_MODES}
        for record, weights in zip(records, responsibilities):
            for mode_name, weight in weights.items():
                action_by_mode[mode_name].append((record["action_params"], weight))
        params = _params_from_components(collected, action_by_mode)
    return params


def summarize_traces(traces: List[Trajectory]) -> Dict[str, float]:
    return {
        "n": len(traces),
        "success_rate": float(np.mean([trace.success for trace in traces])) if traces else 0.0,
        "collision_rate": float(np.mean([trace.collision for trace in traces])) if traces else 0.0,
        "mean_score": float(np.mean([trace.score for trace in traces])) if traces else 0.0,
        "mean_steps": float(np.mean([len(trace.actions) for trace in traces])) if traces else 0.0,
        "mean_cycle_count": float(np.mean([trace.loop_count for trace in traces])) if traces else 0.0,
    }


def evaluate(tasks: List[ParkingTask], params: PolicyParams, reuse: ReuseResult) -> List[Trajectory]:
    policy = build_policy(params, reuse)
    return [simulate_policy(policy, params.__dict__, task) for task in tasks]


def evaluate_baseline(tasks: List[ParkingTask], reuse: ReuseResult) -> List[Trajectory]:
    policy = build_baseline_policy(reuse)
    return [simulate_policy(policy, {"name": "fixed_baseline"}, task) for task in tasks]


def compact_trace(trace: Trajectory) -> Dict[str, object]:
    segments = trace.params.get("loop_free_segments", [])
    return {
        "task_id": trace.task_id,
        "success": trace.success,
        "collision": trace.collision,
        "score": trace.score,
        "steps": len(trace.actions),
        "cycle_count": trace.loop_count,
        "final_state": trace.states[-1] if trace.states else [],
        "mode_counts": {mode: trace.modes.count(mode) for mode in sorted(set(trace.modes))},
        "segment_modes": [segment.get("mode_hint") for segment in segments],
        "segment_durations": [segment.get("duration") for segment in segments],
    }


def serialize_task(task: ParkingTask) -> Dict[str, object]:
    return {
        "task_id": task.task_id,
        "slot_length": task.slot_length,
        "car_length": task.car_length,
        "front_x": task.front_x,
        "back_x": task.back_x,
        "start": task.start.astype(float).tolist(),
        "goal": task.goal.astype(float).tolist(),
        "max_steps": task.max_steps,
    }


def serialize_trajectory(trace: Trajectory) -> Dict[str, object]:
    return {
        "task_id": trace.task_id,
        "states": [[float(value) for value in state] for state in trace.states],
        "observations": [[float(value) for value in obs] for obs in trace.observations],
        "actions": [[float(value) for value in action] for action in trace.actions],
        "modes": list(trace.modes),
        "success": trace.success,
        "collision": trace.collision,
        "score": trace.score,
        "loop_count": trace.loop_count,
        "params": trace.params,
    }


def serialize_trajectories(traces: List[Trajectory]) -> List[Dict[str, object]]:
    return [serialize_trajectory(trace) for trace in traces]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def plot_trajectories(outpath: Path, train_eval: List[Trajectory], test_eval: List[Trajectory], train_tasks: List[ParkingTask], test_tasks: List[ParkingTask]) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, title, traces, tasks in [(axes[0], "Train distribution", train_eval[:6], train_tasks[:6]), (axes[1], "Test distribution", test_eval[:6], test_tasks[:6])]:
        for trace, task in zip(traces, tasks):
            states = np.array(trace.states)
            ax.plot(states[:, 0], states[:, 1], color="tab:green" if trace.success else "tab:red", alpha=0.75, linewidth=1.8)
            ax.scatter(states[0, 0], states[0, 1], color="black", s=18)
            ax.scatter(task.goal[0], task.goal[1], color="tab:blue", s=28, marker="x")
            ax.add_patch(plt.Rectangle((task.back_x - task.car_length / 2.0, 0.20), task.car_length, 0.85, color="gray", alpha=0.20))
            ax.add_patch(plt.Rectangle((task.front_x - task.car_length / 2.0, 0.20), task.car_length, 0.85, color="gray", alpha=0.20))
        ax.axhline(0.20, color="saddlebrown", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("longitudinal position x")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("lateral distance from curb y")
    fig.suptitle("Student state-machine trajectories")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_success_rates(outpath: Path, metrics: Dict[str, object]) -> None:
    if plt is None:
        return
    labels = ["Baseline train", "Baseline test", "Teacher train", "Student train", "Student test"]
    values = [metrics[key]["success_rate"] for key in ("baseline_train", "baseline_test", "teacher_train", "student_train", "student_test")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=["#8c8c8c", "#b0b0b0", "#6b8e23", "#1f77b4", "#ff7f0e"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Generalization behavior")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.03, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    repo_root = args.repo_root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics_path = Path(getattr(args, "metrics_output", None) or outdir / "metrics.json").resolve()
    traces_path = Path(getattr(args, "traces_output", None) or outdir / "traces.json").resolve()

    manifest = scan_repo(repo_root)
    reuse = try_reuse_state_machine(repo_root, manifest)
    manifest["reuse"] = {"status": reuse.status, "reason": reuse.reason, "module_path": reuse.module_path}
    save_json(outdir / "repo_manifest.json", manifest)

    train_tasks = make_tasks(args.train_n, "train", rng)
    test_tasks = make_tasks(args.test_n, "test", rng)
    baseline_train = evaluate_baseline(train_tasks, reuse)
    baseline_test = evaluate_baseline(test_tasks, reuse)

    student_params: Optional[PolicyParams] = None
    teacher_traces: List[Trajectory] = []
    history = []
    for outer_iter in range(args.outer_iters):
        teacher_traces = [teacher_optimize_task(task, reuse, rng, args.teacher_iters, student_params) for task in train_tasks]
        student_params = distill_student(teacher_traces)
        train_eval = evaluate(train_tasks, student_params, reuse)
        history.append({"iteration": outer_iter, "teacher_train": summarize_traces(teacher_traces), "student_train": summarize_traces(train_eval), "student_params": student_params.__dict__})

    if student_params is None:
        raise RuntimeError("student training produced no policy")
    train_eval = evaluate(train_tasks, student_params, reuse)
    test_eval = evaluate(test_tasks, student_params, reuse)

    config = {
        "repo_root": str(repo_root),
        "outdir": str(outdir),
        "train_n": args.train_n,
        "test_n": args.test_n,
        "teacher_iters": args.teacher_iters,
        "outer_iters": args.outer_iters,
        "seed": args.seed,
    }
    command = " ".join(sys.argv)
    metrics = {
        "artifact_kind": "parking_psm_training_metrics",
        "command": command,
        "config": config,
        "seed": args.seed,
        "teacher_iters": args.teacher_iters,
        "outer_iters": args.outer_iters,
        "metrics_output": str(metrics_path),
        "traces_output": str(traces_path),
        "student_fit": {"method": "em_style_segment_assignment", "em_iters": 4, "modes": list(STUDENT_MODES)},
        "repo_reuse": manifest["reuse"],
        "learned_thresholds": student_params.__dict__,
        "baseline_train": summarize_traces(baseline_train),
        "baseline_test": summarize_traces(baseline_test),
        "teacher_train": summarize_traces(teacher_traces),
        "student_train": summarize_traces(train_eval),
        "student_test": summarize_traces(test_eval),
        "adaptive_history": history,
        "teacher_trace_examples": [compact_trace(trace) for trace in teacher_traces[:5]],
        "student_train_examples": [compact_trace(trace) for trace in train_eval[:5]],
        "student_test_examples": [compact_trace(trace) for trace in test_eval[:5]],
    }
    trace_payload = {
        "artifact_kind": "parking_psm_training_traces",
        "command": command,
        "config": config,
        "train_tasks": [serialize_task(task) for task in train_tasks],
        "test_tasks": [serialize_task(task) for task in test_tasks],
        "teacher_traces": serialize_trajectories(teacher_traces),
        "student_train_traces": serialize_trajectories(train_eval),
        "student_test_traces": serialize_trajectories(test_eval),
    }
    save_json(metrics_path, metrics)
    save_json(traces_path, trace_payload)
    plot_trajectories(outdir / "trajectories.png", train_eval, test_eval, train_tasks, test_tasks)
    plot_success_rates(outdir / "success_rates.png", metrics)
    return metrics


def verify_metrics(metrics: Dict[str, object]) -> None:
    required = {
        "baseline_test": "baseline evaluation",
        "teacher_train": "loop-free teacher evaluation",
        "student_train": "student train evaluation",
        "student_test": "student test evaluation",
        "learned_thresholds": "learned state-machine parameters",
        "teacher_trace_examples": "teacher trace examples",
        "student_fit": "student fit metadata",
    }
    missing = [name for name in required if name not in metrics]
    if missing:
        raise AssertionError(f"missing metrics: {', '.join(missing)}")

    learned = metrics["learned_thresholds"]
    for name in default_policy_params().__dict__:
        if name not in learned:
            raise AssertionError(f"learned policy is missing parameter {name}")

    examples = metrics["teacher_trace_examples"]
    if not examples:
        raise AssertionError("teacher did not produce trace examples")
    if not any(example.get("segment_modes") and example.get("segment_durations") for example in examples):
        raise AssertionError("teacher examples do not expose loop-free segments")
    if metrics["teacher_train"]["success_rate"] <= 0.0:
        raise AssertionError("loop-free teacher never solved the training tasks")
    if metrics["student_train"]["success_rate"] <= 0.0:
        raise AssertionError("distilled student never solved the training tasks")
    if metrics["student_test"]["success_rate"] <= metrics["baseline_test"]["success_rate"]:
        raise AssertionError("student did not improve over the fixed baseline on the test distribution")
    if metrics["student_fit"].get("method") != "em_style_segment_assignment":
        raise AssertionError("student fit did not use the EM-style segment assignment path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repo-aware SPPIG-inspired simulation.")
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--outdir", default=Path("artifacts/programmatic_policy"), type=Path)
    parser.add_argument("--train-n", default=24, type=int)
    parser.add_argument("--test-n", default=24, type=int)
    parser.add_argument("--teacher-iters", default=3, type=int)
    parser.add_argument("--outer-iters", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--traces-output", default=None, type=Path)
    parser.add_argument("--verify", action="store_true", help="fail if the synthesized policy does not satisfy basic paper-level checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_experiment(args)
    if args.verify:
        verify_metrics(metrics)
    reuse = metrics["repo_reuse"]
    print(f"repo reuse: {reuse['status']} ({reuse['module_path'] or reuse['reason']})")
    print(
        "metrics: "
        f"baseline_test={metrics['baseline_test']['success_rate']:.2f}, "
        f"teacher_train={metrics['teacher_train']['success_rate']:.2f}, "
        f"student_train={metrics['student_train']['success_rate']:.2f}, "
        f"student_test={metrics['student_test']['success_rate']:.2f}"
    )
    print(f"metrics written to {metrics['metrics_output']}")
    print(f"traces written to {metrics['traces_output']}")
    print(f"plots written to {args.outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
