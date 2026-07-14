"""Continuous-control parking benchmark for programmatic policies.

The benchmark follows the tight-parking control interface used by Inala et al.:
each action is longitudinal velocity plus steering angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


Observation = List[float]
Action = List[float]
START_GOAL_X_DISTANCE = 10.0


@dataclass
class ParkingTask:
    task_id: int
    slot_length: float
    car_length: float
    front_x: float
    back_x: float
    start: np.ndarray
    goal: np.ndarray
    start_goal_x_distance: float = START_GOAL_X_DISTANCE
    max_steps: int = 260


@dataclass
class Trajectory:
    task_id: int
    states: List[List[float]]
    observations: List[Observation]
    actions: List[Action]
    modes: List[str]
    success: bool
    collision: bool
    score: float
    loop_count: int
    params: dict


def make_tasks(n: int, split: str, rng: np.random.Generator) -> List[ParkingTask]:
    tasks: List[ParkingTask] = []
    for task_id in range(n):
        if split == "train":
            slot_length = float(rng.uniform(6.7, 7.7))
            start_y = float(rng.uniform(2.75, 3.25))
            heading_noise = float(rng.normal(0.0, 0.03))
        else:
            slot_length = float(rng.uniform(6.0, 8.8))
            start_y = float(rng.uniform(2.35, 3.65))
            heading_noise = float(rng.normal(0.0, 0.07))
        car_length = 4.5
        front_x = slot_length / 2.0 + car_length / 2.0
        back_x = -slot_length / 2.0 - car_length / 2.0
        goal = np.array([0.0, 0.58, 0.0], dtype=float)
        start = np.array([goal[0] - START_GOAL_X_DISTANCE, start_y, heading_noise], dtype=float)
        tasks.append(
            ParkingTask(
                task_id=task_id,
                slot_length=slot_length,
                car_length=car_length,
                front_x=front_x,
                back_x=back_x,
                start=start,
                goal=goal,
                start_goal_x_distance=START_GOAL_X_DISTANCE,
            )
        )
    return tasks


def observe(state: np.ndarray, task: ParkingTask) -> Observation:
    x, y, theta = state
    half = task.car_length / 2.0
    front_gap = task.front_x - x - half
    back_gap = x - task.back_x - half
    goal_dx = task.goal[0] - x
    goal_dy = task.goal[1] - y
    return [
        float(x),
        float(y),
        float(theta),
        float(front_gap),
        float(back_gap),
        float(goal_dx),
        float(goal_dy),
        float(math.hypot(goal_dx, goal_dy)),
    ]


def step_dynamics(state: np.ndarray, action: Sequence[float], dt: float = 0.10) -> np.ndarray:
    """Low-speed bicycle dynamics controlled by velocity and steering."""

    if len(action) != 2:
        raise ValueError(f"parking action must be [velocity, steering], got {len(action)} values")
    speed, steer = float(action[0]), float(action[1])
    wheelbase = 2.70
    x, y, theta = state
    theta = theta + (speed / wheelbase) * math.tan(steer) * dt
    theta = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    x = x + speed * math.cos(theta) * dt
    y = y + speed * math.sin(theta) * dt
    return np.array([x, y, theta], dtype=float)


def collision_or_bounds(state: np.ndarray, task: ParkingTask) -> bool:
    x, y, theta = state
    half = task.car_length / 2.0
    front_gap = task.front_x - x - half
    back_gap = x - task.back_x - half
    return bool(
        (front_gap < 0.12 and y < 1.35)
        or (back_gap < 0.12 and y < 1.35)
        or y < 0.20
        or y > 4.25
        or abs(theta) > 1.35
    )


def is_success(state: np.ndarray, mode: str, task: ParkingTask) -> bool:
    return bool(np.linalg.norm(state[:2] - task.goal[:2]) < 1.00 and abs(state[2]) < 0.30 and mode == "done")
