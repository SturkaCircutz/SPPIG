from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, List, Optional, Protocol, Sequence


Observation = List[float]


class ContinuousPolicy(Protocol):
    def reset(self) -> None:
        ...

    def act(self, observation: Observation) -> float:
        ...


@dataclass
class CartpoleConfig:
    pole_length: float
    horizon_seconds: float
    dt: float = 0.02
    force_limit: float = 10.0
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    theta_limit_radians: float = 12.0 * math.pi / 180.0
    x_limit: float = 2.4

    @property
    def max_steps(self) -> int:
        return int(self.horizon_seconds / self.dt)


@dataclass
class CartpoleResult:
    success: bool
    steps: int
    reward: float
    max_abs_theta: float
    max_abs_x: float


class CartpoleEnv:
    """Continuous-action Cartpole matching the paper's benchmark interface.

    Paper audit:
    - #A = 1, continuous force.
    - #O = 4: x, cart velocity, pole angle, angular velocity.
    - train: time = 5s, pole length = 0.5.
    - test: time = 300s, pole length = 1.0.
    """

    def __init__(self, cfg: CartpoleConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.state = [0.0, 0.0, 0.0, 0.0]

    @classmethod
    def train_env(cls, seed: int = 0) -> "CartpoleEnv":
        return cls(CartpoleConfig(pole_length=0.5, horizon_seconds=5.0), seed=seed)

    @classmethod
    def test_env(cls, seed: int = 0) -> "CartpoleEnv":
        return cls(CartpoleConfig(pole_length=1.0, horizon_seconds=300.0), seed=seed)

    def reset(self, state: Optional[Sequence[float]] = None) -> Observation:
        if state is None:
            self.state = [
                self.rng.uniform(-0.05, 0.05),
                self.rng.uniform(-0.05, 0.05),
                self.rng.uniform(-0.05, 0.05),
                self.rng.uniform(-0.05, 0.05),
            ]
        else:
            self.state = [float(value) for value in state]
        return self.observe()

    def observe(self) -> Observation:
        return list(self.state)

    def step(self, force: float) -> tuple[Observation, float, bool]:
        self.state = cartpole_next_state(self.state, force, self.cfg)
        done = cartpole_done(self.state, self.cfg)
        return self.observe(), 1.0, done

    def rollout(
        self,
        policy: ContinuousPolicy,
        initial_state: Optional[Sequence[float]] = None,
        max_steps: Optional[int] = None,
    ) -> CartpoleResult:
        obs = self.reset(initial_state)
        policy.reset()
        total_reward = 0.0
        max_abs_theta = abs(obs[2])
        max_abs_x = abs(obs[0])
        steps = max_steps if max_steps is not None else self.cfg.max_steps

        for step in range(1, steps + 1):
            obs, reward, done = self.step(policy.act(obs))
            total_reward += reward
            max_abs_theta = max(max_abs_theta, abs(obs[2]))
            max_abs_x = max(max_abs_x, abs(obs[0]))
            if done:
                return CartpoleResult(False, step, total_reward, max_abs_theta, max_abs_x)
        return CartpoleResult(True, steps, total_reward, max_abs_theta, max_abs_x)


def cartpole_next_state(
    state: Sequence[float],
    force: float,
    cfg: CartpoleConfig,
    dt: Optional[float] = None,
) -> Observation:
    x, x_dot, theta, theta_dot = state
    force = max(-cfg.force_limit, min(cfg.force_limit, float(force)))
    step_dt = cfg.dt if dt is None else float(dt)
    total_mass = cfg.cart_mass + cfg.pole_mass
    polemass_length = cfg.pole_mass * cfg.pole_length

    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    theta_acc = (
        cfg.gravity * sintheta - costheta * temp
    ) / (
        cfg.pole_length
        * (4.0 / 3.0 - cfg.pole_mass * costheta * costheta / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass

    return [
        x + step_dt * x_dot,
        x_dot + step_dt * x_acc,
        theta + step_dt * theta_dot,
        theta_dot + step_dt * theta_acc,
    ]


def cartpole_done(state: Sequence[float], cfg: CartpoleConfig) -> bool:
    x, _, theta, _ = state
    return abs(x) > cfg.x_limit or abs(theta) > cfg.theta_limit_radians


class BangBangCartpolePSM:
    """Two-mode constant-action policy in the paper's Cartpole grammar class."""

    def __init__(self, force: float = 10.0) -> None:
        self.force = float(force)
        self.mode = "push_right"

    def reset(self) -> None:
        self.mode = "push_right"

    def act(self, observation: Observation) -> float:
        _, _, theta, theta_dot = observation
        # Depth-2 Boolean switch over observations: angle and angular velocity
        # determine which constant-force mode is active.
        if theta + 0.25 * theta_dot >= 0.0:
            self.mode = "push_right"
        else:
            self.mode = "push_left"
        return self.force if self.mode == "push_right" else -self.force

    def describe(self) -> str:
        return (
            "m0 action=-force, m1 action=+force; "
            "switch by theta + 0.25 * theta_dot >= 0"
        )


def evaluate_cartpole_policy(
    policy: ContinuousPolicy,
    train_rollouts: int = 20,
    test_rollouts: int = 20,
    seed: int = 0,
    test_max_steps: Optional[int] = None,
) -> dict[str, float]:
    train_env = CartpoleEnv.train_env(seed=seed)
    test_env = CartpoleEnv.test_env(seed=seed + 1)
    train_successes = 0
    test_successes = 0
    for _ in range(train_rollouts):
        train_successes += int(train_env.rollout(policy).success)
    for _ in range(test_rollouts):
        result = test_env.rollout(policy, max_steps=test_max_steps)
        test_successes += int(result.success)
    return {
        "train_success_rate": train_successes / train_rollouts,
        "test_success_rate": test_successes / test_rollouts,
    }


def sample_cartpole_initial_states(num_states: int, seed: int) -> List[Observation]:
    env = CartpoleEnv.train_env(seed=seed)
    return [env.reset() for _ in range(num_states)]
