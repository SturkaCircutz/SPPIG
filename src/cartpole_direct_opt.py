from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Dict, List, Sequence, Tuple

from cartpole_env import CartpoleEnv
from cartpole_synthesis import Depth2Switch, SynthesizedCartpolePSM


DIRECT_OPT_THETA_WEIGHTS = (-50.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DIRECT_OPT_OMEGA_WEIGHTS = (-10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
DIRECT_OPT_FORCE_VALUES = (-10.0, 10.0)
DIRECT_OPT_THRESHOLD_SCALE = 0.25


@dataclass
class DirectOptConfig:
    seed: int = 0
    num_train_states: int = 10
    random_candidates: int = 256
    eval_rollouts: int = 20
    test_max_steps: int = 15_000
    quick: bool = False


@dataclass
class DirectOptCandidate:
    theta_weight: float
    omega_weight: float
    threshold: float
    left_force: float
    right_force: float
    train_reward_mean: float
    train_success_rate: float


@dataclass
class DirectOptResult:
    policy: SynthesizedCartpolePSM
    candidate: DirectOptCandidate
    train_success_rate: float
    test_success_rate: float
    train_reward_mean: float
    test_reward_mean: float
    searched_candidates: int
    config: DirectOptConfig
    algorithm_provenance: Dict[str, object]


def cartpole_direct_opt_algorithm_provenance() -> Dict[str, object]:
    return {
        "baseline": "direct_opt",
        "paper_baseline": "Direct-Opt",
        "not_paper_scale": True,
        "search_method": "deterministic_grid_plus_seeded_random_threshold_search",
        "policy_class": "two_mode_constant_action_depth2_linear_switch",
        "selection_objective": "mean_train_horizon_reward_then_success",
        "train_horizon_seconds": CartpoleEnv.train_env().cfg.horizon_seconds,
        "test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "theta_weight_grid": list(DIRECT_OPT_THETA_WEIGHTS),
        "omega_weight_grid": list(DIRECT_OPT_OMEGA_WEIGHTS),
        "force_values": list(DIRECT_OPT_FORCE_VALUES),
        "threshold_scale": DIRECT_OPT_THRESHOLD_SCALE,
        "limitations": (
            "Diagnostic direct optimization over a bounded two-mode CartPole PSM. "
            "It is not the paper's two-hour parallel numerical optimization with "
            "batch restarts over all program parameters."
        ),
    }


def run_cartpole_direct_opt(cfg: DirectOptConfig) -> DirectOptResult:
    rng = random.Random(cfg.seed)
    train_env = CartpoleEnv.train_env(seed=cfg.seed)
    train_states = [train_env.reset() for _ in range(max(1, cfg.num_train_states))]
    candidates = _direct_opt_candidates(train_states, cfg, rng)
    best = max(
        candidates,
        key=lambda candidate: (
            candidate.train_reward_mean,
            candidate.train_success_rate,
            -abs(candidate.threshold),
        ),
    )
    policy = _candidate_policy(best)
    eval_train_env = CartpoleEnv.train_env(seed=100 + cfg.seed)
    eval_test_env = CartpoleEnv.test_env(seed=200 + cfg.seed)
    train_results = [eval_train_env.rollout(policy) for _ in range(cfg.eval_rollouts)]
    test_results = [
        eval_test_env.rollout(policy, max_steps=cfg.test_max_steps)
        for _ in range(cfg.eval_rollouts)
    ]
    train = _summarize_results(train_results)
    test = _summarize_results(test_results)
    return DirectOptResult(
        policy=policy,
        candidate=best,
        train_success_rate=train["success_rate"],
        test_success_rate=test["success_rate"],
        train_reward_mean=train["reward_mean"],
        test_reward_mean=test["reward_mean"],
        searched_candidates=len(candidates),
        config=cfg,
        algorithm_provenance=cartpole_direct_opt_algorithm_provenance(),
    )


def direct_opt_metrics(result: DirectOptResult) -> Dict[str, object]:
    return {
        "config": asdict(result.config),
        "algorithm_provenance": result.algorithm_provenance,
        "policy_description": result.policy.describe(),
        "best_candidate": asdict(result.candidate),
        "searched_candidates": result.searched_candidates,
        "eval_rollouts": result.config.eval_rollouts,
        "test_max_steps": result.config.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "train": {
            "success_rate": result.train_success_rate,
            "reward_mean": result.train_reward_mean,
        },
        "test": {
            "success_rate": result.test_success_rate,
            "reward_mean": result.test_reward_mean,
        },
    }


def _direct_opt_candidates(
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    rng: random.Random,
) -> List[DirectOptCandidate]:
    candidates: List[DirectOptCandidate] = []
    for theta_weight in DIRECT_OPT_THETA_WEIGHTS:
        for omega_weight in DIRECT_OPT_OMEGA_WEIGHTS:
            candidates.append(
                _evaluate_candidate(
                    theta_weight,
                    omega_weight,
                    0.0,
                    min(DIRECT_OPT_FORCE_VALUES),
                    max(DIRECT_OPT_FORCE_VALUES),
                    train_states,
                )
            )
    for _ in range(max(0, cfg.random_candidates)):
        theta_weight = rng.choice(DIRECT_OPT_THETA_WEIGHTS)
        omega_weight = rng.choice(DIRECT_OPT_OMEGA_WEIGHTS)
        threshold = rng.uniform(-DIRECT_OPT_THRESHOLD_SCALE, DIRECT_OPT_THRESHOLD_SCALE)
        left_force = rng.choice(DIRECT_OPT_FORCE_VALUES)
        right_force = -left_force
        candidates.append(
            _evaluate_candidate(
                theta_weight,
                omega_weight,
                threshold,
                left_force,
                right_force,
                train_states,
            )
        )
    return candidates


def _evaluate_candidate(
    theta_weight: float,
    omega_weight: float,
    threshold: float,
    left_force: float,
    right_force: float,
    train_states: List[Sequence[float]],
) -> DirectOptCandidate:
    policy = SynthesizedCartpolePSM(
        left_force,
        right_force,
        Depth2Switch(theta_weight, omega_weight, threshold),
    )
    train_env = CartpoleEnv.train_env(seed=0)
    results = [train_env.rollout(policy, initial_state=state) for state in train_states]
    summary = _summarize_results(results)
    return DirectOptCandidate(
        theta_weight=theta_weight,
        omega_weight=omega_weight,
        threshold=threshold,
        left_force=left_force,
        right_force=right_force,
        train_reward_mean=summary["reward_mean"],
        train_success_rate=summary["success_rate"],
    )


def _candidate_policy(candidate: DirectOptCandidate) -> SynthesizedCartpolePSM:
    return SynthesizedCartpolePSM(
        candidate.left_force,
        candidate.right_force,
        Depth2Switch(candidate.theta_weight, candidate.omega_weight, candidate.threshold),
    )


def _summarize_results(results) -> Dict[str, float]:
    result_list = list(results)
    return {
        "success_rate": sum(result.success for result in result_list) / len(result_list),
        "reward_mean": sum(result.reward for result in result_list) / len(result_list),
    }
