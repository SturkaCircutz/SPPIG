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
    batch_size: int = 10
    batch_refinement_rounds: int = 1
    local_refinement_steps: int = 2
    restart_candidates_on_stall: int = 1
    local_step_fraction: float = 0.25
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
    source: str = "grid"


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
    search_diagnostics: Dict[str, object]


def cartpole_direct_opt_algorithm_provenance() -> Dict[str, object]:
    return {
        "baseline": "direct_opt",
        "paper_baseline": "Direct-Opt",
        "not_paper_scale": True,
        "search_method": "deterministic_grid_seeded_random_search_plus_bounded_batch_restart_refinement",
        "policy_class": "two_mode_constant_action_depth2_linear_switch",
        "selection_objective": "mean_train_horizon_reward_then_success",
        "batch_refinement": "seed_each_batch_from_best_so_far_and_restart_on_stall",
        "paper_batch_size": 10,
        "paper_parallel_threads": 10,
        "paper_time_limit_seconds": 7200,
        "local_parallel_threads": 1,
        "local_time_limit_seconds": None,
        "train_horizon_seconds": CartpoleEnv.train_env().cfg.horizon_seconds,
        "test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "theta_weight_grid": list(DIRECT_OPT_THETA_WEIGHTS),
        "omega_weight_grid": list(DIRECT_OPT_OMEGA_WEIGHTS),
        "force_values": list(DIRECT_OPT_FORCE_VALUES),
        "threshold_scale": DIRECT_OPT_THRESHOLD_SCALE,
        "limitations": (
            "Diagnostic direct optimization over a bounded two-mode CartPole PSM. "
            "It includes a bounded batch/restart local refinement, but is not the "
            "paper's two-hour, ten-thread numerical optimization over the full "
            "continuous one-hot switching grammar."
        ),
    }


def run_cartpole_direct_opt(cfg: DirectOptConfig) -> DirectOptResult:
    rng = random.Random(cfg.seed)
    train_env = CartpoleEnv.train_env(seed=cfg.seed)
    train_states = [train_env.reset() for _ in range(max(1, cfg.num_train_states))]
    candidates, search_diagnostics = _direct_opt_candidates(train_states, cfg, rng)
    best = max(
        candidates,
        key=_candidate_rank_key,
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
        searched_candidates=int(search_diagnostics["evaluated_candidates"]),
        config=cfg,
        algorithm_provenance=cartpole_direct_opt_algorithm_provenance(),
        search_diagnostics=search_diagnostics,
    )


def direct_opt_metrics(result: DirectOptResult) -> Dict[str, object]:
    return {
        "config": asdict(result.config),
        "algorithm_provenance": result.algorithm_provenance,
        "search_diagnostics": result.search_diagnostics,
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
) -> Tuple[List[DirectOptCandidate], Dict[str, object]]:
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
                    "grid",
                )
            )
    for _ in range(max(0, cfg.random_candidates)):
        candidates.append(
            _evaluate_candidate(*_random_candidate_params(rng), train_states, "random_restart")
        )

    best = max(candidates, key=_candidate_rank_key)
    batch_candidates, batch_diagnostics = _batch_restart_refinement_candidates(
        best,
        train_states,
        cfg,
        rng,
    )
    candidates.extend(batch_candidates)
    diagnostics = {
        "grid_candidates": len(DIRECT_OPT_THETA_WEIGHTS) * len(DIRECT_OPT_OMEGA_WEIGHTS),
        "random_candidates": max(0, cfg.random_candidates),
        "batch_refinement_candidates": len(batch_candidates),
        "evaluated_candidates": (
            len(candidates)
            + int(batch_diagnostics["batch_seed_evaluations"])
            + int(batch_diagnostics["batch_local_evaluations"])
            + int(batch_diagnostics["restart_evaluations"])
        ),
        **batch_diagnostics,
    }
    return candidates, diagnostics


def _batch_restart_refinement_candidates(
    seed_candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    rng: random.Random,
) -> Tuple[List[DirectOptCandidate], Dict[str, int]]:
    batches = _direct_opt_batches(train_states, max(1, cfg.batch_size))
    rounds = max(0, cfg.batch_refinement_rounds)
    if rounds == 0 or not batches:
        return [], {
            "batch_count": len(batches),
            "batch_rounds": rounds,
            "batch_seed_evaluations": 0,
            "batch_local_evaluations": 0,
            "restart_evaluations": 0,
            "accepted_batch_improvements": 0,
            "accepted_restarts": 0,
        }

    candidates: List[DirectOptCandidate] = []
    current = seed_candidate
    local_evaluations = 0
    restart_evaluations = 0
    batch_seed_evaluations = 0
    accepted_improvements = 0
    accepted_restarts = 0
    for _ in range(rounds):
        for batch in batches:
            batch_seed_evaluations += 1
            batch_best = _evaluate_candidate(
                current.theta_weight,
                current.omega_weight,
                current.threshold,
                current.left_force,
                current.right_force,
                batch,
                "batch_seed",
            )
            for _ in range(max(0, cfg.local_refinement_steps)):
                neighbors = [
                    _evaluate_candidate(*params, batch, "batch_local_refinement")
                    for params in _local_neighbor_params(batch_best, cfg)
                ]
                local_evaluations += len(neighbors)
                local_best = max(neighbors, key=_candidate_rank_key) if neighbors else batch_best
                if _candidate_rank_key(local_best) > _candidate_rank_key(batch_best):
                    batch_best = local_best
                    accepted_improvements += 1
                    continue
                restarts = [
                    _evaluate_candidate(*_random_candidate_params(rng), batch, "batch_random_restart")
                    for _ in range(max(0, cfg.restart_candidates_on_stall))
                ]
                restart_evaluations += len(restarts)
                restart_best = max(restarts, key=_candidate_rank_key) if restarts else batch_best
                if _candidate_rank_key(restart_best) > _candidate_rank_key(batch_best):
                    batch_best = restart_best
                    accepted_restarts += 1
                    continue
                break
            full_candidate = _evaluate_candidate(
                batch_best.theta_weight,
                batch_best.omega_weight,
                batch_best.threshold,
                batch_best.left_force,
                batch_best.right_force,
                train_states,
                "batch_refinement",
            )
            candidates.append(full_candidate)
            if _candidate_rank_key(full_candidate) > _candidate_rank_key(current):
                current = full_candidate
    return candidates, {
        "batch_count": len(batches),
        "batch_rounds": rounds,
        "batch_seed_evaluations": batch_seed_evaluations,
        "batch_local_evaluations": local_evaluations,
        "restart_evaluations": restart_evaluations,
        "accepted_batch_improvements": accepted_improvements,
        "accepted_restarts": accepted_restarts,
    }


def _direct_opt_batches(
    train_states: List[Sequence[float]],
    batch_size: int,
) -> List[List[Sequence[float]]]:
    return [
        train_states[index : index + batch_size]
        for index in range(0, len(train_states), batch_size)
    ]


def _random_candidate_params(rng: random.Random) -> Tuple[float, float, float, float, float]:
    left_force = rng.choice(DIRECT_OPT_FORCE_VALUES)
    return (
        rng.choice(DIRECT_OPT_THETA_WEIGHTS),
        rng.choice(DIRECT_OPT_OMEGA_WEIGHTS),
        rng.uniform(-DIRECT_OPT_THRESHOLD_SCALE, DIRECT_OPT_THRESHOLD_SCALE),
        left_force,
        -left_force,
    )


def _local_neighbor_params(
    candidate: DirectOptCandidate,
    cfg: DirectOptConfig,
) -> List[Tuple[float, float, float, float, float]]:
    force_lower = min(DIRECT_OPT_FORCE_VALUES)
    force_upper = max(DIRECT_OPT_FORCE_VALUES)
    force_step = (force_upper - force_lower) * max(0.0, cfg.local_step_fraction)
    theta_step = max(1.0, abs(candidate.theta_weight)) * max(0.0, cfg.local_step_fraction)
    omega_step = max(0.25, abs(candidate.omega_weight)) * max(0.0, cfg.local_step_fraction)
    threshold_step = DIRECT_OPT_THRESHOLD_SCALE * max(0.0, cfg.local_step_fraction)
    params: List[Tuple[float, float, float, float, float]] = []
    for delta_theta, delta_omega, delta_threshold, delta_left, delta_right in (
        (theta_step, 0.0, 0.0, 0.0, 0.0),
        (-theta_step, 0.0, 0.0, 0.0, 0.0),
        (0.0, omega_step, 0.0, 0.0, 0.0),
        (0.0, -omega_step, 0.0, 0.0, 0.0),
        (0.0, 0.0, threshold_step, 0.0, 0.0),
        (0.0, 0.0, -threshold_step, 0.0, 0.0),
        (0.0, 0.0, 0.0, force_step, 0.0),
        (0.0, 0.0, 0.0, -force_step, 0.0),
        (0.0, 0.0, 0.0, 0.0, force_step),
        (0.0, 0.0, 0.0, 0.0, -force_step),
    ):
        params.append(
            (
                _clamp(
                    candidate.theta_weight + delta_theta,
                    min(DIRECT_OPT_THETA_WEIGHTS),
                    max(DIRECT_OPT_THETA_WEIGHTS),
                ),
                _clamp(
                    candidate.omega_weight + delta_omega,
                    min(DIRECT_OPT_OMEGA_WEIGHTS),
                    max(DIRECT_OPT_OMEGA_WEIGHTS),
                ),
                _clamp(
                    candidate.threshold + delta_threshold,
                    -DIRECT_OPT_THRESHOLD_SCALE,
                    DIRECT_OPT_THRESHOLD_SCALE,
                ),
                _clamp(candidate.left_force + delta_left, force_lower, force_upper),
                _clamp(candidate.right_force + delta_right, force_lower, force_upper),
            )
        )
    return list(dict.fromkeys(params))


def _candidate_rank_key(candidate: DirectOptCandidate) -> Tuple[float, float, float]:
    return (
        candidate.train_reward_mean,
        candidate.train_success_rate,
        -abs(candidate.threshold),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _evaluate_candidate(
    theta_weight: float,
    omega_weight: float,
    threshold: float,
    left_force: float,
    right_force: float,
    train_states: List[Sequence[float]],
    source: str = "grid",
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
        source=source,
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
