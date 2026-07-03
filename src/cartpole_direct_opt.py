from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Dict, List, Sequence, Tuple

from cartpole_env import (
    PAPER_EVAL_ROLLOUTS,
    CartpoleEnv,
    cartpole_reward_spec,
    cartpole_space_spec,
    summarize_cartpole_results,
)
from cartpole_synthesis import (
    BooleanTreeSwitch,
    Depth2Switch,
    ObservationPredicate,
    SynthesizedCartpolePSM,
)


DIRECT_OPT_THETA_WEIGHTS = (-50.0, -20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DIRECT_OPT_OMEGA_WEIGHTS = (-10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
DIRECT_OPT_FORCE_VALUES = (-10.0, 10.0)
DIRECT_OPT_THRESHOLD_SCALE = 0.25
DIRECT_OPT_BOOLEAN_THRESHOLD_GRIDS = (
    (-0.5, 0.0, 0.5),
    (-0.5, 0.0, 0.5),
    (-0.05, 0.0, 0.05),
    (-0.5, 0.0, 0.5),
)
DIRECT_OPT_BOOLEAN_TOP_STUMPS = 4
DIRECT_OPT_OBSERVATION_FEATURES = ("x", "cart_velocity", "theta", "omega")
DIRECT_OPT_RELATIONS = (">=", "<=")
DIRECT_OPT_TREE_OPERATORS = ("leaf", "and", "or")
PAPER_DIRECT_OPT_BATCH_SIZE = 10
PAPER_DIRECT_OPT_PARALLEL_THREADS = 10
PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS = 7200


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
    eval_rollouts: int = PAPER_EVAL_ROLLOUTS
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
    switch_kind: str = "linear"
    first_feature: int | None = None
    first_relation: str | None = None
    first_threshold: float | None = None
    second_feature: int | None = None
    second_relation: str | None = None
    second_threshold: float | None = None
    operator: str | None = None
    first_feature_one_hot: Tuple[int, ...] = ()
    first_relation_one_hot: Tuple[int, ...] = ()
    second_feature_one_hot: Tuple[int, ...] = ()
    second_relation_one_hot: Tuple[int, ...] = ()
    operator_one_hot: Tuple[int, ...] = ()
    first_appendix_b3_alpha_s: float | None = None
    first_appendix_b3_feature_weights: Tuple[float, ...] = ()
    first_appendix_b3_alpha_0: float | None = None
    second_appendix_b3_alpha_s: float | None = None
    second_appendix_b3_feature_weights: Tuple[float, ...] = ()
    second_appendix_b3_alpha_0: float | None = None


@dataclass
class DirectOptResult:
    policy: SynthesizedCartpolePSM
    candidate: DirectOptCandidate
    train_success_rate: float
    test_success_rate: float
    train_reward_mean: float
    test_reward_mean: float
    train_steps_mean: float
    test_steps_mean: float
    train_survival_seconds_mean: float
    test_survival_seconds_mean: float
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
        "policy_class": "two_mode_constant_action_linear_or_depth2_boolean_tree_switch",
        "selection_objective": "mean_combined_reward_over_selected_initial_states_then_success",
        "batch_refinement": "seed_each_batch_from_best_so_far_and_restart_on_stall",
        "paper_batch_size": PAPER_DIRECT_OPT_BATCH_SIZE,
        "paper_parallel_threads": PAPER_DIRECT_OPT_PARALLEL_THREADS,
        "paper_time_limit_seconds": PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS,
        "local_parallel_threads": 1,
        "local_time_limit_seconds": None,
        "switch_search_space": "linear_theta_omega_grid_plus_bounded_boolean_tree_predicates_with_one_hot_metadata",
        "candidate_accounting": (
            "searched_candidates and evaluated_candidates count candidate evaluation calls; "
            "train_rollout_evaluations counts individual selected-state train rollouts"
        ),
        "boolean_tree_depth": 2,
        "boolean_tree_features": list(DIRECT_OPT_OBSERVATION_FEATURES),
        "boolean_tree_relations": list(DIRECT_OPT_RELATIONS),
        "boolean_tree_operator_choices": list(DIRECT_OPT_TREE_OPERATORS),
        "boolean_tree_threshold_grids": [list(grid) for grid in DIRECT_OPT_BOOLEAN_THRESHOLD_GRIDS],
        "boolean_tree_expansion": "evaluate_all_stumps_then_depth2_expansions_from_top_training_reward_stumps",
        "one_hot_switch_encoding": (
            "records Appendix B.3 continuous one-hot vertex fields plus bounded discrete metadata for "
            "feature, relation, and depth-2 tree operator choices; "
            "does not optimize the paper's continuous one-hot relaxation"
        ),
        "local_refinement": "linear_weight_threshold_force_neighbors_or_boolean_threshold_force_neighbors",
        "train_horizon_seconds": CartpoleEnv.train_env().cfg.horizon_seconds,
        "test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "theta_weight_grid": list(DIRECT_OPT_THETA_WEIGHTS),
        "omega_weight_grid": list(DIRECT_OPT_OMEGA_WEIGHTS),
        "force_values": list(DIRECT_OPT_FORCE_VALUES),
        "threshold_scale": DIRECT_OPT_THRESHOLD_SCALE,
        "limitations": (
            "Diagnostic direct optimization over a bounded two-mode CartPole PSM. "
            "It optimizes mean train-horizon reward over the selected initial states and includes "
            "bounded Boolean-tree switch candidates with one-hot metadata and batch/restart local "
            "refinement, but is not the paper's two-hour, ten-thread continuous numerical "
            "optimization over the full one-hot switching grammar."
        ),
    }


def cartpole_direct_opt_protocol_status(cfg: DirectOptConfig) -> Dict[str, object]:
    paper_test_env = CartpoleEnv.test_env()
    configured_paper_batch_size = cfg.batch_size == PAPER_DIRECT_OPT_BATCH_SIZE
    uses_paper_batch_size = configured_paper_batch_size and cfg.num_train_states >= PAPER_DIRECT_OPT_BATCH_SIZE
    paper_test_horizon = cfg.test_max_steps == paper_test_env.cfg.max_steps
    paper_eval_rollouts = cfg.eval_rollouts == PAPER_EVAL_ROLLOUTS
    return {
        "paper_baseline": "Direct-Opt",
        "paper_batch_size": PAPER_DIRECT_OPT_BATCH_SIZE,
        "selected_batch_size": cfg.batch_size,
        "configured_paper_batch_size": configured_paper_batch_size,
        "uses_paper_batch_size": uses_paper_batch_size,
        "paper_parallel_threads": PAPER_DIRECT_OPT_PARALLEL_THREADS,
        "selected_parallel_threads": 1,
        "uses_paper_parallel_threads": False,
        "paper_time_limit_seconds": PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS,
        "selected_time_limit_seconds": None,
        "uses_paper_time_limit": False,
        "full_continuous_one_hot_switch_grammar": False,
        "bounded_one_hot_switch_metadata": True,
        "appendix_b3_one_hot_vertex_metadata": True,
        "linear_switch_encoding": True,
        "batch_optimization_seeded_from_best_so_far": cfg.batch_refinement_rounds > 0,
        "random_restart_on_stall": cfg.restart_candidates_on_stall > 0,
        "optimizes_combined_reward_over_selected_initial_states": True,
        "combined_reward_aggregation": "mean_train_horizon_reward_over_selected_initial_states",
        "optimizes_combined_reward_over_all_selected_initial_states": True,
        "optimizes_full_initial_state_distribution": False,
        "selected_train_initial_states": cfg.num_train_states,
        "paper_test_horizon_steps": paper_test_env.cfg.max_steps,
        "selected_test_max_steps": cfg.test_max_steps,
        "uses_full_test_horizon": paper_test_horizon,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": cfg.eval_rollouts,
        "uses_paper_eval_rollouts": paper_eval_rollouts,
        "quick_diagnostic": cfg.quick,
        "paper_scale_direct_opt_protocol": False,
        "limitation": (
            "Bounded local Direct-Opt diagnostic: records batch/restart structure and one-hot "
            "metadata, but does not run the paper's ten-thread, two-hour continuous optimization "
            "over the full one-hot switching grammar."
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
        train_steps_mean=train["steps_mean"],
        test_steps_mean=test["steps_mean"],
        train_survival_seconds_mean=train["survival_seconds_mean"],
        test_survival_seconds_mean=test["survival_seconds_mean"],
        searched_candidates=int(search_diagnostics["evaluated_candidates"]),
        config=cfg,
        algorithm_provenance=cartpole_direct_opt_algorithm_provenance(),
        search_diagnostics=search_diagnostics,
    )


def direct_opt_metrics(result: DirectOptResult) -> Dict[str, object]:
    return {
        "config": asdict(result.config),
        "algorithm_provenance": result.algorithm_provenance,
        "paper_protocol_status": cartpole_direct_opt_protocol_status(result.config),
        "search_diagnostics": result.search_diagnostics,
        "policy_description": result.policy.describe(),
        "best_candidate": asdict(result.candidate),
        "searched_candidates": result.searched_candidates,
        "eval_rollouts": result.config.eval_rollouts,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "uses_paper_eval_rollouts": result.config.eval_rollouts == PAPER_EVAL_ROLLOUTS,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
        "test_max_steps": result.config.test_max_steps,
        "paper_test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "train": {
            "success_rate": result.train_success_rate,
            "reward_mean": result.train_reward_mean,
            "steps_mean": result.train_steps_mean,
            "survival_seconds_mean": result.train_survival_seconds_mean,
        },
        "test": {
            "success_rate": result.test_success_rate,
            "reward_mean": result.test_reward_mean,
            "steps_mean": result.test_steps_mean,
            "survival_seconds_mean": result.test_survival_seconds_mean,
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
    boolean_candidates, boolean_diagnostics = _boolean_tree_candidates(train_states)
    candidates.extend(boolean_candidates)

    best = max(candidates, key=_candidate_rank_key)
    batch_candidates, batch_diagnostics = _batch_restart_refinement_candidates(
        best,
        train_states,
        cfg,
        rng,
    )
    candidates.extend(batch_candidates)
    evaluated_candidate_calls = (
        len(candidates)
        + int(batch_diagnostics["batch_seed_evaluations"])
        + int(batch_diagnostics["batch_local_evaluations"])
        + int(batch_diagnostics["restart_evaluations"])
    )
    full_train_rollout_evaluations = len(candidates) * len(train_states)
    batch_rollout_evaluations = (
        int(batch_diagnostics["batch_seed_rollout_evaluations"])
        + int(batch_diagnostics["batch_local_rollout_evaluations"])
        + int(batch_diagnostics["restart_rollout_evaluations"])
    )
    diagnostics = {
        "grid_candidates": len(DIRECT_OPT_THETA_WEIGHTS) * len(DIRECT_OPT_OMEGA_WEIGHTS),
        "random_candidates": max(0, cfg.random_candidates),
        **boolean_diagnostics,
        "batch_refinement_candidates": len(batch_candidates),
        "candidate_evaluation_calls": evaluated_candidate_calls,
        "evaluated_candidates": evaluated_candidate_calls,
        "evaluated_candidates_units": "candidate_evaluation_calls",
        "full_train_candidate_evaluation_calls": len(candidates),
        "batch_candidate_evaluation_calls": (
            int(batch_diagnostics["batch_seed_evaluations"])
            + int(batch_diagnostics["batch_local_evaluations"])
            + int(batch_diagnostics["restart_evaluations"])
        ),
        "train_rollout_evaluations": full_train_rollout_evaluations + batch_rollout_evaluations,
        "full_train_rollout_evaluations": full_train_rollout_evaluations,
        "batch_rollout_evaluations": batch_rollout_evaluations,
        **batch_diagnostics,
    }
    return candidates, diagnostics


def _boolean_tree_candidates(
    train_states: List[Sequence[float]],
) -> Tuple[List[DirectOptCandidate], Dict[str, int]]:
    stumps = [BooleanTreeSwitch(predicate) for predicate in _direct_opt_predicates()]
    stump_candidates = [
        _evaluate_boolean_candidate(
            stump,
            min(DIRECT_OPT_FORCE_VALUES),
            max(DIRECT_OPT_FORCE_VALUES),
            train_states,
            "boolean_stump",
        )
        for stump in stumps
    ]
    top_stumps = sorted(stump_candidates, key=_candidate_rank_key, reverse=True)[:DIRECT_OPT_BOOLEAN_TOP_STUMPS]
    depth2_switches: List[BooleanTreeSwitch] = []
    for candidate in top_stumps:
        switch = _candidate_switch(candidate)
        if not isinstance(switch, BooleanTreeSwitch):
            continue
        for predicate in _direct_opt_predicates():
            depth2_switches.append(BooleanTreeSwitch(switch.first, predicate, "and"))
            depth2_switches.append(BooleanTreeSwitch(switch.first, predicate, "or"))
    depth2_candidates = [
        _evaluate_boolean_candidate(
            switch,
            min(DIRECT_OPT_FORCE_VALUES),
            max(DIRECT_OPT_FORCE_VALUES),
            train_states,
            "boolean_depth2",
        )
        for switch in _unique_boolean_switches(depth2_switches)
    ]
    return stump_candidates + depth2_candidates, {
        "boolean_stump_candidates": len(stump_candidates),
        "boolean_depth2_candidates": len(depth2_candidates),
        "boolean_top_stumps_for_depth2": len(top_stumps),
        "boolean_candidates_with_one_hot_metadata": len(stump_candidates) + len(depth2_candidates),
        "boolean_candidates_with_appendix_b3_vertex_metadata": len(stump_candidates) + len(depth2_candidates),
    }


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
            "batch_seed_rollout_evaluations": 0,
            "batch_local_evaluations": 0,
            "batch_local_rollout_evaluations": 0,
            "restart_evaluations": 0,
            "restart_rollout_evaluations": 0,
            "accepted_batch_improvements": 0,
            "accepted_restarts": 0,
        }

    candidates: List[DirectOptCandidate] = []
    current = seed_candidate
    local_evaluations = 0
    restart_evaluations = 0
    restart_rollout_evaluations = 0
    batch_seed_evaluations = 0
    batch_seed_rollout_evaluations = 0
    batch_local_rollout_evaluations = 0
    accepted_improvements = 0
    accepted_restarts = 0
    for _ in range(rounds):
        for batch in batches:
            batch_seed_evaluations += 1
            batch_seed_rollout_evaluations += len(batch)
            batch_best = _reevaluate_candidate(current, batch, "batch_seed")
            for _ in range(max(0, cfg.local_refinement_steps)):
                neighbors = _local_neighbor_candidates(batch_best, batch, cfg)
                local_evaluations += len(neighbors)
                batch_local_rollout_evaluations += len(neighbors) * len(batch)
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
                restart_rollout_evaluations += len(restarts) * len(batch)
                restart_best = max(restarts, key=_candidate_rank_key) if restarts else batch_best
                if _candidate_rank_key(restart_best) > _candidate_rank_key(batch_best):
                    batch_best = restart_best
                    accepted_restarts += 1
                    continue
                break
            full_candidate = _reevaluate_candidate(batch_best, train_states, "batch_refinement")
            candidates.append(full_candidate)
            if _candidate_rank_key(full_candidate) > _candidate_rank_key(current):
                current = full_candidate
    return candidates, {
        "batch_count": len(batches),
        "batch_rounds": rounds,
        "batch_seed_evaluations": batch_seed_evaluations,
        "batch_seed_rollout_evaluations": batch_seed_rollout_evaluations,
        "batch_local_evaluations": local_evaluations,
        "batch_local_rollout_evaluations": batch_local_rollout_evaluations,
        "restart_evaluations": restart_evaluations,
        "restart_rollout_evaluations": restart_rollout_evaluations,
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


def _local_neighbor_candidates(
    candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
) -> List[DirectOptCandidate]:
    if candidate.switch_kind == "boolean_tree":
        return _boolean_local_neighbor_candidates(candidate, train_states, cfg)
    return [
        _evaluate_candidate(*params, train_states, "batch_local_refinement")
        for params in _linear_local_neighbor_params(candidate, cfg)
    ]


def _linear_local_neighbor_params(
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


def _boolean_local_neighbor_candidates(
    candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
) -> List[DirectOptCandidate]:
    switch = _candidate_switch(candidate)
    if not isinstance(switch, BooleanTreeSwitch):
        return []
    force_lower = min(DIRECT_OPT_FORCE_VALUES)
    force_upper = max(DIRECT_OPT_FORCE_VALUES)
    force_step = (force_upper - force_lower) * max(0.0, cfg.local_step_fraction)
    threshold_switches: List[BooleanTreeSwitch] = []
    for direction in (-1.0, 1.0):
        threshold_switches.append(
            _boolean_switch_with_threshold_delta(switch, 0, direction * cfg.local_step_fraction)
        )
        if switch.second is not None:
            threshold_switches.append(
                _boolean_switch_with_threshold_delta(switch, 1, direction * cfg.local_step_fraction)
            )
    params = [
        (
            local_switch,
            _clamp(candidate.left_force + delta_left, force_lower, force_upper),
            _clamp(candidate.right_force + delta_right, force_lower, force_upper),
        )
        for local_switch in [switch] + _unique_boolean_switches(threshold_switches)
        for delta_left, delta_right in (
            (0.0, 0.0),
            (force_step, 0.0),
            (-force_step, 0.0),
            (0.0, force_step),
            (0.0, -force_step),
        )
    ]
    return [
        _evaluate_boolean_candidate(
            local_switch,
            left_force,
            right_force,
            train_states,
            "batch_local_refinement",
        )
        for local_switch, left_force, right_force in _unique_boolean_neighbor_params(params)
    ]


def _candidate_rank_key(candidate: DirectOptCandidate) -> Tuple[float, float, float]:
    return (
        candidate.train_reward_mean,
        candidate.train_success_rate,
        -_candidate_threshold_magnitude(candidate),
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
    switch = Depth2Switch(theta_weight, omega_weight, threshold)
    return _evaluate_switch_candidate(
        switch,
        left_force,
        right_force,
        train_states,
        source,
        theta_weight=theta_weight,
        omega_weight=omega_weight,
        threshold=threshold,
    )


def _evaluate_boolean_candidate(
    switch: BooleanTreeSwitch,
    left_force: float,
    right_force: float,
    train_states: List[Sequence[float]],
    source: str,
) -> DirectOptCandidate:
    return _evaluate_switch_candidate(
        switch,
        left_force,
        right_force,
        train_states,
        source,
        theta_weight=0.0,
        omega_weight=0.0,
        threshold=0.0,
    )


def _evaluate_switch_candidate(
    switch: Depth2Switch | BooleanTreeSwitch,
    left_force: float,
    right_force: float,
    train_states: List[Sequence[float]],
    source: str,
    theta_weight: float,
    omega_weight: float,
    threshold: float,
) -> DirectOptCandidate:
    policy = SynthesizedCartpolePSM(
        left_force,
        right_force,
        switch,
    )
    train_env = CartpoleEnv.train_env(seed=0)
    results = [train_env.rollout(policy, initial_state=state) for state in train_states]
    summary = _summarize_results(results)
    return _candidate_from_switch(
        switch,
        left_force,
        right_force,
        summary["reward_mean"],
        summary["success_rate"],
        source,
        theta_weight=theta_weight,
        omega_weight=omega_weight,
        threshold=threshold,
    )


def _candidate_from_switch(
    switch: Depth2Switch | BooleanTreeSwitch,
    left_force: float,
    right_force: float,
    train_reward_mean: float,
    train_success_rate: float,
    source: str,
    theta_weight: float = 0.0,
    omega_weight: float = 0.0,
    threshold: float = 0.0,
) -> DirectOptCandidate:
    if isinstance(switch, BooleanTreeSwitch):
        first_alpha_s, first_feature_weights, first_alpha_0 = _appendix_b3_predicate_encoding(switch.first)
        second_alpha_s: float | None = None
        second_feature_weights: Tuple[float, ...] = ()
        second_alpha_0: float | None = None
        if switch.second is not None:
            second_alpha_s, second_feature_weights, second_alpha_0 = _appendix_b3_predicate_encoding(switch.second)
        return DirectOptCandidate(
            theta_weight=theta_weight,
            omega_weight=omega_weight,
            threshold=threshold,
            left_force=left_force,
            right_force=right_force,
            train_reward_mean=train_reward_mean,
            train_success_rate=train_success_rate,
            source=source,
            switch_kind="boolean_tree",
            first_feature=switch.first.feature_index,
            first_relation=switch.first.relation,
            first_threshold=switch.first.threshold,
            second_feature=switch.second.feature_index if switch.second is not None else None,
            second_relation=switch.second.relation if switch.second is not None else None,
            second_threshold=switch.second.threshold if switch.second is not None else None,
            operator=switch.operator if switch.second is not None else None,
            first_feature_one_hot=_one_hot(switch.first.feature_index, len(DIRECT_OPT_OBSERVATION_FEATURES)),
            first_relation_one_hot=_relation_one_hot(switch.first.relation),
            second_feature_one_hot=(
                _one_hot(switch.second.feature_index, len(DIRECT_OPT_OBSERVATION_FEATURES))
                if switch.second is not None
                else ()
            ),
            second_relation_one_hot=(
                _relation_one_hot(switch.second.relation)
                if switch.second is not None
                else ()
            ),
            operator_one_hot=_operator_one_hot(switch.operator if switch.second is not None else "leaf"),
            first_appendix_b3_alpha_s=first_alpha_s,
            first_appendix_b3_feature_weights=first_feature_weights,
            first_appendix_b3_alpha_0=first_alpha_0,
            second_appendix_b3_alpha_s=second_alpha_s,
            second_appendix_b3_feature_weights=second_feature_weights,
            second_appendix_b3_alpha_0=second_alpha_0,
        )
    return DirectOptCandidate(
        theta_weight=theta_weight,
        omega_weight=omega_weight,
        threshold=threshold,
        left_force=left_force,
        right_force=right_force,
        train_reward_mean=train_reward_mean,
        train_success_rate=train_success_rate,
        source=source,
    )


def _candidate_policy(candidate: DirectOptCandidate) -> SynthesizedCartpolePSM:
    return SynthesizedCartpolePSM(
        candidate.left_force,
        candidate.right_force,
        _candidate_switch(candidate),
    )


def _candidate_switch(candidate: DirectOptCandidate) -> Depth2Switch | BooleanTreeSwitch:
    if candidate.switch_kind == "boolean_tree":
        first = ObservationPredicate(
            int(candidate.first_feature),
            str(candidate.first_relation),
            float(candidate.first_threshold),
        )
        second = None
        if candidate.second_feature is not None:
            second = ObservationPredicate(
                int(candidate.second_feature),
                str(candidate.second_relation),
                float(candidate.second_threshold),
            )
        return BooleanTreeSwitch(first, second, candidate.operator or "and")
    return Depth2Switch(candidate.theta_weight, candidate.omega_weight, candidate.threshold)


def _reevaluate_candidate(
    candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    source: str,
) -> DirectOptCandidate:
    return _evaluate_switch_candidate(
        _candidate_switch(candidate),
        candidate.left_force,
        candidate.right_force,
        train_states,
        source,
        theta_weight=candidate.theta_weight,
        omega_weight=candidate.omega_weight,
        threshold=candidate.threshold,
    )


def _direct_opt_predicates() -> List[ObservationPredicate]:
    predicates: List[ObservationPredicate] = []
    for feature_index, thresholds in enumerate(DIRECT_OPT_BOOLEAN_THRESHOLD_GRIDS):
        for threshold in thresholds:
            predicates.append(ObservationPredicate(feature_index, ">=", threshold))
            predicates.append(ObservationPredicate(feature_index, "<=", threshold))
    return predicates


def _one_hot(index: int, width: int) -> Tuple[int, ...]:
    return tuple(1 if item == index else 0 for item in range(width))


def _relation_one_hot(relation: str) -> Tuple[int, ...]:
    return _one_hot(DIRECT_OPT_RELATIONS.index(relation), len(DIRECT_OPT_RELATIONS))


def _operator_one_hot(operator: str) -> Tuple[int, ...]:
    return _one_hot(DIRECT_OPT_TREE_OPERATORS.index(operator), len(DIRECT_OPT_TREE_OPERATORS))


def _appendix_b3_predicate_encoding(predicate: ObservationPredicate) -> Tuple[float, Tuple[float, ...], float]:
    feature_weights = tuple(
        float(value)
        for value in _one_hot(predicate.feature_index, len(DIRECT_OPT_OBSERVATION_FEATURES))
    )
    if predicate.relation == "<=":
        return 1.0, feature_weights, predicate.threshold
    if predicate.relation == ">=":
        return -1.0, feature_weights, -predicate.threshold
    raise ValueError(f"unknown relation: {predicate.relation}")


def _boolean_switch_with_threshold_delta(
    switch: BooleanTreeSwitch,
    predicate_index: int,
    step_fraction: float,
) -> BooleanTreeSwitch:
    first = switch.first
    second = switch.second
    if predicate_index == 0:
        first = _predicate_with_threshold_delta(first, step_fraction)
    elif predicate_index == 1 and second is not None:
        second = _predicate_with_threshold_delta(second, step_fraction)
    return BooleanTreeSwitch(first, second, switch.operator)


def _predicate_with_threshold_delta(
    predicate: ObservationPredicate,
    step_fraction: float,
) -> ObservationPredicate:
    thresholds = DIRECT_OPT_BOOLEAN_THRESHOLD_GRIDS[predicate.feature_index]
    span = max(thresholds) - min(thresholds)
    step = span * step_fraction
    threshold = _clamp(predicate.threshold + step, min(thresholds), max(thresholds))
    return predicate.with_threshold(threshold)


def _unique_boolean_switches(switches: List[BooleanTreeSwitch]) -> List[BooleanTreeSwitch]:
    unique: Dict[str, BooleanTreeSwitch] = {}
    for switch in switches:
        unique.setdefault(switch.describe(), switch)
    return list(unique.values())


def _unique_boolean_neighbor_params(
    params: List[Tuple[BooleanTreeSwitch, float, float]],
) -> List[Tuple[BooleanTreeSwitch, float, float]]:
    unique: Dict[Tuple[str, float, float], Tuple[BooleanTreeSwitch, float, float]] = {}
    for switch, left_force, right_force in params:
        unique.setdefault((switch.describe(), left_force, right_force), (switch, left_force, right_force))
    return list(unique.values())


def _candidate_threshold_magnitude(candidate: DirectOptCandidate) -> float:
    if candidate.switch_kind != "boolean_tree":
        return abs(candidate.threshold)
    thresholds = [
        value
        for value in (candidate.first_threshold, candidate.second_threshold)
        if value is not None
    ]
    return sum(abs(value) for value in thresholds)


def _summarize_results(results) -> Dict[str, float]:
    return summarize_cartpole_results(results)
