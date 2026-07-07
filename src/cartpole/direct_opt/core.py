from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import random
import time
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
DIRECT_OPT_CONTINUOUS_ONE_HOT_TOP_LEAVES = 4
DIRECT_OPT_CONTINUOUS_ONE_HOT_FEATURE_MIXES = (
    (0.0, 0.0, 0.5, 0.5),
    (0.0, 0.0, 0.75, 0.25),
    (0.0, 0.0, 0.25, 0.75),
    (0.25, 0.25, 0.25, 0.25),
)
DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS = (-0.5, 0.0, 0.5)
DIRECT_OPT_CONTINUOUS_ONE_HOT_ALPHA_S_STEP_SCALE = 1.0
DIRECT_OPT_CONTINUOUS_ONE_HOT_WEIGHT_STEP_SCALE = 0.25
PAPER_DIRECT_OPT_BATCH_SIZE = 10
PAPER_DIRECT_OPT_PARALLEL_THREADS = 10
PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS = 7200


@dataclass(frozen=True)
class DirectOptContinuousOneHotPredicate:
    alpha_s: float
    feature_weights: Tuple[float, ...]
    alpha_0: float

    def evaluate(self, observation: Sequence[float]) -> bool:
        weighted_observation = sum(
            weight * value
            for weight, value in zip(self.feature_weights, observation)
        )
        return self.alpha_s * weighted_observation <= self.alpha_0

    def describe(self) -> str:
        terms = " + ".join(
            f"{weight:.3f}*o[{index}]"
            for index, weight in enumerate(self.feature_weights)
            if abs(weight) > 1e-12
        )
        return f"{self.alpha_s:.3f}*({terms or '0.000'}) <= {self.alpha_0:.3f}"


@dataclass(frozen=True)
class DirectOptContinuousOneHotSwitch:
    first: DirectOptContinuousOneHotPredicate
    second: DirectOptContinuousOneHotPredicate | None = None
    operator: str = "leaf"

    def decide(self, observation: Sequence[float]) -> int:
        first_enabled = self.first.evaluate(observation)
        if self.second is None or self.operator == "leaf":
            return 1 if first_enabled else 0
        second_enabled = self.second.evaluate(observation)
        if self.operator == "and":
            return 1 if first_enabled and second_enabled else 0
        if self.operator == "or":
            return 1 if first_enabled or second_enabled else 0
        raise ValueError(f"unknown DirectOptContinuousOneHotSwitch operator: {self.operator}")

    def describe(self) -> str:
        if self.second is None or self.operator == "leaf":
            return f"mode=1 if {self.first.describe()}, else mode=0"
        return (
            f"mode=1 if {self.first.describe()} {self.operator} "
            f"{self.second.describe()}, else mode=0"
        )


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
    train_distribution_rerank_candidates: int = 0
    train_distribution_rerank_rollouts: int = 0
    parallel_threads: int = 1
    time_limit_seconds: float | None = None
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
    continuous_one_hot_alpha_s: float | None = None
    continuous_one_hot_feature_weights: Tuple[float, ...] = ()
    continuous_one_hot_alpha_0: float | None = None
    continuous_one_hot_operator: str | None = None
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
    selected_train_initial_states: List[Sequence[float]]
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
        "search_method": "deterministic_grid_seeded_continuous_one_hot_random_search_plus_bounded_batch_restart_refinement",
        "policy_class": "two_mode_constant_action_linear_depth2_boolean_or_continuous_one_hot_switch",
        "selection_objective": "mean_combined_reward_over_selected_initial_states_then_success",
        "train_distribution_reranking": (
            "optionally reevaluate top selected-state candidates on seeded CartPole train-distribution "
            "rollouts before final policy selection; this is sampled distribution evidence, not the "
            "paper's full continuous initial-state optimization"
        ),
        "batch_refinement": "seed_each_batch_from_best_so_far_and_restart_on_stall_with_continuous_one_hot_candidates",
        "search_stopping": "stop_after_training_solution_or_parallel_chunk_or_time_limit",
        "paper_batch_size": PAPER_DIRECT_OPT_BATCH_SIZE,
        "paper_parallel_threads": PAPER_DIRECT_OPT_PARALLEL_THREADS,
        "paper_time_limit_seconds": PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS,
        "local_parallel_threads": "configurable_via_parallel_threads",
        "local_time_limit_seconds": "configurable_via_time_limit_seconds",
        "switch_search_space": "linear_theta_omega_grid_plus_bounded_boolean_tree_predicates_plus_bounded_continuous_one_hot_leaf_depth2_mixtures",
        "candidate_accounting": (
            "searched_candidates and evaluated_candidates count candidate evaluation calls; "
            "train_rollout_evaluations counts individual selected-state train rollouts; "
            "train_distribution_rerank_rollout_evaluations counts optional sampled "
            "train-distribution rerank rollouts; total_train_rollout_evaluations counts both"
        ),
        "optimization_trace": (
            "metrics record the exact selected train initial states and a compact "
            "batch-refinement trace with batch seed, local-neighbor, restart, and full-train "
            "reevaluation decisions"
        ),
        "boolean_tree_depth": 2,
        "boolean_tree_features": list(DIRECT_OPT_OBSERVATION_FEATURES),
        "boolean_tree_relations": list(DIRECT_OPT_RELATIONS),
        "boolean_tree_operator_choices": list(DIRECT_OPT_TREE_OPERATORS),
        "boolean_tree_threshold_grids": [list(grid) for grid in DIRECT_OPT_BOOLEAN_THRESHOLD_GRIDS],
        "boolean_tree_expansion": "evaluate_all_stumps_then_depth2_expansions_from_top_training_reward_stumps",
        "continuous_one_hot_feature_mixes": [list(weights) for weights in DIRECT_OPT_CONTINUOUS_ONE_HOT_FEATURE_MIXES],
        "continuous_one_hot_thresholds": list(DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS),
        "continuous_one_hot_top_leaves_for_depth2": DIRECT_OPT_CONTINUOUS_ONE_HOT_TOP_LEAVES,
        "continuous_one_hot_alpha_s_step_scale": DIRECT_OPT_CONTINUOUS_ONE_HOT_ALPHA_S_STEP_SCALE,
        "continuous_one_hot_weight_step_scale": DIRECT_OPT_CONTINUOUS_ONE_HOT_WEIGHT_STEP_SCALE,
        "continuous_one_hot_candidate_family": "bounded_appendix_b3_alpha_s_feature_mix_leaf_and_depth2_predicates",
        "continuous_one_hot_expansion": "evaluate_all_leaf_mixtures_then_depth2_expansions_from_top_training_reward_leaves",
        "random_restart_encoding": "bounded_appendix_b3_continuous_one_hot_alpha_s_simplex_feature_weights_threshold_force",
        "one_hot_switch_encoding": (
            "evaluates a bounded Appendix B.3 continuous leaf/depth2 feature-mixture candidate family, samples "
            "bounded continuous one-hot random restarts, and records continuous one-hot vertex fields plus "
            "bounded discrete metadata for feature, relation, and depth-2 tree operator choices; does not "
            "optimize the paper's full continuous one-hot relaxation"
        ),
        "local_refinement": (
            "linear_weight_threshold_force_neighbors_or_boolean_threshold_force_neighbors_or_"
            "continuous_one_hot_operator_alpha_s_weight_threshold_force_neighbors"
        ),
        "train_horizon_seconds": CartpoleEnv.train_env().cfg.horizon_seconds,
        "test_horizon_steps": CartpoleEnv.test_env().cfg.max_steps,
        "theta_weight_grid": list(DIRECT_OPT_THETA_WEIGHTS),
        "omega_weight_grid": list(DIRECT_OPT_OMEGA_WEIGHTS),
        "force_values": list(DIRECT_OPT_FORCE_VALUES),
        "threshold_scale": DIRECT_OPT_THRESHOLD_SCALE,
        "limitations": (
            "Diagnostic direct optimization over a bounded two-mode CartPole PSM. "
            "It optimizes mean train-horizon reward over the selected initial states and includes "
            "bounded Boolean-tree switch candidates, a bounded continuous one-hot leaf/depth2 "
            "feature-mixture candidate family, bounded continuous one-hot random restarts, optional "
            "sampled train-distribution reranking, and bounded one-hot operator/alpha_s/weight/"
            "threshold/force local refinement, but is not the paper's "
            "two-hour, ten-thread continuous numerical optimization over the full one-hot "
            "switching grammar."
        ),
    }


def cartpole_direct_opt_protocol_status(cfg: DirectOptConfig) -> Dict[str, object]:
    paper_test_env = CartpoleEnv.test_env()
    configured_paper_batch_size = cfg.batch_size == PAPER_DIRECT_OPT_BATCH_SIZE
    uses_paper_batch_size = configured_paper_batch_size and cfg.num_train_states >= PAPER_DIRECT_OPT_BATCH_SIZE
    selected_parallel_threads = max(1, int(cfg.parallel_threads))
    selected_time_limit_seconds = cfg.time_limit_seconds
    paper_test_horizon = cfg.test_max_steps == paper_test_env.cfg.max_steps
    paper_eval_rollouts = cfg.eval_rollouts == PAPER_EVAL_ROLLOUTS
    selected_rerank_candidates = max(0, int(cfg.train_distribution_rerank_candidates))
    selected_rerank_rollouts = max(0, int(cfg.train_distribution_rerank_rollouts))
    uses_train_distribution_reranking = selected_rerank_candidates > 0 and selected_rerank_rollouts > 0
    full_continuous_one_hot_switch_grammar = False
    full_batch_optimization = (
        uses_paper_batch_size
        and cfg.batch_refinement_rounds > 0
        and cfg.restart_candidates_on_stall > 0
    )
    uses_paper_parallel_threads = selected_parallel_threads == PAPER_DIRECT_OPT_PARALLEL_THREADS
    uses_paper_time_limit = selected_time_limit_seconds == PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS
    optimizes_full_initial_state_distribution = False
    direct_opt_requirements = {
        "paper_batch_size_and_batch_refinement": full_batch_optimization,
        "paper_parallel_threads": uses_paper_parallel_threads,
        "paper_time_limit": uses_paper_time_limit,
        "full_continuous_one_hot_switch_grammar": full_continuous_one_hot_switch_grammar,
        "full_initial_state_distribution": optimizes_full_initial_state_distribution,
        "full_test_horizon": paper_test_horizon,
        "paper_eval_rollouts": paper_eval_rollouts,
    }
    missing_requirements = [
        requirement
        for requirement, satisfied in direct_opt_requirements.items()
        if not satisfied
    ]
    paper_scale_direct_opt_protocol = not missing_requirements
    return {
        "paper_baseline": "Direct-Opt",
        "paper_batch_size": PAPER_DIRECT_OPT_BATCH_SIZE,
        "selected_batch_size": cfg.batch_size,
        "configured_paper_batch_size": configured_paper_batch_size,
        "uses_paper_batch_size": uses_paper_batch_size,
        "paper_parallel_threads": PAPER_DIRECT_OPT_PARALLEL_THREADS,
        "selected_parallel_threads": selected_parallel_threads,
        "uses_paper_parallel_threads": uses_paper_parallel_threads,
        "paper_time_limit_seconds": PAPER_DIRECT_OPT_TIME_LIMIT_SECONDS,
        "selected_time_limit_seconds": selected_time_limit_seconds,
        "uses_paper_time_limit": uses_paper_time_limit,
        "full_continuous_one_hot_switch_grammar": full_continuous_one_hot_switch_grammar,
        "bounded_one_hot_switch_metadata": True,
        "bounded_continuous_one_hot_switch_relaxation": True,
        "appendix_b3_one_hot_vertex_metadata": True,
        "linear_switch_encoding": True,
        "batch_optimization_seeded_from_best_so_far": cfg.batch_refinement_rounds > 0,
        "random_restart_on_stall": cfg.restart_candidates_on_stall > 0,
        "full_batch_optimization": full_batch_optimization,
        "stops_when_training_solution_found": True,
        "optimizes_combined_reward_over_selected_initial_states": True,
        "combined_reward_aggregation": "mean_train_horizon_reward_over_selected_initial_states",
        "optimizes_combined_reward_over_all_selected_initial_states": True,
        "uses_sampled_train_distribution_reranking": uses_train_distribution_reranking,
        "selected_train_distribution_rerank_candidates": selected_rerank_candidates,
        "selected_train_distribution_rerank_rollouts": selected_rerank_rollouts,
        "train_distribution_rerank_objective": (
            "mean_train_horizon_reward_over_seeded_train_distribution_rollouts_then_success"
        ),
        "optimizes_full_initial_state_distribution": optimizes_full_initial_state_distribution,
        "selected_train_initial_states": cfg.num_train_states,
        "paper_test_horizon_steps": paper_test_env.cfg.max_steps,
        "selected_test_max_steps": cfg.test_max_steps,
        "uses_full_test_horizon": paper_test_horizon,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": cfg.eval_rollouts,
        "uses_paper_eval_rollouts": paper_eval_rollouts,
        "quick_diagnostic": cfg.quick,
        "direct_opt_protocol_requirements": direct_opt_requirements,
        "missing_direct_opt_protocol_requirements": missing_requirements,
        "paper_scale_direct_opt_protocol": paper_scale_direct_opt_protocol,
        "limitation": (
            "Bounded local Direct-Opt diagnostic: records batch/restart structure and evaluates "
            "a bounded continuous one-hot leaf/depth2 feature-mixture candidate family with local "
            "alpha_s and feature-weight neighbors, but does not run the "
            "paper's ten-thread, two-hour continuous optimization over the full one-hot "
            "switching grammar."
        ),
    }


def run_cartpole_direct_opt(cfg: DirectOptConfig) -> DirectOptResult:
    rng = random.Random(cfg.seed)
    train_env = CartpoleEnv.train_env(seed=cfg.seed)
    train_states = [train_env.reset() for _ in range(max(1, cfg.num_train_states))]
    candidates, search_diagnostics = _direct_opt_candidates(train_states, cfg, rng)
    selected_state_best = max(candidates, key=_candidate_rank_key)
    best, rerank_diagnostics = _train_distribution_rerank_candidates(
        candidates,
        cfg,
        selected_state_best,
    )
    search_diagnostics["selected_state_best_candidate"] = _candidate_trace_summary(selected_state_best)
    search_diagnostics["train_distribution_rerank"] = rerank_diagnostics
    rerank_rollout_evaluations = int(rerank_diagnostics["train_rollout_evaluations"])
    search_diagnostics["train_distribution_rerank_rollout_evaluations"] = rerank_rollout_evaluations
    search_diagnostics["total_train_rollout_evaluations"] = (
        int(search_diagnostics["train_rollout_evaluations"]) + rerank_rollout_evaluations
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
        selected_train_initial_states=train_states,
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
        "selected_train_initial_states": _serialize_initial_states(result.selected_train_initial_states),
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
    deadline = _DirectOptDeadline(cfg.time_limit_seconds)
    grid_params = [
        (
            theta_weight,
            omega_weight,
            0.0,
            min(DIRECT_OPT_FORCE_VALUES),
            max(DIRECT_OPT_FORCE_VALUES),
            "grid",
        )
        for theta_weight in DIRECT_OPT_THETA_WEIGHTS
        for omega_weight in DIRECT_OPT_OMEGA_WEIGHTS
    ]
    grid_candidates = _evaluate_candidate_params(
        grid_params,
        train_states,
        cfg,
        deadline,
        allow_first=True,
        stop_on_solution=True,
    )
    candidates.extend(grid_candidates)
    stopped_after_grid = deadline.expired()
    solution_found_phase = _direct_opt_solution_phase(grid_candidates, "grid")
    random_candidates = _continuous_one_hot_random_restart_candidates(
        0 if stopped_after_grid or solution_found_phase is not None else max(0, cfg.random_candidates),
        train_states,
        cfg,
        rng,
        deadline,
        "random_restart",
        stop_on_solution=True,
    )
    candidates.extend(random_candidates)
    stopped_after_random = deadline.expired()
    solution_found_phase = solution_found_phase or _direct_opt_solution_phase(random_candidates, "random")
    boolean_candidates, boolean_diagnostics = (
        _boolean_tree_candidates(train_states, cfg, deadline, stop_on_solution=True)
        if not stopped_after_random and solution_found_phase is None
        else ([], _empty_boolean_diagnostics())
    )
    candidates.extend(boolean_candidates)
    stopped_after_boolean = deadline.expired()
    solution_found_phase = solution_found_phase or _direct_opt_solution_phase(boolean_candidates, "boolean")
    continuous_one_hot_candidates, continuous_one_hot_diagnostics = (
        _continuous_one_hot_candidates(train_states, cfg, deadline, stop_on_solution=True)
        if not stopped_after_boolean and solution_found_phase is None
        else ([], _empty_continuous_one_hot_diagnostics())
    )
    candidates.extend(continuous_one_hot_candidates)
    solution_found_phase = solution_found_phase or _direct_opt_solution_phase(
        continuous_one_hot_candidates,
        "continuous_one_hot",
    )

    best = max(candidates, key=_candidate_rank_key)
    stopped_before_batch = deadline.expired()
    planned_batch_count = len(_direct_opt_batches(train_states, max(1, cfg.batch_size)))
    batch_candidates, batch_diagnostics = (
        _batch_restart_refinement_candidates(
            best,
            train_states,
            cfg,
            rng,
            deadline,
        )
        if not stopped_before_batch and solution_found_phase is None
        else ([], _empty_batch_diagnostics(planned_batch_count, max(0, cfg.batch_refinement_rounds), deadline))
    )
    candidates.extend(batch_candidates)
    solution_found_phase = solution_found_phase or _direct_opt_solution_phase(batch_candidates, "batch_refinement")
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
        "grid_candidates": len(grid_candidates),
        "random_candidates": len(random_candidates),
        "random_restart_switch_kind": "continuous_one_hot",
        "random_restart_candidates_with_appendix_b3_metadata": len(random_candidates),
        **boolean_diagnostics,
        **continuous_one_hot_diagnostics,
        "batch_refinement_candidates": len(batch_candidates),
        "parallel_threads": max(1, int(cfg.parallel_threads)),
        "uses_parallel_candidate_evaluation": max(1, int(cfg.parallel_threads)) > 1,
        "time_limit_seconds": cfg.time_limit_seconds,
        "time_limit_reached": deadline.expired(),
        "solution_found": solution_found_phase is not None,
        "solution_found_phase": solution_found_phase,
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


def _direct_opt_solution_phase(candidates: Sequence[DirectOptCandidate], phase: str) -> str | None:
    return phase if any(_candidate_solves_training(candidate) for candidate in candidates) else None


def _candidate_solves_training(candidate: DirectOptCandidate) -> bool:
    return candidate.train_success_rate >= 1.0


class _DirectOptDeadline:
    def __init__(self, time_limit_seconds: float | None) -> None:
        self.time_limit_seconds = time_limit_seconds
        self.started_at = time.monotonic()

    def expired(self) -> bool:
        if self.time_limit_seconds is None:
            return False
        return time.monotonic() - self.started_at >= max(0.0, float(self.time_limit_seconds))


def _empty_boolean_diagnostics() -> Dict[str, int]:
    return {
        "boolean_stump_candidates": 0,
        "boolean_depth2_candidates": 0,
        "boolean_top_stumps_for_depth2": 0,
        "boolean_candidates_with_one_hot_metadata": 0,
        "boolean_candidates_with_appendix_b3_vertex_metadata": 0,
    }


def _empty_continuous_one_hot_diagnostics() -> Dict[str, int]:
    return {
        "continuous_one_hot_leaf_candidates": 0,
        "continuous_one_hot_depth2_candidates": 0,
        "continuous_one_hot_top_leaves_for_depth2": 0,
        "continuous_one_hot_candidates": 0,
        "continuous_one_hot_candidates_with_appendix_b3_metadata": 0,
    }


def _empty_batch_diagnostics(
    batch_count: int,
    rounds: int,
    deadline: "_DirectOptDeadline | None" = None,
) -> Dict[str, object]:
    return {
        "batch_count": batch_count,
        "batch_rounds": rounds,
        "batch_seed_evaluations": 0,
        "batch_seed_rollout_evaluations": 0,
        "batch_local_evaluations": 0,
        "batch_local_rollout_evaluations": 0,
        "restart_evaluations": 0,
        "restart_rollout_evaluations": 0,
        "accepted_batch_improvements": 0,
        "accepted_restarts": 0,
        "batch_time_limit_reached": bool(deadline and deadline.expired()),
        "batch_solution_found": False,
        "batch_refinement_trace": [],
    }


def _evaluate_candidate_params(
    params: List[Tuple[float, float, float, float, float, str]],
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
    allow_first: bool = False,
    stop_on_solution: bool = False,
) -> List[DirectOptCandidate]:
    parallel_threads = max(1, int(cfg.parallel_threads))
    if parallel_threads == 1 or len(params) <= 1:
        candidates: List[DirectOptCandidate] = []
        for theta_weight, omega_weight, threshold, left_force, right_force, source in params:
            if not candidates and not allow_first and deadline is not None and deadline.expired():
                break
            if candidates and deadline is not None and deadline.expired():
                break
            candidate = _evaluate_candidate(
                theta_weight,
                omega_weight,
                threshold,
                left_force,
                right_force,
                train_states,
                source,
            )
            candidates.append(candidate)
            if stop_on_solution and _candidate_solves_training(candidate):
                break
        return candidates
    candidates = []
    start_index = 0
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        while start_index < len(params):
            if candidates and deadline is not None and deadline.expired():
                break
            if not candidates and not allow_first and deadline is not None and deadline.expired():
                break
            chunk = params[start_index : start_index + parallel_threads]
            start_index += parallel_threads
            futures = [
                executor.submit(
                    _evaluate_candidate,
                    theta_weight,
                    omega_weight,
                    threshold,
                    left_force,
                    right_force,
                    train_states,
                    source,
                )
                for theta_weight, omega_weight, threshold, left_force, right_force, source in chunk
            ]
            chunk_candidates = [future.result() for future in futures]
            candidates.extend(chunk_candidates)
            if stop_on_solution and _direct_opt_solution_phase(chunk_candidates, "parallel_chunk") is not None:
                break
    return candidates


def _boolean_tree_candidates(
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig | None = None,
    deadline: _DirectOptDeadline | None = None,
    stop_on_solution: bool = False,
) -> Tuple[List[DirectOptCandidate], Dict[str, int]]:
    stumps = [BooleanTreeSwitch(predicate) for predicate in _direct_opt_predicates()]
    eval_cfg = cfg or DirectOptConfig()
    stump_candidates = _evaluate_boolean_candidates(
        [
            (
                stump,
                min(DIRECT_OPT_FORCE_VALUES),
                max(DIRECT_OPT_FORCE_VALUES),
                "boolean_stump",
            )
            for stump in stumps
        ],
        train_states,
        eval_cfg,
        deadline,
        stop_on_solution=stop_on_solution,
    )
    if stop_on_solution and _direct_opt_solution_phase(stump_candidates, "boolean_stump") is not None:
        return stump_candidates, {
            "boolean_stump_candidates": len(stump_candidates),
            "boolean_depth2_candidates": 0,
            "boolean_top_stumps_for_depth2": 0,
            "boolean_candidates_with_one_hot_metadata": len(stump_candidates),
            "boolean_candidates_with_appendix_b3_vertex_metadata": len(stump_candidates),
        }
    top_stumps = sorted(stump_candidates, key=_candidate_rank_key, reverse=True)[:DIRECT_OPT_BOOLEAN_TOP_STUMPS]
    depth2_switches: List[BooleanTreeSwitch] = []
    for candidate in top_stumps:
        switch = _candidate_switch(candidate)
        if not isinstance(switch, BooleanTreeSwitch):
            continue
        for predicate in _direct_opt_predicates():
            depth2_switches.append(BooleanTreeSwitch(switch.first, predicate, "and"))
            depth2_switches.append(BooleanTreeSwitch(switch.first, predicate, "or"))
    depth2_candidates = _evaluate_boolean_candidates(
        [
            (
                switch,
                min(DIRECT_OPT_FORCE_VALUES),
                max(DIRECT_OPT_FORCE_VALUES),
                "boolean_depth2",
            )
            for switch in _unique_boolean_switches(depth2_switches)
        ],
        train_states,
        eval_cfg,
        deadline,
        stop_on_solution=stop_on_solution,
    )
    return stump_candidates + depth2_candidates, {
        "boolean_stump_candidates": len(stump_candidates),
        "boolean_depth2_candidates": len(depth2_candidates),
        "boolean_top_stumps_for_depth2": len(top_stumps),
        "boolean_candidates_with_one_hot_metadata": len(stump_candidates) + len(depth2_candidates),
        "boolean_candidates_with_appendix_b3_vertex_metadata": len(stump_candidates) + len(depth2_candidates),
    }


def _continuous_one_hot_candidates(
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
    stop_on_solution: bool = False,
) -> Tuple[List[DirectOptCandidate], Dict[str, int]]:
    predicates = [
        DirectOptContinuousOneHotPredicate(alpha_s, tuple(feature_weights), alpha_0)
        for feature_weights in DIRECT_OPT_CONTINUOUS_ONE_HOT_FEATURE_MIXES
        for alpha_s in (-1.0, 1.0)
        for alpha_0 in DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS
    ]
    leaf_candidates = _evaluate_continuous_one_hot_candidates(
        [
            (
                DirectOptContinuousOneHotSwitch(predicate),
                min(DIRECT_OPT_FORCE_VALUES),
                max(DIRECT_OPT_FORCE_VALUES),
                "continuous_one_hot",
            )
            for predicate in predicates
        ],
        train_states,
        cfg,
        deadline,
        stop_on_solution=stop_on_solution,
    )
    if stop_on_solution and _direct_opt_solution_phase(leaf_candidates, "continuous_one_hot") is not None:
        candidate_count = len(leaf_candidates)
        return leaf_candidates, {
            "continuous_one_hot_leaf_candidates": len(leaf_candidates),
            "continuous_one_hot_depth2_candidates": 0,
            "continuous_one_hot_top_leaves_for_depth2": 0,
            "continuous_one_hot_candidates": candidate_count,
            "continuous_one_hot_candidates_with_appendix_b3_metadata": candidate_count,
        }
    top_leaves = sorted(leaf_candidates, key=_candidate_rank_key, reverse=True)[:DIRECT_OPT_CONTINUOUS_ONE_HOT_TOP_LEAVES]
    depth2_switches: List[DirectOptContinuousOneHotSwitch] = []
    if not (deadline is not None and deadline.expired()):
        for candidate in top_leaves:
            switch = _candidate_switch(candidate)
            if not isinstance(switch, DirectOptContinuousOneHotSwitch):
                continue
            for predicate in predicates:
                depth2_switches.append(DirectOptContinuousOneHotSwitch(switch.first, predicate, "and"))
                depth2_switches.append(DirectOptContinuousOneHotSwitch(switch.first, predicate, "or"))
    depth2_candidates = _evaluate_continuous_one_hot_candidates(
        [
            (
                switch,
                min(DIRECT_OPT_FORCE_VALUES),
                max(DIRECT_OPT_FORCE_VALUES),
                "continuous_one_hot_depth2",
            )
            for switch in _unique_continuous_one_hot_switches(depth2_switches)
        ],
        train_states,
        cfg,
        deadline,
        stop_on_solution=stop_on_solution,
    )
    candidate_count = len(leaf_candidates) + len(depth2_candidates)
    return leaf_candidates + depth2_candidates, {
        "continuous_one_hot_leaf_candidates": len(leaf_candidates),
        "continuous_one_hot_depth2_candidates": len(depth2_candidates),
        "continuous_one_hot_top_leaves_for_depth2": len(top_leaves),
        "continuous_one_hot_candidates": candidate_count,
        "continuous_one_hot_candidates_with_appendix_b3_metadata": candidate_count,
    }


def _evaluate_continuous_one_hot_candidates(
    params: List[Tuple[DirectOptContinuousOneHotSwitch, float, float, str]],
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
    stop_on_solution: bool = False,
) -> List[DirectOptCandidate]:
    parallel_threads = max(1, int(cfg.parallel_threads))
    if parallel_threads == 1 or len(params) <= 1:
        candidates: List[DirectOptCandidate] = []
        for switch, left_force, right_force, source in params:
            if deadline is not None and deadline.expired():
                break
            candidate = _evaluate_continuous_one_hot_candidate(
                switch,
                left_force,
                right_force,
                train_states,
                source,
            )
            candidates.append(candidate)
            if stop_on_solution and _candidate_solves_training(candidate):
                break
        return candidates
    candidates = []
    start_index = 0
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        while start_index < len(params):
            if deadline is not None and deadline.expired():
                break
            chunk = params[start_index : start_index + parallel_threads]
            start_index += parallel_threads
            futures = [
                executor.submit(
                    _evaluate_continuous_one_hot_candidate,
                    switch,
                    left_force,
                    right_force,
                    train_states,
                    source,
                )
                for switch, left_force, right_force, source in chunk
            ]
            chunk_candidates = [future.result() for future in futures]
            candidates.extend(chunk_candidates)
            if stop_on_solution and _direct_opt_solution_phase(chunk_candidates, "parallel_chunk") is not None:
                break
    return candidates


def _evaluate_boolean_candidates(
    params: List[Tuple[BooleanTreeSwitch, float, float, str]],
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
    stop_on_solution: bool = False,
) -> List[DirectOptCandidate]:
    parallel_threads = max(1, int(cfg.parallel_threads))
    if parallel_threads == 1 or len(params) <= 1:
        candidates: List[DirectOptCandidate] = []
        for switch, left_force, right_force, source in params:
            if not candidates and deadline is not None and deadline.expired():
                break
            if candidates and deadline is not None and deadline.expired():
                break
            candidate = _evaluate_boolean_candidate(switch, left_force, right_force, train_states, source)
            candidates.append(candidate)
            if stop_on_solution and _candidate_solves_training(candidate):
                break
        return candidates
    candidates = []
    start_index = 0
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        while start_index < len(params):
            if deadline is not None and deadline.expired():
                break
            chunk = params[start_index : start_index + parallel_threads]
            start_index += parallel_threads
            futures = [
                executor.submit(
                    _evaluate_boolean_candidate,
                    switch,
                    left_force,
                    right_force,
                    train_states,
                    source,
                )
                for switch, left_force, right_force, source in chunk
            ]
            chunk_candidates = [future.result() for future in futures]
            candidates.extend(chunk_candidates)
            if stop_on_solution and _direct_opt_solution_phase(chunk_candidates, "parallel_chunk") is not None:
                break
    return candidates


def _batch_restart_refinement_candidates(
    seed_candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    rng: random.Random,
    deadline: _DirectOptDeadline | None = None,
) -> Tuple[List[DirectOptCandidate], Dict[str, object]]:
    batches = _direct_opt_batches(train_states, max(1, cfg.batch_size))
    rounds = max(0, cfg.batch_refinement_rounds)
    if rounds == 0 or not batches:
        return [], _empty_batch_diagnostics(len(batches), rounds, deadline)

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
    batch_solution_found = False
    batch_trace: List[Dict[str, object]] = []
    for round_index in range(rounds):
        for batch_index, batch in enumerate(batches):
            if deadline is not None and deadline.expired():
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
                    "batch_time_limit_reached": True,
                    "batch_solution_found": batch_solution_found,
                    "batch_refinement_trace": batch_trace,
                }
            batch_seed_evaluations += 1
            batch_seed_rollout_evaluations += len(batch)
            batch_best = _reevaluate_candidate(current, batch, "batch_seed")
            trace_entry: Dict[str, object] = {
                "round_index": round_index,
                "batch_index": batch_index,
                "selected_initial_state_indices": list(
                    range(
                        batch_index * max(1, cfg.batch_size),
                        min(len(train_states), batch_index * max(1, cfg.batch_size) + len(batch)),
                    )
                ),
                "seed_from_full_train_best": _candidate_trace_summary(current),
                "batch_seed_result": _candidate_trace_summary(batch_best),
                "local_steps": [],
            }
            local_step_traces: List[Dict[str, object]] = []
            for local_step_index in range(max(0, cfg.local_refinement_steps)):
                if deadline is not None and deadline.expired():
                    break
                neighbors = _local_neighbor_candidates(batch_best, batch, cfg, deadline)
                local_evaluations += len(neighbors)
                batch_local_rollout_evaluations += len(neighbors) * len(batch)
                local_best = max(neighbors, key=_candidate_rank_key) if neighbors else batch_best
                step_trace: Dict[str, object] = {
                    "step_index": local_step_index,
                    "local_evaluations": len(neighbors),
                    "best_local_candidate": (
                        _candidate_trace_summary(local_best)
                        if neighbors
                        else None
                    ),
                    "accepted_local_improvement": False,
                    "restart_evaluations": 0,
                    "best_restart_candidate": None,
                    "accepted_restart": False,
                    "stopped_after_no_improvement": False,
                }
                if _candidate_rank_key(local_best) > _candidate_rank_key(batch_best):
                    batch_best = local_best
                    accepted_improvements += 1
                    step_trace["accepted_local_improvement"] = True
                    local_step_traces.append(step_trace)
                    continue
                restarts = _continuous_one_hot_random_restart_candidates(
                    max(0, cfg.restart_candidates_on_stall),
                    batch,
                    cfg,
                    rng,
                    deadline,
                    "batch_random_restart",
                )
                restart_evaluations += len(restarts)
                restart_rollout_evaluations += len(restarts) * len(batch)
                restart_best = max(restarts, key=_candidate_rank_key) if restarts else batch_best
                step_trace["restart_evaluations"] = len(restarts)
                step_trace["best_restart_candidate"] = (
                    _candidate_trace_summary(restart_best)
                    if restarts
                    else None
                )
                if _candidate_rank_key(restart_best) > _candidate_rank_key(batch_best):
                    batch_best = restart_best
                    accepted_restarts += 1
                    step_trace["accepted_restart"] = True
                    local_step_traces.append(step_trace)
                    continue
                step_trace["stopped_after_no_improvement"] = True
                local_step_traces.append(step_trace)
                break
            trace_entry["local_steps"] = local_step_traces
            if deadline is not None and deadline.expired():
                trace_entry["full_train_reevaluation_skipped_due_to_time_limit"] = True
                batch_trace.append(trace_entry)
                continue
            full_candidate = _reevaluate_candidate(batch_best, train_states, "batch_refinement")
            candidates.append(full_candidate)
            accepted_full_train = _candidate_rank_key(full_candidate) > _candidate_rank_key(current)
            trace_entry["full_train_reevaluation"] = _candidate_trace_summary(full_candidate)
            trace_entry["accepted_full_train_improvement"] = accepted_full_train
            trace_entry["solution_found"] = _candidate_solves_training(full_candidate)
            batch_trace.append(trace_entry)
            if accepted_full_train:
                current = full_candidate
            if _candidate_solves_training(full_candidate):
                batch_solution_found = True
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
                    "batch_time_limit_reached": bool(deadline and deadline.expired()),
                    "batch_solution_found": batch_solution_found,
                    "batch_refinement_trace": batch_trace,
                }
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
        "batch_time_limit_reached": bool(deadline and deadline.expired()),
        "batch_solution_found": batch_solution_found,
        "batch_refinement_trace": batch_trace,
    }


def _direct_opt_batches(
    train_states: List[Sequence[float]],
    batch_size: int,
) -> List[List[Sequence[float]]]:
    return [
        train_states[index : index + batch_size]
        for index in range(0, len(train_states), batch_size)
    ]


def _continuous_one_hot_random_restart_candidates(
    count: int,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    rng: random.Random,
    deadline: _DirectOptDeadline | None,
    source: str,
    stop_on_solution: bool = False,
) -> List[DirectOptCandidate]:
    params = [
        (*_random_continuous_one_hot_params(rng), source)
        for _ in range(max(0, count))
    ]
    return _evaluate_continuous_one_hot_candidates(
        params,
        train_states,
        cfg,
        deadline,
        stop_on_solution=stop_on_solution,
    )


def _random_continuous_one_hot_params(
    rng: random.Random,
) -> Tuple[DirectOptContinuousOneHotSwitch, float, float]:
    left_force = rng.choice(DIRECT_OPT_FORCE_VALUES)
    return _random_continuous_one_hot_switch(rng), left_force, -left_force


def _random_continuous_one_hot_switch(rng: random.Random) -> DirectOptContinuousOneHotSwitch:
    operator = rng.choice(DIRECT_OPT_TREE_OPERATORS)
    first = _random_continuous_one_hot_predicate(rng)
    second = None if operator == "leaf" else _random_continuous_one_hot_predicate(rng)
    return DirectOptContinuousOneHotSwitch(first, second, operator=operator)


def _random_continuous_one_hot_predicate(rng: random.Random) -> DirectOptContinuousOneHotPredicate:
    return DirectOptContinuousOneHotPredicate(
        rng.uniform(-1.0, 1.0),
        _random_simplex_weights(rng, len(DIRECT_OPT_OBSERVATION_FEATURES)),
        rng.uniform(
            min(DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS),
            max(DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS),
        ),
    )


def _random_simplex_weights(rng: random.Random, width: int) -> Tuple[float, ...]:
    raw = [rng.random() for _ in range(max(1, width))]
    return _normalize_continuous_one_hot_weights(raw)


def _local_neighbor_candidates(
    candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
) -> List[DirectOptCandidate]:
    if candidate.switch_kind == "continuous_one_hot":
        return _continuous_one_hot_local_neighbor_candidates(candidate, train_states, cfg, deadline)
    if candidate.switch_kind == "boolean_tree":
        return _boolean_local_neighbor_candidates(candidate, train_states, cfg)
    return _evaluate_candidate_params(
        [
            (*params, "batch_local_refinement")
            for params in _linear_local_neighbor_params(candidate, cfg)
        ],
        train_states,
        cfg,
        deadline,
    )


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
            "batch_local_refinement",
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
    return _evaluate_boolean_candidates(
        _unique_boolean_neighbor_params(params),
        train_states,
        cfg,
    )


def _continuous_one_hot_local_neighbor_candidates(
    candidate: DirectOptCandidate,
    train_states: List[Sequence[float]],
    cfg: DirectOptConfig,
    deadline: _DirectOptDeadline | None = None,
) -> List[DirectOptCandidate]:
    switch = _candidate_switch(candidate)
    if not isinstance(switch, DirectOptContinuousOneHotSwitch):
        return []
    force_lower = min(DIRECT_OPT_FORCE_VALUES)
    force_upper = max(DIRECT_OPT_FORCE_VALUES)
    force_step = (force_upper - force_lower) * max(0.0, cfg.local_step_fraction)
    threshold_step = DIRECT_OPT_THRESHOLD_SCALE * max(0.0, cfg.local_step_fraction)
    threshold_lower = min(DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS)
    threshold_upper = max(DIRECT_OPT_CONTINUOUS_ONE_HOT_THRESHOLDS)
    switches = [switch]
    for direction in (-1.0, 1.0):
        switches.append(
            DirectOptContinuousOneHotSwitch(
                DirectOptContinuousOneHotPredicate(
                    switch.first.alpha_s,
                    switch.first.feature_weights,
                    _clamp(switch.first.alpha_0 + direction * threshold_step, threshold_lower, threshold_upper),
                ),
                switch.second,
                operator=switch.operator,
            )
        )
        if switch.second is not None:
            switches.append(
                DirectOptContinuousOneHotSwitch(
                    switch.first,
                    DirectOptContinuousOneHotPredicate(
                        switch.second.alpha_s,
                        switch.second.feature_weights,
                        _clamp(switch.second.alpha_0 + direction * threshold_step, threshold_lower, threshold_upper),
                    ),
                operator=switch.operator,
            )
        )
    for first_weights in _continuous_one_hot_weight_neighbors(switch.first.feature_weights, cfg):
        switches.append(
            DirectOptContinuousOneHotSwitch(
                DirectOptContinuousOneHotPredicate(
                    switch.first.alpha_s,
                    first_weights,
                    switch.first.alpha_0,
                ),
                switch.second,
                operator=switch.operator,
            )
        )
    for first_alpha_s in _continuous_one_hot_alpha_s_neighbors(switch.first.alpha_s, cfg):
        switches.append(
            DirectOptContinuousOneHotSwitch(
                DirectOptContinuousOneHotPredicate(
                    first_alpha_s,
                    switch.first.feature_weights,
                    switch.first.alpha_0,
                ),
                switch.second,
                operator=switch.operator,
            )
        )
    if switch.second is not None:
        for operator in _continuous_one_hot_operator_neighbors(switch.operator):
            switches.append(
                DirectOptContinuousOneHotSwitch(
                    switch.first,
                    switch.second,
                    operator=operator,
                )
            )
        for second_weights in _continuous_one_hot_weight_neighbors(switch.second.feature_weights, cfg):
            switches.append(
                DirectOptContinuousOneHotSwitch(
                    switch.first,
                    DirectOptContinuousOneHotPredicate(
                        switch.second.alpha_s,
                        second_weights,
                        switch.second.alpha_0,
                    ),
                    operator=switch.operator,
                )
            )
        for second_alpha_s in _continuous_one_hot_alpha_s_neighbors(switch.second.alpha_s, cfg):
            switches.append(
                DirectOptContinuousOneHotSwitch(
                    switch.first,
                    DirectOptContinuousOneHotPredicate(
                        second_alpha_s,
                        switch.second.feature_weights,
                        switch.second.alpha_0,
                    ),
                    operator=switch.operator,
                )
            )
    params = [
        (
            local_switch,
            _clamp(candidate.left_force + delta_left, force_lower, force_upper),
            _clamp(candidate.right_force + delta_right, force_lower, force_upper),
            "batch_local_refinement",
        )
        for local_switch in switches
        for delta_left, delta_right in (
            (0.0, 0.0),
            (force_step, 0.0),
            (-force_step, 0.0),
            (0.0, force_step),
            (0.0, -force_step),
        )
    ]
    unique: Dict[
        Tuple[str, float, float, str],
        Tuple[DirectOptContinuousOneHotSwitch, float, float, str],
    ] = {}
    for local_switch, left_force, right_force, source in params:
        unique.setdefault(
            (local_switch.describe(), left_force, right_force, source),
            (local_switch, left_force, right_force, source),
        )
    return _evaluate_continuous_one_hot_candidates(list(unique.values()), train_states, cfg, deadline)


def _continuous_one_hot_operator_neighbors(operator: str) -> List[str]:
    return [candidate for candidate in DIRECT_OPT_TREE_OPERATORS if candidate != operator]


def _continuous_one_hot_alpha_s_neighbors(
    alpha_s: float,
    cfg: DirectOptConfig,
) -> List[float]:
    step = DIRECT_OPT_CONTINUOUS_ONE_HOT_ALPHA_S_STEP_SCALE * max(0.0, cfg.local_step_fraction)
    if step <= 0.0:
        return []
    candidates = [
        _clamp(alpha_s + step, -1.0, 1.0),
        _clamp(alpha_s - step, -1.0, 1.0),
        -alpha_s,
    ]
    return [value for value in dict.fromkeys(candidates) if abs(value - alpha_s) > 1e-12]


def _continuous_one_hot_weight_neighbors(
    feature_weights: Tuple[float, ...],
    cfg: DirectOptConfig,
) -> List[Tuple[float, ...]]:
    step = DIRECT_OPT_CONTINUOUS_ONE_HOT_WEIGHT_STEP_SCALE * max(0.0, cfg.local_step_fraction)
    if step <= 0.0:
        return []
    neighbors: List[Tuple[float, ...]] = []
    width = len(feature_weights)
    for source_index in range(width):
        if feature_weights[source_index] <= 0.0:
            continue
        for target_index in range(width):
            if target_index == source_index:
                continue
            shifted = list(feature_weights)
            delta = min(step, shifted[source_index])
            shifted[source_index] -= delta
            shifted[target_index] += delta
            neighbors.append(_normalize_continuous_one_hot_weights(shifted))
    return list(dict.fromkeys(neighbors))


def _normalize_continuous_one_hot_weights(weights: Sequence[float]) -> Tuple[float, ...]:
    clipped = [max(0.0, float(weight)) for weight in weights]
    total = sum(clipped)
    if total <= 0.0:
        width = max(1, len(clipped))
        return tuple(1.0 / width for _ in range(width))
    return tuple(weight / total for weight in clipped)


def _candidate_rank_key(candidate: DirectOptCandidate) -> Tuple[float, float, float]:
    return (
        candidate.train_reward_mean,
        candidate.train_success_rate,
        -_candidate_threshold_magnitude(candidate),
    )


def _train_distribution_rerank_candidates(
    candidates: Sequence[DirectOptCandidate],
    cfg: DirectOptConfig,
    selected_state_best: DirectOptCandidate,
) -> Tuple[DirectOptCandidate, Dict[str, object]]:
    candidate_count = max(0, int(cfg.train_distribution_rerank_candidates))
    rollout_count = max(0, int(cfg.train_distribution_rerank_rollouts))
    if candidate_count <= 0 or rollout_count <= 0:
        return selected_state_best, {
            "enabled": False,
            "requested_candidates": candidate_count,
            "requested_rollouts": rollout_count,
            "evaluated_candidates": 0,
            "train_rollout_evaluations": 0,
            "sample_seed": None,
            "sampled_initial_states": [],
            "selected_candidate_changed": False,
            "selection_source": selected_state_best.source,
            "evaluated": [],
        }

    top_candidates = sorted(candidates, key=_candidate_rank_key, reverse=True)[:candidate_count]
    sample_seed = 10_000 + cfg.seed
    sample_env = CartpoleEnv.train_env(seed=sample_seed)
    sampled_initial_states = [sample_env.reset() for _ in range(rollout_count)]
    evaluations: List[Dict[str, object]] = []
    best_candidate = selected_state_best
    best_key: Tuple[float, float, float, float] | None = None
    for index, candidate in enumerate(top_candidates):
        policy = _candidate_policy(candidate)
        env = CartpoleEnv.train_env(seed=sample_seed + index + 1)
        summary = _summarize_results(
            [env.rollout(policy, initial_state=state) for state in sampled_initial_states]
        )
        key = (
            float(summary["reward_mean"]),
            float(summary["success_rate"]),
            candidate.train_reward_mean,
            candidate.train_success_rate,
        )
        evaluations.append(
            {
                "rank": index,
                "source": candidate.source,
                "switch_kind": candidate.switch_kind,
                "selected_state_train_reward_mean": candidate.train_reward_mean,
                "selected_state_train_success_rate": candidate.train_success_rate,
                "train_distribution_reward_mean": summary["reward_mean"],
                "train_distribution_success_rate": summary["success_rate"],
                "train_distribution_steps_mean": summary["steps_mean"],
                "train_distribution_survival_seconds_mean": summary["survival_seconds_mean"],
                "policy_description": policy.describe(),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    return best_candidate, {
        "enabled": True,
        "requested_candidates": candidate_count,
        "requested_rollouts": rollout_count,
        "evaluated_candidates": len(top_candidates),
        "train_rollout_evaluations": len(top_candidates) * rollout_count,
        "sample_seed": sample_seed,
        "sampled_initial_states": _serialize_initial_states(sampled_initial_states),
        "selected_candidate_changed": best_candidate is not selected_state_best,
        "selection_source": best_candidate.source,
        "evaluated": evaluations,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _serialize_initial_states(states: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(value) for value in state] for state in states]


def _candidate_trace_summary(candidate: DirectOptCandidate) -> Dict[str, object]:
    return {
        "source": candidate.source,
        "switch_kind": candidate.switch_kind,
        "train_reward_mean": candidate.train_reward_mean,
        "train_success_rate": candidate.train_success_rate,
        "rank_key": list(_candidate_rank_key(candidate)),
        "policy_description": _candidate_policy(candidate).describe(),
    }


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


def _evaluate_continuous_one_hot_candidate(
    switch: DirectOptContinuousOneHotSwitch,
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
    switch: Depth2Switch | BooleanTreeSwitch | DirectOptContinuousOneHotSwitch,
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
    switch: Depth2Switch | BooleanTreeSwitch | DirectOptContinuousOneHotSwitch,
    left_force: float,
    right_force: float,
    train_reward_mean: float,
    train_success_rate: float,
    source: str,
    theta_weight: float = 0.0,
    omega_weight: float = 0.0,
    threshold: float = 0.0,
) -> DirectOptCandidate:
    if isinstance(switch, DirectOptContinuousOneHotSwitch):
        return DirectOptCandidate(
            theta_weight=theta_weight,
            omega_weight=omega_weight,
            threshold=threshold,
            left_force=left_force,
            right_force=right_force,
            train_reward_mean=train_reward_mean,
            train_success_rate=train_success_rate,
            source=source,
            switch_kind="continuous_one_hot",
            continuous_one_hot_alpha_s=switch.first.alpha_s,
            continuous_one_hot_feature_weights=switch.first.feature_weights,
            continuous_one_hot_alpha_0=switch.first.alpha_0,
            continuous_one_hot_operator=switch.operator,
            first_appendix_b3_alpha_s=switch.first.alpha_s,
            first_appendix_b3_feature_weights=switch.first.feature_weights,
            first_appendix_b3_alpha_0=switch.first.alpha_0,
            second_appendix_b3_alpha_s=(
                switch.second.alpha_s if switch.second is not None else None
            ),
            second_appendix_b3_feature_weights=(
                switch.second.feature_weights if switch.second is not None else ()
            ),
            second_appendix_b3_alpha_0=(
                switch.second.alpha_0 if switch.second is not None else None
            ),
            operator_one_hot=_operator_one_hot(switch.operator),
        )
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


def _candidate_switch(
    candidate: DirectOptCandidate,
) -> Depth2Switch | BooleanTreeSwitch | DirectOptContinuousOneHotSwitch:
    if candidate.switch_kind == "continuous_one_hot":
        second = None
        if candidate.second_appendix_b3_alpha_s is not None:
            second = DirectOptContinuousOneHotPredicate(
                float(candidate.second_appendix_b3_alpha_s),
                tuple(candidate.second_appendix_b3_feature_weights),
                float(candidate.second_appendix_b3_alpha_0),
            )
        return DirectOptContinuousOneHotSwitch(
            DirectOptContinuousOneHotPredicate(
                float(candidate.continuous_one_hot_alpha_s),
                tuple(candidate.continuous_one_hot_feature_weights),
                float(candidate.continuous_one_hot_alpha_0),
            ),
            second,
            operator=candidate.continuous_one_hot_operator or "leaf",
        )
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


def _unique_continuous_one_hot_switches(
    switches: List[DirectOptContinuousOneHotSwitch],
) -> List[DirectOptContinuousOneHotSwitch]:
    unique: Dict[str, DirectOptContinuousOneHotSwitch] = {}
    for switch in switches:
        unique.setdefault(switch.describe(), switch)
    return list(unique.values())


def _unique_boolean_neighbor_params(
    params: List[Tuple[BooleanTreeSwitch, float, float, str]],
) -> List[Tuple[BooleanTreeSwitch, float, float, str]]:
    unique: Dict[Tuple[str, float, float, str], Tuple[BooleanTreeSwitch, float, float, str]] = {}
    for switch, left_force, right_force, source in params:
        unique.setdefault(
            (switch.describe(), left_force, right_force, source),
            (switch, left_force, right_force, source),
        )
    return list(unique.values())


def _candidate_threshold_magnitude(candidate: DirectOptCandidate) -> float:
    if candidate.switch_kind == "continuous_one_hot":
        thresholds = [
            value
            for value in (candidate.continuous_one_hot_alpha_0, candidate.second_appendix_b3_alpha_0)
            if value is not None
        ]
        return sum(abs(value) for value in thresholds)
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
