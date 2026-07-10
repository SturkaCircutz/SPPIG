from __future__ import annotations

import bisect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple

from cartpole_env import (
    CARTPOLE_PSM_MODE_UPDATE_ORDER,
    PAPER_EVAL_ROLLOUTS,
    CartpoleConfig,
    CartpoleEnv,
    Observation,
    cartpole_done,
    cartpole_next_state,
    cartpole_reward_spec,
    cartpole_space_spec,
)

from .synthesis_config import *
from .synthesis_programs import *

def synthesize_cartpole_student_with_history(
    cfg: CartpoleSynthesisConfig,
) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleTrace], List[CartpoleSynthesisIteration]]:
    rng = random.Random(cfg.seed)
    env = CartpoleEnv.train_env(seed=cfg.seed)
    initial_states = [env.reset() for _ in range(cfg.num_initial_states)]
    student: ProbabilisticCartpoleStudent | None = None
    traces: List[CartpoleTrace] = []
    history: List[CartpoleSynthesisIteration] = []
    # Alternate between a teacher that searches for high-reward traces and a
    # student fit that makes later teacher traces easier to explain with the PSM.
    for iteration in range(max(1, cfg.teacher_student_iters)):
        traces = _optimize_loop_free_traces_for_initial_states(
            initial_states,
            env.cfg,
            cfg,
            rng,
            student,
        )
        student, student_fit_history = fit_probabilistic_cartpole_student_with_history(traces, cfg)
        history.append(CartpoleSynthesisIteration(iteration + 1, traces, student, student_fit_history))
    if student is None:
        raise RuntimeError("Cartpole synthesis did not produce a student policy")
    return student, traces, history


def _optimize_loop_free_traces_for_initial_states(
    initial_states: List[Observation],
    env_cfg: CartpoleConfig,
    cfg: CartpoleSynthesisConfig,
    rng: random.Random,
    student: ProbabilisticCartpoleStudent | None,
) -> List[CartpoleTrace]:
    parallel_workers = max(1, int(cfg.parallel_trace_workers))
    if parallel_workers == 1 or len(initial_states) <= 1:
        return [
            _optimize_loop_free_trace(initial_state, env_cfg, cfg, rng, student)
            for initial_state in initial_states
        ]

    trace_seeds = [rng.randrange(2**63) for _ in initial_states]
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [
            executor.submit(
                _optimize_loop_free_trace,
                initial_state,
                env_cfg,
                cfg,
                random.Random(trace_seed),
                student,
            )
            for initial_state, trace_seed in zip(initial_states, trace_seeds)
        ]
        return [future.result() for future in futures]


def fit_probabilistic_cartpole_student(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> ProbabilisticCartpoleStudent:
    student, _ = fit_probabilistic_cartpole_student_with_history(traces, cfg)
    return student


def fit_probabilistic_cartpole_student_with_history(
    traces: List[CartpoleTrace],
    cfg: CartpoleSynthesisConfig,
) -> tuple[ProbabilisticCartpoleStudent, List[CartpoleStudentFitStep]]:
    """Fit the Cartpole student using Gaussian action-parameter distributions.

    This implements the action-distribution part of the paper's EM-style
    student step for Cartpole's constant-action grammar. The latent segment
    responsibilities are initialized from action likelihoods, then each bounded
    EM iteration repeats switch-timing forward-backward responsibilities with
    action-distribution refits before one switch-parameter M-step.
    Switch timing still uses local Gaussian mean/std refinement rather than the
    paper's full continuous M-step.
    """

    segments_by_trace = _segments_from_traces(traces)
    segments = [segment for trace_segments in segments_by_trace for segment in trace_segments]
    left_default = min(cfg.force_values)
    right_default = max(cfg.force_values)
    action_distributions = {
        0: GaussianScalar(left_default, 1.0),
        1: GaussianScalar(right_default, 1.0),
    }
    responsibilities: List[Tuple[float, float]] = []
    switch_pair_responsibilities: List[Tuple[float, float, float, float]] = []
    switch: SwitchProgram | None = None
    switch_parameter_distributions: List[GaussianScalar] = []
    transition_switches: Dict[Tuple[int, int], SwitchProgram] = {}
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] = {}
    fit_history: List[CartpoleStudentFitStep] = []

    for iteration in range(max(1, cfg.student_em_iters)):
        if iteration == 0 or cfg.student_switch_responsibility_passes <= 0:
            responsibilities = _action_likelihood_responsibilities(segments, action_distributions)
            responsibilities = _condition_initial_mode_responsibilities(segments_by_trace, responsibilities)
            action_distributions = _fit_action_distributions(
                segments,
                responsibilities,
                left_default,
                right_default,
            )
            if cfg.student_switch_responsibility_passes <= 0:
                switch, switch_parameter_distributions = _fit_student_switch(
                    traces,
                    segments_by_trace,
                    responsibilities,
                    switch_pair_responsibilities or None,
                    cfg,
                )
                transition_switches, transition_switch_parameter_distributions = _fit_transition_switches(
                    traces,
                    segments_by_trace,
                    responsibilities,
                    switch_pair_responsibilities or None,
                    switch,
                    switch_parameter_distributions,
                    cfg,
                )
                step_transition_switches: Dict[Tuple[int, int], SwitchProgram] = {}
                step_transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] = {}
            elif switch is None:
                bootstrap = _bootstrap_probabilistic_student(cfg)
                switch = bootstrap.switch
                switch_parameter_distributions = list(bootstrap.switch_parameter_distributions)
                step_transition_switches = transition_switches
                step_transition_switch_parameter_distributions = transition_switch_parameter_distributions
            phase = "action_likelihood_initialization" if iteration == 0 else "action_likelihood_refit"
            fit_history.append(
                _student_fit_step(
                    traces,
                    iteration + 1,
                    0,
                    phase,
                    responsibilities,
                    switch_pair_responsibilities,
                    action_distributions,
                    switch,
                    switch_parameter_distributions,
                    step_transition_switches,
                    step_transition_switch_parameter_distributions,
                )
            )

        if cfg.student_switch_responsibility_passes <= 0:
            fit_history.append(
                _student_fit_step(
                    traces,
                    iteration + 1,
                    0,
                    "switch_condition_m_step",
                    responsibilities,
                    switch_pair_responsibilities,
                    action_distributions,
                    switch,
                    switch_parameter_distributions,
                    transition_switches,
                    transition_switch_parameter_distributions,
                )
            )
            continue
        if switch is None:
            raise RuntimeError("Cartpole student EM requires an initialized switch")
        for pass_index in range(cfg.student_switch_responsibility_passes):
            responsibilities, switch_pair_responsibilities = _refine_responsibilities_and_switch_pairs_with_timing(
                segments_by_trace,
                action_distributions,
                switch,
                switch_parameter_distributions,
                transition_switches,
                transition_switch_parameter_distributions,
            )
            action_distributions = _fit_action_distributions(
                segments,
                responsibilities,
                left_default,
                right_default,
            )
            fit_history.append(
                _student_fit_step(
                    traces,
                    iteration + 1,
                    pass_index + 1,
                    "switch_timing_responsibility_refit",
                    responsibilities,
                    switch_pair_responsibilities,
                    action_distributions,
                    switch,
                    switch_parameter_distributions,
                    transition_switches,
                    transition_switch_parameter_distributions,
                )
            )
        switch, switch_parameter_distributions = _fit_student_switch(
            traces,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities or None,
            cfg,
        )
        transition_switches, transition_switch_parameter_distributions = _fit_transition_switches(
            traces,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities or None,
            switch,
            switch_parameter_distributions,
            cfg,
        )
        fit_history.append(
            _student_fit_step(
                traces,
                iteration + 1,
                cfg.student_switch_responsibility_passes,
                "switch_condition_m_step",
                responsibilities,
                switch_pair_responsibilities,
                action_distributions,
                switch,
                switch_parameter_distributions,
                transition_switches,
                transition_switch_parameter_distributions,
            )
        )

    if switch is None:
        switch, switch_parameter_distributions = _fit_student_switch(
            traces,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities or None,
            cfg,
        )
    if not transition_switches:
        transition_switches, transition_switch_parameter_distributions = _fit_transition_switches(
            traces,
            segments_by_trace,
            responsibilities,
            switch_pair_responsibilities or None,
            switch,
            switch_parameter_distributions,
            cfg,
        )

    threshold_distribution = (
        switch_parameter_distributions[0]
        if switch_parameter_distributions
        else GaussianScalar(_switch_default_threshold(switch), 1.0)
    )
    student = ProbabilisticCartpoleStudent(
        action_distributions,
        switch,
        threshold_distribution,
        switch_parameter_distributions,
        responsibilities,
        transition_switches,
        transition_switch_parameter_distributions,
        list(switch_pair_responsibilities),
    )
    return student, fit_history


def _student_fit_step(
    traces: List[CartpoleTrace],
    em_iteration: int,
    responsibility_pass: int,
    phase: str,
    responsibilities: List[Tuple[float, float]],
    switch_pair_responsibilities: List[Tuple[float, float, float, float]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None,
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None = None,
) -> CartpoleStudentFitStep:
    trace_log_likelihood = _student_fit_trace_log_likelihood(
        traces,
        responsibilities,
        action_distributions,
        switch,
        switch_parameter_distributions,
        transition_switches,
        transition_switch_parameter_distributions,
    )
    return CartpoleStudentFitStep(
        em_iteration=em_iteration,
        responsibility_pass=responsibility_pass,
        phase=phase,
        trace_log_likelihood=trace_log_likelihood,
        mean_trace_log_likelihood=(
            trace_log_likelihood / len(traces)
            if traces
            else 0.0
        ),
        responsibilities=list(responsibilities),
        switch_pair_responsibilities=list(switch_pair_responsibilities),
        action_distributions=dict(action_distributions),
        switch=switch,
        switch_parameter_distributions=list(switch_parameter_distributions),
        transition_switches=dict(transition_switches or {}),
        transition_switch_parameter_distributions={
            transition: list(distributions)
            for transition, distributions in (transition_switch_parameter_distributions or {}).items()
        },
    )


def _student_fit_trace_log_likelihood(
    traces: List[CartpoleTrace],
    responsibilities: List[Tuple[float, float]],
    action_distributions: Dict[int, GaussianScalar],
    switch: SwitchProgram,
    switch_parameter_distributions: List[GaussianScalar],
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None,
    transition_switch_parameter_distributions: Dict[Tuple[int, int], List[GaussianScalar]] | None = None,
) -> float:
    threshold_distribution = (
        switch_parameter_distributions[0]
        if switch_parameter_distributions
        else GaussianScalar(_switch_default_threshold(switch), 1.0)
    )
    student = ProbabilisticCartpoleStudent(
        dict(action_distributions),
        switch,
        threshold_distribution,
        list(switch_parameter_distributions),
        list(responsibilities),
        dict(transition_switches or {}),
        {
            transition: list(distributions)
            for transition, distributions in (transition_switch_parameter_distributions or {}).items()
        },
    )
    return sum(_trace_log_probability(trace, student) for trace in traces)


def _next_cartpole_mode(
    current_mode: int,
    observation: Observation,
    selector_switch: SwitchProgram,
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None = None,
) -> int:
    if transition_switches:
        transition = (current_mode, 1 - current_mode)
        switch = transition_switches.get(transition)
        if switch is not None and switch.decide(observation) == 1:
            return 1 - current_mode
        return current_mode
    return selector_switch.decide(observation)


def _deterministic_transition_switches(
    student: ProbabilisticCartpoleStudent,
) -> Dict[Tuple[int, int], SwitchProgram]:
    if not student.transition_switches:
        return {}
    distributions_by_transition = student.transition_switch_parameter_distributions or {}
    return {
        transition: _switch_with_distribution_means(
            switch,
            distributions_by_transition.get(transition, []),
        )
        for transition, switch in student.transition_switches.items()
    }


def _sample_transition_switches(
    student: ProbabilisticCartpoleStudent,
    rng: random.Random,
) -> Dict[Tuple[int, int], SwitchProgram]:
    if not student.transition_switches:
        return {}
    distributions_by_transition = student.transition_switch_parameter_distributions or {}
    return {
        transition: _sample_switch(
            switch,
            distributions_by_transition.get(transition, []),
            rng,
        )
        for transition, switch in student.transition_switches.items()
    }


def _transition_switch_descriptions(
    transition_switches: Dict[Tuple[int, int], SwitchProgram] | None,
) -> Dict[str, str]:
    if not transition_switches:
        return {}
    return {
        f"{source}->{target}": _directed_transition_description(source, target, switch)
        for (source, target), switch in sorted(transition_switches.items())
    }


def _directed_transition_description(source: int, target: int, switch: SwitchProgram) -> str:
    return f"fire {source}->{target} if {_switch_condition_description(switch)}"


def _switch_condition_description(switch: SwitchProgram) -> str:
    if isinstance(switch, Depth2Switch):
        return (
            f"{switch.theta_weight:.3f}*theta + {switch.omega_weight:.3f}*omega "
            f">= {switch.threshold:.3f}"
        )
    if isinstance(switch, LinearObservationSwitch):
        return switch.describe().removeprefix("mode=1 if ").removesuffix(", else mode=0")
    if isinstance(switch, BooleanTreeSwitch):
        if switch.second is None:
            return switch.first.describe()
        return f"{switch.first.describe()} {switch.operator} {switch.second.describe()}"
    return switch.describe()


def _action_likelihood_responsibilities(
    segments: List[CartpoleSegment],
    action_distributions: Dict[int, GaussianScalar],
) -> List[Tuple[float, float]]:
    # The actions are observed, but their latent mode assignments are softened
    # so ambiguous segments can influence both constant-action primitives.
    return [
        _mode_responsibilities(segment.action_parameter, action_distributions)
        for segment in segments
    ]


def _condition_initial_mode_responsibilities(
    segments_by_trace: List[List[CartpoleSegment]],
    responsibilities: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    conditioned = list(responsibilities)
    offset = 0
    for trace_segments in segments_by_trace:
        if trace_segments and offset < len(conditioned):
            conditioned[offset] = (1.0, 0.0)
        offset += len(trace_segments)
    return conditioned


def _bootstrap_probabilistic_student(cfg: CartpoleSynthesisConfig) -> ProbabilisticCartpoleStudent:
    threshold = GaussianScalar(TEACHER_BOOTSTRAP_SWITCH_THRESHOLD, TEACHER_BOOTSTRAP_SWITCH_STD)
    return ProbabilisticCartpoleStudent(
        action_distributions={
            0: GaussianScalar(min(cfg.force_values), TEACHER_BOOTSTRAP_ACTION_STD),
            1: GaussianScalar(max(cfg.force_values), TEACHER_BOOTSTRAP_ACTION_STD),
        },
        switch=Depth2Switch(
            TEACHER_BOOTSTRAP_SWITCH_THETA_WEIGHT,
            TEACHER_BOOTSTRAP_SWITCH_OMEGA_WEIGHT,
            TEACHER_BOOTSTRAP_SWITCH_THRESHOLD,
        ),
        switch_threshold_distribution=threshold,
        switch_parameter_distributions=[threshold],
        responsibilities=[],
    )
