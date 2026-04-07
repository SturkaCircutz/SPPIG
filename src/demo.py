"""Runnable demonstration of the simplified adaptive-teaching prototype."""

import argparse
from typing import Dict, Iterable, List, Tuple

from psm import ConstantAction, Mode, ProgrammaticStateMachine, describe_policy
from student import StudentConfig
from teacher import TeacherConfig
from toy_env import RepetitionToyEnv, RolloutSummary, ToyState
from train import alternating_train


def build_constant_policy(name: str, value: float) -> ProgrammaticStateMachine:
    """Create a one-mode baseline policy with a constant scalar action."""

    return ProgrammaticStateMachine(
        modes={name: Mode(name=name, action_fn=ConstantAction([value]))},
        start_mode=name,
        end_mode=name,
    )


def make_initial_student() -> ProgrammaticStateMachine:
    """Return the intentionally weak student used to bootstrap training."""

    return build_constant_policy(name="mode_0", value=0.0)


def evaluate_candidates(
    env: RepetitionToyEnv,
    candidates: Iterable[Tuple[str, ProgrammaticStateMachine]],
) -> List[Dict[str, object]]:
    """Evaluate several policies on the train and test splits."""

    rows: List[Dict[str, object]] = []
    for label, policy in candidates:
        row: Dict[str, object] = {"label": label}
        row.update(env.evaluate_policy(policy, split="train"))
        row.update(env.evaluate_policy(policy, split="test"))
        rows.append(row)
    return rows


def format_metrics_table(rows: List[Dict[str, object]]) -> str:
    """Render evaluation metrics as a compact plain-text table."""

    headers = [
        ("label", "policy"),
        ("train_success_rate", "train succ"),
        ("test_success_rate", "test succ"),
        ("train_avg_return", "train ret"),
        ("test_avg_return", "test ret"),
        ("test_avg_steps", "test steps"),
    ]

    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_rows.append(
            [
                str(row["label"]),
                f"{float(row['train_success_rate']):.2f}",
                f"{float(row['test_success_rate']):.2f}",
                f"{float(row['train_avg_return']):.2f}",
                f"{float(row['test_avg_return']):.2f}",
                f"{float(row['test_avg_steps']):.2f}",
            ]
        )

    widths = []
    for index, (_, header) in enumerate(headers):
        widths.append(max(len(header), *(len(row[index]) for row in formatted_rows)))

    lines = [
        "  ".join(header.ljust(widths[index]) for index, (_, header) in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in formatted_rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def format_history(history: List[Dict[str, float]]) -> str:
    """Render the outer-loop metrics in a readable multi-line format."""

    lines = []
    for metrics in history:
        lines.append(
            "iter={iteration} traces={num_traces} avg_trace_reward={avg_trace_reward:.2f} "
            "train_succ={train_success_rate:.2f} test_succ={test_success_rate:.2f}".format(**metrics)
        )
    return "\n".join(lines)


def format_rollout(summary: RolloutSummary) -> str:
    """Render a rollout trace so the learned repetition pattern is easy to inspect."""

    lines = [
        (
            "initial remaining_steps={remaining} success={success} total_reward={reward:.2f}"
        ).format(
            remaining=summary.initial_state.remaining_steps,
            success=summary.success,
            reward=summary.total_reward,
        )
    ]
    for step in summary.steps:
        lines.append(
            "t={t:02d} mode={mode} action={action:.1f} obs=[{remaining:.0f}, {armed:.0f}] "
            "reward={reward:.1f} next_mode={next_mode} next_remaining={next_remaining}".format(
                t=step.step_index,
                mode=step.mode_name,
                action=step.action[0],
                remaining=step.observation[0],
                armed=step.observation[1],
                reward=step.reward,
                next_mode=step.next_mode_name,
                next_remaining=step.next_state.remaining_steps,
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Expose a few knobs so the prototype is easy to rerun and inspect."""

    parser = argparse.ArgumentParser(description="Run the toy adaptive-teaching prototype.")
    parser.add_argument("--seed", type=int, default=7, help="Seed used by the toy environment.")
    parser.add_argument("--outer-iters", type=int, default=3, help="Number of alternating optimization rounds.")
    parser.add_argument("--num-traces", type=int, default=12, help="Teacher traces collected per round.")
    parser.add_argument("--max-steps", type=int, default=16, help="Maximum teacher rollout horizon.")
    return parser.parse_args()


def main() -> None:
    """Train the toy student, compare it with baselines, and print example rollouts."""

    args = parse_args()
    env = RepetitionToyEnv(seed=args.seed)
    teacher_cfg = TeacherConfig(num_traces=args.num_traces, max_steps=args.max_steps, student_action_weight=0.0)
    student_cfg = StudentConfig(num_modes=3, min_guard_examples=1)

    student, history = alternating_train(
        env=env,
        initial_student=make_initial_student(),
        grammar=None,
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        num_outer_iters=args.outer_iters,
    )

    baselines = [
        ("initial_zero", make_initial_student()),
        ("always_arm", build_constant_policy(name="always_arm", value=1.0)),
        ("learned_student", student),
    ]
    evaluation_rows = evaluate_candidates(env, baselines)

    print("Training history:")
    print(format_history(history))

    print("\nBaseline comparison:")
    print(format_metrics_table(evaluation_rows))

    print("\nLearned policy:")
    print(describe_policy(student))

    print("\nExample train rollout:")
    print(format_rollout(env.rollout_policy(student, ToyState(remaining_steps=3, armed=False))))

    print("\nExample test rollout:")
    print(format_rollout(env.rollout_policy(student, ToyState(remaining_steps=6, armed=False))))


if __name__ == "__main__":
    main()
