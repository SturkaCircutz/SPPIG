from __future__ import annotations

from shutil import get_terminal_size

from shuttle_env import ShuttleLineEnv
from student import StudentConfig
from teacher import TeacherConfig
from train import alternating_train


def main() -> None:
    env = ShuttleLineEnv(length=5, train_crossings=[2, 3], test_crossings=[6, 8])
    teacher_cfg = TeacherConfig(num_traces=8, random_seed=13, max_steps=200)
    student_cfg = StudentConfig(num_modes=3, action_grammar="constant", switch_grammar="axis_threshold")

    policy, history = alternating_train(
        env=env,
        initial_student=None,
        grammar=None,
        teacher_cfg=teacher_cfg,
        student_cfg=student_cfg,
        num_outer_iters=3,
    )

    print("Training history")
    for item in history:
        print(
            f"  iter={item['iteration']} traces={item['num_traces']} "
            f"avg_teacher_reward={item['avg_teacher_reward']:.3f}"
        )

    print("\nLearned policy")
    print(policy.describe())

    print("\nEvaluation")
    width = max(72, get_terminal_size((72, 20)).columns)
    print("-" * min(width, 96))
    print("crossings | split | success | steps | reward | final_mode")
    print("-" * min(width, 96))
    for split, crossings_set in (("train", env.train_crossings), ("test", env.test_crossings)):
        for crossings in crossings_set:
            result = env.evaluate_policy(policy, crossings)
            print(
                f"{crossings:9d} | {split:5s} | {str(result.success):7s} | "
                f"{result.steps:5d} | {result.reward:.1f}    | {result.final_mode}"
            )


if __name__ == "__main__":
    main()
