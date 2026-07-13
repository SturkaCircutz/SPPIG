from teacher import optimize_teacher
from student import fit_student_from_traces


def alternating_train(env, initial_student, grammar, teacher_cfg, student_cfg, num_outer_iters):
    student = initial_student
    history = []
    for iteration in range(num_outer_iters):
        traces = optimize_teacher(env, student, teacher_cfg)
        student = fit_student_from_traces(traces, grammar, student_cfg)
        avg_reward = sum(trace.reward for trace in traces) / len(traces) if traces else 0.0
        history.append(
            {
                "iteration": iteration,
                "num_traces": len(traces),
                "avg_teacher_reward": avg_reward,
            }
        )
    return student, history
