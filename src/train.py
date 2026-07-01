from teacher import optimize_teacher
from student import fit_student_from_traces
from adaptive_teaching_sim import evaluate


def alternating_train(env, initial_student, grammar, teacher_cfg, student_cfg, num_outer_iters):
    student = initial_student
    history = []
    for iteration in range(num_outer_iters):
        traces = optimize_teacher(
            env["tasks"],
            env["reuse"],
            env["rng"],
            teacher_cfg,
            student,
        )
        student = fit_student_from_traces(traces, grammar, student_cfg)
        train_eval = evaluate(env["tasks"], student, env["reuse"])
        history.append(
            {
                "iteration": iteration,
                "num_traces": len(traces),
                "train_success_rate": sum(trace.success for trace in train_eval) / max(1, len(train_eval)),
            }
        )
    return student, history
