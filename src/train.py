"""High-level alternating training loop for the simplified prototype."""

from teacher import optimize_teacher
from student import fit_student_from_traces


def alternating_train(env, initial_student, grammar, teacher_cfg, student_cfg, num_outer_iters):
    """Alternate between teacher trace collection and student policy fitting."""

    student = initial_student
    history = []
    for iteration in range(num_outer_iters):
        traces = optimize_teacher(env, student, teacher_cfg)
        student = fit_student_from_traces(traces, grammar, student_cfg)
        metrics = {
            "iteration": iteration,
            "num_traces": len(traces),
            "avg_trace_reward": sum(trace.reward for trace in traces) / max(len(traces), 1),
        }
        if hasattr(env, "evaluate_policy"):
            metrics.update(env.evaluate_policy(student, split="train"))
            metrics.update(env.evaluate_policy(student, split="test"))
        history.append(metrics)
    return student, history
