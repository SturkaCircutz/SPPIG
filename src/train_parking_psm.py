"""Train a programmatic state machine on the parking benchmark."""

from __future__ import annotations

from adaptive_teaching_sim import (  # noqa: F401
    LoopFreeProgram,
    compact_trace,
    distill_student,
    evaluate,
    evaluate_baseline,
    main,
    parse_args,
    plot_success_rates,
    plot_trajectories,
    run_experiment,
    save_json,
    serialize_task,
    serialize_trajectories,
    serialize_trajectory,
    simulate_loop_free,
    simulate_policy,
    summarize_traces,
    teacher_optimize_task,
    verify_metrics,
)


if __name__ == "__main__":
    raise SystemExit(main())
