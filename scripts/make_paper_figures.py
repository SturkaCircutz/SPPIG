from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cartpole_env import PAPER_EVAL_ROLLOUTS  # noqa: E402

RESULTS_CSV = os.path.join(ROOT, "artifacts", "results", "cartpole_results.csv")
SUMMARY_CSV = os.path.join(ROOT, "artifacts", "results", "cartpole_summary.csv")
OUT_DIR = os.path.join(ROOT, "essay", "figures")
TABLE_TEX = os.path.join(ROOT, "essay", "cartpole_results_table.tex")
POLICY_TEX = os.path.join(ROOT, "essay", "cartpole_policy_fragment.tex")
FIGURE19_TEX = os.path.join(ROOT, "essay", "cartpole_figure19_reference_fragment.tex")
ABSTRACT_RESULTS_TEX = os.path.join(ROOT, "essay", "cartpole_abstract_results.tex")
PPO_METRICS_GLOBS = [
    os.path.join(ROOT, "artifacts", "cartpole_ppo_*_metrics.json"),
    os.path.join(ROOT, "artifacts", "results", "metrics", "*.json"),
    os.path.join(ROOT, "artifacts", "ppo_sweep", "metrics", "*.json"),
]
PSM_METRICS_GLOBS = [
    os.path.join(ROOT, "artifacts", "cartpole_psm*_metrics.json"),
    os.path.join(ROOT, "artifacts", "results", "metrics", "psm_seed*.json"),
]
FIGURE19_METRICS_GLOBS = [
    os.path.join(ROOT, "artifacts", "results", "metrics", "figure19*.json"),
]
LINEAR_SWITCH_RE = re.compile(
    r"mode=1 if (?P<theta>[-+]?\d+(?:\.\d+)?)\*theta \+ "
    r"(?P<omega>[-+]?\d+(?:\.\d+)?)\*omega >= "
    r"(?P<threshold>[-+]?\d+(?:\.\d+)?)"
)
PAPER_TEST_HORIZON_STEPS = 15_000
LOCAL_DIAGNOSTIC_NOTE = (
    "Local diagnostic artifacts only; not a paper-scale reproduction of the "
    "10^7-timestep, five-seed, 1000-rollout PPO/PPO-LSTM protocol."
)
LOCAL_DIAGNOSTIC_TEX_NOTE = LOCAL_DIAGNOSTIC_NOTE.replace("10^7", r"10\textsuperscript{7}")


def read_results() -> list[dict[str, str]]:
    path = SUMMARY_CSV if os.path.exists(SUMMARY_CSV) else RESULTS_CSV
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(row: dict[str, str], name: str) -> float:
    return float(row.get(f"{name}_mean") or row[name])


def metric_or_none(row: dict[str, str], name: str) -> float | None:
    value = row.get(f"{name}_mean") or row.get(name)
    return float(value) if value not in (None, "") else None


def artifact_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def truthy_csv(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def row_metrics_path(row: dict[str, str]) -> str:
    return row.get("best_metrics_output") or row.get("metrics_output") or ""


def row_traces_path(row: dict[str, str]) -> str:
    return row.get("best_traces_output") or row.get("traces_output") or ""


def row_has_result_artifact(row: dict[str, str]) -> bool:
    path = row_metrics_path(row) or row.get("checkpoint")
    return bool(path and os.path.exists(artifact_path(path)))


def require_result_artifacts(rows: list[dict[str, str]]) -> None:
    missing = [row["policy"] for row in rows if not row_has_result_artifact(row)]
    if missing:
        raise FileNotFoundError(
            "missing result artifacts for generated paper claims: "
            + ", ".join(missing)
        )
    non_paper_horizon = [
        row["policy"]
        for row in rows
        if not row.get("test_horizon_steps")
        or int(float(row["test_horizon_steps"])) != PAPER_TEST_HORIZON_STEPS
    ]
    if non_paper_horizon:
        raise ValueError(
            "result rows lack paper 300-second test-horizon provenance: "
            + ", ".join(non_paper_horizon)
        )
    missing_rollout_provenance = [
        row["policy"]
        for row in rows
        if not row.get("eval_rollouts")
    ]
    if missing_rollout_provenance:
        raise ValueError(
            "result rows lack evaluation-rollout provenance: "
            + ", ".join(missing_rollout_provenance)
        )
    missing_protocol_status: list[str] = []
    for row in rows:
        metrics_path = row_metrics_path(row)
        if not metrics_path:
            missing_protocol_status.append(row["policy"])
            continue
        with open(artifact_path(metrics_path), encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "paper_protocol_status" not in metrics:
            missing_protocol_status.append(row["policy"])
    if missing_protocol_status:
        raise ValueError(
            "result metrics lack paper-protocol status: "
            + ", ".join(missing_protocol_status)
        )
    missing_psm_trace_artifacts: list[str] = []
    incomplete_psm_trace_artifacts: list[str] = []
    for row in rows:
        if row.get("policy") != "Synthesized PSM diagnostic":
            continue
        metrics_path = row_metrics_path(row)
        trace_path = row_traces_path(row)
        if not trace_path:
            if metrics_path:
                with open(artifact_path(metrics_path), encoding="utf-8") as handle:
                    trace_path = json.load(handle).get("traces_output") or ""
        if not trace_path or not os.path.exists(artifact_path(trace_path)):
            missing_psm_trace_artifacts.append(row["policy"])
            continue
        with open(artifact_path(trace_path), encoding="utf-8") as handle:
            trace_payload = json.load(handle)
        trace_history = trace_payload.get("trace_history")
        if (
            not isinstance(trace_history, list)
            or not trace_history
            or trace_history[-1].get("traces") != trace_payload.get("traces")
        ):
            incomplete_psm_trace_artifacts.append(row["policy"])
    if missing_psm_trace_artifacts:
        raise FileNotFoundError(
            "synthesized PSM rows lack full teacher-trace artifacts: "
            + ", ".join(missing_psm_trace_artifacts)
        )
    if incomplete_psm_trace_artifacts:
        raise ValueError(
            "synthesized PSM trace artifacts lack per-iteration trace history: "
            + ", ".join(incomplete_psm_trace_artifacts)
        )
    paper_scale_rollout_mismatch = [
        row["policy"]
        for row in rows
        if truthy_csv(row.get("paper_scale_result"))
        and int(float(row["eval_rollouts"])) != PAPER_EVAL_ROLLOUTS
    ]
    if paper_scale_rollout_mismatch:
        raise ValueError(
            "paper-scale result rows must use 1000 evaluation rollouts: "
            + ", ".join(paper_scale_rollout_mismatch)
        )


def display_policy(name: str) -> str:
    return name.replace("Programmatic state machine", "Programmatic PSM").replace("PPO-LSTM", "PPO-LSTM")


def find_policy_row(rows: list[dict[str, str]], policy_name: str) -> dict[str, str] | None:
    for row in rows:
        if row["policy"] == policy_name:
            return row
    return None


def percent(value: float) -> str:
    return f"{100.0 * value:.0f}\\%"


def write_results_table(rows: list[dict[str, str]], outpath: str = TABLE_TEX) -> None:
    lines = [
        "% Generated by scripts/make_paper_figures.py from CartPole result artifacts.",
        f"% {LOCAL_DIAGNOSTIC_NOTE}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & Train succ. & Test succ. & Train rew. & Test rew. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{display_policy(row['policy'])} & "
            f"{metric(row, 'train_success'):.2f} & "
            f"{metric(row, 'test_success'):.2f} & "
            f"{metric(row, 'train_reward'):.1f} & "
            f"{metric(row, 'test_reward'):.1f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\par\\smallskip\\noindent\\emph{{Note:}} {LOCAL_DIAGNOSTIC_TEX_NOTE}",
            "",
        ]
    )
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_abstract_results(rows: list[dict[str, str]], outpath: str = ABSTRACT_RESULTS_TEX) -> bool:
    ppo = find_policy_row(rows, "PPO MLP")
    psm = find_policy_row(rows, "Programmatic state machine")
    if ppo is None or psm is None:
        lines = [
            "% Generated by scripts/make_paper_figures.py; required result rows were unavailable.",
            "Local diagnostic result rows were unavailable for this build.",
            "",
        ]
        with open(outpath, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return False

    ppo_train = metric(ppo, "train_success")
    ppo_test = metric(ppo, "test_success")
    ppo_test_reward = metric(ppo, "test_reward")
    psm_train = metric(psm, "train_success")
    psm_test = metric(psm, "test_success")
    psm_test_reward = metric(psm, "test_reward")
    lines = [
        "% Generated by scripts/make_paper_figures.py from CartPole result artifacts.",
        f"% {LOCAL_DIAGNOSTIC_NOTE}",
        (
            f"In local diagnostics, feed-forward PPO reaches {percent(ppo_train)} training success "
            f"and obtains {percent(ppo_test)} success on the full 300-second test horizon with "
            f"mean test reward {ppo_test_reward:.1f}. The fixed programmatic state machine reaches "
            f"{percent(psm_train)} training success, obtains {percent(psm_test)} full-horizon "
            f"test success, and has mean test reward {psm_test_reward:.1f}."
        ),
        "",
    ]
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return True


def plot_success_bars(rows: list[dict[str, str]]) -> None:
    labels = [row["policy"].replace("Programmatic state machine", "Programmatic PSM") for row in rows]
    train = [metric(row, "train_success") for row in rows]
    test = [metric(row, "test_success") for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - width / 2, train, width, label="Train: 5s, len=0.5", color="#3b6ea8")
    ax.bar(x + width / 2, test, width, label="Test: 300s, len=1.0", color="#b84a4a")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cartpole_success_rates.png"), dpi=220)
    plt.close(fig)


def plot_survival_rewards(rows: list[dict[str, str]]) -> None:
    labels = [row["policy"].replace("Programmatic state machine", "Programmatic PSM") for row in rows]
    rewards = [metric_or_none(row, "test_steps") or metric(row, "test_reward") for row in rows]
    palette = ["#6f8fb8", "#8aa777", "#c58b47", "#9b6fa8", "#c76f5b"]
    colors = [palette[index % len(palette)] for index in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(np.arange(len(labels)), rewards, color=colors)
    ax.axhline(15_000, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.text(2.2, 14_650, "full 300s horizon", ha="right", va="top", fontsize=8)
    ax.set_ylabel("Mean test survival reward")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cartpole_test_survival_reward.png"), dpi=220)
    plt.close(fig)


def read_psm_metric_files(patterns: list[str] | None = None) -> list[dict[str, object]]:
    metric_files: list[dict[str, object]] = []
    for pattern in patterns or PSM_METRICS_GLOBS:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("policy_description"):
                metric_files.append({"path": path, "payload": payload})
    return sorted(metric_files, key=psm_metric_priority)


def psm_metric_priority(metric_file: dict[str, object]) -> tuple[int, str]:
    payload = metric_file["payload"]
    status = payload.get("paper_protocol_status", {})
    if (
        status.get("policy_source") == "fixed_two_mode_program_parameters"
        and status.get("uses_full_test_horizon") is True
    ):
        return (0, str(metric_file["path"]))
    if status:
        return (1, str(metric_file["path"]))
    return (2, str(metric_file["path"]))


def read_figure19_metric_files(patterns: list[str] | None = None) -> list[dict[str, object]]:
    metric_files: list[dict[str, object]] = []
    for pattern in patterns or FIGURE19_METRICS_GLOBS:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            status = payload.get("paper_protocol_status", {})
            parameters = payload.get("program_parameters", {})
            if (
                status.get("policy_source") == "paper_figure19_manual_transcription"
                and parameters.get("figure") == "SPPIG paper Figure 19"
            ):
                metric_files.append({"path": path, "payload": payload})
    return metric_files


def parse_linear_switch(description: str) -> tuple[float, float, float] | None:
    match = LINEAR_SWITCH_RE.search(description)
    if match is None:
        return None
    theta_weight = float(match.group("theta"))
    omega_weight = float(match.group("omega"))
    threshold = float(match.group("threshold"))
    if omega_weight == 0.0:
        return None
    return theta_weight, omega_weight, threshold


def linear_switch_latex(linear_switch: tuple[float, float, float]) -> str:
    theta_weight, omega_weight, threshold = linear_switch
    sign = "+" if omega_weight >= 0.0 else "-"
    return f"{theta_weight:g}\\theta_t {sign} {abs(omega_weight):g}\\dot{{\\theta}}_t \\ge {threshold:g}"


def linear_switch_mathtext(linear_switch: tuple[float, float, float]) -> str:
    return linear_switch_latex(linear_switch).replace("_t", "").replace("\\ge", "\\geq")


def mode1_region_is_above_boundary(linear_switch: tuple[float, float, float]) -> bool:
    _, omega_weight, _ = linear_switch
    return omega_weight > 0.0


def write_policy_fragment(metric_files: list[dict[str, object]], outpath: str = POLICY_TEX) -> bool:
    for metric_file in metric_files:
        description = str(metric_file["payload"].get("policy_description", ""))
        linear_switch = parse_linear_switch(description)
        if linear_switch is None:
            continue
        lines = [
            "% Generated by scripts/make_paper_figures.py from a PSM metrics artifact.",
            "\\[",
            "  a(x_t) =",
            "  \\begin{cases}",
            f"    +10, & {linear_switch_latex(linear_switch)},\\\\",
            "    -10, & \\text{otherwise}.",
            "  \\end{cases}",
            "\\]",
            "",
        ]
        with open(outpath, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return True
    lines = [
        "% Generated by scripts/make_paper_figures.py; no linear PSM metrics artifact was available.",
        "No linear programmatic switch metrics artifact was available for this diagnostic run.",
        "",
    ]
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return False


def figure19_expr_latex(expr: str) -> str:
    return (
        expr.replace("theta", r"\theta_t")
        .replace("omega", r"\dot{\theta}_t")
        .replace(" >= ", r" \ge ")
        .replace(" and ", r" \wedge ")
    )


def texttt_latex(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_figure19_reference_fragment(
    metric_files: list[dict[str, object]],
    outpath: str = FIGURE19_TEX,
) -> bool:
    for metric_file in metric_files:
        payload = metric_file["payload"]
        parameters = payload.get("program_parameters", {})
        status = payload.get("paper_protocol_status", {})
        modes = parameters.get("modes", {})
        start = parameters.get("start", {})
        if not modes or not start:
            continue
        m1 = modes["m1"]
        m2 = modes["m2"]
        lines = [
            "% Generated by scripts/make_paper_figures.py from a Figure 19 reference metrics artifact.",
            "\\[",
            "  \\begin{aligned}",
            f"    m_0 &\\to m_1 \\text{{ if }} {figure19_expr_latex(start['m1'])},\\\\",
            f"    a_{{m_1}} &= {m1['action']:g},\\quad "
            f"m_1 \\to m_2 \\text{{ if }} {figure19_expr_latex(m1['switch_to_m2'])},\\\\",
            f"    a_{{m_2}} &= {m2['action']:g},\\quad "
            f"m_2 \\to m_1 \\text{{ if }} {figure19_expr_latex(m2['switch_to_m1'])}.",
            "  \\end{aligned}",
            "\\]",
            (
                "\\noindent\\emph{Reference provenance:} manual visual transcription of "
                "SPPIG Figure 19; the metrics mark "
                "\\texttt{synthesized\\_by\\_current\\_algorithm=false} and "
                f"\\texttt{{policy\\_source={texttt_latex(status.get('policy_source'))}}}."
            ),
            "",
        ]
        with open(outpath, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return True
    lines = [
        "% Generated by scripts/make_paper_figures.py; no Figure 19 reference metrics artifact was available.",
        "No manually transcribed Figure 19 CartPole reference metrics artifact was available for this build.",
        "",
    ]
    with open(outpath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return False


def plot_switch_boundary(metric_files: list[dict[str, object]], outpath: str | None = None) -> bool:
    linear_switch = None
    for metric_file in metric_files:
        description = str(metric_file["payload"].get("policy_description", ""))
        linear_switch = parse_linear_switch(description)
        if linear_switch is not None:
            break
    if linear_switch is None:
        return False
    theta_weight, omega_weight, threshold = linear_switch
    theta = np.linspace(-0.22, 0.22, 200)
    omega = (threshold - theta_weight * theta) / omega_weight

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(theta, omega, color="#2f2f2f", linewidth=2.0)
    if mode1_region_is_above_boundary(linear_switch):
        ax.fill_between(theta, omega, 2.5, color="#3b6ea8", alpha=0.15, label="push right")
        ax.fill_between(theta, -2.5, omega, color="#b84a4a", alpha=0.15, label="push left")
    else:
        ax.fill_between(theta, -2.5, omega, color="#3b6ea8", alpha=0.15, label="push right")
        ax.fill_between(theta, omega, 2.5, color="#b84a4a", alpha=0.15, label="push left")
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel(r"pole angle $\theta$")
    ax.set_ylabel(r"angular velocity $\dot{\theta}$")
    ax.set_title(
        rf"Programmatic switch: ${linear_switch_mathtext(linear_switch)}$"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath or os.path.join(OUT_DIR, "programmatic_switch_boundary.png"), dpi=220)
    plt.close(fig)
    return True


def read_ppo_metric_files(patterns: list[str] | None = None) -> list[dict[str, object]]:
    metric_files: list[dict[str, object]] = []
    for pattern in patterns or PPO_METRICS_GLOBS:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            history = payload.get("eval_history", [])
            if history:
                metric_files.append({"path": path, "payload": payload})
    return metric_files


def metric_label(metric_file: dict[str, object]) -> str:
    payload = metric_file["payload"]
    config = payload.get("config", {})
    policy = str(config.get("policy_type", "ppo")).upper()
    seed = config.get("seed")
    return f"{policy} seed {seed}" if seed is not None else policy


def plot_ppo_training_curves(metric_files: list[dict[str, object]], outpath: str | None = None) -> bool:
    if not metric_files:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for metric_file in metric_files:
        payload = metric_file["payload"]
        history = payload["eval_history"]
        timesteps = [entry["timesteps"] for entry in history]
        train_success = [entry["train_success_rate"] for entry in history]
        test_success = [entry["test_success_rate"] for entry in history]
        label = metric_label(metric_file)
        ax.plot(timesteps, train_success, linewidth=1.8, label=f"{label} train")
        ax.plot(timesteps, test_success, linewidth=1.4, linestyle="--", label=f"{label} test")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Success rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath or os.path.join(OUT_DIR, "cartpole_ppo_training_curves.png"), dpi=220)
    plt.close(fig)
    return True


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_results()
    require_result_artifacts(rows)
    psm_metric_files = read_psm_metric_files()
    figure19_metric_files = read_figure19_metric_files()
    write_results_table(rows)
    write_abstract_results(rows)
    write_policy_fragment(psm_metric_files)
    write_figure19_reference_fragment(figure19_metric_files)
    plot_success_bars(rows)
    plot_survival_rewards(rows)
    plot_switch_boundary(psm_metric_files)
    plot_ppo_training_curves(read_ppo_metric_files())


if __name__ == "__main__":
    main()
