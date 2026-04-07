from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "ink": "#10243E",
    "slate": "#5F6C7B",
    "sand": "#F4EFE6",
    "card": "#FFFCF7",
    "shadow": "#D7D0C6",
    "train": "#1C8C73",
    "test": "#C65A3A",
    "agent": "#2F6BFF",
    "gold": "#E0A33A",
    "sky": "#DDEAF7",
    "rose": "#F9E2DA",
    "mint": "#DDF1E8",
    "line": "#CAD3DD",
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
    }
)


def _add_card(ax, x, y, w, h, facecolor=PALETTE["card"], edgecolor=PALETTE["ink"], shadow=True):
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.10, y - 0.10),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.28",
                linewidth=0,
                facecolor=PALETTE["shadow"],
                alpha=0.35,
                zorder=0,
            )
        )

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.28",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=1,
    )
    ax.add_patch(card)
    return card


def _add_chip(ax, x, y, text, facecolor, textcolor="white", fontsize=8):
    chip = FancyBboxPatch(
        (x, y),
        1.20,
        0.42,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=0,
        facecolor=facecolor,
        zorder=5,
    )
    ax.add_patch(chip)
    ax.text(x + 0.60, y + 0.21, text, ha="center", va="center", fontsize=fontsize, color=textcolor, weight="bold", zorder=6)


def _add_car(ax, x, y, w, h, color, wheel="#283240", z=4):
    body = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0,
        edgecolor="white",
        facecolor=color,
        zorder=z,
    )
    ax.add_patch(body)
    for wx in (x + 0.28, x + w - 0.28):
        ax.add_patch(Circle((wx, y - 0.05), 0.08, color=wheel, zorder=z + 1))
        ax.add_patch(Circle((wx, y + h + 0.05), 0.08, color=wheel, zorder=z + 1))
    return body


def _draw_gap_marker(ax, x0, x1, y, label, color):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="<->", lw=1.4, color=color, shrinkA=0, shrinkB=0),
        zorder=7,
    )
    ax.text((x0 + x1) / 2, y + 0.18, label, ha="center", va="bottom", fontsize=8, color=color, weight="bold", zorder=7)


def _draw_parking_panel(ax, gap, title, subtitle, accent, chip_text):
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    _add_card(ax, 0.15, 0.15, 9.7, 6.2, facecolor=PALETTE["card"])
    ax.add_patch(FancyBboxPatch((0.55, 1.00), 8.9, 3.8, boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=0, facecolor=PALETTE["sand"], zorder=2))
    ax.plot([0.75, 9.20], [1.55, 1.55], color=PALETTE["line"], lw=1.8, zorder=3)
    ax.plot([0.75, 9.20], [4.25, 4.25], color=PALETTE["line"], lw=1.8, zorder=3)
    ax.plot([1.00, 9.00], [2.90, 2.90], color="#D8CFC2", lw=1.0, ls=(0, (4, 4)), zorder=3)

    _add_chip(ax, 0.65, 5.55, chip_text, accent)
    ax.text(2.05, 5.72, title, fontsize=11, weight="bold", color=PALETTE["ink"], va="center", zorder=6)
    ax.text(0.70, 5.05, subtitle, fontsize=8.5, color=PALETTE["slate"], zorder=6)

    front_x = 6.05 + gap / 2.0
    back_x = 6.05 - gap / 2.0 - 1.75
    _add_car(ax, front_x, 2.25, 1.75, 0.95, color=PALETTE["ink"], z=4)
    _add_car(ax, back_x, 2.25, 1.75, 0.95, color=PALETTE["ink"], z=4)
    _add_car(ax, 4.10, 2.30, 1.55, 0.86, color=PALETTE["agent"], z=5)

    _draw_gap_marker(ax, back_x + 1.75, front_x, 1.15, f"gap {gap:.1f}", accent)

    ax.scatter([4.90], [2.75], s=24, color=PALETTE["agent"], zorder=8)
    ax.add_patch(Circle((8.35, 4.05), 0.18, fill=False, linewidth=1.7, edgecolor=accent, zorder=7))
    ax.add_patch(Circle((8.35, 4.05), 0.05, color=accent, zorder=7))
    ax.add_patch(
        FancyArrowPatch(
            (4.90, 2.75),
            (8.05, 3.90),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2.2,
            color=accent,
            zorder=7,
        )
    )
    ax.text(4.25, 2.00, "start", fontsize=8, color=PALETTE["slate"], zorder=7)
    ax.text(8.10, 4.35, "goal", fontsize=8, color=accent, weight="bold", zorder=7)


def _draw_mode_node(ax, xy, label, face, edge=PALETTE["ink"]):
    ax.add_patch(Circle(xy, 0.11, facecolor=face, edgecolor=edge, linewidth=1.4, zorder=3))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=9, weight="bold", color=PALETTE["ink"], zorder=4)


def make_running_example():
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.08]})

    panels = [
        (2.8, "Train A", "wide parking interval", PALETTE["train"], "TRAIN"),
        (2.2, "Train B", "moderate parking interval", PALETTE["train"], "TRAIN"),
        (1.7, "Train C", "narrow but seen at train time", PALETTE["train"], "TRAIN"),
        (1.1, "Test", "tighter geometry than training", PALETTE["test"], "TEST"),
    ]
    for ax, (gap, title, subtitle, accent, chip_text) in zip(axes[:4], panels):
        _draw_parking_panel(ax, gap=gap, title=title, subtitle=subtitle, accent=accent, chip_text=chip_text)

    ax = axes[4]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _add_card(ax, 0.03, 0.05, 0.94, 0.90, facecolor="#F8FAFD", shadow=True)
    ax.text(0.09, 0.88, "Learned State Machine", fontsize=11, weight="bold", color=PALETTE["ink"], zorder=6)
    ax.text(0.09, 0.82, "A compact controller reuses two maneuver modes", fontsize=8.3, color=PALETTE["slate"], zorder=6)

    nodes = {
        "start": (0.20, 0.55),
        "align": (0.47, 0.74),
        "reverse": (0.47, 0.34),
        "exit": (0.78, 0.55),
    }
    _draw_mode_node(ax, nodes["start"], "S", PALETTE["sky"])
    _draw_mode_node(ax, nodes["align"], "A", "#E6EEF9")
    _draw_mode_node(ax, nodes["reverse"], "R", "#E6EEF9")
    _draw_mode_node(ax, nodes["exit"], "E", PALETTE["mint"])

    arrows = [
        ("start", "align", "enter lane", 0.0),
        ("align", "reverse", "front near", 0.0),
        ("reverse", "align", "rear near", 0.0),
        ("align", "exit", "clear lane", 0.0),
    ]
    for src, dst, label, rad in arrows:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.8,
                color=PALETTE["ink"],
                zorder=2,
            )
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.07, label, fontsize=7.5, color=PALETTE["slate"], ha="center", zorder=5)

    ax.add_patch(
        FancyArrowPatch(
            nodes["reverse"],
            nodes["align"],
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.8,
            color=PALETTE["gold"],
            zorder=2,
        )
    )
    ax.text(0.09, 0.12, "The loop A  R  A  R repeats until the exit guard fires.", fontsize=8.1, color=PALETTE["slate"], zorder=5)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.98, bottom=0.04, wspace=0.16)
    fig.savefig(FIG_DIR / "running_example.png", dpi=280, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_flow_card(ax, x, y, w, h, title, subtitle, accent, chip):
    _add_card(ax, x, y, w, h, facecolor=PALETTE["card"], edgecolor=PALETTE["ink"])
    _add_chip(ax, x + 0.22, y + h - 0.54, chip, accent)
    ax.text(x + 0.22, y + h - 0.82, title, fontsize=11, weight="bold", color=PALETTE["ink"], zorder=6)
    ax.text(x + 0.22, y + h - 1.14, subtitle, fontsize=8.2, color=PALETTE["slate"], zorder=6)


def _draw_teacher_icon(ax, x, y):
    for offset, color in ((0.00, PALETTE["train"]), (0.18, PALETTE["gold"]), (0.36, PALETTE["agent"])):
        ax.add_patch(
            FancyArrowPatch(
                (x, y - offset),
                (x + 1.15, y + 0.18 - offset),
                connectionstyle="arc3,rad=0.2",
                arrowstyle="-|>",
                mutation_scale=12,
                lw=2.1,
                color=color,
                zorder=4,
            )
        )


def _draw_student_icon(ax, x, y):
    ax.add_patch(Circle((x, y), 0.18, facecolor=PALETTE["sky"], edgecolor=PALETTE["ink"], lw=1.2, zorder=4))
    ax.add_patch(Circle((x + 0.62, y + 0.34), 0.18, facecolor="#E6EEF9", edgecolor=PALETTE["ink"], lw=1.2, zorder=4))
    ax.add_patch(Circle((x + 0.62, y - 0.34), 0.18, facecolor="#E6EEF9", edgecolor=PALETTE["ink"], lw=1.2, zorder=4))
    ax.add_patch(Circle((x + 1.22, y), 0.18, facecolor=PALETTE["mint"], edgecolor=PALETTE["ink"], lw=1.2, zorder=4))
    for p1, p2, color in (
        ((x + 0.16, y + 0.02), (x + 0.45, y + 0.26), PALETTE["ink"]),
        ((x + 0.16, y - 0.02), (x + 0.45, y - 0.26), PALETTE["gold"]),
        ((x + 0.80, y + 0.22), (x + 1.04, y + 0.04), PALETTE["train"]),
    ):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.8, color=color, zorder=3))


def make_adaptive_teaching():
    fig, ax = plt.subplots(figsize=(12.8, 4.9))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(0.35, 4.82, "Adaptive Teaching Pipeline", fontsize=15, weight="bold", color=PALETTE["ink"])
    ax.text(0.35, 4.48, "Teacher trajectories stay reward-seeking while the student compresses them into a reusable programmatic policy.", fontsize=9.3, color=PALETTE["slate"])

    _draw_flow_card(ax, 0.35, 1.15, 2.45, 2.55, "Sample Starts", "Draw train-distribution\ninitial states", PALETTE["train"], "DATA")
    for px, py in ((0.95, 2.05), (1.55, 2.55), (2.10, 1.78)):
        ax.add_patch(Circle((px, py), 0.11, color=PALETTE["train"], zorder=4))
    ax.add_patch(Rectangle((0.75, 1.55), 1.65, 0.10, color=PALETTE["line"], zorder=3))

    _draw_flow_card(ax, 3.20, 1.15, 2.90, 2.55, "Teacher Rollouts", "Open-loop trajectories\noptimize local progress", PALETTE["gold"], "TEACHER")
    _draw_teacher_icon(ax, 3.75, 2.55)
    ax.text(3.70, 1.55, "high reward, low-level control", fontsize=8.1, color=PALETTE["slate"], zorder=6)

    _draw_flow_card(ax, 6.55, 1.15, 2.95, 2.55, "Trace Labeling", "Record observations,\nactions, and mode hints", PALETTE["agent"], "TRACE")
    for row, y in enumerate((2.65, 2.15, 1.65)):
        ax.add_patch(FancyBboxPatch((7.00, y), 2.00, 0.24, boxstyle="round,pad=0.01,rounding_size=0.06", linewidth=0, facecolor="#EDF2F8", zorder=3))
        for col, color in enumerate((PALETTE["agent"], PALETTE["gold"], PALETTE["train"], PALETTE["test"])):
            ax.add_patch(Rectangle((7.15 + 0.42 * col, y + 0.04), 0.28, 0.16, linewidth=0, facecolor=color if col <= row else "#D5DEE7", zorder=4))

    _draw_flow_card(ax, 9.95, 1.15, 2.95, 2.55, "Student Fit", "Mean actions + symbolic\nthreshold guards", PALETTE["train"], "STUDENT")
    _draw_student_icon(ax, 10.45, 2.18)
    ax.text(10.36, 1.55, "compact PSM with reusable modes", fontsize=8.1, color=PALETTE["slate"], zorder=6)

    ax.add_patch(
        FancyBboxPatch((10.60, 4.05), 2.45, 0.62, boxstyle="round,pad=0.02,rounding_size=0.18", linewidth=1.1, edgecolor=PALETTE["ink"], facecolor="#F6FAF8")
    )
    ax.text(11.83, 4.36, "student regularizes the next teacher batch", ha="center", va="center", fontsize=8.6, color=PALETTE["ink"])

    for x1, x2 in ((2.82, 3.18), (6.13, 6.53), (9.48, 9.92)):
        ax.add_patch(
            FancyArrowPatch((x1, 2.43), (x2, 2.43), arrowstyle="-|>", mutation_scale=14, lw=2.3, color=PALETTE["ink"], zorder=4)
        )

    ax.add_patch(
        FancyArrowPatch(
            (11.40, 3.78),
            (4.70, 3.78),
            connectionstyle="arc3,rad=0.34",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=2.2,
            color=PALETTE["gold"],
            zorder=3,
        )
    )

    fig.savefig(FIG_DIR / "adaptive_teaching.png", dpi=280, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_car_shift(ax, x, y, scale):
    _add_car(ax, x + 0.15, y + 0.18, 0.85, 0.42, PALETTE["ink"], z=5)
    _add_car(ax, x + 1.45 + scale * 0.12, y + 0.18, 0.85, 0.42, PALETTE["ink"], z=5)
    _add_car(ax, x + 0.95, y + 0.20, 0.72, 0.38, PALETTE["agent"], z=6)
    _draw_gap_marker(ax, x + 1.00, x + 1.45 + scale * 0.12, y + 0.02, "", PALETTE["gold"])


def _draw_quad_shift(ax, x, y, length):
    ax.add_patch(Rectangle((x + 0.10, y + 0.18), length, 0.26, facecolor="#E7EEF5", edgecolor="none", zorder=3))
    for cx in (x + 0.55, x + 1.05, x + 1.55, x + 2.05):
        if cx < x + 0.10 + length - 0.18:
            ax.add_patch(Circle((cx, y + 0.31), 0.05, color=PALETTE["gold"], zorder=4))
    ax.add_patch(Polygon([[x + 0.20, y + 0.57], [x + 0.34, y + 0.31], [x + 0.48, y + 0.57]], closed=True, color=PALETTE["agent"], zorder=5))


def _draw_quadpo_shift(ax, x, y, dense=False):
    ax.add_patch(Rectangle((x + 0.12, y + 0.16), 2.18, 0.34, facecolor="#EEF4FA", edgecolor="none", zorder=3))
    centers = [x + 0.55, x + 1.15, x + 1.75] if not dense else [x + 0.42, x + 0.86, x + 1.30, x + 1.74, x + 2.05]
    for cx in centers:
        ax.add_patch(Circle((cx, y + 0.33), 0.08, facecolor=PALETTE["test"], edgecolor="white", lw=0.8, zorder=4))
    ax.add_patch(Polygon([[x + 0.18, y + 0.70], [x + 0.32, y + 0.42], [x + 0.46, y + 0.70]], closed=True, color=PALETTE["agent"], zorder=5))


def _draw_pendulum_shift(ax, x, y, heavy=False):
    anchor_x = x + 1.18
    anchor_y = y + 0.78
    radius = 0.78
    ax.add_patch(Circle((anchor_x, anchor_y), 0.05, color=PALETTE["ink"], zorder=5))
    angle = -38 if not heavy else -58
    ax.add_patch(Arc((anchor_x, anchor_y), radius * 1.75, radius * 1.75, theta1=235, theta2=320, color=PALETTE["line"], lw=1.0, zorder=3))
    x2 = anchor_x + radius * 0.62
    y2 = anchor_y - radius * 0.78
    if heavy:
        x2 = anchor_x + radius * 0.82
        y2 = anchor_y - radius * 1.02
    ax.plot([anchor_x, x2], [anchor_y, y2], color=PALETTE["ink"], lw=1.7, zorder=4)
    ax.add_patch(Circle((x2, y2), 0.14 if not heavy else 0.18, color=PALETTE["gold"], zorder=5))


def _draw_cartpole_shift(ax, x, y, long=False):
    ax.add_patch(Rectangle((x + 0.25, y + 0.18), 1.85 if not long else 2.15, 0.08, facecolor=PALETTE["line"], edgecolor="none", zorder=3))
    cart_x = x + 0.88 if not long else x + 1.05
    ax.add_patch(FancyBboxPatch((cart_x, y + 0.22), 0.42, 0.20, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=PALETTE["agent"], edgecolor="white", lw=0.8, zorder=5))
    ax.plot([cart_x + 0.21, cart_x + (0.52 if not long else 0.72)], [y + 0.42, y + (0.95 if not long else 1.02)], color=PALETTE["ink"], lw=1.6, zorder=5)


def _draw_swimmer_shift(ax, x, y, stretch=False):
    segment = 0.34 if not stretch else 0.48
    points = [(x + 0.24, y + 0.50), (x + 0.24 + segment, y + 0.72), (x + 0.24 + 2 * segment, y + 0.46), (x + 0.24 + 3 * segment, y + 0.68)]
    for p1, p2 in zip(points, points[1:]):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=PALETTE["agent"], lw=3.0, solid_capstyle="round", zorder=5)
    for px, py in points:
        ax.add_patch(Circle((px, py), 0.05, color=PALETTE["gold"], zorder=6))


def _draw_benchmark_tile(ax, name, desc, icon_fn):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    _add_card(ax, 0.18, 0.18, 9.64, 5.80, facecolor=PALETTE["card"])
    ax.text(0.62, 5.26, name, fontsize=11.5, weight="bold", color=PALETTE["ink"])
    ax.text(0.62, 4.88, desc, fontsize=8.5, color=PALETTE["slate"])

    _add_chip(ax, 0.62, 4.15, "TRAIN", PALETTE["train"], fontsize=7.8)
    _add_chip(ax, 5.30, 4.15, "TEST", PALETTE["test"], fontsize=7.8)
    _add_card(ax, 0.62, 1.00, 3.75, 2.75, facecolor="#F7FBF9", edgecolor=PALETTE["line"], shadow=False)
    _add_card(ax, 5.05, 1.00, 4.13, 2.75, facecolor="#FFF7F4", edgecolor=PALETTE["line"], shadow=False)
    icon_fn(ax, 1.10, 1.55, False)
    icon_fn(ax, 5.58, 1.55, True)


def make_benchmark_overview():
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    tiles = [
        ("Car", "train-test shift in parking gap", _draw_car_shift),
        ("Quad", "course length extrapolation", _draw_quad_shift),
        ("QuadPO", "hidden periodic obstacles", _draw_quadpo_shift),
        ("Pendulum", "mass and dynamics shift", _draw_pendulum_shift),
        ("CartPole", "longer horizon and track", _draw_cartpole_shift),
        ("Swimmer", "morphology segment shift", _draw_swimmer_shift),
    ]

    for ax, (name, desc, icon_fn) in zip(axes.flatten(), tiles):
        _draw_benchmark_tile(ax, name, desc, icon_fn)

    fig.subplots_adjust(left=0.03, right=0.985, top=0.97, bottom=0.04, wspace=0.12, hspace=0.18)
    fig.savefig(FIG_DIR / "benchmark_overview.png", dpi=280, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    make_running_example()
    make_adaptive_teaching()
    make_benchmark_overview()
    print(f"saved figures to {FIG_DIR}")
