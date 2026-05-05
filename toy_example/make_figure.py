#!/usr/bin/env python3
"""
Paper figure for the toy mode-averaging section.

Panels:
  (a) Training setup for L=4 with overlapping l=3 chunks
  (b) CompDiffuser samples at L=10
  (c) RCD samples at L=10
  (d) Success rate vs horizon L=4..20

Usage:
  python make_figure.py
  python make_figure.py --output figure2_toy.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


BOUNDARY_STD = 0.2
INTERIOR_STD = 0.1
NUM_STDS = 3.0
TRAJ_Y_MAX = 1.5

BOUNDARY_RADIUS = NUM_STDS * BOUNDARY_STD
INTERIOR_RADIUS = NUM_STDS * INTERIOR_STD

COLORS = {
    "paper": "#ffffff",
    "panel": "#ffffff",
    "ink": "#2c2a28",
    "muted": "#7b756d",
    "grid": "#d0d0d0",
    "overlap": "#e0e0e0",
    "chunk_a": "#d6a44f",
    "chunk_b": "#7e95c9",
    "mode_pos": "#d95f5f",
    "mode_neg": "#4d68b2",
    "invalid": "#b8afa6",
    "base": "#c94153",
    "rcd": "#1f5d94",
    "gain": "#cfdceb",
}

CHUNK_COLORS = {
    "start": "#2980b9",
    "overlap": "#27ae60",
    "end": "#e74c3c",
    "density": "#c8c8c8",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def evaluate_valid(samples: np.ndarray, num_stds: float = NUM_STDS) -> np.ndarray:
    boundary_r = num_stds * BOUNDARY_STD
    interior_r = num_stds * INTERIOR_STD
    start_ok = np.abs(samples[:, 0]) < boundary_r
    end_ok = np.abs(samples[:, -1]) < boundary_r
    interior = samples[:, 1:-1]
    interior_ok = np.all(
        np.minimum(np.abs(interior - 1.0), np.abs(interior + 1.0)) < interior_r,
        axis=1,
    )
    signs = np.sign(interior)
    signs[signs == 0] = 1
    same_mode = np.all(signs == signs[:, [0]], axis=1)
    return start_ok & end_ok & interior_ok & same_mode


def load_summary(results_dir: Path, horizon: int, method: str) -> dict:
    with (results_dir / f"L{horizon}_{method}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(results_dir: Path, horizon: int, method: str) -> np.ndarray:
    return np.load(results_dir / f"L{horizon}_{method}.npy")


def classify_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = evaluate_valid(samples)
    mode_mean = samples[:, 1:-1].mean(axis=1)
    pos = valid & (mode_mean > 0)
    neg = valid & ~pos
    invalid = ~valid
    return pos, neg, invalid


def add_panel_tag(ax: plt.Axes, tag: str, title: str) -> None:
    ax.text(
        0.0,
        1.08,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.16,
        1.08,
        f" {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        color=COLORS["ink"],
    )


def style_clean_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(color=COLORS["muted"], labelcolor=COLORS["ink"])


def draw_reference_densities(ax: plt.Axes, total_dim: int) -> None:
    ys = np.linspace(-TRAJ_Y_MAX, TRAJ_Y_MAX, 400)
    norm_scale = 0.035
    for pos in range(total_dim):
        if pos in (0, total_dim - 1):
            centers, sigma = [0.0], BOUNDARY_STD
        else:
            centers, sigma = [-1.0, 1.0], INTERIOR_STD
        density = (
            sum(np.exp(-0.5 * ((ys - c) / sigma) ** 2) / sigma for c in centers)
            * norm_scale
        )
        ax.fill_betweenx(
            ys,
            pos,
            pos + density,
            color=CHUNK_COLORS["density"],
            alpha=1.0,
            lw=0,
            zorder=0,
        )


def draw_schematic(ax: plt.Axes) -> None:
    style_clean_axes(ax)
    rng = np.random.default_rng(42)
    total_dim = 5
    xs = np.arange(total_dim)

    ax.set_facecolor("#ffffff")
    draw_reference_densities(ax, total_dim)
    ax.set_xlim(-0.5, total_dim - 1 + 0.5)
    ax.set_ylim(-TRAJ_Y_MAX, TRAJ_Y_MAX)
    ax.set_xticks(xs)
    ax.set_xticklabels([rf"$x_{{{i + 1}}}$" for i in xs], fontsize=14)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r"$-1$", r"$0$", r"$+1$"], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="x", color="#b0b0b0", alpha=0.9, lw=1.15)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Invisible spacer to match height of "Valid Rate" text in panels (b),(c)
    ax.text(
        0.02,
        1.06,
        " ",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="none", edgecolor="none"),
    )

    n_per_mode = 3
    for mode in (+1, -1):
        for _ in range(n_per_mode):
            vals = [
                rng.normal(0.0, BOUNDARY_STD),
                rng.normal(mode, INTERIOR_STD),
                rng.normal(mode, INTERIOR_STD),
            ]
            ax.plot(
                [0, 1, 2],
                vals,
                "-o",
                color=CHUNK_COLORS["start"],
                ms=4.5,
                lw=2.0,
                alpha=0.65,
                zorder=3,
            )

    for mode in (+1, -1):
        for _ in range(n_per_mode):
            vals = [
                rng.normal(mode, INTERIOR_STD),
                rng.normal(mode, INTERIOR_STD),
                rng.normal(mode, INTERIOR_STD),
            ]
            ax.plot(
                [1, 2, 3],
                vals,
                "-o",
                color=CHUNK_COLORS["overlap"],
                ms=4.5,
                lw=2.0,
                alpha=0.65,
                zorder=3,
            )

    for mode in (+1, -1):
        for _ in range(n_per_mode):
            vals = [
                rng.normal(mode, INTERIOR_STD),
                rng.normal(mode, INTERIOR_STD),
                rng.normal(0.0, BOUNDARY_STD),
            ]
            ax.plot(
                [2, 3, 4],
                vals,
                "-o",
                color=CHUNK_COLORS["end"],
                ms=4.5,
                lw=2.0,
                alpha=0.65,
                zorder=3,
            )

    ax.scatter(
        [0],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=110,
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        [0], [0], marker="o", facecolor="black", edgecolor="black", s=30, zorder=6
    )
    ax.scatter(
        [4],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=110,
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        [4], [0], marker="*", facecolor="black", edgecolor="black", s=105, zorder=6
    )


def draw_valid_bands(ax: plt.Axes, total_dim: int) -> None:
    for pos in range(total_dim):
        if pos in (0, total_dim - 1):
            patch = Rectangle(
                (pos - 0.42, -BOUNDARY_RADIUS),
                0.84,
                2 * BOUNDARY_RADIUS,
                facecolor="#ebe6dd",
                edgecolor="none",
                alpha=0.95,
                zorder=0,
            )
            ax.add_patch(patch)
        else:
            for center in (-1.0, 1.0):
                patch = Rectangle(
                    (pos - 0.42, center - INTERIOR_RADIUS),
                    0.84,
                    2 * INTERIOR_RADIUS,
                    facecolor="#ebe6dd",
                    edgecolor="none",
                    alpha=0.95,
                    zorder=0,
                )
                ax.add_patch(patch)


def draw_trajectory_panel(
    ax: plt.Axes,
    samples: np.ndarray,
    summary: dict,
    title: str,
    annotate_invalid: bool = False,
    show_header: bool = True,
    max_samples: int = 200,
) -> None:
    style_clean_axes(ax)
    if samples.shape[0] > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(samples.shape[0], max_samples, replace=False)
        samples = samples[idx]
    total_dim = samples.shape[1]
    xs = np.arange(total_dim)
    valid = evaluate_valid(samples)

    boundary_r = NUM_STDS * BOUNDARY_STD
    interior_r = NUM_STDS * INTERIOR_STD

    ax.set_facecolor("#ffffff")
    draw_reference_densities(ax, total_dim)

    def _in_mode(val: float, pos: int) -> bool:
        if pos in (0, total_dim - 1):
            return abs(val) < boundary_r
        return min(abs(val - 1.0), abs(val + 1.0)) < interior_r

    segs_ok, segs_bad = [], []
    for i in range(samples.shape[0]):
        traj = samples[i]
        for j in range(total_dim - 1):
            seg = [[j, traj[j]], [j + 1, traj[j + 1]]]
            ok = _in_mode(traj[j], j) and _in_mode(traj[j + 1], j + 1)
            if ok and 0 < j and j + 1 < total_dim - 1:
                ok = (1.0 if traj[j] >= 0 else -1.0) == (
                    1.0 if traj[j + 1] >= 0 else -1.0
                )
            (segs_ok if ok else segs_bad).append(seg)

    if segs_ok:
        ax.add_collection(
            LineCollection(
                segs_ok,
                colors="#2ca25f",
                alpha=0.18,
                linewidths=1.0,
                zorder=2,
            )
        )
    if segs_bad:
        ax.add_collection(
            LineCollection(
                segs_bad,
                colors="#f28482",
                alpha=0.22,
                linewidths=1.0,
                zorder=2,
            )
        )

    ax.scatter(
        [0],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=115,
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        [0],
        [0],
        marker="o",
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        s=32,
        zorder=6,
    )
    ax.scatter(
        [total_dim - 1],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=115,
        linewidth=1.4,
        zorder=5,
    )
    ax.scatter(
        [total_dim - 1],
        [0],
        marker="*",
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        s=112,
        zorder=6,
    )

    ax.set_xlim(-0.5, total_dim - 0.5)
    ax.set_ylim(-TRAJ_Y_MAX, TRAJ_Y_MAX)
    ax.set_xticks(xs)
    ax.set_xticklabels([rf"$x_{{{i + 1}}}$" for i in xs], fontsize=14)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", color="#b0b0b0", alpha=0.9, linewidth=1.15)
    for spine in ax.spines.values():
        spine.set_visible(False)

    stats = f"Valid Rate {summary['success_rate']:.1%}"
    ax.text(
        0.02,
        1.06,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#f0f0f0", edgecolor="none"),
    )

    if annotate_invalid:
        pass

    if show_header:
        add_panel_tag(ax, title.split()[0], " ".join(title.split()[1:]))


def draw_success_panel(
    ax: plt.Axes,
    horizons: list[int],
    base: list[float],
    rcd: list[float],
    show_header: bool = True,
) -> None:
    style_clean_axes(ax)
    ax.set_xlim(min(horizons) - 0.2, max(horizons) + 0.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.85, alpha=0.8)
    ax.fill_between(horizons, base, rcd, color="#b0b0b0", alpha=0.45, zorder=1)
    ax.plot(
        horizons,
        rcd,
        color="#1a1a1a",
        lw=2.8,
        marker="s",
        ms=6.0,
        linestyle="-",
        label="RCD",
        zorder=4,
    )
    ax.plot(
        horizons,
        base,
        color="#555555",
        lw=2.8,
        marker="o",
        ms=6.0,
        linestyle="dotted",
        label="CompDiffuser",
        zorder=3,
    )
    ax.set_xlabel("Inference Horizon", fontsize=17)
    ax.set_ylabel("Valid Rate", fontsize=17)
    ax.legend(frameon=False, loc="lower left", fontsize=13.5, handlelength=2.5)

    if show_header:
        add_panel_tag(ax, "(d)", "Success vs. horizon")


def save_panel_figure(
    draw_fn,
    pdf_path: Path,
    png_path: Path,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["paper"])
    draw_fn(ax)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03, dpi=320)
    plt.close(fig)


def build_standalone_panels(results_dir: Path, output_dir: Path) -> None:
    horizons = list(range(4, 21))
    base_curve = [
        load_summary(results_dir, horizon, "base")["success_rate"]
        for horizon in horizons
    ]
    rcd_curve = [
        load_summary(results_dir, horizon, "rcd")["success_rate"]
        for horizon in horizons
    ]

    base_l10 = load_samples(results_dir, 10, "base")
    rcd_l10 = load_samples(results_dir, 10, "rcd")
    base_l10_summary = load_summary(results_dir, 10, "base")
    rcd_l10_summary = load_summary(results_dir, 10, "rcd")

    save_panel_figure(
        draw_schematic,
        output_dir / "panel_a_training_segments.pdf",
        output_dir / "panel_a_training_segments.png",
        figsize=(3.5, 3.2),
    )
    save_panel_figure(
        lambda ax: draw_trajectory_panel(
            ax,
            base_l10,
            base_l10_summary,
            "(b) CompDiffuser",
            annotate_invalid=True,
            show_header=False,
        ),
        output_dir / "panel_b_compdiffuser.pdf",
        output_dir / "panel_b_compdiffuser.png",
        figsize=(3.5, 3.2),
    )
    save_panel_figure(
        lambda ax: draw_trajectory_panel(
            ax,
            rcd_l10,
            rcd_l10_summary,
            "(c) RCD",
            show_header=False,
        ),
        output_dir / "panel_c_rcd.pdf",
        output_dir / "panel_c_rcd.png",
        figsize=(3.5, 3.2),
    )
    save_panel_figure(
        lambda ax: draw_success_panel(
            ax,
            horizons,
            base_curve,
            rcd_curve,
            show_header=False,
        ),
        output_dir / "panel_d_valid_plan_rate_curve.pdf",
        output_dir / "panel_d_valid_plan_rate_curve.png",
        figsize=(3.8, 3.8),
    )


def build_figure(
    results_dir: Path, output: Path, output_png: Path | None = None
) -> None:
    horizons = list(range(4, 21))
    base_summaries = [
        load_summary(results_dir, horizon, "base") for horizon in horizons
    ]
    rcd_summaries = [load_summary(results_dir, horizon, "rcd") for horizon in horizons]

    base_curve = [summary["success_rate"] for summary in base_summaries]
    rcd_curve = [summary["success_rate"] for summary in rcd_summaries]

    base_l10 = load_samples(results_dir, 10, "base")
    rcd_l10 = load_samples(results_dir, 10, "rcd")
    base_l10_summary = load_summary(results_dir, 10, "base")
    rcd_l10_summary = load_summary(results_dir, 10, "rcd")

    fig = plt.figure(figsize=(12.8, 3.7), facecolor=COLORS["paper"])
    gs = fig.add_gridspec(1, 4, width_ratios=[1.02, 1.18, 1.18, 1.30], wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    pos = ax_d.get_position()
    new_height = pos.height * 0.92
    ax_d.set_position(
        [pos.x0, pos.y0 + 0.5 * (pos.height - new_height), pos.width, new_height]
    )

    draw_schematic(ax_a)
    add_panel_tag(ax_a, "(a)", "Training segments")

    draw_trajectory_panel(
        ax_b,
        base_l10,
        base_l10_summary,
        "(b) CompDiffuser",
        annotate_invalid=True,
    )
    draw_trajectory_panel(
        ax_c,
        rcd_l10,
        rcd_l10_summary,
        "(c) RCD",
    )

    draw_success_panel(ax_d, horizons, base_curve, rcd_curve)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    if output_png is not None:
        fig.savefig(output_png, bbox_inches="tight", pad_inches=0.04, dpi=320)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir / "results"
    default_output = script_dir / "toy_mode_averaging.pdf"

    parser = argparse.ArgumentParser(
        description="Generate the toy mode-averaging paper figure."
    )
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--output-png", type=Path, default=script_dir / "toy_mode_averaging.png"
    )
    parser.add_argument(
        "--standalone-dir",
        type=Path,
        default=script_dir / "standalone_panels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_standalone_panels(args.results_dir, args.standalone_dir)
    print(f"Saved standalone panels -> {args.standalone_dir}")


if __name__ == "__main__":
    main()
