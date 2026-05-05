"""
plot.py — Compare toy long-horizon inference methods including RCD.

Generates a 2×2 grid:
  Row 1: L=4
  Row 2: L=12
  Columns: CompDiffuser, RCD

Each panel shows 200 sample trajectories coloured green (valid) or coral (invalid),
with reference density profiles and success rate annotation.

Usage:
  python plot.py --results-dir results --output figure_rcd.png
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


BOUNDARY_STD = 0.2
INTERIOR_STD = 0.1


def evaluate_valid(samples: np.ndarray, num_stds: float = 3.0) -> np.ndarray:
    """Valid iff every point within num_stds*σ of its mode AND all interior same mode."""
    boundary_r = num_stds * BOUNDARY_STD
    interior_r = num_stds * INTERIOR_STD
    start_ok = np.abs(samples[:, 0]) < boundary_r
    end_ok = np.abs(samples[:, -1]) < boundary_r
    interior = samples[:, 1:-1]
    interior_ok = np.all(
        np.minimum(np.abs(interior - 1.0), np.abs(interior + 1.0)) < interior_r, axis=1
    )
    signs = np.sign(interior)
    signs[signs == 0] = 1
    same_mode = np.all(signs == signs[:, [0]], axis=1)
    return start_ok & end_ok & interior_ok & same_mode


def draw_reference_densities(ax: plt.Axes, total_dim: int) -> None:
    ys = np.linspace(-1.65, 1.65, 400)
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
            color="#8ecae6",
            alpha=0.45,
            lw=0,
            zorder=0,
        )


def plot_panel(
    ax: plt.Axes, samples: np.ndarray, title: str, num_stds: float = 3.0
) -> float:
    total_dim = samples.shape[1]
    xs = np.arange(total_dim)
    valid = evaluate_valid(samples, num_stds)
    success = float(valid.mean())

    boundary_r = num_stds * BOUNDARY_STD
    interior_r = num_stds * INTERIOR_STD

    ax.set_facecolor("#f1f2f4")
    draw_reference_densities(ax, samples.shape[1])

    def _in_mode(val, pos):
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
            LineCollection(segs_ok, colors="#2ca25f", alpha=0.16, linewidths=1.0, zorder=2)
        )
    if segs_bad:
        ax.add_collection(
            LineCollection(segs_bad, colors="#f28482", alpha=0.16, linewidths=1.0, zorder=2)
        )

    ax.scatter(
        [0],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=120,
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        [0], [0], marker="o", facecolor="#264653", edgecolor="#264653", s=35, zorder=6
    )
    ax.scatter(
        [total_dim - 1],
        [0],
        marker="o",
        facecolor="white",
        edgecolor="black",
        s=120,
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        [total_dim - 1],
        [0],
        marker="*",
        facecolor="#264653",
        edgecolor="#264653",
        s=120,
        zorder=6,
    )

    ax.set_xlim(-0.5, samples.shape[1] - 0.5)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$x_{{{i}}}$" for i in xs], fontsize=9)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", color="#9ecae1", alpha=0.5, linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{title}  (success={success:.0%})", fontsize=11, weight="bold")
    return success


def main():
    parser = argparse.ArgumentParser(description="Plot toy comparison figure")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="figure_rcd.png")
    parser.add_argument("--num-stds", type=float, default=3.0)
    args = parser.parse_args()

    horizons = [4, 12]
    methods = [
        ("base", "CompDiffuser"),
        ("rcd", "RCD"),
    ]

    fig, axes = plt.subplots(
        len(horizons), len(methods), figsize=(6.0 * len(methods), 3.6 * len(horizons))
    )
    fig.patch.set_facecolor("white")

    for row, horizon in enumerate(horizons):
        for col, (method_key, method_name) in enumerate(methods):
            path = os.path.join(args.results_dir, f"L{horizon}_{method_key}.npy")
            if not os.path.exists(path):
                print(f"WARNING: {path} not found, skipping.")
                axes[row, col].text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                    fontsize=14,
                )
                axes[row, col].set_title(f"{method_name} (L={horizon})")
                continue

            samples = np.load(path)
            title = f"{method_name}  (L={horizon})"
            success = plot_panel(axes[row, col], samples, title, args.num_stds)
            print(f"{title}: success={success:.1%}  ({samples.shape[0]} samples)")

    fig.tight_layout(pad=1.5)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
