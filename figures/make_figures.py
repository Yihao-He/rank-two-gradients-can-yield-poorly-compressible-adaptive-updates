from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 6.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def read_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fig_analytic(rows):
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ns = (32, 64, 128, 256, 512)
    for color, n in zip(COLORS, ns):
        subset = sorted([r for r in rows if int(r["n"]) == n and int(r["rank"]) < n],
                        key=lambda r: int(r["rank"]))
        ax.plot([int(r["rank"]) for r in subset],
                [float(r["best_rank_relative_frobenius_residual"]) for r in subset],
                marker="o", markersize=2.5, linewidth=.95, color=color, label=f"n={n}")
    guide_r = np.array([4, 128], dtype=float)
    ax.plot(guide_r, .3*(guide_r/4)**(-.5), color="#555555", linestyle="--", linewidth=.8,
            label=r"reference slope $r^{-1/2}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Approximation rank r")
    ax.set_ylabel("Best relative Frobenius residual")
    ax.grid(which="both", alpha=.18, linewidth=.45)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(pad=.6)
    fig.savefig(FIGURES / "fig1_analytic_residual.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gpu(rows):
    ranks = [2, 4, 8, 16, 32]
    specs = [
        ("last_layer_down_proj", "G32", "last block gradient", "#0072B2", "-"),
        ("last_layer_down_proj", "U32", "last block adaptive update", "#D55E00", "-"),
        ("layer_minus_4_down_proj", "G32", "layer -4 gradient", "#0072B2", "--"),
        ("layer_minus_4_down_proj", "U32", "layer -4 adaptive update", "#D55E00", "--"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for block, key, label, color, style in specs:
        row = next(r for r in rows if r["block"] == block and r["matrix"] == key)
        ax.plot(ranks, [float(row[f"E{r}"]) for r in ranks],
                marker="o", markersize=2.7, linewidth=1., color=color, linestyle=style, label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Approximation rank r")
    ax.set_ylabel("Best relative Frobenius residual")
    ax.grid(which="both", alpha=.18, linewidth=.45)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=.6)
    fig.savefig(FIGURES / "fig2_gpu_block_residuals.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    fig_analytic(read_csv("analytic_residual_curve.csv"))
    fig_gpu(read_csv("optimizer_stage_diagnostics.csv"))
    print("Wrote fig1_analytic_residual.pdf and fig2_gpu_block_residuals.pdf")


if __name__ == "__main__":
    main()
