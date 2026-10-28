"""
Fig 8 -- Recovery of shared evolutionary parameters across designs M1-M5.
Source: Round 4, Cell 19 ("round4_primary_parameter_recovery_final"). No training.

Input
  round4/final/tables/round4_primary_multischedule_all_seeds_metrics.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

CSV = ps.ROUND4_TABLES / "round4_primary_multischedule_all_seeds_metrics.csv"
ps.require(CSV)

DESIGNS = ["M1", "M2", "M3", "M4", "M5"]
CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLORS = {d: CYCLE[i] for i, d in enumerate(DESIGNS)}
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P"}
TRUE_MU, TRUE_COST = 1.0e-4, 0.10

runs = pd.read_csv(CSV)


def panel(ax, metric, title, ylabel, log=False, true_value=None):
    x = np.arange(len(DESIGNS), dtype=float)
    means, sds = [], []
    for i, d in enumerate(DESIGNS):
        v = runs.loc[runs["design_id"] == d, metric].astype(float).dropna().to_numpy()
        jit = np.zeros(1) if len(v) <= 1 else np.linspace(-0.115, 0.115, len(v))
        ax.scatter(np.full(len(v), x[i]) + jit, v, s=14, marker=MARKERS[d], facecolor=COLORS[d],
                   edgecolor="black", linewidth=0.3, alpha=0.27, zorder=2)
        means.append(np.mean(v))
        sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    ax.plot(x, means, color="black", lw=1.0, alpha=0.78, zorder=3)
    for i, d in enumerate(DESIGNS):
        ax.errorbar(x[i], means[i], yerr=sds[i], fmt=MARKERS[d], ms=5.4, mfc=COLORS[d], mec="black",
                    mew=0.7, ecolor="black", elinewidth=0.9, capsize=2.8, capthick=0.8, zorder=5)
    if true_value is not None:
        ax.axhline(true_value, ls="--", lw=1.0, color="black", zorder=1)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(DESIGNS)
    ax.set_xlim(-0.38, len(DESIGNS) - 0.62)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", which="major", alpha=0.20, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ps.frame(ax)


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 5.4))
panel(axes[0, 0], "mutation_probability", "Mutation-probability recovery",
      r"Estimated mutation probability, $\hat{\mu}$", log=True, true_value=TRUE_MU)
panel(axes[0, 1], "resistance_fitness_cost", "Resistance-fitness-cost recovery",
      r"Estimated resistance fitness cost, $\hat{c}_R$", true_value=TRUE_COST)
panel(axes[1, 0], "mu_error_percent", "Mutation-probability error",
      r"Relative error in $\mu$ (%)", log=True)
panel(axes[1, 1], "cost_error_percent", "Fitness-cost error",
      r"Relative error in $c_R$ (%)")
for ax, letter in zip(axes.ravel(), "ABCD"):
    ps.panel_label(ax, letter, x=-0.15, y=1.03)

box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.55", linewidth=0.5, alpha=0.92)
axes[0, 0].text(0.975, 0.955, r"Truth: $\mu^\star=10^{-4}$", transform=axes[0, 0].transAxes,
                ha="right", va="top", fontsize=ps.FS_ANNOT, bbox=box, zorder=8)
axes[0, 1].text(0.975, 0.955, r"Truth: $c_R^\star=0.10$", transform=axes[0, 1].transAxes,
                ha="right", va="top", fontsize=ps.FS_ANNOT, bbox=box, zorder=8)

fig.legend(handles=[
    Line2D([0], [0], marker="o", ls="None", mfc="0.65", mec="black", alpha=0.30, ms=4.5, label="Individual seed"),
    Line2D([0], [0], marker="D", ls="-", color="black", mfc="white", mec="black", ms=4.5, lw=1.0,
           label="Mean ± sample SD")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.6,
    handletextpad=0.4)
fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.065, top=0.905, wspace=0.32, hspace=0.42)
ps.save_plos(fig, 8)
