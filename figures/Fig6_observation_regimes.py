"""
Fig 6 -- Effect of phenotype-observation richness on hidden-state reconstruction.
Source: Round 3, Cell 19 (final refined version, "..._state_distributions_v5"). No training.

Input
  round3/final/tables/round3_observation_regime_all_seeds_metrics.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

CSV = ps.ROUND3_TABLES / "round3_observation_regime_all_seeds_metrics.csv"
ps.require(CSV)

REGIMES = ["O1", "O2", "O3", "O4"]
TICKS = ["O1\n(0 pts)", "O2\n(1 pt)", "O3\n(3 pts)", "O4\n(21 pts)"]
CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COLORS = {r: CYCLE[i] for i, r in enumerate(REGIMES)}
MARKERS = {"O1": "o", "O2": "s", "O3": "^", "O4": "D"}

runs = pd.read_csv(CSV)
runs = runs[runs["regime"].astype(str).isin(REGIMES)]
runs = runs.drop_duplicates(subset=["regime", "seed_id"], keep="last")


def fmt_value(v, metric):
    if metric == "rmse_fraction" and v != 0:
        e = int(np.floor(np.log10(abs(v))))
        return rf"${v / 10 ** e:.2f}\times10^{{{e}}}$"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def panel(ax, metric, ylabel, title, log=False):
    x = np.arange(len(REGIMES), dtype=float)
    means, sds = [], []
    for i, r in enumerate(REGIMES):
        v = runs.loc[runs["regime"].astype(str) == r, metric].astype(float).dropna().to_numpy()
        jit = np.zeros(1) if len(v) == 1 else np.linspace(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), x[i]) + jit, v, s=14, marker=MARKERS[r], facecolor=COLORS[r],
                   edgecolor="black", linewidth=0.3, alpha=0.38, zorder=2)
        means.append(np.mean(v))
        sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    means, sds = np.array(means), np.array(sds)
    ax.plot(x, means, color="black", lw=1.0, alpha=0.78, zorder=3)
    for i, r in enumerate(REGIMES):
        if log:
            yerr = sds[i]
        else:  # non-negative metric: clip the lower bar at zero
            yerr = np.array([[min(sds[i], max(means[i], 0.0))], [sds[i]]])
        ax.errorbar(x[i], means[i], yerr=yerr, fmt=MARKERS[r], ms=5.2, mfc=COLORS[r], mec="black",
                    mew=0.7, ecolor="black", elinewidth=0.9, capsize=2.8, capthick=0.8, zorder=5)
        # value label centred just above the upper error-bar cap (never on the line)
        top = means[i] + sds[i]
        ax.annotate(fmt_value(means[i], metric), (x[i], top), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=ps.FS_ANNOT,
                    fontweight="bold", annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="0.40",
                              linewidth=0.5, alpha=0.94), zorder=7)
    tops = means + sds
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, tops.max()) * 2.2)          # head-room for labels
    else:
        hi = max(ax.get_ylim()[1], tops.max())
        ax.set_ylim(bottom=0.0, top=hi * 1.15)              # head-room for labels
    ax.set_xticks(x)
    ax.set_xticklabels(TICKS)
    ax.set_xlim(-0.45, len(REGIMES) - 0.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ps.frame(ax)


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 5.6))
specs = [("rmse_sensitive", "Sensitive-state RMSE", r"Sensitive state, $S(t)$", False),
         ("rmse_resistant", "Resistant-state RMSE", r"Resistant state, $R(t)$", False),
         ("rmse_total", "Total-population RMSE", r"Total population, $S(t)+R(t)$", False),
         ("rmse_fraction", "Resistant-fraction RMSE", r"Resistant fraction, $q_R(t)$", True)]
for letter, ax, (metric, ylabel, title, log) in zip("ABCD", axes.ravel(), specs):
    panel(ax, metric, ylabel, title, log)
    ps.panel_label(ax, letter, x=-0.16, y=1.03)
fig.align_ylabels(axes[:, 0])
fig.align_ylabels(axes[:, 1])
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.945, wspace=0.32, hspace=0.50)
ps.save_plos(fig, 6)
