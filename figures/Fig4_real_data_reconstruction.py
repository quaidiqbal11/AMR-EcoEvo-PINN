"""
Fig 4 -- Experimental-data reconstruction with the inverse parameters fixed.
Source: real-data notebook, standalone publication-figure cell (final P2 model).
No model needed: reads the source-data CSVs that cell exported.

Inputs
  Real_PINN_Results/figures/publication_ready/source_data/final_MAIN_total_load_source_data.csv
  Real_PINN_Results/figures/publication_ready/source_data/final_MAIN_qR48_source_data.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import plos_style as ps

TOTAL_CSV = ps.REAL_SOURCE_DATA / "final_MAIN_total_load_source_data.csv"
Q48_CSV = ps.REAL_SOURCE_DATA / "final_MAIN_qR48_source_data.csv"
ps.require(TOTAL_CSV, Q48_CSV)

DOSE_COLORS = {0.0: "#4D4D4D", 30.0: "#009E73", 40.0: "#0072B2", 50.0: "#CC79A7", 100.0: "#D55E00"}
MODEL = "#111111"
DOSES = [0.0, 30.0, 40.0, 50.0, 100.0]
TREATED = [30.0, 40.0, 50.0, 100.0]

total = pd.read_csv(TOTAL_CSV)
q48 = pd.read_csv(Q48_CSV)
total["dose_ug_ml"] = total["dose_ug_ml"].astype(float)
q48["dose_ug_ml"] = q48["dose_ug_ml"].astype(float)

fig, axes = plt.subplots(2, 3, figsize=(ps.FULL_WIDTH, 5.3))
axes = axes.ravel()

for i, dose in enumerate(DOSES):
    ax = axes[i]
    d = total[total["dose_ug_ml"] == dose]
    obs = d[d["series"] == "observed_replicate"]
    model = d[d["series"] == "final_P2_PINN"].sort_values("time_h")
    train, val = obs[obs["split"] == "train"], obs[obs["split"] == "val"]
    ax.scatter(train["time_h"], train["value"], s=12, color=DOSE_COLORS[dose], alpha=0.34,
               edgecolor="none", zorder=2)
    ax.scatter(val["time_h"], val["value"], s=18, marker="D", facecolor="white",
               edgecolor=DOSE_COLORS[dose], linewidth=0.8, zorder=4)
    stats = obs.groupby("time_h")["value"].agg(["mean", "std"]).reset_index()
    ax.errorbar(stats["time_h"], stats["mean"], yerr=stats["std"], fmt="o", ms=3.5, mfc="white",
                mec="#111111", mew=0.8, ecolor="#111111", elinewidth=0.8, capsize=2.0,
                capthick=0.7, zorder=5)
    ax.plot(model["time_h"], model["value"], color=MODEL, lw=1.5, zorder=6)
    horizon = float(obs["time_h"].max())
    ax.set_xlim(-0.02 * max(horizon, 1.0), horizon * 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_title(f"{dose:g} µg/mL", fontweight="bold", pad=3)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"$\log_{10}(B+1)$")
    ps.open_axes(ax)
    ax.grid(axis="y", linewidth=0.4, alpha=0.14)
    ps.panel_label(ax, "ABCDE"[i], x=-0.30, y=1.0)

# Panel F -- resistant fraction at 48 h
ax = axes[5]
rng = np.random.default_rng(20260822)
x = np.arange(len(TREATED), dtype=float)
model_q = []
for j, dose in enumerate(TREATED):
    d = q48[q48["dose_ug_ml"] == dose]
    reps = d.loc[d["series"] == "observed_replicate", "qR_48h"].to_numpy(float)
    ax.scatter(x[j] + rng.normal(0.0, 0.045, len(reps)), reps, s=15, color=DOSE_COLORS[dose],
               alpha=0.55, edgecolor="white", linewidth=0.3, zorder=2)
    sd = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
    ax.errorbar(x[j] - 0.08, float(np.mean(reps)), yerr=sd, fmt="o", ms=3.8, mfc="white",
                mec="#111111", mew=0.8, ecolor="#111111", elinewidth=0.8, capsize=2.5, zorder=4)
    model_q.append(float(d.loc[d["series"] == "final_P2_PINN", "qR_48h"].iloc[0]))
ax.plot(x + 0.08, model_q, color=MODEL, lw=1.1, marker="D", ms=4.0, mfc="white", mec=MODEL,
        mew=0.8, zorder=5)
ax.set_xticks(x, ["30", "40", "50", "100"])
ax.set_xlabel("Streptomycin dose (µg/mL)")
ax.set_ylabel(r"Resistant fraction $q_R(48\,\mathrm{h})$")
ax.set_ylim(-0.03, 1.04)
ps.open_axes(ax)
ax.grid(axis="y", linewidth=0.4, alpha=0.14)
ps.panel_label(ax, "F", x=-0.30, y=1.0)
ax.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", ms=4.5, mfc="0.65", mec="white", label="Experimental replicate"),
    Line2D([0], [0], marker="o", ls="none", ms=4.5, mfc="white", mec="#111111", label="Observed mean ± SD"),
    Line2D([0], [0], marker="D", color=MODEL, ms=4.5, mfc="white", label="Hard PINN")],
    loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.2)

fig.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", ms=4.5, mfc="0.55", mec="none", alpha=0.5, label="Training replicate"),
    Line2D([0], [0], marker="D", ls="none", ms=4.5, mfc="white", mec="0.35", label="Held-out replicate"),
    Line2D([0], [0], marker="o", ls="none", ms=4.5, mfc="white", mec="#111111", label="Observed mean ± SD"),
    Line2D([0], [0], color=MODEL, lw=1.5, label="Hard PINN")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
    columnspacing=1.2, handlelength=1.5)

fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.875, wspace=0.50, hspace=0.55)
ps.save_plos(fig, 4)
