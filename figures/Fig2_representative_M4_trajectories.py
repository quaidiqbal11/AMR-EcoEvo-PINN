"""
Fig 2 -- Representative state reconstruction, four-environment sparse-phenotype design (M4).
Source: Round 4, Cell 20 (Figure A). No training; reads saved CSVs.

Inputs
  round4/final/tables/round4_primary_multischedule_all_seeds_metrics.csv
  round4/final/predictions/<label>_all_schedule_predictions.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, MaxNLocator

import plos_style as ps

PRIMARY_CSV = ps.ROUND4_TABLES / "round4_primary_multischedule_all_seeds_metrics.csv"
ps.require(PRIMARY_CSV)

SCHEDULE_ORDER = ["untreated", "early_continuous", "late_continuous", "pulsed"]
SCHEDULE_NAMES = {"untreated": "Untreated", "early_continuous": "Early continuous",
                  "late_continuous": "Late continuous", "pulsed": "Pulsed"}
# Treatment-ON intervals (h), as in SCHEDULES_R4 (horizon 0-20 h)
TREATMENT_ON = {"untreated": [], "early_continuous": [(5.0, 20.0)],
                "late_continuous": [(10.0, 20.0)], "pulsed": [(5.0, 8.0), (12.0, 15.0)]}

# --- Objective representative M4 seed (same rule as the notebook: closest to the
#     per-metric median in MAD-scaled distance) ---------------------------------
primary = pd.read_csv(PRIMARY_CSV)
m4 = primary[primary["design_id"] == "M4"].reset_index(drop=True)
distance = np.zeros(len(m4))
for metric in ["rmse_resistant", "rmse_fraction", "mu_error_percent", "cost_error_percent"]:
    v = m4[metric].astype(float).to_numpy()
    med = np.median(v)
    distance += ((v - med) / max(np.median(np.abs(v - med)), 1e-12)) ** 2
label = str(m4.iloc[int(np.argmin(distance))]["label"])
pred_csv = ps.ROUND4_PREDICTIONS / f"{label}_all_schedule_predictions.csv"
ps.require(pred_csv)
pred = pd.read_csv(pred_csv)
print("Representative M4 run:", label)


def style_axis(ax, fraction=False):
    ax.grid(alpha=0.16, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ps.frame(ax)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-3, 3) if fraction else (-4, 4))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.get_offset_text().set_fontsize(ps.FS_TICK)


fig, axes = plt.subplots(4, 3, figsize=(ps.FULL_WIDTH, 7.2), sharex=True)
ref_color, pinn_color = "#1f77b4", "#ff7f0e"
columns = [("sensitive", "predicted_sensitive"), ("resistant", "predicted_resistant"),
           ("resistant_fraction", "predicted_resistant_fraction")]

for r, schedule in enumerate(SCHEDULE_ORDER):
    frame_s = pred[pred["schedule"] == schedule].sort_values("time")
    if frame_s.empty:
        raise RuntimeError(f"No trajectory rows for schedule '{schedule}' in {pred_csv}")
    for c, (true_col, pred_col) in enumerate(columns):
        ax = axes[r, c]
        for left, right in TREATMENT_ON[schedule]:
            ax.axvspan(left, right, facecolor="0.82", edgecolor="none", alpha=0.30, zorder=0)
        ax.plot(frame_s["time"], frame_s[true_col], color=ref_color, lw=1.4, zorder=3)
        ax.plot(frame_s["time"], frame_s[pred_col], color=pinn_color, lw=1.25, ls="--", zorder=4)
        style_axis(ax, fraction=(c == 2))

for ax in axes[-1, :]:
    ax.set_xlabel("Time (h)")

fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.865, wspace=0.36, hspace=0.52)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

titles = [r"Sensitive population, $S(t)$", r"Resistant population, $R(t)$",
          r"Resistant fraction, $q_R(t)$"]
for c, title in enumerate(titles):
    pos = axes[0, c].get_position()
    fig.text(0.5 * (pos.x0 + pos.x1), 0.925, title, ha="center", va="center",
             fontsize=ps.FS_TITLE, fontweight="bold")
for r, schedule in enumerate(SCHEDULE_ORDER):
    pos = axes[r, 0].get_position()
    fig.text(pos.x0, pos.y1 + 0.008, f"{'ABCD'[r]}  {SCHEDULE_NAMES[schedule]}",
             ha="left", va="bottom", fontsize=ps.FS_TITLE, fontweight="bold")
# Shared column y labels, placed just left of the widest tick labels in each column
inv = fig.transFigure.inverted()
for c, ylab in enumerate(["Population", "Population", "Resistant fraction"]):
    left_edge = min(inv.transform(axes[r, c].yaxis.get_tightbbox(renderer))[0, 0] for r in range(4))
    y_mid = 0.5 * (axes[1, c].get_position().y1 + axes[2, c].get_position().y0)
    fig.text(left_edge - 0.008, y_mid, ylab, rotation=90, ha="right", va="center",
             fontsize=ps.FS_LABEL)

fig.legend(handles=[Line2D([0], [0], color=ref_color, lw=1.4, label="Reference"),
                    Line2D([0], [0], color=pinn_color, lw=1.25, ls="--", label="Hard PINN"),
                    Patch(facecolor="0.82", edgecolor="none", alpha=0.5, label="Treatment ON")],
           loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
           columnspacing=1.6, handlelength=2.4)

ps.save_plos(fig, 2)
