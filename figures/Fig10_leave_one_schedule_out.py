"""
Fig 10 -- Schedule-specific information from leave-one-schedule-out refits
(paired % change relative to the full M4 design).
Source: Round 4, Cell 23C. No training; recomputes the paired changes from the
saved leave-one-schedule-out metrics exactly as Cell 23C does
(falls back to the reported values if that table is absent).

Input
  round4/final/tables/round4_leave_one_schedule_out_metrics.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

CSV = ps.ROUND4_TABLES / "round4_leave_one_schedule_out_metrics.csv"

METRICS = ["rmse_resistant", "rmse_fraction", "mu_error_percent", "cost_error_percent"]
METRIC_LABELS = [r"RMSE $R$", r"RMSE $q_R$", r"Error in $\mu$", r"Error in $c_R$"]
OMITTED = {"NO_UNTREATED": "Untreated", "NO_EARLY": "Early continuous",
           "NO_LATE": "Late continuous", "NO_PULSED": "Pulsed"}
ROW_ORDER = ["Untreated", "Early continuous", "Late continuous", "Pulsed"]

# Values reported in the manuscript (mean paired % change over seeds); used only
# when the leave-one-schedule-out table from notebook 01 is not available.
PUBLISHED = np.array([[90.1, 122.0, 68.6, 33.3],
                      [-0.5, -35.1, 166.7, -76.5],
                      [128.9, 185.6, 21.8, 73.8],
                      [55.2, 95.1, 186.9, 2.9]])

if CSV.exists():
    loo = pd.read_csv(CSV)
    full = (loo[loo["configuration"] == "FULL"][["seed_id", *METRICS]]
            .rename(columns={m: f"{m}_full" for m in METRICS}))
    paired = loo[loo["configuration"] != "FULL"].merge(full, on="seed_id", how="inner",
                                                       validate="many_to_one")
    paired["omitted"] = paired["configuration"].map(OMITTED)
    for m in METRICS:
        paired[f"{m}_chg"] = 100.0 * (paired[m] - paired[f"{m}_full"]) / paired[f"{m}_full"]
    matrix = (paired.groupby("omitted")[[f"{m}_chg" for m in METRICS]].mean()
              .reindex(ROW_ORDER).to_numpy(float))
else:
    print(f"Fig10: {CSV.name} not found -- drawing the values reported in the manuscript.")
    matrix = PUBLISHED

LIMIT = 200.0
fig, ax = plt.subplots(figsize=(6.0, 3.4))
image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-LIMIT, vmax=LIMIT)
ax.set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
ax.set_yticks(np.arange(len(ROW_ORDER)), ROW_ORDER)
ax.set_ylabel("Omitted schedule")
ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ROW_ORDER), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.45)
ax.tick_params(which="minor", bottom=False, left=False)
for r in range(matrix.shape[0]):
    for c in range(matrix.shape[1]):
        v = matrix[r, c]
        ax.text(c, r, f"{v:+.1f}%", ha="center", va="center", fontsize=9,
                fontweight="bold", color="white" if abs(v) >= 95.0 else "black")
cbar = fig.colorbar(image, ax=ax, pad=0.025, ticks=np.arange(-200, 201, 50))
cbar.set_label("Change after schedule omission\nvs. full M4 (%)")
cbar.ax.tick_params(labelsize=ps.FS_TICK)
fig.subplots_adjust(left=0.22, right=0.96, top=0.97, bottom=0.10)
ps.save_plos(fig, 10)
