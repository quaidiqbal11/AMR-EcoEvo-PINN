"""
Fig 5 -- Experimental-data generalization and dose-design sensitivity.
Source: real-data notebook, "cell 9 + cell 10 figures" (R3.8 / R3.9). No training.

Inputs
  Real dataset/Ramsayer_qR_48h_EXACT_linked.csv
  Real_PINN_Results/tables/final_R38_11FR_P1P2_shared_leave_one_dose_qR_raw.csv
  Real_PINN_Results/tables/final_R39_11FR_nested_treatment_diversity_BASE_ONLY_raw.csv
  Real_PINN_Results/tables/final_R39_11FR_nested_treatment_diversity_BASE_ONLY_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

Q48_CSV = ps.REAL_DATA_DIR / "Ramsayer_qR_48h_EXACT_linked.csv"
R38_RAW = ps.REAL_TABLES / "final_R38_11FR_P1P2_shared_leave_one_dose_qR_raw.csv"
R39_RAW = ps.REAL_TABLES / "final_R39_11FR_nested_treatment_diversity_BASE_ONLY_raw.csv"
R39_SUM = ps.REAL_TABLES / "final_R39_11FR_nested_treatment_diversity_BASE_ONLY_summary.csv"
ps.require(Q48_CSV, R38_RAW, R39_RAW, R39_SUM)

q48, r38, r39_raw, r39_sum = (pd.read_csv(p) for p in (Q48_CSV, R38_RAW, R39_RAW, R39_SUM))

TREATED = [30.0, 40.0, 50.0, 100.0]
DESIGNS = ["R1", "R2", "R3", "R4", "R5"]
DESIGN_TICKS = ["30\n(1 dose)", "30, 40\n(2 doses)", "30–50\n(3 doses)",
                "30–100\n(4 doses)", "0–100\n(5 doses)"]
DOSE_COLORS = {30.0: "#4C78A8", 40.0: "#72B7B2", 50.0: "#B279A2", 100.0: "#E45756"}
DESIGN_COLORS = {"R1": "#4C78A8", "R2": "#F58518", "R3": "#54A24B", "R4": "#E45756", "R5": "#9D79BC"}


def clean(ax):
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ps.open_axes(ax)


def design_panel(ax, raw_col, ylabel, title, log=False):
    x = np.arange(len(DESIGNS))
    for i, design in enumerate(DESIGNS):
        vals = r39_raw.loc[r39_raw["design"] == design, raw_col].dropna().to_numpy(float)
        jit = np.linspace(-0.075, 0.075, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=16, alpha=0.72, color=DESIGN_COLORS[design],
                   edgecolor="white", linewidth=0.35, zorder=3)
    means, sds = [], []
    for design in DESIGNS:
        row = r39_sum.loc[r39_sum["design"] == design].iloc[0]
        means.append(float(row[f"{raw_col}_mean"]))
        sd = row[f"{raw_col}_sd"]
        sds.append(0.0 if pd.isna(sd) else float(sd))
    ax.errorbar(x, means, yerr=sds, fmt="D-", color="black", mfc="white", mec="black", ms=4.0,
                capsize=2.5, lw=1.1, zorder=5)
    ax.set_xticks(x, DESIGN_TICKS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    if log:
        ax.set_yscale("log")
    clean(ax)


fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_WIDTH, 5.9))
axA, axB, axC, axD = axes.ravel()

# Panel A -- leave-one-dose phenotype generalization
rng = np.random.default_rng(20260822)
for x, dose in enumerate(TREATED):
    obs = q48.loc[np.isclose(q48["dose_ug_ml"].to_numpy(float), dose), "resistant_fraction_qR"].to_numpy(float)
    pred = r38.loc[np.isclose(r38["held_dose_ug_ml"].to_numpy(float), dose), "predicted_qR_48"].to_numpy(float)
    axA.scatter(np.full(len(obs), x) + rng.normal(0.0, 0.045, len(obs)), obs, s=16,
                color=DOSE_COLORS[dose], alpha=0.55, edgecolor="white", linewidth=0.35, zorder=2)
    axA.errorbar(x - 0.10, float(np.mean(obs)), yerr=float(np.std(obs, ddof=1)) if len(obs) > 1 else 0.0,
                 fmt="o", color="black", mfc="white", ms=4.2, capsize=2.5, lw=1.0, zorder=5)
    jit = np.linspace(-0.035, 0.035, len(pred)) if len(pred) > 1 else np.zeros(len(pred))
    axA.scatter(np.full(len(pred), x + 0.10) + jit, pred, s=16, marker="D", facecolor="none",
                edgecolor="black", linewidth=0.7, zorder=4)
    axA.errorbar(x + 0.10, float(np.mean(pred)), yerr=float(np.std(pred, ddof=1)) if len(pred) > 1 else 0.0,
                 fmt="D", color="black", mfc="black", ms=4.0, capsize=2.5, lw=1.0, zorder=6)
axA.set_xticks(range(4), ["30", "40", "50", "100"])
axA.set_xlabel("Held-out streptomycin dose (µg/mL)")
axA.set_ylabel(r"Resistant fraction $q_R(48\,\mathrm{h})$")
axA.set_ylim(-0.05, 1.05)
axA.set_title("Leave-one-dose phenotype generalization", loc="left", fontweight="bold", pad=4)
clean(axA)
axA.legend(handles=[
    Line2D([0], [0], marker="o", color="none", mfc="0.55", ms=4.5, label="Hidden experimental replicate"),
    Line2D([0], [0], marker="o", color="black", mfc="white", ms=4.2, label="Observed mean ± SD"),
    Line2D([0], [0], marker="D", color="black", mfc="black", ms=4.0, label="Holdout prediction mean ± seed SD")],
    loc="lower right", frameon=True, framealpha=0.9, borderaxespad=0.3)
ps.panel_label(axA, "A", x=-0.16, y=1.02)

design_panel(axB, "validation_rmse_qR", r"Validation RMSE of $q_R$",
             "Phenotype error across nested treatment designs")
ps.panel_label(axB, "B", x=-0.16, y=1.02)
design_panel(axC, "mutation_probability", r"Mutation probability $\mu$",
             r"Design sensitivity of $\mu$", log=True)
ps.panel_label(axC, "C", x=-0.16, y=1.02)
design_panel(axD, "resistance_fitness_cost", r"Resistance fitness cost $c_R$",
             r"Design sensitivity of $c_R$")
ps.panel_label(axD, "D", x=-0.16, y=1.02)

fig.subplots_adjust(left=0.095, right=0.985, bottom=0.10, top=0.94, wspace=0.30, hspace=0.62)
ps.save_plos(fig, 5)
