"""
Fig 10 -- Independent validation at the unpaired treated 86-h endpoint.
Source: real-data notebook, "cell 12 figures" (R3.10, Figure C). No training.

Inputs
  Real dataset/Ramsayer_qR_86h_ENDPOINT_unpaired.csv
  Real_PINN_Results/tables/final_R310_11FR_86h_unpaired_distribution_forecast_population_forecasts_from_48h.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

Q86_CSV = ps.REAL_DATA_DIR / "Ramsayer_qR_86h_ENDPOINT_unpaired.csv"
FORECAST_CSV = (ps.REAL_TABLES /
                "final_R310_11FR_86h_unpaired_distribution_forecast_population_forecasts_from_48h.csv")
ps.require(Q86_CSV, FORECAST_CSV)

DOSES = [30.0, 40.0, 50.0, 100.0]
C_OBS, C_FORECAST, C_BASE = "#222222", "#0072B2", "#8A8A8A"
OFF_OBS, OFF_FORECAST, OFF_BASE = -0.18, 0.00, 0.18

q86 = pd.read_csv(Q86_CSV)
for col in ["dose_ug_ml", "log10_population_plus1", "resistant_fraction_qR"]:
    q86[col] = pd.to_numeric(q86[col], errors="coerce")
q86 = q86[q86["dose_ug_ml"].isin(DOSES)].dropna(
    subset=["dose_ug_ml", "log10_population_plus1", "resistant_fraction_qR"])
forecast = pd.read_csv(FORECAST_CSV)

rng = np.random.default_rng(20261125)
x = np.arange(len(DOSES), dtype=float)


def mean_sd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def points(ax, xc, v, color, marker, alpha):
    xs = xc + rng.uniform(-0.03, 0.03, len(v))
    if marker == "x":
        ax.scatter(xs, v, s=14, marker="x", linewidths=0.7, color=color, alpha=alpha, zorder=2)
    else:
        ax.scatter(xs, v, s=16, marker=marker, facecolors="white", edgecolors=color,
                   linewidths=0.8, alpha=alpha, zorder=2)


def summary(ax, xs, m, s, color, marker, ls):
    ax.errorbar(xs, m, yerr=s, fmt=marker + ls, color=color,
                markerfacecolor=color if marker == "x" else "white", markeredgecolor=color,
                markeredgewidth=0.9, markersize=4.6, linewidth=1.4, elinewidth=0.9, capsize=2.5,
                capthick=0.8, zorder=5)


def panel(ax, obs_col, fc_col, base_col, ylabel, letter, fraction):
    stats = {"obs": [], "fc": [], "base": []}
    for i, dose in enumerate(DOSES):
        o = q86.loc[q86["dose_ug_ml"] == dose, obs_col].to_numpy(float)
        f = forecast.loc[forecast["dose_ug_ml"] == dose, fc_col].to_numpy(float)
        b = forecast.loc[forecast["dose_ug_ml"] == dose, base_col].to_numpy(float)
        points(ax, x[i] + OFF_OBS, o, C_OBS, "o", 0.42)
        points(ax, x[i] + OFF_FORECAST, f, C_FORECAST, "D", 0.45)
        points(ax, x[i] + OFF_BASE, b, C_BASE, "x", 0.38)
        stats["obs"].append(mean_sd(o))
        stats["fc"].append(mean_sd(f))
        stats["base"].append(mean_sd(b))
    for key, off, color, marker, ls in [("obs", OFF_OBS, C_OBS, "o", "-"),
                                        ("fc", OFF_FORECAST, C_FORECAST, "D", "--"),
                                        ("base", OFF_BASE, C_BASE, "x", ":")]:
        m, s = np.array(stats[key]).T
        summary(ax, x + off, m, s, color, marker, ls)
    ax.set_xticks(x, [str(int(d)) for d in DOSES])
    ax.set_xlabel("Streptomycin dose (µg/mL)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    ps.open_axes(ax)
    ax.tick_params(direction="in")
    ps.panel_label(ax, letter, x=-0.17, y=1.0)
    if fraction:
        ax.set_ylim(-0.02, 1.08)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(-0.2, lo), hi + 0.35)


fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 3.3))
panel(axes[0], "log10_population_plus1", "forecast_log10_B86", "persistence_log10_B86",
      r"$\log_{10}(B_{86}+1)$", "A", False)
panel(axes[1], "resistant_fraction_qR", "forecast_qR86", "persistence_qR86",
      r"Resistant fraction $q_R(86)$", "B", True)
fig.legend(handles=[
    Line2D([0], [0], marker="o", ls="-", color=C_OBS, mfc="white", mec=C_OBS, mew=0.9, lw=1.4, ms=4.6,
           label="Observed 86 h (unpaired)"),
    Line2D([0], [0], marker="D", ls="--", color=C_FORECAST, mfc="white", mec=C_FORECAST, mew=0.9, lw=1.4,
           ms=4.6, label="Mechanistic forecast"),
    Line2D([0], [0], marker="x", ls=":", color=C_BASE, lw=1.4, ms=4.6, label="48-h persistence baseline")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.1,
    columnspacing=1.4, handletextpad=0.5)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.85, wspace=0.30)
ps.save_plos(fig, 10)
