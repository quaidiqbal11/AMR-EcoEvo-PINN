"""
Fig 3 -- Local identifiability geometry across nested designs M1-M5.
Source: Round 4, Cell 22 (Figure 22A). No training; reads the plot-value CSV
that Cell 22 exported.

Input
  round4/final/tables/round4_identifiability_geometry_plot_values.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plos_style as ps

# Re-apply the PLOS style (Arial, 8-11 pt). Needed when this runs in a notebook
# after another figure (e.g. Fig 1) has changed matplotlib's fonts.
plt.rcParams.update(ps.RC)

CSV = ps.ROUND4_TABLES / "round4_identifiability_geometry_plot_values.csv"
ps.require(CSV)

DESIGNS = ["M1", "M2", "M3", "M4", "M5"]
df = pd.read_csv(CSV)
df["design_id"] = pd.Categorical(df["design_id"].astype(str), DESIGNS, ordered=True)
df = df.sort_values("design_id").reset_index(drop=True)
x = np.arange(len(df), dtype=float)

sv_max = df["largest_singular_value"].to_numpy(float)
sv_min = df["smallest_singular_value"].to_numpy(float)
kappa = df["condition_number"].to_numpy(float)
corr = np.abs(df["absolute_parameter_correlation"].to_numpy(float))

fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH, 2.75))
lw, ms = 1.4, 4.5

axes[0].plot(x, sv_max, marker="o", lw=lw, ms=ms, label=r"$\sigma_{\max}$")
axes[0].plot(x, sv_min, marker="s", lw=lw, ms=ms, label=r"$\sigma_{\min}$")
axes[0].set_yscale("log")
axes[0].set_ylabel("Normalized sensitivity\nsingular value")
axes[0].set_title("Sensitivity spectrum", loc="left", fontweight="bold", pad=5)
axes[0].legend(loc="upper left", frameon=False, handlelength=1.8, handletextpad=0.4)

axes[1].plot(x, kappa, marker="D", lw=lw, ms=ms - 0.3)
axes[1].set_ylabel(r"Condition number, $\kappa$")
axes[1].set_title("Sensitivity conditioning", loc="left", fontweight="bold", pad=5)
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
pad = max(0.20, 0.08 * (kappa.max() - kappa.min()))
axes[1].set_ylim(max(0.0, kappa.min() - pad), kappa.max() + pad)

axes[2].plot(x, corr, marker="^", lw=lw, ms=ms + 0.3)
axes[2].set_ylim(0.0, 1.02)
axes[2].set_ylabel(r"Local correlation, $|\rho(\log_{10}\mu,\,c_R)|$")
axes[2].set_title("Local parameter coupling", loc="left", fontweight="bold", pad=5)
axes[2].yaxis.set_major_locator(MaxNLocator(nbins=6))

for letter, ax in zip("ABC", axes):
    ax.set_xticks(x)
    ax.set_xticklabels(df["design_id"].astype(str))
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ps.frame(ax)
    ps.panel_label(ax, letter, x=-0.30, y=1.02)

fig.subplots_adjust(left=0.105, right=0.99, bottom=0.11, top=0.87, wspace=0.52)
ps.save_plos(fig, 3)
