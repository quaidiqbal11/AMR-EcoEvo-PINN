"""
Fig 7 -- Observation-objective geometry in the (mu, c_R) plane, designs M1 and M4.
Source: Round 4, Cell 24. Reads the enriched profile-surface CSV (Cell 24 adds
mse and log10_delta_mse to it), so no ODE solves are needed.

Input
  round4/final/tables/round4_profile_surfaces_M1_M4.csv
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import plos_style as ps

CSV = ps.ROUND4_TABLES / "round4_profile_surfaces_M1_M4.csv"
ps.require(CSV)

TRUE_LOG10_MU = math.log10(1.0e-4)
TRUE_COST = 0.10
DESIGNS = ["M1", "M4"]
TITLES = {"M1": "M1 — single schedule", "M4": "M4 — four schedules"}

surface = pd.read_csv(CSV)
surface["design_id"] = surface["design_id"].astype(str)
if "log10_delta_mse" not in surface.columns:
    raise KeyError("Column 'log10_delta_mse' missing: run notebook Cell 24 once so it "
                   "writes the enriched profile table.")

pivots = {d: surface[surface["design_id"] == d]
          .pivot(index="cost", columns="log10_mu", values="log10_delta_mse")
          .sort_index(axis=0).sort_index(axis=1) for d in DESIGNS}
all_z = np.concatenate([p.to_numpy(float).ravel() for p in pivots.values()])
all_z = all_z[np.isfinite(all_z)]
vmin, vmax = float(np.nanpercentile(all_z, 2.0)), float(np.nanmax(all_z))
if vmax <= vmin:
    vmax = vmin + 1.0
levels = np.linspace(vmin, vmax, 30)
line_levels = levels[3::5]

fig = plt.figure(figsize=(ps.FULL_WIDTH, 3.4))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.05], wspace=0.14)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
cax = fig.add_subplot(gs[0, 2])

contour = None
for ax, design, letter in zip([ax1, ax2], DESIGNS, "AB"):
    piv = pivots[design]
    xv, yv, zv = piv.columns.to_numpy(float), piv.index.to_numpy(float), piv.to_numpy(float)
    contour = ax.contourf(xv, yv, zv, levels=levels, cmap="viridis", extend="both")
    ax.contour(xv, yv, zv, levels=line_levels, colors="black", linewidths=0.3, alpha=0.18)
    ax.axvline(TRUE_LOG10_MU, ls=":", lw=0.8, color="black", alpha=0.48, zorder=3)
    ax.axhline(TRUE_COST, ls=":", lw=0.8, color="black", alpha=0.48, zorder=3)
    ax.plot(TRUE_LOG10_MU, TRUE_COST, marker="*", ms=9, mfc="white", mec="black", mew=0.9,
            ls="None", zorder=6)
    frame = surface[surface["design_id"] == design]
    best = frame.loc[frame["mse"].idxmin()]
    ax.plot(float(best["log10_mu"]), float(best["cost"]), marker="X", ms=6.5, color="tab:orange",
            ls="None", zorder=7)
    ax.set_title(TITLES[design], loc="left", fontweight="bold", pad=4)
    ax.set_xlabel(r"$\log_{10}(\mu)$")
    ax.set_xlim(-5.5, -2.5)
    ax.set_ylim(0.0, 0.30)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ps.frame(ax)
    ps.panel_label(ax, letter, x=-0.12, y=1.02)

ax1.set_ylabel(r"Resistance fitness cost, $c_R$")
ax2.tick_params(axis="y", labelleft=False)
cbar = fig.colorbar(contour, cax=cax)
cbar.set_label(r"$\log_{10}(\Delta\mathrm{MSE}+10^{-14})$")
cbar.ax.tick_params(labelsize=ps.FS_TICK)
# ticks only inside the colour range (no tick on the extend triangles)
cb_ticks = [t for t in MaxNLocator(nbins=6).tick_values(vmin, vmax) if vmin - 1e-9 <= t <= vmax + 1e-9]
cbar.set_ticks(cb_ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in cb_ticks])

fig.legend(handles=[
    Line2D([0], [0], marker="*", ls="None", ms=8, mfc="white", mec="black", label="True parameter pair"),
    Line2D([0], [0], marker="X", ls="None", ms=6, color="tab:orange", label="Finite-grid minimum")],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.6,
    handletextpad=0.4)
fig.subplots_adjust(left=0.085, right=0.905, bottom=0.14, top=0.83)
ps.save_plos(fig, 7)
