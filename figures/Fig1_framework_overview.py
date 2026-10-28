#!/usr/bin/env python3
"""
Figure 1 -- Proposed simplex- and capacity-preserving PINN framework.

Layout (left to right, plus a bottom bar)
-----------------------------------------
1. Eco-evolutionary dynamics and observations: S and R compartments, mutation
   flux mu, environmental exposure A_e(t), dense total load B(t) and sparse
   resistant fraction q_R(t).
2. Constrained PINN (one per environment e): network phi_e, sigma/softmax heads,
   simplex-capacity map S = K n p_S, R = K n p_R, and the exact constraints.
3. Shared biological parameters psi = (mu, c_R) coupling the networks
   phi_1 ... phi_E.
4. Training objective L = l_B L_B + l_q L_q + l_f L_phys + l_reg L_reg.

Output
------
Fig1.pdf  vector version for the LaTeX manuscript
Fig1.png  raster preview
Fig1.tif  PLOS submission file (RGB, LZW compression)
The figure is 7.5 in wide (PLOS maximum), text >= 8 pt, raster files at 300 dpi
by default.

Usage
-----
    python Fig1_framework_overview.py
    python Fig1_framework_overview.py --out-dir figures --dpi 600

Requirements: numpy, matplotlib, pillow.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
from PIL import Image

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
try:                                    # run as a script
    DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"
except NameError:                       # pasted into a Jupyter cell
    DEFAULT_OUT_DIR = Path.cwd() / "output"
DEFAULT_DPI = 300

# The drawing is laid out on a 1920 x 830 grid (y downwards), scaled to 7.5 in.
GRID_W, GRID_H = 1920.0, 830.0
FIG_WIDTH_IN = 7.5
FIG_HEIGHT_IN = FIG_WIDTH_IN * GRID_H / GRID_W   # 3.24 in

# Font sizes (pt)
FS_TITLE = 10.0
FS_SYMBOL = 10.5
FS_EQ = 10.0
FS_TEXT = 9.0
FS_SMALL = 8.5


@dataclass(frozen=True)
class Style:
    ink: str = "#1A1A1A"
    grey: str = "#8C8C8C"
    arrow_block: str = "#5B8FC7"
    # panel 1 (blue)
    p1_fill: str = "#EDF5FD"
    p1_edge: str = "#A9CDEE"
    p1_band: str = "#D9E9FA"
    # panel 2 (green)
    p2_fill: str = "#F0F9EC"
    p2_edge: str = "#AEDA9F"
    p2_band: str = "#DFF1D6"
    p2_box: str = "#DCF0D2"
    p2_box_edge: str = "#7FBF6A"
    # panel 3 (orange)
    p3_fill: str = "#FEF5EC"
    p3_edge: str = "#F2BB8C"
    p3_band: str = "#FBE2CB"
    # objective bar
    obj_fill: str = "#EEEEF1"
    obj_edge: str = "#A3A3A8"
    # compartments
    s_fill: str = "#D3E4F8"
    s_edge: str = "#2F6DB5"
    r_fill: str = "#FCE3CC"
    r_edge: str = "#E07B22"
    a_fill: str = "#EEEEEE"
    a_edge: str = "#9A9A9A"
    head_fill: str = "#EDF5FD"
    # network
    node_in: str = "#D3E4F8"
    node_hidden: str = "#E4E4E4"
    node_edge: str = "#6E6E6E"
    # data
    dense: str = "#2F6DB5"
    sparse: str = "#C8102E"
    # environment networks
    env_colors: tuple = (("#D3E4F8", "#2F6DB5"),
                         ("#EADCF7", "#9B59D0"),
                         ("#FAD6EA", "#D23C8C"))


STYLE = Style()


def configure_matplotlib() -> None:
    """Times-like serif text with STIX mathematics; TrueType fonts in the PDF."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = [name for name in ("Times New Roman", "Times", "Liberation Serif", "STIXGeneral")
             if name in available] or ["serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


# -----------------------------------------------------------------------------
# Drawing helpers (grid coordinates, y increases downwards)
# -----------------------------------------------------------------------------
class Canvas:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, GRID_W)
        self.ax.set_ylim(GRID_H, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    # --- primitives ----------------------------------------------------------
    def box(self, x0, y0, x1, y1, fill, edge, lw=1.0, radius=14, z=1):
        patch = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
        self.ax.add_patch(patch)
        return patch

    def text(self, x, y, s, size=FS_TEXT, weight="normal", ha="center", va="center",
             color=None, z=6, **kw):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                            color=color or STYLE.ink, zorder=z, **kw)

    def arrow(self, p0, p1, color=None, lw=1.1, dashed=False, rad=0.0, both=False,
              head=7, z=4, shrink=0):
        style = "<|-|>" if both else "-|>"
        patch = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=head,
                                connectionstyle=f"arc3,rad={rad}",
                                color=color or STYLE.ink, linewidth=lw,
                                linestyle=(0, (3, 2.2)) if dashed else "-",
                                shrinkA=shrink, shrinkB=shrink, zorder=z)
        self.ax.add_patch(patch)
        return patch

    def polyline_arrow(self, points, color=None, lw=1.1, head=7, z=4):
        """Elbow connector: straight segments through `points`, arrow head at the end."""
        color = color or STYLE.ink
        xs, ys = zip(*points[:-1])
        self.ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=z)
        self.arrow(points[-2], points[-1], color=color, lw=lw, head=head, z=z)

    def block_arrow(self, x0, x1, y, width=26, head_len=20, head_width=46):
        """Thick right-pointing arrow linking two panels."""
        hw, tw = head_width / 2, width / 2
        verts = [(x0, y - tw), (x1 - head_len, y - tw), (x1 - head_len, y - hw), (x1, y),
                 (x1 - head_len, y + hw), (x1 - head_len, y + tw), (x0, y + tw), (x0, y - tw)]
        codes = [MPath.MOVETO] + [MPath.LINETO] * 6 + [MPath.CLOSEPOLY]
        self.ax.add_patch(PathPatch(MPath(verts, codes), facecolor=STYLE.arrow_block,
                                    edgecolor="none", zorder=7))

    def node(self, x, y, r, fill, edge=None, alpha=1.0, z=5):
        self.ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge or STYLE.node_edge,
                                 linewidth=0.9, alpha=alpha, zorder=z))

    def brace(self, x, y0, y1, depth=16, lw=1.2):
        """Right-facing curly brace '}' spanning y0..y1 at horizontal position x."""
        ym, h = 0.5 * (y0 + y1), y1 - y0
        # four cubic Bezier segments: top end -> upper arm -> tip -> lower arm -> bottom end
        verts = [(x, y0),
                 (x + depth * 0.75, y0), (x + depth * 0.5, y0 + 8), (x + depth * 0.5, y0 + 0.18 * h),
                 (x + depth * 0.5, ym - 0.12 * h), (x + depth * 0.55, ym - 3), (x + depth * 1.1, ym),
                 (x + depth * 0.55, ym + 3), (x + depth * 0.5, ym + 0.12 * h), (x + depth * 0.5, y1 - 0.18 * h),
                 (x + depth * 0.5, y1 - 8), (x + depth * 0.75, y1), (x, y1)]
        codes = [MPath.MOVETO] + [MPath.CURVE4] * 12
        self.ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none",
                                    edgecolor=STYLE.ink, linewidth=lw, zorder=5))

    # --- output ----------------------------------------------------------------
    def save(self, out_dir: Path, stem: str = "Fig1", dpi: int = DEFAULT_DPI) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf, png, tif = (out_dir / f"{stem}.{ext}" for ext in ("pdf", "png", "tif"))
        self.fig.savefig(pdf)
        self.fig.savefig(png, dpi=dpi)
        buffer = BytesIO()
        self.fig.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        Image.open(buffer).convert("RGB").save(tif, compression="tiff_lzw", dpi=(dpi, dpi))
        plt.close(self.fig)
        return [pdf, png, tif]


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------
def draw_dynamics_panel(c: Canvas) -> None:
    s = STYLE
    c.box(10, 12, 628, 628, s.p1_fill, s.p1_edge, lw=1.0, radius=18, z=0)
    c.box(22, 20, 615, 100, s.p1_band, "none", radius=14, z=0.5)
    c.text(312, 59, "Eco-evolutionary dynamics\nand observations", size=FS_TITLE,
           weight="bold", linespacing=1.05)

    # compartments
    c.box(108, 205, 242, 288, s.s_fill, s.s_edge, lw=1.3, radius=10, z=3)
    c.box(390, 205, 524, 288, s.r_fill, s.r_edge, lw=1.3, radius=10, z=3)
    c.text(175, 247, "$S$", size=FS_SYMBOL + 1)
    c.text(457, 247, "$R$", size=FS_SYMBOL + 1)

    # growth/death self-loops
    c.text(167, 139, "growth / death", size=FS_SMALL)
    c.text(470, 139, "growth / death", size=FS_SMALL)
    c.arrow((150, 204), (202, 204), color=s.s_edge, lw=1.2, rad=-1.9, head=7)
    c.arrow((432, 204), (484, 204), color=s.r_edge, lw=1.2, rad=-1.9, head=7)

    # mutation flux
    c.arrow((243, 247), (389, 247), lw=1.3, head=8)
    c.text(315, 207, r"$\mu$", size=FS_SYMBOL)

    # environmental exposure
    c.box(256, 318, 376, 383, s.a_fill, s.a_edge, lw=1.0, radius=9, z=3)
    c.text(316, 351, "$A_e(t)$", size=FS_SYMBOL)
    c.arrow((254, 342), (176, 291), lw=0.9, dashed=True, both=True, head=6, color="#3A3A3A")
    c.arrow((378, 342), (458, 291), lw=0.9, dashed=True, both=True, head=6, color="#3A3A3A")
    c.text(316, 404, "environmental exposure", size=FS_SMALL)

    # separator
    c.ax.plot([26, 612], [432, 432], color="#A8A8A8", lw=0.8, ls=(0, (4, 3)), zorder=2)

    # observations: dense B(t), sparse q_R(t)
    c.text(180, 470, "$B(t) = S + R$", size=FS_EQ)
    c.text(483, 470, "$q_R(t) = R/B$", size=FS_EQ)
    for x0, x1 in ((70, 290), (374, 593)):
        c.ax.plot([x0, x0, x1], [498, 593, 593], color=s.ink, lw=1.4,
                  solid_joinstyle="miter", zorder=3)
    c.text(42, 551, "$B$", size=FS_EQ)
    c.text(346, 551, "$q_R$", size=FS_EQ)
    c.text(180, 613, "$t$", size=FS_EQ)
    c.text(483, 613, "$t$", size=FS_EQ)

    xb = np.linspace(75, 283, 24)
    u = (xb - 75) / (283 - 75)
    # dip, rebound, peak and decline (illustrative dense load trajectory)
    yb = np.interp(u, [0, 0.07, 0.17, 0.32, 0.55, 0.75, 1.0],
                   [549, 566, 580, 551, 523, 545, 553])
    c.ax.scatter(xb, yb, s=4.5, color=s.dense, zorder=4, linewidths=0)

    xq = [378, 414, 437, 484, 527, 576, 585]
    yq = [546, 571, 551, 529, 538, 553, 535]
    c.ax.scatter(xq, yq, s=5.5, color=s.sparse, zorder=4, linewidths=0)


def draw_pinn_panel(c: Canvas) -> None:
    s = STYLE
    c.box(660, 12, 1437, 628, s.p2_fill, s.p2_edge, lw=1.0, radius=18, z=0)
    c.box(672, 22, 1425, 90, s.p2_band, "none", radius=14, z=0.5)
    c.text(1048, 53, r"Constrained PINN  (one per environment $\boldsymbol{e}$)",
           size=FS_TITLE, weight="bold")

    # input
    c.text(698, 256, "$(t, e)$", size=FS_SYMBOL)
    c.arrow((738, 256), (782, 256), lw=1.3, head=8)

    # network phi_e: 3 x 3 nodes
    layers_x = (815, 868, 921)
    nodes_y = (185, 252, 320)
    r = 19
    for xa, xb in zip(layers_x[:-1], layers_x[1:]):
        for ya in nodes_y:
            for yb in nodes_y:
                c.ax.plot([xa, xb], [ya, yb], color="#555555", lw=0.55, zorder=4)
    for i, x in enumerate(layers_x):
        for y in nodes_y:
            fill = s.node_in if i == 0 else s.node_hidden
            alpha = 0.55 if i == 1 else 1.0
            c.node(x, y, r, fill, alpha=alpha)
    for y in nodes_y:                       # short input arrows
        c.arrow((786, y + (256 - y) * 0.45), (797, y + (256 - y) * 0.05), lw=0.7, head=4.5,
                color="#555555", dashed=(y != 252))
    c.text(870, 378, r"$\phi_e$", size=FS_SYMBOL + 1)

    # heads
    c.box(1018, 185, 1115, 240, s.head_fill, "#3A3A3A", lw=1.0, radius=8, z=3)
    c.box(1015, 280, 1121, 332, s.head_fill, "#3A3A3A", lw=1.0, radius=8, z=3)
    c.text(1066, 212, r"$\sigma(\cdot)$", size=FS_EQ)
    c.text(1068, 306, "softmax", size=FS_SMALL - 0.5)
    for y0, y1 in ((185, 212), (252, 212)):
        c.arrow((940, y0 + (y1 - y0) * 0.15), (1016, y1), lw=0.8, head=5)
    for y0, y1 in ((252, 300), (320, 300)):
        c.arrow((940, y0 + (y1 - y0) * 0.15), (1013, y1), lw=0.8, head=5)
    c.arrow((1116, 212), (1150, 212), lw=0.9, head=5)
    c.arrow((1122, 306), (1146, 306), lw=0.9, head=5)
    c.text(1170, 212, "$n$", size=FS_SYMBOL)
    c.text(1187, 309, "$p_S, p_R$", size=FS_EQ)

    # brace and simplex-capacity map
    c.brace(1228, 180, 352, depth=24)
    c.box(1264, 195, 1428, 343, s.p2_box, s.p2_box_edge, lw=1.1, radius=10, z=3)
    c.text(1345, 244, "$S = K n p_S$", size=FS_EQ)
    c.text(1345, 298, "$R = K n p_R$", size=FS_EQ)

    # exact constraints
    c.box(697, 433, 1398, 593, s.p2_box, s.p2_box_edge, lw=1.1, radius=14, z=3)
    c.text(1047, 470, "Constraints", size=FS_TITLE, weight="bold")
    c.text(1047, 535, r"$S, R \geq 0,\qquad S + R \leq K,\qquad p_S + p_R = 1$",
           size=FS_SYMBOL)


def draw_shared_panel(c: Canvas) -> None:
    s = STYLE
    c.box(1462, 12, 1913, 628, s.p3_fill, s.p3_edge, lw=1.0, radius=18, z=0)
    c.box(1472, 20, 1905, 168, s.p3_band, "none", radius=14, z=0.5)
    c.text(1689, 52, "Shared biological parameters", size=FS_TITLE - 0.5, weight="bold")
    c.text(1689, 123, r"$\psi\ =\ (\mu, c_R)$", size=FS_SYMBOL + 2)
    c.arrow((1678, 169), (1678, 249), lw=1.1, head=7)

    rows = ((288, "$e = 1$", r"$\phi_1$", "$S_1, R_1$", 251, 324),
            (380, "$e = 2$", r"$\phi_2$", "$S_2, R_2$", 343, 416),
            (532, "$e = E$", r"$\phi_E$", "$S_E, R_E$", 491, 571))
    for (y, e_lbl, phi_lbl, out_lbl, y0, y1), (fill, edge) in zip(rows, s.env_colors):
        c.box(1620, y0, 1730, y1, fill, edge, lw=1.2, radius=9, z=3)
        c.text(1675, (y0 + y1) / 2, phi_lbl, size=FS_SYMBOL + 1)
        c.text(1570, y, e_lbl, size=FS_EQ, ha="right")
        c.arrow((1576, y), (1618, y), lw=1.0, head=6)
        c.arrow((1732, y), (1782, y), lw=1.0, head=6)
        c.text(1842, y + 2, out_lbl, size=FS_EQ)
    for x in (1675, 1835):
        c.text(x, 452, r"$\vdots$", size=FS_SYMBOL)


def draw_objective_bar(c: Canvas) -> None:
    s = STYLE
    c.box(230, 683, 1690, 805, s.obj_fill, s.obj_edge, lw=1.0, radius=12, z=1)
    c.text(422, 730, "Training objective", size=FS_TITLE + 0.5, weight="bold")
    terms = ((652, r"$\mathcal{L}$"), (688, "$=$"), (780, r"$\lambda_B\mathcal{L}_B$"),
             (882, "$+$"), (990, r"$\lambda_q\mathcal{L}_q$"), (1097, "$+$"),
             (1222, r"$\lambda_f\mathcal{L}_{\mathrm{phys}}$"), (1342, "$+$"),
             (1462, r"$\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}$"))
    for x, t in terms:
        c.text(x, 736, t, size=FS_EQ + 0.5)

    # connectors: data and shared parameters feed the objective; objective trains the PINN
    c.polyline_arrow([(146, 640), (146, 731), (228, 731)], lw=1.1)
    c.polyline_arrow([(1775, 640), (1775, 731), (1692, 731)], lw=1.1)
    c.arrow((1003, 682), (1003, 629), lw=1.1, head=7)


def build_figure() -> Canvas:
    configure_matplotlib()
    c = Canvas()
    draw_dynamics_panel(c)
    draw_pinn_panel(c)
    draw_shared_panel(c)
    draw_objective_bar(c)
    c.block_arrow(626, 661, 285)
    c.block_arrow(1438, 1473, 283)
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 1 (proposed framework).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output folder (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="raster resolution")
    args, _ = parser.parse_known_args()      # tolerate Jupyter's own arguments
    paths = build_figure().save(args.out_dir, dpi=args.dpi)
    print(f"Figure 1 ({FIG_WIDTH_IN} x {FIG_HEIGHT_IN:.2f} in, {args.dpi} dpi):")
    for p in paths:
        print("  " + str(p))


if __name__ == "__main__":
    main()
