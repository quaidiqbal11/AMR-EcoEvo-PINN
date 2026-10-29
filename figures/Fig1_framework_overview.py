"""
Fig 1 - Simplex- and capacity-preserving PINN (flow-diagram version).
Q. Iqbal, Z. Shah (HBKU).

Layout follows a left-to-right network diagram: feature vectors are drawn as stacks of cells
with their size above, operations as narrow upright boxes, and the training loss is fed by a
bracket from the network outputs. Network size (4 hidden layers x 64 units) matches the notebooks.

Writes Fig1.pdf, Fig1.png and Fig1.tif (300 dpi) to figures/output/ when run from the
repository, or to the Desktop when pasted into a notebook cell.
"""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

try:                                   # run as a file inside the repository
    OUT_DIR = Path(__file__).resolve().parent / "output"
except NameError:                      # pasted into a notebook cell
    OUT_DIR = Path.home() / "Desktop"
DPI = 300
W, H = 1080, 520                       # drawing grid (y runs downwards)
FIG_W = 7.5                            # inches (PLOS full width)

INK = "#111111"
BLUE, BLUE_L = "#2B6CB0", "#A9C8EC"    # sensitive / hidden features
ORANGE_L = "#F6C08F"                   # resistant
GREEN_L = "#B7DFA8"                    # abundance n
PURPLE_L = "#CDB8E8"                   # shared parameters
GREY_L = "#C9C9C9"                     # residual / inputs
LW = 1.8

font = [f for f in ("Arial", "Liberation Sans", "DejaVu Sans")
        if f in {x.name for x in font_manager.fontManager.ttflist}][0]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font],
                     "mathtext.fontset": "stixsans", "pdf.fonttype": 42})

fig = plt.figure(figsize=(FIG_W, FIG_W * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


# ----------------------------------------------------------------------------- helpers
def txt(x, y, s, size=8.5, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, color=INK, zorder=10, **kw)


def rbox(x0, y0, x1, y1, label=None, size=8.5, vertical=False, lw=LW):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=9",
                                fc="white", ec=INK, lw=lw, zorder=4))
    if label:
        txt((x0 + x1) / 2, (y0 + y1) / 2, label, size=size, rotation=90 if vertical else 0,
            linespacing=1.15)


def arrow(p, q, lw=LW, head=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=INK, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=5))


def line(xs, ys, lw=LW):
    ax.plot(xs, ys, color=INK, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=5)


def vec(x, yc, colors, dim=None, label=None, cell=22, label_gap=16):
    """Column vector of cells centred at (x, yc); returns (top, bottom)."""
    n = len(colors)
    top = yc - n * cell / 2
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x - cell / 2, top + i * cell), cell, cell, fc=c, ec=INK, lw=1.3,
                               zorder=6))
    bottom = top + n * cell
    if dim is not None:
        txt(x, top - 14, dim)
    if label:
        txt(x, bottom + label_gap, label, va="top", linespacing=1.15)
    return top, bottom


def documents(x, y, s=1.0):
    """Stack of three pages with text lines (data icon)."""
    w, h = 46 * s, 58 * s
    for k in (2, 1, 0):
        x0, y0 = x - w / 2 + 7 * k, y - h / 2 - 7 * k
        ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=INK, lw=1.4, zorder=6 - k * 0.1))
    x0, y0 = x - w / 2, y - h / 2
    for j in range(5):
        yy = y0 + 12 * s + j * 9 * s
        ax.plot([x0 + 8 * s, x0 + w - 8 * s - (12 * s if j == 4 else 0)], [yy, yy], color=INK,
                lw=1.6, solid_capstyle="butt", zorder=8)


# ----------------------------------------------------------------------------- top row: network
Y = 185                                                    # main flow line

# hard-constraint guarantee (top left)
txt(30, 34, "Hard constraints, exact for all $t$:", ha="left", style="italic")
txt(30, 62, "$S, R \\geq 0$,   $S + R \\leq K$,   $p_S + p_R = 1$", ha="left")

# input (t, e)
vec(48, Y, [GREY_L, GREY_L], dim="2", label="Input\n$(t, e)$")
arrow((61, Y), (102, Y))

# environment-specific state network
rbox(104, Y - 40, 250, Y + 40, "State network\n$\\phi_e$  (4 \u00d7 64)")
arrow((252, Y), (295, Y))

# hidden features
vec(316, Y, [BLUE_L] * 5, dim="64", label="Hidden\nfeatures")

# fork to the two heads
YN, YP = 110, 268
line([329, 355], [Y, Y])
line([355, 355], [YN, YP])
arrow((355, YN), (386, YN))
arrow((355, YP), (386, YP))
rbox(388, YN - 42, 424, YN + 42, "$\\sigma(\\cdot)$", vertical=True)
rbox(388, YP - 50, 424, YP + 50, "softmax", vertical=True)
arrow((426, YN), (466, YN))
arrow((426, YP), (466, YP))
vec(488, YN, [GREEN_L], dim="1", label="Abundance $n$", label_gap=8)
vec(488, YP, [BLUE_L, ORANGE_L], dim="2", label="Composition\n$p_S, p_R$", label_gap=8)

# merge into the simplex-capacity layer
line([501, 566], [YN, YN])
line([501, 566], [YP, YP])
line([566, 566], [YN, YP])
arrow((566, Y), (596, Y))
rbox(598, Y - 92, 650, Y + 92, "$S = K\\,n\\,p_S$\n$R = K\\,n\\,p_R$", vertical=True)
txt(624, Y + 102, "Simplex\u2013capacity\nlayer", va="top", linespacing=1.1)
arrow((652, Y), (694, Y))

# states
vec(716, Y, [BLUE_L, ORANGE_L], dim="2", label="States\n$S, R$")
arrow((729, Y), (771, Y))

# physics residual with shared parameters
rbox(773, Y - 92, 825, Y + 92, "ODE residual\nautodiff", vertical=True)
vec(799, 46, [PURPLE_L, PURPLE_L], cell=20)
txt(780, 46, "Shared $\\psi = (\\log_{10}\\mu,\\ c_R)$", ha="right")
arrow((799, 66), (799, Y - 94))
arrow((827, Y), (869, Y))
vec(892, Y, [GREY_L] * 3, label="Physics\nresidual")

# ----------------------------------------------------------------------------- bottom row: loss
YB = 372                                                   # bracket level
line([716, 716], [Y + 88, YB])
line([892, 892], [Y + 85, YB])
line([716, 892], [YB, YB])
arrow((804, YB), (804, 396))
rbox(600, 398, 1010, 466,
     "Composite loss\n$\\mathcal{L} = \\lambda_B\\mathcal{L}_B + \\lambda_q\\mathcal{L}_q"
     " + \\lambda_f\\mathcal{L}_{\\mathrm{phys}} + \\lambda_{\\mathrm{reg}}\\mathcal{L}_{\\mathrm{reg}}$")

documents(330, 424, s=0.9)
txt(330, 466, "Observations\n$B(t)$ dense, $q_R(t)$ sparse", va="top", linespacing=1.15)
arrow((370, 432), (598, 432))

# ----------------------------------------------------------------------------- save
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig1.pdf")
fig.savefig(OUT_DIR / "Fig1.png", dpi=DPI)
buf = BytesIO()
fig.savefig(buf, format="png", dpi=DPI)
buf.seek(0)
Image.open(buf).convert("RGB").save(OUT_DIR / "Fig1.tif", compression="tiff_lzw", dpi=(DPI, DPI))
plt.close(fig)
print("Saved Fig1.pdf, Fig1.png, Fig1.tif to", OUT_DIR)
