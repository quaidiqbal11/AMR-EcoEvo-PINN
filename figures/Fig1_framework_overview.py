"""
Fig 1 - simplex- and capacity-preserving PINN framework.
Q. Iqbal, Z. Shah (HBKU). Writes Fig1.pdf, Fig1.png and Fig1.tif to figures/output/.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from PIL import Image

try:                                   # figures/output/ inside the repository
    OUT_DIR = Path(__file__).resolve().parent / "output"
except NameError:                      # pasted into a notebook: save to the Desktop
    OUT_DIR = Path.home() / "Desktop"
DPI = 300
W, H = 1920, 860                 # drawing grid, y runs downwards
FIG_W = 7.5                      # inches (PLOS full width)

BLUE, ORANGE = "#2B6CB0", "#D9730D"
BLUE_L, ORANGE_L = "#E6EEF8", "#FBEADB"
GREEN, GREEN_L = "#2E7D32", "#E8F3E6"          # abundance n / hard constraints
RED = "#B71C1C"                                # drug exposure and gradients
PURPLE, PURPLE_L = "#6A3D9A", "#EFE8F6"
TEAL, TEAL_L = "#00796B", "#E0F2EF"
MAGENTA, MAGENTA_L = "#AD1457", "#FBE6EF"
GOLD_L = "#FFF4D6"
GREY, INK = "#7A7A7A", "black"

sans = [f for f in ("Arial", "Liberation Sans", "DejaVu Sans")
        if f in {x.name for x in font_manager.fontManager.ttflist}][0]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [sans],
                     "mathtext.fontset": "stixsans", "pdf.fonttype": 42,
                     "axes.linewidth": 0.8})

fig = plt.figure(figsize=(FIG_W, FIG_W * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def txt(x, y, s, size=8.5, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, **kw)


def rect(x0, y0, x1, y1, fc="white", ec=INK, lw=0.8, ls="-"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def arrow(p, q, c=INK, lw=0.8, ls="-", rad=0, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=c,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=3))


def line(xs, ys, c=INK, lw=0.8, ls="-"):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="butt", zorder=3)


def panel(x, y, letter, title):
    txt(x, y, letter, size=11, weight="bold", ha="left")
    txt(x + 40, y + 1, title, size=8.5, ha="left")


def inset(x0, y0, x1, y1):
    """Small data plot placed in grid coordinates."""
    a = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])
    a.set_xticks([])
    a.set_yticks([])
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.patch.set_alpha(0)
    return a


# ---------------------------------------------------------------- A: model + data
panel(20, 32, "A", "Model and observations")

rect(110, 190, 240, 270, fc=BLUE_L, ec=BLUE, lw=1.0)
rect(390, 190, 520, 270, fc=ORANGE_L, ec=ORANGE, lw=1.0)
txt(175, 230, "$S$", size=11)
txt(455, 230, "$R$", size=11)

arrow((150, 190), (200, 190), c=BLUE, rad=-1.8)                 # growth / death loops
arrow((430, 190), (480, 190), c=ORANGE, rad=-1.8)
txt(175, 112, "$b_S(1-\\mu)\\Lambda,\\ d_S$", size=8)
txt(455, 112, "$b_R\\Lambda,\\ d_R$", size=8)

arrow((240, 230), (390, 230), lw=1.0)
txt(315, 210, "$\\mu$", size=10)

rect(260, 320, 370, 370, fc="#FBEAEA", ec=RED, ls=(0, (3, 2)))
txt(315, 345, "$A_e(t)$", size=9.5)
arrow((262, 330), (180, 272), c=RED, ls=(0, (3, 2)))
arrow((368, 330), (450, 272), c=RED, ls=(0, (3, 2)))
txt(315, 395, "drug exposure in environment $e$", size=8, color=RED)

t = np.linspace(0, 1, 200)
B = 0.55 - 0.25 * np.exp(-((t - 0.15) / 0.09) ** 2) + 0.35 * np.exp(-((t - 0.5) / 0.16) ** 2)
q = 0.15 + 0.75 / (1 + np.exp(-(t - 0.45) / 0.08))

a1 = inset(70, 470, 290, 590)
a1.plot(t, B, color="#9DB9DE", lw=0.8)
a1.plot(t[::9], B[::9] + np.random.default_rng(1).normal(0, 0.02, len(t[::9])), "o",
        ms=2.0, color=BLUE, mec="none")
a1.set_ylim(0, 1)
a1.set_xlabel("$t$", fontsize=8.5, labelpad=1)
a1.set_ylabel("$B$", fontsize=8.5, rotation=0, labelpad=7, va="center")
txt(180, 448, "$B=S+R$ (dense)", size=8)

a2 = inset(370, 470, 590, 590)
a2.plot(t, q, color="#F2C49B", lw=0.8)
idx = [10, 48, 80, 105, 130, 170, 192]
a2.plot(t[idx], q[idx] + np.random.default_rng(2).normal(0, 0.04, len(idx)), "o",
        ms=2.6, color=ORANGE, mec="none")
a2.set_ylim(0, 1.05)
a2.set_xlabel("$t$", fontsize=8.5, labelpad=1)
a2.set_ylabel("$q_R$", fontsize=8.5, rotation=0, labelpad=9, va="center")
txt(480, 448, "$q_R=R/B$ (sparse)", size=8)

# ---------------------------------------------------------------- B: constrained PINN
line([640, 640], [20, 630], c="0.8")
panel(660, 32, "B", "Constrained PINN, one per environment $e$")

txt(700, 245, "$(t,e)$", size=9.5)
arrow((728, 245), (770, 245))
xs, ys = (800, 860, 920), (175, 245, 315)
for xa, xb in zip(xs[:-1], xs[1:]):
    for ya in ys:
        for yb in ys:
            line([xa, xb], [ya, yb], c="0.7", lw=0.5)
for x, fc in zip(xs, (BLUE_L, "#F0F0F0", "#F0F0F0")):
    for y in ys:
        ax.add_patch(Circle((x, y), 15, fc=fc, ec=INK, lw=0.8, zorder=4))
txt(860, 365, "$\\phi_e$", size=10.5)

rect(990, 182, 1100, 228, fc=GREEN_L, ec=GREEN)
rect(990, 262, 1100, 308, fc=PURPLE_L, ec=PURPLE)
txt(1045, 205, "$\\sigma(\\cdot)$", size=9.5)
txt(1045, 285, "softmax", size=8)
for y0, y1 in ((175, 205), (245, 205), (245, 285), (315, 285)):
    arrow((935, y0), (990, y1), lw=0.7, head=5)
arrow((1100, 205), (1135, 205), lw=0.7, head=5)
arrow((1100, 285), (1135, 285), lw=0.7, head=5)
txt(1145, 205, "$n$", size=9.5, ha="left", color=GREEN)
txt(1145, 285, "$p_S$", size=9.5, ha="left", color=BLUE)
txt(1172, 285, ",", size=9.5, ha="left")
txt(1190, 285, "$p_R$", size=9.5, ha="left", color=ORANGE)

line([1238, 1248, 1248, 1238], [180, 180, 310, 310])            # square bracket
line([1248, 1262], [245, 245])
txt(1272, 225, "$S = K\\,n\\,p_S$", size=9.5, ha="left", color=BLUE)
txt(1272, 268, "$R = K\\,n\\,p_R$", size=9.5, ha="left", color=ORANGE)

rect(700, 440, 1410, 580, fc=GREEN_L, ec=GREEN, lw=0.8)
txt(1055, 475, "Hard constraints (exact for every $t$)", size=8.5, color=GREEN)
txt(1055, 530, "$S,R \\geq 0, \\qquad S+R \\leq K, \\qquad p_S+p_R = 1$", size=10)

# ---------------------------------------------------------------- C: shared parameters
line([1440, 1440], [20, 630], c="0.8")
panel(1460, 32, "C", "Shared parameters")

rect(1565, 88, 1795, 142, fc=GOLD_L, ec="#B8860B")
txt(1680, 115, "$\\psi = (\\mu,\\ c_R)$", size=11)
rows = [(250, "1"), (350, "2"), (520, "E")]
line([1680, 1680], [142, 185])
line([1580, 1580], [185, 520])
line([1580, 1680], [185, 185])
env = [(TEAL_L, TEAL), (PURPLE_L, PURPLE), (MAGENTA_L, MAGENTA)]
for (y, k), (fc, ec) in zip(rows, env):
    arrow((1580, y), (1625, y), head=5)
    rect(1625, y - 30, 1735, y + 30, fc=fc, ec=ec, lw=1.0)
    txt(1680, y, f"$\\phi_{k}$", size=10.5)
    arrow((1735, y), (1775, y), head=5)
    txt(1785, y, f"$S_{k},\\ R_{k}$", size=9.5, ha="left")
    txt(1555, y, f"$e={k}$", size=8.5, ha="right", color=GREY)
txt(1680, 440, "$\\vdots$", size=10)
txt(1810, 440, "$\\vdots$", size=10)

# ---------------------------------------------------------------- D: training objective
line([20, 1900], [660, 660], c="0.8")
panel(20, 700, "D", "Training objective")

eq = [(700, "$\\mathcal{L}(\\phi,\\psi)\\ =$", None, INK),
      (900, "$\\lambda_B\\,\\mathcal{L}_B$", "misfit to $B$", BLUE),
      (990, "$+$", None, INK),
      (1090, "$\\lambda_q\\,\\mathcal{L}_q$", "misfit to $q_R$", ORANGE),
      (1180, "$+$", None, INK),
      (1300, "$\\lambda_f\\,\\mathcal{L}_{\\rm phys}$", "ODE residual (A)", GREEN),
      (1420, "$+$", None, INK),
      (1540, "$\\lambda_{\\rm reg}\\,\\mathcal{L}_{\\rm reg}$", "regularisation", GREY)]
for x, s, note, c in eq:
    txt(x, 745, s, size=10.5, color=c)
    if note:
        txt(x, 795, note, size=8, color=c)

arrow((560, 610), (560, 720), c=GREY, lw=0.7, head=5)            # data enter the loss
txt(570, 640, "data", size=8, color=GREY, ha="left")
arrow((1055, 695), (1055, 600), c=RED, lw=0.9, head=6)           # gradient updates
txt(1065, 625, "$\\nabla_{\\phi_e}\\mathcal{L}$", size=8.5, color=RED, ha="left")
arrow((1820, 695), (1820, 600), c=RED, lw=0.9, head=6)
txt(1830, 625, "$\\nabla_{\\psi}\\mathcal{L}$", size=8.5, color=RED, ha="left")

# ---------------------------------------------------------------- save
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "Fig1.pdf")
fig.savefig(OUT_DIR / "Fig1.png", dpi=DPI)
buf = BytesIO()
fig.savefig(buf, format="png", dpi=DPI)
buf.seek(0)
Image.open(buf).convert("RGB").save(OUT_DIR / "Fig1.tif", compression="tiff_lzw", dpi=(DPI, DPI))
plt.close(fig)
print("Saved Fig1.pdf, Fig1.png, Fig1.tif to", OUT_DIR)
