"""
plos_style.py -- shared settings for re-drawing the AMR-PINN figures to
PLOS Computational Biology figure requirements.

PLOS figure rules applied here
  * width 2.63-7.5 in, height <= 8.75 in (figures are drawn at FINAL size,
    so 8 pt in the script is 8 pt in print -- no later shrinking)
  * text 8-12 pt, Arial (falls back to Liberation Sans, which is metric-identical)
  * TIFF, RGB, LZW compression, 600 dpi  (+ vector PDF for LaTeX)

All folders are relative to the repository root (SECTION 1).
"""

from io import BytesIO
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# =============================================================================
# SECTION 1 -- FOLDERS (relative to the repository; nothing to edit)
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
# Optional override, e.g. to draw the figures from results stored elsewhere.
if os.environ.get("AMR_REPO_ROOT"):
    REPO_ROOT = Path(os.environ["AMR_REPO_ROOT"])

SYNTHETIC_RESULTS = REPO_ROOT / "results" / "synthetic"      # written by notebook 01
ROUND3_TABLES = SYNTHETIC_RESULTS / "round3" / "final" / "tables"
ROUND4_TABLES = SYNTHETIC_RESULTS / "round4" / "final" / "tables"
ROUND4_PREDICTIONS = SYNTHETIC_RESULTS / "round4" / "final" / "predictions"

REAL_DATA_DIR = REPO_ROOT / "data" / "real"                   # Ramsayer CSVs
REAL_RESULTS = REPO_ROOT / "results" / "real_data"            # written by notebook 02
REAL_TABLES = REAL_RESULTS / "tables"
REAL_SOURCE_DATA = REAL_RESULTS / "figures" / "publication_ready" / "source_data"

OUT_DIR = REPO_ROOT / "figures" / "output"

# =============================================================================
# SECTION 2 -- SIZES (inches) AND FONTS (points)
# =============================================================================
FULL_WIDTH = 7.5          # PLOS maximum width
MAX_HEIGHT = 8.75         # PLOS maximum height

FS_TICK = 8.0             # tick labels           (PLOS minimum = 8 pt)
FS_LABEL = 9.0            # axis labels
FS_TITLE = 9.0            # panel titles
FS_LEGEND = 8.0           # legends
FS_ANNOT = 8.0            # in-panel annotations
FS_PANEL = 11.0           # panel letters A, B, C ...


def _pick_font():
    """Arial if installed (Windows/macOS), otherwise Liberation Sans."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans"):
        if name in available:
            return name
    raise RuntimeError(
        "Neither Arial nor Liberation Sans is installed; PLOS requires Arial."
    )


FONT = _pick_font()

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.cal": FONT,
    "mathtext.tt": FONT,
    "font.size": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
    "figure.titlesize": FS_TITLE,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
plt.rcParams.update(RC)


# =============================================================================
# SECTION 3 -- HELPERS
# =============================================================================
def require(*paths):
    """Stop with a clear message if an input file is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Input file(s) not found -- run the notebooks first (see README.md):\n  "
            + "\n  ".join(missing)
        )


def frame(ax, lw=0.8):
    """Black frame on all four sides (Round 3/4 figure style)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(lw)


def open_axes(ax):
    """Top/right spines removed (real-data figure style)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", ha="left", va="bottom")


def save_plos(fig, number):
    """
    Save FigN.pdf (vector, for LaTeX) and FigN.tif (600 dpi, RGB, LZW, for PLOS).
    The figure is saved at exactly its figsize -- no bbox cropping -- so the
    printed size and font sizes are exactly as designed.
    """
    w, h = fig.get_size_inches()
    if w > FULL_WIDTH + 1e-6 or h > MAX_HEIGHT + 1e-6:
        raise ValueError(f"Fig{number} is {w:.2f} x {h:.2f} in; PLOS max is 7.5 x 8.75 in.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"Fig{number}.pdf"
    tif_path = OUT_DIR / f"Fig{number}.tif"
    fig.savefig(pdf_path)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=600)
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(tif_path, compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)
    print(f"Fig{number}: {w:.2f} x {h:.2f} in  ->  {pdf_path.name}, {tif_path.name}  ({OUT_DIR})")
