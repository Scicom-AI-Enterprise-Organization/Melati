"""The look of every Melati chart, the preprint's (preprint/plot_figs.py) and the Hugging Face cards' (asr/release/
card.py, emotion/release.py, nonverbal/release.py, scripts/dataset_card.py), so a chart on a card and in the paper read
as one: Inter (the paper's typeface, scicom.sty), Scicom's chart palette (its eight validated series colours in a
fixed order, its muted text and border greys), thin solid grid lines, no tick marks, the legend under the axes, and
an arrow to the better end of an axis that has one.

Inter comes from TeX Live where the paper is built, and from fonts/ on a machine with no TeX: a copy of TeX Live's
Inter files (SIL Open Font License), gitignored, copied there with the code. Without either, DejaVu Sans, with a
warning, since a chart in another face no longer matches the paper.
"""
import subprocess
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _inter():
    dirs = []
    try:
        reg = subprocess.run(["kpsewhich", "Inter-Regular.otf"], capture_output=True, text=True).stdout.strip()
        if reg:
            dirs.append(Path(reg).parent)
    except OSError:
        pass
    dirs.append(ROOT / "fonts")
    for d in dirs:
        files = sorted(d.glob("Inter-*.otf"))
        if files:
            for f in files:
                font_manager.fontManager.addfont(str(f))
            return "Inter"
    warnings.warn("Inter not found (no TeX Live, nothing in fonts/): charts are drawn in DejaVu Sans")
    return "DejaVu Sans"


FONT = _inter()
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.cal": FONT,      # else matplotlib looks up a cursive system face for every mathtext string
    "axes.unicode_minus": True,
    "legend.frameon": False,
})

# Scicom platform tokens (light theme), as hex
FG, TEXT, MUTED = "#18181b", "#3f3f46", "#6b7280"          # --foreground, gray-700, --muted-foreground
BORDER, RULE, BAND = "#e5e7eb", "#d4d4d8", "#f3f4f6"        # --border, gray-300, bg-muted/50
BRAND, ORANGE = "#183b8d", "#f58220"                         # the logo's navy and orange
S1, S2, S3, S4, S5, S6, S7, S8 = ("#3b82f6", "#0095a2", "#21763c", "#c67700", "#7a4aba", "#d44d8c", "#79821d",
                                  "#414aa1")                 # --series-1..8, fixed order, never cycled
GRID = dict(color=BORDER, linewidth=0.6, linestyle="-")
TICK = MUTED


def frame(ax, xlab=None, ylab=None, axis="both"):
    ax.grid(True, axis=axis, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=TICK, labelsize=8, length=0, pad=4)
    if xlab:
        ax.set_xlabel(xlab, fontsize=8.5, color=TEXT, labelpad=7)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5, color=TEXT, labelpad=7)


def better(label, lower=True, word="better"):
    """An axis label that says which way is better, with an arrow to the better end. A y label is turned a quarter, so
    the same arrows point down (lower) and up (higher) there (Voice-Isolator's preprint, the same helper)."""
    return f"{label}  ← {word}" if lower else f"{label}  {word} →"


CARD_DPI = 300      # a card's figure is shown at PX_PER_IN, so 300 dpi stays sharp on a 2x screen
PX_PER_IN = 100     # every figure on a card at one scale: its 8 pt text shows at about 11 px, as the page's own text


def hf_url(repo, path, repo_type="model"):
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}/resolve/main/{path}"


def card_img(png, alt, url):
    """A card's <img> at one scale for every figure: the PNG's width in inches (it is saved at CARD_DPI) times
    PX_PER_IN. A markdown image is drawn as wide as the page's column, so a wide chart was blown up past its size and
    went soft on a 2x screen, and two charts of different widths showed their text at two sizes."""
    import struct
    with open(png, "rb") as f:
        width = struct.unpack(">I", f.read(24)[16:20])[0]       # the PNG header's IHDR width
    return f'<img src="{url}" alt="{alt}" width="{round(width / CARD_DPI * PX_PER_IN)}">'


def legend_below(ax, ncol=2, pad=0.18, **kw):
    """Legends go under the axes: inside the frame they cover the lines a reader came for."""
    leg = ax.legend(fontsize=7.8, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -pad), borderaxespad=0.0,
                    handlelength=1.8, columnspacing=1.6, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)
