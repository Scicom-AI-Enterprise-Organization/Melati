#!/usr/bin/env python3
"""One chart per file, for the preprint, drawn only from result files under ../results/.

    ../.venv/bin/python plot_figs.py

Every figure is a single panel in the paper's typeface, Inter (the Scicom preprint style, scicom.sty), with the
legend under the axes, in Scicom's chart palette: its eight validated series colours in a fixed order, its muted
text and border greys, thin solid grid lines. The
style lives in melati/plotstyle.py, which the Hugging Face cards' figures share. Nothing is typed in by hand.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
# the style every Melati chart shares, here and on the Hugging Face cards: font, palette, frame, legend, arrows
from melati.plotstyle import (BAND, BORDER, BRAND, FG, GRID, MUTED, ORANGE, RULE, S1, S2, S3, S4, S5, S6,  # noqa: E402,F401
                              S7, S8, TEXT, TICK, better, frame, legend_below)
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

RES = ROOT / "results"
IMG = HERE / "img"
IMG.mkdir(parents=True, exist_ok=True)
DPI = 200
# systems -> (label, colour, marker): one stable slot each, Melati in slot 1 (the brand primary)
SYSTEMS = {
    "melati": ("Melati-Pipe", S1, "o"),
    "emilia_upstream": ("Emilia-Pipe (upstream)", S4, "s"),
    "emilia_v1": ("Malaysian-Emilia v1", S5, "v"),
    "emilia_v2": ("Malaysian-Emilia v2 (mesolitica fork)", S3, "^"),
    "base": ("whisper-large-v3", S4, "s"),
    "halluc": ("hallucination recipe", S5, "v"),
    "halluc_cv22": ("hallucination recipe + CV22 replay", S1, "o"),
}


def save(fig, name):
    fig.savefig(IMG / name, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"-> preprint/img/{name}")


def J(rel):
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def bars(ax, groups, series, values, colours, width=0.8, fmt=None, horizontal=False, label_size=6.8):
    """Grouped bars: values[s][g]; a value written over each bar (fmt), None drawn as a gap."""
    n = len(series)
    w = width / n
    for i, (s, c) in enumerate(zip(series, colours)):
        xs = np.arange(len(groups)) + (i - (n - 1) / 2) * w
        vs = [v if v is not None else 0 for v in values[i]]
        bb = (ax.barh if horizontal else ax.bar)(xs, vs, w * 0.92, color=c, label=s, zorder=3)
        if fmt:
            for x, v, raw in zip(xs, vs, values[i]):
                if raw is None:
                    continue
                if horizontal:
                    ax.text(v, x, " " + fmt(raw), va="center", ha="left", fontsize=label_size, color=TEXT)
                else:
                    ax.text(x, v, fmt(raw), ha="center", va="bottom", fontsize=label_size, color=TEXT)
    (ax.set_yticks if horizontal else ax.set_xticks)(np.arange(len(groups)))
    (ax.set_yticklabels if horizontal else ax.set_xticklabels)(groups)


pct = lambda v: f"{100 * v:.0f}%"
pct1 = lambda v: f"{100 * v:.1f}%"
f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"


# ----------------------------------------------------------------------------------------------- diagrams

def box(ax, x, y, w, h, title, lines=(), colour=S1, title_size=8.2, line_size=6.6):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                                   linewidth=0.9, edgecolor=colour, facecolor="white", zorder=2))
    ax.add_patch(matplotlib.patches.Rectangle((x, y + h - 0.004), w, 0.004, color=colour, zorder=3))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size, color=FG,
            fontweight="semibold", zorder=4)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - i * 0.042, l, ha="center", va="top", fontsize=line_size, color=TEXT,
                zorder=4)


def arrow(ax, x0, y0, x1, y1, colour=TEXT):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1,
                                                                  mutation_scale=10, shrinkA=0, shrinkB=0), zorder=5)


def fig_pipeline():
    """The stages, what each decides, and the model server behind it."""
    stages = [("A. Front", ["one gain per recording", "events: PANNs", "separation only where", "background is heard"]),
              ("B. Diarize", ["who spoke when", "Who-Spoke-When", "(vLLM)"]),
              ("C. Segment", ["single speaker", "cut only in pauses", "overlap removed,", "edges backed off"]),
              ("D. Transcribe", ["our whisper-large-v3", "merge (vLLM)", "loops re-decoded", "language checked"]),
              ("E. Align", ["word start and end", "Where-Words-Begin-", "and-End"]),
              ("F. Quality", ["SOTA-MOS, DNSMOS", "on the mix and on", "speech + effects"]),
              ("G. Tag", ["emotion: our 5 heads,", "MERaLiON-SER and", "emotion2vec+; events:", "Qwen3-Omni"]),
              ("H. Fill", ["voiced gaps", "transcribed alone:", "filler or dropped", "word"]),
              ("I. Export", ["24 kHz FLAC", "one row per segment", "tags rendered", "Drama-3 style"])]
    colours = [S1, S2, S2, S1, S2, S3, S5, S5, S4]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, h, gap = 0.162, 0.34, 0.042               # a gap wide enough to show the arrow between two stages
    top, bottom = 0.62, 0.06
    x_of = lambda col, row: 0.008 + col * (w + gap) + (0.5 * (w + gap) if row else 0)
    for i, (t, ls) in enumerate(stages):
        row, col = divmod(i, 5)
        x, y = x_of(col, row), (top if row == 0 else bottom)
        box(ax, x, y, w, h, t, ls, colour=colours[i], title_size=7.8, line_size=5.9)
        if col < 4 and i < len(stages) - 1:
            arrow(ax, x + w + 0.006, y + h / 2, x + w + gap - 0.006, y + h / 2)
    # the turn from E (end of row 1) to F (start of row 2): down, across, down, an elbow rather than a diagonal
    # that ran under F's border and hid its head
    xe, xf = x_of(4, 0) + w / 2, x_of(0, 1) + w / 2
    mid = (top + bottom + h) / 2
    ax.plot([xe, xe, xf, xf], [top - 0.012, mid, mid, mid], color=TEXT, lw=1.1, zorder=5, solid_capstyle="round")
    arrow(ax, xf, mid, xf, bottom + h + 0.014)
    save(fig, "fig_pipeline.png")


def fig_prefix():
    """What the decoder reads at each position: at inference, in training with the bug, in training fixed."""
    rows = [("inference", ["<|startoftranscript|>", "<|ms|>", "<|transcribe|>", "<|notimestamps|>", "And", "then"], 3),
            ("training, bug", ["<|startoftranscript|>", "pad", "pad", "pad", "pad", "And"], 4),
            ("training, fixed", ["<|startoftranscript|>", "<|ms|>", "<|transcribe|>", "<|notimestamps|>", "And", "then"], 3)]
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    ax.set_xlim(-1.55, 6); ax.set_ylim(-0.4, 3.0); ax.axis("off")
    for r, (name, toks, first) in enumerate(rows):
        y = 2.25 - r * 0.95
        ax.text(-1.5, y + 0.22, name, ha="left", va="center", fontsize=8, color=FG, fontweight="semibold")
        for k, t in enumerate(toks):
            bad = t == "pad"
            colour = S6 if bad else (S1 if t.startswith("<|") else S3)
            ax.add_patch(matplotlib.patches.FancyBboxPatch((k * 0.98, y), 0.9, 0.44,
                         boxstyle="round,pad=0.01,rounding_size=0.05", linewidth=0.9, edgecolor=colour,
                         facecolor=BAND if bad else "white", zorder=2))
            ax.text(k * 0.98 + 0.45, y + 0.22, t.replace("<|startoftranscript|>", "<|sot|>"), ha="center",
                    va="center", fontsize=5.6 if len(t) > 12 else 6.6, color=S6 if bad else TEXT, zorder=3)
        ax.annotate("first word predicted", xy=(first * 0.98 + 0.45, y), xytext=(first * 0.98 + 0.45, y - 0.28),
                    ha="center", va="center", fontsize=6.2, color=MUTED,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7))
    for k in range(6):
        ax.text(k * 0.98 + 0.45, 2.88, f"position {k}", ha="center", va="center", fontsize=6.4, color=MUTED)
    save(fig, "fig_prefix.png")


def fig_gapmethod():
    """Illustration (synthetic signal): aligned words, the gaps between them, the candidates, the labels."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 4.0, 400)
    env = np.full_like(t, -62.0) + rng.normal(0, 1.2, t.size)
    words = [(0.25, 0.7, "so"), (0.78, 1.25, "we"), (1.95, 2.45, "went"), (2.5, 3.0, "home")]
    for a, b, _ in words:
        env[(t >= a) & (t < b)] = -24 + rng.normal(0, 2.0, ((t >= a) & (t < b)).sum())
    env[(t >= 1.38) & (t < 1.80)] = -40 + rng.normal(0, 1.5, ((t >= 1.38) & (t < 1.80)).sum())   # a breath
    env[(t >= 3.18) & (t < 3.62)] = -30 + rng.normal(0, 2.0, ((t >= 3.18) & (t < 3.62)).sum())   # a laugh
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    frame(ax, "time (s)", "frame level (dB)", axis="y")
    ax.plot(t, env, color=TEXT, lw=0.9, zorder=3)
    thr = -45
    ax.axhline(thr, color=S4, lw=0.9, ls="--", zorder=2)
    ax.text(0.02, thr + 1.0, "candidate level:\nspeech - 40 dB, noise floor + 6 dB", ha="left", va="bottom",
            fontsize=6.0, color=S4)
    for a, b, w in words:
        ax.add_patch(matplotlib.patches.Rectangle((a, -14), b - a, 6, color=S1, alpha=0.9, zorder=3))
        ax.text((a + b) / 2, -11, w, ha="center", va="center", fontsize=7, color="white", zorder=4)
    for (a, b), lab in (((1.38, 1.80), "candidate:\nanswer C = breath"), ((3.18, 3.62), "candidate:\nanswer D = laugh")):
        ax.axvspan(a, b, color=S5, alpha=0.12, zorder=1)
        ax.text((a + b) / 2, -53, lab, ha="center", va="center", fontsize=6.3, color=S5)
    ax.set_ylim(-70, -6)
    save(fig, "fig_gapmethod.png")


def fig_segment():
    """Illustration (synthetic turns): cuts only in pauses, overlap removed from both speakers."""
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 2.5); ax.axis("off")
    a = [(0.3, 6.4), (7.0, 11.8), (14.6, 19.7)]
    b = [(5.9, 7.3), (11.4, 14.9)]
    for y, turns, c, name in ((1.6, a, S1, "speaker A"), (0.8, b, S2, "speaker B")):
        ax.text(-0.05, y + 0.15, name, ha="right", va="center", fontsize=7.2, color=TEXT)
        for s, e in turns:
            ax.add_patch(matplotlib.patches.Rectangle((s, y), e - s, 0.3, color=c, zorder=2))
    for s, e in ((5.9, 6.4), (7.0, 7.3), (11.4, 11.8), (14.6, 14.9)):
        ax.add_patch(matplotlib.patches.Rectangle((s, 0.75), e - s, 1.2, color=S6, alpha=0.18, zorder=3))
    ax.text(13.1, 2.15, "overlap: removed from both speakers", ha="center", fontsize=6.5, color=S6)
    for x in (3.1, 9.4, 17.2):
        ax.plot([x, x], [1.55, 1.95], color=FG, lw=1.0, zorder=4)
    ax.text(3.1, 2.15, "cut in a pause", ha="center", fontsize=6.5, color=FG)
    ax.plot([0, 20], [0.2, 0.2], color=RULE, lw=0.8)
    for x in range(0, 21, 5):
        ax.text(x, -0.15, f"{x} s", ha="center", fontsize=6.4, color=MUTED)
    ax.text(10, -0.5, "illustration: segments end in pauses, fit Whisper's 30 s window, and never carry a "
            "second voice", ha="center", fontsize=6.4, color=MUTED)
    save(fig, "fig_segment.png")


# ----------------------------------------------------------------------------------------------- audit

def fig_ltas():
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    frame(ax, "frequency (kHz)", "level relative to 1-4 kHz (dB)")
    for key, label, c in (("audit_emilia_v1", "Malaysian-Emilia v1 (upstream export)", S4),
                          ("audit_emilia_v2", "Malaysian-Emilia v2 (mesolitica fork)", S3)):
        d = J(f"emilia/{key}.json")
        if not d or not d.get("ltas_db"):
            continue
        db = np.array(d["ltas_db"])
        f = np.fft.rfftfreq(2 * (len(db) - 1), 1 / d["ltas_sr"]) / 1000
        ref = np.median(db[(f >= 1) & (f <= 4)])
        ax.plot(f, db - ref, color=c, lw=1.3, label=label)
    ax.set_ylim(-110, 15)
    legend_below(ax, ncol=1, pad=0.2)
    save(fig, "fig_ltas.png")


def fig_fullscale():
    v1, v2, vox = J("emilia/audit_emilia_v1.json"), J("emilia/audit_emilia_v2.json"), compare_file("vox")
    names = ["Malaysian-Emilia v1", "Malaysian-Emilia v2", "Emilia-Pipe\n(VoxConverse)", "Melati\n(VoxConverse)"]
    vals = [v1["share_peak_at_full_scale"], v2["share_peak_at_full_scale"],
            vox["pipelines"]["emilia"]["audio"]["share_peak_at_full_scale"],
            vox["pipelines"]["melati_all"]["audio"]["share_peak_at_full_scale"]]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    frame(ax, None, better("peaking at full scale"), axis="y")
    bars(ax, names, ["share"], [vals], [S4], width=0.55, fmt=pct1)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "fig_fullscale.png")


# ----------------------------------------------------------------------------------------------- separation

NV_CLASSES = ["breathes", "laughs", "coughs", "sighs", "sneezes", "cries", "screams", "burps"]


def fig_stems():
    d = J("separation/nv_injection.json")
    rows = [d.get(f"{c}/none") or {} for c in NV_CLASSES]
    speech = [r.get("dbs_speech") for r in rows]
    vocal = [(r.get("dbs_speech") or 0) + (r.get("dbs_sfx") or 0) for r in rows]
    uvr = [r.get("uvr_vocals") for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    frame(ax, None, better("vocalisation energy kept", lower=False), axis="y")
    bars(ax, NV_CLASSES, ["DBS-TIGER speech stem", "DBS-TIGER speech + effects (Melati)", "UVR-MDX vocals (Emilia)"],
         [speech, vocal, uvr], [S6, S1, S4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.12)
    legend_below(ax, ncol=3, pad=0.13)
    save(fig, "fig_stems.png")


def fig_dnsmos():
    d = J("separation/nv_injection.json")
    rows = [d.get(f"{c}/none") or {} for c in NV_CLASSES]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    frame(ax, None, "DNSMOS OVRL change", axis="y")
    xs = np.arange(len(NV_CLASSES))
    ax.bar(xs, [r.get("dnsmos_delta") for r in rows], 0.55, color=S4, zorder=3)
    for x, r in zip(xs, rows):
        ax.text(x, r.get("dnsmos_delta", 0) - 0.02, f"{100 * r.get('dnsmos_cross_3', 0):.0f}% fall\nunder 3.0",
                ha="center", va="top", fontsize=6.1, color=TEXT)
    ax.set_xticks(xs); ax.set_xticklabels(NV_CLASSES)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_ylim(min(r.get("dnsmos_delta", 0) for r in rows) - 0.22, 0.05)
    save(fig, "fig_dnsmos.png")


# ----------------------------------------------------------------------------------------------- ASR

# The checkpoints and their numbers live in asr_runs.py (shared with make_tables.py); a colour per run here.
from asr_runs import ASR_RUNS, RELEASE, asr  # noqa: E402
COLOUR = {"base": S4, "bug": S5, "fix": S2, "fix25": S1, "fix50": S1, "distill": S6, "label": S6, "distill25": S3,
          "label25": S3, "d25bl": S7, "ttsbl": S7, "tts": S1, "attn": S8, "lr1e-4": S8, "full2e-5": S8,
          "full5e-6": S8}
COLOUR.update({k: S6 for k in ASR_RUNS if k == "merge" or k.startswith("m_")})
ASR = {k: (path, lab, COLOUR[k]) for k, (path, lab) in ASR_RUNS.items()}


def present(keys):
    return [k for k in keys if asr(k)]


SHORT = {"merge": "released merge", "tts": "Indic + KL", "full5e-6": "full fine-tune, 5e-6"}


def short(k):
    """A run's name as a figure shows it: the registry's label, shortened where it would not fit a tick."""
    return SHORT.get(k, ASR[k][1])


def wrap(label, width=15):
    """An x tick label broken over lines at spaces, so neighbouring labels never collide."""
    import textwrap
    return "\n".join(textwrap.wrap(label, width)) or label


def fig_halluc():
    arms = [("silence_hr", "silence"), ("music_hr", "music"), ("nonspeech_hr", "other non-speech"),
            ("wild_words", "real voice-free audio")]
    ks = present(["base", "bug", "fix", "fix25", RELEASE])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    frame(ax, None, better("clips with any text written"), axis="y")
    bars(ax, [a[1] for a in arms], [short(k) for k in ks], [[asr(k).get(a[0]) for a in arms] for k in ks],
         [ASR[k][2] if k != RELEASE else S6 for k in ks], fmt=pct1, label_size=5.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.12)
    legend_below(ax, ncol=3, pad=0.13)
    save(fig, "fig_halluc.png")


def fig_stop_langs():
    """FLEURS clips returned empty, per language: the fine-tune without replay against the ones with it."""
    ks = present(["fix", "fix25", RELEASE])
    if not ks:
        return
    by = {k: asr(k)["fail"].get("fleurs_empty_by_lang") or {} for k in ks}
    langs = sorted(by[ks[0]], key=lambda l: (-by[ks[0]][l], l))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    frame(ax, None, better("FLEURS clips returned empty"), axis="y")
    x = np.arange(len(langs))
    for k, c, m in zip(ks, (S2, S1, S6), ("o", "s", "D")):
        ax.scatter(x, [by[k].get(l, 0) for l in langs], s=14, color=c, marker=m, label=short(k), zorder=3,
                   linewidths=0)
    ax.set_xticks(x); ax.set_xticklabels(langs, rotation=90, fontsize=6.2)
    ax.set_xlim(-0.8, len(langs) - 0.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.03, 1.05)
    legend_below(ax, ncol=3, pad=0.2)
    save(fig, "fig_stop_langs.png")


STOP_ARMS = [("librispeech_test_clean", "LibriSpeech", S1), ("genuine", "Malaysian speech", S2),
             ("fleurs", "FLEURS", S6), ("cv22", "Common Voice 22", S4)]


def fig_false_stop():
    """Speech clips returned empty, per arm, for every checkpoint: the cost of being allowed to stop."""
    ks = present(["base", "fix", "attn", "full5e-6", "fix25", "fix50", "distill25", "label25", "d25bl", "ttsbl",
                  RELEASE])
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(ks) + 1.1))
    frame(ax, better("speech clips returned empty"), None, axis="x")
    ys = np.arange(len(ks))[::-1]
    for j, (arm, lab, c) in enumerate(STOP_ARMS):
        ax.scatter([(asr(k).get("empty_on_speech") or {}).get(arm, np.nan) for k in ks], ys + (j - 1.5) * 0.13,
                   s=16, color=c, label=lab, zorder=3, linewidths=0)
    ax.set_yticks(ys); ax.set_yticklabels([ASR[k][1] for k in ks], fontsize=7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(-0.005, None)
    legend_below(ax, ncol=4, pad=0.12)
    save(fig, "fig_false_stop.png")


def fig_noise():
    """Speech in noise returned empty, by signal-to-noise ratio."""
    ks = present(["base", "fix", "fix25", RELEASE])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    frame(ax, "signal-to-noise ratio (dB)", better("clips returned empty"))
    for k, c, m in zip(ks, (S4, S2, S1, S6), ("s", "o", "^", "D")):
        d = asr(k)["fail"].get("noise_empty_by_snr") or {}
        xs = sorted(d, key=float)
        ax.plot([float(v) for v in xs], [d[v] for v in xs], marker=m, ms=4, lw=1.4, color=c, label=short(k))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.01, None)
    legend_below(ax, ncol=2, pad=0.2)
    save(fig, "fig_noise.png")


def fig_lid(metric="macro_cer", name="fig_lid.png", ylab="FLEURS CER, language mean", fmt=f3):
    ks = present(["base", "bug", "fix", "fix25", "fix50", "distill25", "label25", RELEASE])
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    frame(ax, None, better(ylab), axis="y")
    det = [(asr(k)["lid"].get("fleurs") or {}).get(metric) for k in ks]
    forc = [(asr(k)["lid"].get("fleurs_forced") or {}).get(metric) for k in ks]
    bars(ax, [wrap(short(k), 11) for k in ks], ["language detected by the model", "language forced"],
         [det, forc], [S6, S1], width=0.62, fmt=fmt, label_size=5.6)
    ax.tick_params(axis="x", labelsize=6.2)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, name)


def fig_lid_acc():
    ks = present(list(ASR))
    rows = [(ASR[k][1], ASR[k][2], (asr(k)["lid"].get("fleurs") or {}).get("lid_accuracy")) for k in ks]
    rows = [r for r in rows if r[2] is not None]
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1]
    # dots, not bars: the axis does not start at zero, and a bar's length would claim it does
    ax.scatter([r[2] for r in rows], ys, s=40, color=[r[1] for r in rows], zorder=4)
    for y, r in zip(ys, rows):
        ax.text(r[2] + 0.004, y, f"{100 * r[2]:.1f}%", va="center", fontsize=6.6, color=TEXT)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    base = (asr("base") or {}).get("lid", {}).get("fleurs", {}).get("lid_accuracy")
    if base:
        ax.axvline(base, color=RULE, lw=0.9, ls="--", zorder=2)
    frame(ax, better("FLEURS language identified correctly", lower=False), None, axis="both")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "fig_lid_acc.png")


def fig_sweep():
    """LoRA against full fine-tuning on the same mix (no replay): FLEURS CER and FLEURS clips returned empty."""
    ks = present(["base", "fix", "lr1e-4", "attn", "full5e-6", "full2e-5"])
    labs = ["whisper-large-v3" if k == "base" else ASR[k][1].replace("prefix fed", "LoRA, lr 5e-4") for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    frame(ax, None, better("CER, or share of clips"), axis="y")
    bars(ax, [wrap(l) for l in labs], ["FLEURS CER, language detected", "FLEURS clips returned empty"],
         [[asr(k).get("fleurs_cer_macro") for k in ks], [(asr(k).get("empty_on_speech") or {}).get("fleurs") for k in ks]],
         [S6, S2], width=0.62, fmt=f3, label_size=6.0)
    ax.tick_params(axis="x", labelsize=6.8)
    legend_below(ax, ncol=2, pad=0.26)
    save(fig, "fig_sweep.png")


def fig_runaway():
    """The reduplication arm: a syllable said n times, written more than 1.5n times (a loop) or not at all."""
    ks = present(["base", "fix", "fix25", "fix50", "distill25", "label25", "d25bl", "ttsbl", RELEASE])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    frame(ax, None, better("reduplication clips"), axis="y")
    bars(ax, [wrap(short(k), 11) for k in ks], ["written past 1.5x (a loop)", "nothing written"],
         [[asr(k).get("redup_runaway") for k in ks], [asr(k).get("redup_empty") for k in ks]], [S6, S2], width=0.62,
         fmt=pct, label_size=5.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=6.0)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "fig_runaway.png")


def fig_perlang():
    """FLEURS CER per language, base against the released model (language detected; the forced one beside)."""
    b, r = asr("base"), asr(RELEASE)
    if not (b and r):
        return
    base = (b["lid"].get("fleurs") or {}).get("per_lang", {})
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    frame(ax, better("whisper-large-v3, language detected (CER)"), better(f"{short(RELEASE)} (CER)"))
    cap = 1.2
    for arm, c, lab in (("fleurs", S6, "language detected"), ("fleurs_forced", S1, "language forced")):
        pl = (r["lid"].get(arm) or {}).get("per_lang", {})
        ks = sorted(set(pl) & set(base))
        ax.scatter([min(base[k], cap) for k in ks], [min(pl[k], cap) for k in ks], s=13, color=c, label=lab, zorder=3,
                   alpha=0.85, linewidths=0)
    ax.plot([0, cap], [0, cap], color=RULE, lw=0.9, zorder=2)
    ax.set_xlim(0, cap); ax.set_ylim(0, cap + 0.05)
    ax.text(cap - 0.02, cap - 0.08, "worse than base above the line", ha="right", fontsize=6.4, color=MUTED)
    ax.text(cap - 0.02, 0.05, f"CER above {cap} drawn at {cap}", ha="right", fontsize=6.2, color=MUTED)
    legend_below(ax, ncol=2, pad=0.14)
    save(fig, "fig_perlang.png")


SUP_MODELS = [("base", "whisper-large-v3"), ("fix", "prefix fed"), ("fix25", "+ 25% Common Voice"),
              ("distill25", "+ 25% CV, language KL"), ("label25", "+ 25% CV, language label"),
              ("d25bl", "+ 25% CV, language KL,\nbase's blank languages")]


def fig_suppress():
    """Voice-free clips with words written, pooled over silence, music and other non-speech, decoded twice: with
    Hugging Face's default suppression (no <|endoftext|> as the first token) and as vLLM serves (none)."""
    d = J("asr/suppress_check.json")
    if not d:
        return

    def pooled(m, mode):
        arms = (d.get(m) or {}).get(mode) or {}
        if len(arms) < 3:
            return None
        return sum(a["words_rate"] * a["n"] for a in arms.values()) / sum(a["n"] for a in arms.values())

    rows = [(lab, pooled(k, "hf"), pooled(k, "vllm")) for k, lab in SUP_MODELS if k in d]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    frame(ax, better("voice-free clips with any text written (silence, music, other non-speech)"), None, axis="x")
    bars(ax, [r[0] for r in rows][::-1], ["Hugging Face default (no stop as the first token)", "as vLLM serves it"],
         [[r[1] for r in rows][::-1], [r[2] for r in rows][::-1]], [S6, S1], width=0.7, fmt=pct1, horizontal=True,
         label_size=6.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.12)
    ax.tick_params(axis="y", labelsize=7.2)
    h, l = ax.get_legend_handles_labels()            # legend top to bottom, as the bars in each pair
    legend_below(ax, ncol=1, pad=0.2, handles=h[::-1], labels=l[::-1])
    save(fig, "fig_suppress.png")



def fig_fillers():
    """PodcastFillers, inside the regions Melati keeps: the same pipeline with base Whisper and with the released merge."""
    b, f = J("fillers/podcastfillers_base.json"), J("fillers/podcastfillers_rel.json")
    if not (b and f):
        return
    groups = [("filler", "precision", "precision"), ("filler", "recall", "recall")]
    vals = [[d["scopes"]["inside"][c][m] for c, m, _ in groups] for d in (b, f)]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    frame(ax, None, better("inside the kept regions", lower=False), axis="y")
    bars(ax, [g[2] for g in groups], ["whisper-large-v3", "released merge"], vals,
         [S4, S1], width=0.62, fmt=pct, label_size=6.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    legend_below(ax, ncol=2, pad=0.27)
    save(fig, "fig_fillers.png")


# ----------------------------------------------------------------------------------------------- emotion heads

HEAD_LABELS = [("meralion", "MERaLiON-SER", S4), ("e2v", "emotion2vec+ large", S4),
               ("meralion + e2v", "the two averaged (the old tagger)", S4),
               ("wavlm", "head on WavLM-large", S2), ("e2v_head", "head on emotion2vec+ features", S2),
               ("w2vbert", "head on w2v-BERT 2.0", S2), ("whisper_ft", "head on Melati's Whisper fine-tune", S2),
               ("whisper", "head on Whisper-large-v3", S2),
               ("whisper + whisper_ft + w2vbert + wavlm + e2v + meralion + e2v", "all seven averaged", S1)]


def fig_emo_heads():
    """Held-out UAR for every model on the same 7,791 clips: the released classifiers, Melati's heads, the mix."""
    ens = (J("emotion/ensemble_all.json") or {}).get("results", {})
    if not ens:
        return
    alias = {"e2v_head": "e2v"}           # in ensemble_all, the head on e2v features is the member named e2v
    cur = J("emotion/current.json") or {}
    rows = []
    for key, lab, c in HEAD_LABELS:
        if key in ("meralion", "e2v", "meralion + e2v"):
            r = cur.get(key.replace(" ", "")) if key != "meralion + e2v" else cur.get("meralion+e2v")
            v = r and r["held_out"]["all"]["uar"]
        elif key.startswith("whisper + "):
            v = ens.get(key, {}).get("held_out", {}).get("uar")
        else:
            r = J(f"emotion/probe_{alias.get(key, key)}.json") or {}
            v = r.get("held_out", {}).get("all", {}).get("uar")
        if v is not None:
            rows.append((lab, c, v))
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 0.9))
    ys = np.arange(len(rows))[::-1]
    ax.scatter([r[2] for r in rows], ys, s=42, color=[r[1] for r in rows], zorder=4)
    for y, r in zip(ys, rows):
        ax.text(r[2] + 0.003, y, f"{r[2]:.3f}", va="center", fontsize=6.8, color=TEXT)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    frame(ax, better("unweighted average recall, held-out clips (nEMO; NonverbalTTS's VoxCeleb half)", lower=False),
          None, axis="both")
    save(fig, "fig_emo_heads.png")


def fig_emo_mer():
    """Each head on the held-out clips as released and with MER2023 trained in, and on MER2023 when it is left out."""
    names = [("whisper", "Whisper-large-v3"), ("whisper_ft", "Whisper fine-tune"), ("w2vbert", "w2v-BERT 2.0"),
             ("wavlm", "WavLM-large"), ("e2v", "emotion2vec+ features")]
    live = {h: J(f"emotion/probe_{h}.json") for h, _ in names}
    mer = {h: J(f"emotion/mer2023/probe_{h}.json") for h, _ in names}
    if not all(mer.values()):
        return
    groups = [lab for _, lab in names]
    vals = [[live[h]["held_out"]["all"]["uar"] for h, _ in names],
            [mer[h]["held_out"]["all"]["uar"] for h, _ in names],
            [mer[h]["loco"]["mer2023"]["uar"] for h, _ in names]]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    frame(ax, None, better("unweighted average recall", lower=False), axis="y")
    bars(ax, groups, ["held-out clips, as released", "held-out clips, MER2023 trained in",
                      "MER2023, left out of training"], vals, [S4, S1, S2], width=0.78, fmt=f2, label_size=5.8)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylim(0, 0.75)
    legend_below(ax, ncol=2, pad=0.2)
    save(fig, "fig_emo_mer.png")


def fig_emo_recall():
    """Recall at 0.75 precision (re-weighted to podcast rates; threshold chosen on one half, measured on the other)."""
    ens = (J("emotion/ensemble_all.json") or {}).get("results", {})
    a, b = ens.get("meralion + e2v"), ens.get(HEAD_LABELS[-1][0])
    if not (a and b):
        return
    labs = ["happy", "sad", "angry", "surprised", "fearful"]
    val = lambda r, l: ((r.get("thresholds_cv") or {}).get(l) or {}).get("recall")
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    frame(ax, None, better("recall at 0.75 precision", lower=False), axis="y")
    bars(ax, labs, ["MERaLiON-SER + emotion2vec+ (the old tagger)", "all seven averaged"],
         [[val(a, l) for l in labs], [val(b, l) for l in labs]], [S4, S1], width=0.62,
         fmt=lambda v: f"{100 * v:.0f}%", label_size=6.8)
    for i, l in enumerate(labs):                       # a class with no threshold reaching 0.75 is never tagged
        if val(a, l) is None:
            ax.text(i - 0.155, 0.01, "never\ntagged", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.75)
    legend_below(ax, ncol=2, pad=0.15)
    save(fig, "fig_emo_recall.png")


# ----------------------------------------------------------------------------------------------- emotion

EMO_MODELS = [("meralion", "MERaLiON-SER", S2), ("e2v", "emotion2vec+ large", S5),
              ("meralion+e2v", "average of the two", S1), ("omni", "Qwen3-Omni", S4)]


def fig_emo(metric="acc", name="fig_emo_acc.png", ylab="accuracy"):
    d = J("emotion/eval.json")
    ms = [(lab, c, d["models"][k]) for k, lab, c in EMO_MODELS if k in d["models"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    frame(ax, None, better(ylab, lower=False), axis="y")
    bars(ax, [m[0] for m in ms], ["all 4,160 clips (12 corpora seen by emotion2vec+)", "held out (nEMO, NonverbalTTS)"],
         [[m[2][metric] for m in ms], [m[2]["held_out"][metric] for m in ms]], [RULE, S1], width=0.6, fmt=f2,
         label_size=6.6)
    ax.set_ylim(0, 1.0)
    legend_below(ax, ncol=1, pad=0.16)
    save(fig, name)


def fig_emo_corpus():
    d = J("emotion/eval.json")
    seen = set(d["seen"].get("e2v", []))
    corp = sorted(d["corpora"], key=lambda c: (c in seen, c))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    frame(ax, None, better("accuracy", lower=False), axis="y")
    vals = [[d["models"][k]["per_corpus"][c]["acc"] for c in corp] for k in ("meralion", "e2v")]
    bars(ax, corp, ["MERaLiON-SER", "emotion2vec+ large"], vals, [S2, S5], width=0.7)
    n_held = sum(c not in seen for c in corp)
    ax.axvspan(-0.5, n_held - 0.5, color=BAND, zorder=0)
    ax.text((n_held - 1) / 2, 1.02, "held out", ha="center", fontsize=6.8, color=TEXT)
    ax.text(n_held + (len(corp) - n_held - 1) / 2, 1.02, "in EmoBox, which emotion2vec+ was fine-tuned on",
            ha="center", fontsize=6.8, color=MUTED)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", labelsize=6.4, rotation=35)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "fig_emo_corpus.png")


def fig_emo_prec():
    d = J("emotion/eval.json")
    m = d["models"]["meralion+e2v"]
    cur = m.get("curves_held_out") or {}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    frame(ax, "threshold on the label's probability", better("precision at podcast rates", lower=False))
    colours = {"happy": S1, "angry": S6, "sad": S2, "surprised": S4, "fearful": S5}
    for lab, rows in cur.items():
        rows = [r for r in rows if r[4] >= 10]
        if not rows:
            continue
        ax.plot([r[0] for r in rows], [r[1] for r in rows], color=colours.get(lab, MUTED), lw=1.4, marker="o",
                ms=2.5, label=f"{lab} ({100 * d['priors'][lab]:.0f}% assumed)")
        t = (m["thresholds"].get(lab) or {}).get("threshold")
        if t is not None:
            r = next(r for r in rows if abs(r[0] - t) < 1e-6)
            ax.scatter([t], [r[1]], s=46, facecolors="none", edgecolors=colours.get(lab), linewidths=1.3, zorder=4)
    ax.axhline(d["target_precision"], color=FG, lw=0.8, ls="--")
    ax.text(0.305, d["target_precision"] + 0.015, "bar for tagging", fontsize=6.6, color=FG)
    ax.set_ylim(0, 1)
    legend_below(ax, ncol=3, pad=0.2)
    save(fig, "fig_emo_prec.png")


# ----------------------------------------------------------------------------------------------- non-verbal

MIN_N = 5      # a share resting on fewer tags than this is not drawn


def nv_methods():
    """[(method, {class: (clip precision, clip recall, share of tags at the right word)})], each value None
    when it rests on fewer than MIN_N clips or tags."""
    keep = lambda v, n: v if n >= MIN_N else None
    out = []
    panns = J("nonverbal/nonverbaltts_panns.json")
    if panns:
        best = {}
        for f, cls in panns["by_floor"].items():
            for c, r in cls.items():
                f1 = 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
                if c not in best or f1 > best[c][0]:
                    best[c] = (f1, keep(r["precision"], r["tp"] + r["fp"]), keep(r["recall"], r["tp"] + r["fn"]),
                               keep(r.get("placement"), r["tp"] + r["fp"]))
        out.append(("PANNs events", {c: v[1:] for c, v in best.items()}))
    for key, lab in (("nonverbaltts_omni", "Qwen3-Omni transcribes and tags"),
                     ("nonverbaltts_omni_strict", "same, prompt that asks for every breath")):
        d = J(f"nonverbal/{key}.json")
        if d:
            out.append((lab, {c: (keep(r["precision"], r["tp"] + r["fp"]), keep(r["recall"], r["tp"] + r["fn"]),
                                  keep(r.get("placement"), r.get("placed", 0)))
                              for c, r in d["per_class"].items()}))
    g = J("nonverbal/nonverbaltts_gaps.json")
    if g:
        out.append(("word gaps + Qwen3-Omni, thresholds from train",
                    {c: (keep(r["clip_precision"], r["clip_tp"] + r["clip_fp"]),
                         keep(r["clip_recall"], r["clip_tp"] + r["clip_fn"]), keep(r["tag_precision"], r["hyp"]))
                     for c, r in g["test"]["calibrated"].items()}))
    h = (J("nonverbal/gap_eval_rel2.json") or {}).get(NV_RULE)
    if h:
        out.append(("word gaps (new rule) + Melati's head trained with NonverbalTTS, thresholds from train",
                    {c: (keep(r["clip_precision"], r["clip_tp"] + r["clip_fp"]),
                         keep(r["clip_recall"], r["clip_tp"] + r["clip_fn"]), keep(r["tag_precision"], r["hyp"]))
                     for c, r in h["test"].items()}))
    return out


NV_RULE = '{"floor_db": 2}'             # the candidate rule the pipeline ships (melati.nv_gaps FLOOR_DB)


NV3 = ["laugh", "cough", "breath"]


def fig_nv(idx, name, ylab):
    ms = nv_methods()
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    frame(ax, None, better(ylab, lower=False), axis="y")
    cols = [S4, S5, S6, S7, S1]
    bars(ax, NV3, [m[0] for m in ms], [[(m[1].get(c) or (None, None, None))[idx] for c in NV3] for m in ms],
         cols[-len(ms):] if len(ms) < len(cols) else cols, fmt=pct, label_size=5.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.1)
    ax.text(0.995, 1.06, f"no bar: fewer than {MIN_N} clips or tags behind it", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=6.2, color=MUTED)
    legend_below(ax, ncol=2, pad=0.13)
    save(fig, name)


def labelled_precision(d, cls):
    """Precision counting only tags where PodcastFillers has a label of any kind (it labelled only what its own
    detector proposed): matched / (matched + unmatched tags overlapping some label)."""
    m = d["scopes"]["inside"].get(cls) or {}
    if not m.get("pred"):
        return None
    hit = round((m.get("precision") or 0) * m["pred"])
    wrong = sum(v for k, v in (d.get("unmatched_predictions_overlap") or {}).get(cls, {}).items() if k != "(no label)")
    return hit / max(hit + wrong, 1)


def fig_nv_pf():
    """PodcastFillers' test episodes through export, inside the kept regions: the same candidates labelled by
    Qwen3-Omni and by Melati's heads, each at thresholds chosen on NonverbalTTS train."""
    q, h, e, f = (J(f"fillers/{k}.json") for k in ("podcastfillers_rel", "podcastfillers_nvrel2",
                                                    "podcastfillers_nvheads_events", "podcastfillers_nvrel4f"))
    if not (q and h):
        return
    groups = ["laugh\nprecision", "laugh precision\nwhere labelled", "laugh\nrecall",
              "breath\nprecision", "breath precision\nwhere labelled", "breath\nrecall"]
    runs = [("Qwen3-Omni", q, S4)] + ([("heads, hand-cut events only", e, S7)] if e else []) + \
        [("first released head", h, S5)] + ([("released model: gap head + frame head", f, S1)] if f else [])

    def row(d):
        out = []
        for cls in ("laughter", "breath"):
            m = d["scopes"]["inside"].get(cls) or {}
            tagged = bool(m.get("pred"))
            out += [m.get("precision") if tagged else None, labelled_precision(d, cls),
                    m.get("recall") if tagged else None]
        return out
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    frame(ax, None, better("inside the kept regions", lower=False), axis="y")
    bars(ax, groups, [r[0] for r in runs], [row(d) for _, d, _ in runs], [r[2] for r in runs], width=0.78, fmt=pct,
         label_size=5.8)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "fig_nv_pf.png")


def fig_nv_frames():
    """The frame head against gap candidates where each test set can score both: AMI's test meetings (its marks, a
    tag right within 0.1 s) and NonverbalTTS's test clips (its word-boundary tags, right within one word), each model
    at the thresholds it ships with."""
    fe, fn = J("nonverbal/frames3_eval_pf.json"), J("nonverbal/frames3_nvtts.json")
    prev_ami = ((J("nonverbal/ami_eval_rel2.json") or {}).get("test") or {}).get("nonverbaltts_thresholds")
    prev_nv = ((J("nonverbal/gap_eval_rel2.json") or {}).get(NV_RULE) or {}).get("test")
    if not (fe and fn and prev_ami and prev_nv):
        return
    lt, ct = fe["podcastfillers"]["threshold"], fe["ami"]["cough"]["threshold"]
    al = fe["ami"]["laugh"].get("test_at_pf_threshold") or fe["ami"]["laugh"]["test"]
    ac = fe["ami"]["cough"]["test"]
    nl, nc = fn["test"]["laugh"][str(lt)], fn["test"]["cough"][str(ct)]
    groups = ["AMI laughs", "AMI laughs\nover speech", "AMI coughs", "NonverbalTTS\nlaughs", "NonverbalTTS\ncoughs"]
    gap = [prev_ami["laugh"]["recall"], 0.0, prev_ami["cough"]["recall"], prev_nv["laugh"]["tag_recall"],
           prev_nv["cough"]["tag_recall"]]
    gap_p = [prev_ami["laugh"]["precision"], None, prev_ami["cough"]["precision"], prev_nv["laugh"]["tag_precision"],
             prev_nv["cough"]["tag_precision"]]
    frm = [al["recall"], al["recall_in_speech"], ac["recall"], nl["recall"], nc["recall"]]
    frm_p = [al["precision"], None, ac["precision"], nl["precision"], nc["precision"]]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    frame(ax, None, better("found", lower=False), axis="y")
    bars(ax, groups, ["gap candidates, first released head", "frame head, released"], [gap, frm], [S5, S1],
         fmt=pct, label_size=6.2)
    for i, (pa, pb) in enumerate(zip(gap_p, frm_p)):
        for x, pv, v in ((i - 0.2, pa, gap[i]), (i + 0.2, pb, frm[i])):
            if pv is not None and v > 0.12:
                ax.text(x, v / 2, f"{100 * pv:.0f}%\nprecise", ha="center", va="center", fontsize=5.4, color="white",
                        linespacing=1.1)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    legend_below(ax, ncol=2, pad=0.22)
    save(fig, "fig_nv_frames.png")


def fig_laugh_sweep():
    """Laughs on PodcastFillers' test episodes through the pipeline at every threshold of the frame head
    (nonverbal/sweep_frames.py), the first released head and Qwen3-Omni as the single points they ship at: a higher
    precision that lies on the curve is a stricter threshold, not a better model."""
    sw, fe = J("nonverbal/frames_sweep_pf.json"), J("nonverbal/frames3_eval_pf.json")
    if not (sw and fe):
        return
    pts = sorted((float(t), v) for t, v in sw["laugh"].items())
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    frame(ax, better("laughs found", lower=False), better("precise where labelled", lower=False))
    ax.plot([v["recall"] for _, v in pts], [v["precision_labelled"] for _, v in pts], color=S1, lw=1.6, marker="o",
            ms=3.2, zorder=3, label="frame head, threshold by threshold")
    where = {fe["podcastfillers"]["threshold"]: (5, -11), fe["ami"]["laugh"]["threshold"]: (7, 3), 0.99: (-9, 7)}
    for t, v in pts:
        if t in where:
            ax.annotate(f"{t:g}", (v["recall"], v["precision_labelled"]), textcoords="offset points", xytext=where[t],
                        fontsize=6.6, color=S1)
    for name, label, c in (("previous release", "first released head", S5), ("Qwen3-Omni", "Qwen3-Omni", MUTED)):
        o = sw["others"][name]
        ax.scatter(o["recall"], o["precision_labelled"], s=30, color=c, zorder=4, label=label, edgecolor="white",
                   linewidth=0.6)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.7, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    legend_below(ax, ncol=2, pad=0.26)
    save(fig, "fig_laugh_sweep.png")


def fig_audioset():
    """AudioSet's taggers (PretrainedSED's BEATs, ATST-F and M2D averaged) on NonverbalTTS at the thresholds the export
    uses: found in each half of the clips, the threshold chosen on the other half, precision written in the bar."""
    d, cfg = J("nonverbal/audioset_BEATs+ATST-F+M2D.json"), J("dataset/export_config.json") or {}
    th = cfg.get("audioset_thresholds") or {}
    if not (d and th):
        return
    cls = [(c, n, lab) for c, n, lab in (("sniff", "sniffs", "sniffs"), ("throat_clear", "clears throat", "throat\nclearing"),
                                          ("cough", "coughs", "coughs"), ("laugh", "laughs", "laughs"))
           if c in th and d["nvtts"].get(n + "/halves")]
    halves = [[d["nvtts"][n + "/halves"][k][str(th[c])] for c, n, _ in cls] for k in (0, 1)]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    frame(ax, None, better("found", lower=False), axis="y")
    bars(ax, [f"{lab}\n(at {th[c]:g})" for c, _, lab in cls], ["one half", "the other half"],
         [[m["recall"] for m in h] for h in halves], [S1, S2], fmt=pct, label_size=6.4)
    for k, h in enumerate(halves):                    # precision, one short line inside the bar: it fits a 10% bar
        for i, m in enumerate(h):
            if m["precision"] is not None and (m["recall"] or 0) >= 0.1:
                ax.text(i + (k - 0.5) * 0.4, m["recall"] / 2, f"{100 * m['precision']:.0f}%", ha="center",
                        va="center", fontsize=6.2, color="white")
    ax.tick_params(axis="x", labelsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "fig_audioset.png")


AS_RESULTS = "audioset/student_v5_release_fp32.json"
# each with the release it became on Scicom-intl/Melati-AudioSet-EAT-Large (the repo's commit history)
AS_STAGES = (("train_clean_e10", "clean_e10/best.pt", "v1"), ("train_big_from_v1", "big_from_v1/best.pt", "v2"),
             ("train_frames1_fw0", "frames1_fw0/clip.pt", "v4"), ("train_v5_big6", "v5_big6/best.pt", "v5"))
AS_FRAMES = ("train_frames1", "frames1/clip.pt", 3, "v3")   # stage 3 trained with the frame head beside it
AS_NAMES = {"ced-base": "CED-base", "eat-large": "EAT-large", "sslam": "SSLAM", "eat-base": "EAT-base", "ast": "AST",
            "beats": "BEATs"}
AS_M = "melati:/mnt/data/melati/models/audioset/"      # results keys: where the box kept each checkpoint, not a path
AS_SIX = (f"beats + ced-base + eat-large + {AS_M}frames1/clip.pt + {AS_M}frames1_fw0/clip.pt + sslam")


def fig_as_map():
    """Every AudioSet tagger on the same evaluation clips in fp32 (audioset/evaluate.py): the released ones, the
    average the student learned from, and the student after each stage. A dot each: the gaps are fractions of a
    point, which bars from zero would hide."""
    r = J(AS_RESULTS)
    if not r:
        return
    pts = [(AS_NAMES[k], v, MUTED, "o") for k, v in r["models"].items() if k in AS_NAMES]
    pts.append(("average of three teachers", r["averages"]["ced-base + eat-large + sslam"], S4, "D"))
    if AS_SIX in r["averages"]:
        pts.append(("average of six teachers", r["averages"][AS_SIX], S4, "D"))
    for n, (_, ckpt, rel) in enumerate(AS_STAGES, 1):
        v = r["models"].get(f"{AS_M}{ckpt}")
        if v is not None:
            pts.append((f"Melati student, stage {n} ({rel})", v, S1 if n == len(AS_STAGES) else S2, "o"))
    v = r["models"].get(f"{AS_M}{AS_FRAMES[1]}")
    if v is not None:
        pts.append((f"stage {AS_FRAMES[2]} with the frame head ({AS_FRAMES[3]})", v, MUTED, "o"))
    pts.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    frame(ax, better("mAP on AudioSet's evaluation set", lower=False), None, axis="x")
    for i, (lab, v, c, mk) in enumerate(pts):
        ax.plot([0.45, v], [i, i], color=RULE, lw=0.8, zorder=1)
        ax.scatter(v, i, s=34, color=c, marker=mk, zorder=3)
        ax.text(v + 0.0012, i, f"{v:.4f}", va="center", fontsize=6.8, color=TEXT)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([p[0] for p in pts], fontsize=7.2)
    ax.set_xlim(0.45, max(p[1] for p in pts) + 0.008)
    save(fig, "fig_as_map.png")


def fig_as_train():
    """The student's mAP at each evaluation (bf16, the moving average), stage after stage, against the best released
    single model in fp32 (bf16 reads at most a tenth of a point under fp32 for these taggers)."""
    r = J(AS_RESULTS)
    logs = [[json.loads(line) for line in (RES / "audioset" / f"{log}.jsonl").read_text().splitlines()]
            for log, *_ in AS_STAGES if (RES / "audioset" / f"{log}.jsonl").exists()]
    if not (r and len(logs) == len(AS_STAGES)):
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    frame(ax, "training step", better("mAP (bf16)", lower=False))
    at, ys, starts, first = 0, [], [], None
    for k, rec in enumerate(logs):
        starts.append(at)
        a = next(x["args"] for x in rec if "args" in x)
        pool = J(f"audioset/pool_{a['pool']}.json") or {}
        first = first or pool.get("teachers")
        pts = [(at + x["step"], x.get("map_ema", x.get("map", x.get("clip_map")))) for x in rec
               if x.get("step", -1) >= 0 and (k == 0 or x["step"] > 0)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=(S1, S2, S5, S3)[k % 4], lw=1.6, marker="o", ms=3,
                zorder=3, label=f"stage {k + 1} ({AS_STAGES[k][2]}): {float(a['epochs']):g} epochs on "
                f"{pool.get('clips', 0):,} clips"
                + (", frame head" if float(a.get("frame_weight", 0)) > 0 else "")
                + (f", {len(pool['teachers'])} teachers" if pool.get("teachers") not in (None, first) else ""))
        if k:
            ax.axvline(at, color=RULE, lw=0.8, ls=":", zorder=1)
        ys += [p[1] for p in pts]
        at = pts[-1][0]
    twin = RES / "audioset" / f"{AS_FRAMES[0]}.jsonl"
    if twin.exists():                       # the same stage with the frame head beside it, from the same start
        rec = [json.loads(line) for line in twin.read_text().splitlines()]
        pts = [(starts[AS_FRAMES[2] - 1] + x["step"], x["clip_map"]) for x in rec if x.get("step", -1) > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, lw=1.2, ls="--", marker="o", ms=2.5, zorder=2,
                label=f"stage {AS_FRAMES[2]} with the frame head ({AS_FRAMES[3]})")
        ys += [p[1] for p in pts]
    best = max(v for k, v in r["models"].items() if k in AS_NAMES)
    ax.axhline(best, color=S4, lw=1.0, ls="--", zorder=2, label=f"best released model, CED-base ({best:.4f}, fp32)")
    ax.set_ylim(min(ys + [best]) - 0.002, max(ys + [best]) + 0.003)
    legend_below(ax, ncol=2, pad=0.3)
    save(fig, "fig_as_train.png")


def fig_full_run():
    """The full run's hours of audio finished against wall-clock hours (scripts/full_timeline.py), with the changes
    made while it ran marked: what a pilot never shows is which server a whole corpus waits on."""
    t = J("dataset/timeline.json")
    if not t:
        return
    xs = [p["wall_h"] for p in t["points"]]
    ys = [p["audio_h"] for p in t["points"]]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    frame(ax, "hours since the run started", "hours of audio finished")
    ax.plot(xs, ys, color=S1, linewidth=1.6, zorder=3)
    evs = t.get("events") or []
    top = max(ys)
    for i, ev in enumerate(evs, 1):      # numbered, at four staggered heights: the changes came close together
        ax.axvline(ev["wall_h"], color=MUTED, linewidth=0.7, linestyle="--", zorder=2)
        ax.text(ev["wall_h"], top * (0.45 + 0.15 * ((i - 1) % 4)), f" {i}", fontsize=6.8, color=FG, va="center",
                ha="left")
    ax.set_xlim(0, max(xs)); ax.set_ylim(0, top * 1.05)
    if evs:                              # the key under the axes, three changes a line
        note = "\n".join("     ".join(f"{i} {ev['label']}" for i, ev in enumerate(evs, 1) if k <= i - 1 < k + 3)
                         for k in range(0, len(evs), 3))
        ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=6.6, color=TEXT, ha="left", va="top",
                linespacing=1.5)
    save(fig, "fig_full_run.png")


def fig_dataset_tags():
    """Melati v1's tags per hour of exported speech (scripts/build_hf_dataset.py's stats)."""
    d = J("dataset/stats.json")
    if not d:
        return
    ev, emo = d["events_per_hour"], d["emotion_spans"]
    groups = ["breaths", "laughs", "coughs", "fillers", "happy", "sad", "angry", "surprised", "fearful"]
    vals = [ev.get("breathes", 0), ev.get("laughs", 0), ev.get("coughs", 0), d["fillers_per_hour"]]
    vals += [emo.get(k, 0) / d["hours"] for k in ("happy", "sad", "angry", "surprised", "fearful")]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    frame(ax, None, "per hour of speech", axis="y")
    cols = [S1, S1, S1, S2, S4, S4, S4, S4, S4]
    ax.bar(range(len(groups)), vals, 0.62, color=cols, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, color=TEXT)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    save(fig, "fig_dataset_tags.png")


GAP_RULES = [("{}", "floor + 6 dB\n80 ms (old)"), ('{"floor_db": 3}', "floor + 3 dB\n80 ms"),
             ('{"floor_db": 2}', "floor + 2 dB\n80 ms"), ('{"floor_db": 2, "min_active": 0.04}', "floor + 2 dB\n40 ms")]


def fig_nv_miss():
    """Why NonverbalTTS test breaths have no candidate, under the old rule and the new one."""
    old, new = J("nonverbal/gap_diagnose_test_old.json"), J("nonverbal/gap_diagnose_test_new.json")
    if not (old and new):
        return
    parts = [("found", "a candidate at the boundary", S1), ("too brief", "sound above the threshold, too brief", S2),
             ("too quiet", "below the noise-floor threshold", S4), ("no gap", "no gap between the words", S6)]
    rows = [("new rule", new["breath"]["why"]), ("old rule", old["breath"]["why"])]
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    frame(ax, "share of the reference breaths", None, axis="x")
    for y, (lab, why) in enumerate(rows):
        n, left = sum(why.values()), 0.0
        for key, _, c in parts:
            v = why.get(key, 0) / n
            ax.barh(y, v, left=left, color=c, height=0.6, zorder=3)
            if v >= 0.05:
                ax.text(left + v / 2, y, pct(v), ha="center", va="center", fontsize=6.8, color="white")
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles = [matplotlib.patches.Patch(color=c, label=lab) for _, lab, c in parts]
    legend_below(ax, ncol=2, pad=0.42, handles=handles, labels=[h.get_label() for h in handles])
    save(fig, "fig_nv_miss.png")


def fig_nv_rules():
    """Each rule on NonverbalTTS test: breaths with a candidate, and breaths tagged after the head."""
    sw, ev = J("nonverbal/gap_sweep_test.json"), J("nonverbal/gap_eval_v3head.json")
    if not (sw and ev):
        return
    rules = [(k, lab) for k, lab in GAP_RULES if k in sw and k in ev]
    cand = [sw[k]["candidate_recall"].get("breath") for k, _ in rules]
    tagged = [(ev[k]["test"].get("breath") or {}).get("tag_recall") for k, _ in rules]
    prec = [(ev[k]["test"].get("breath") or {}).get("tag_precision") for k, _ in rules]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    frame(ax, None, better("share of the test breaths", lower=False), axis="y")
    groups = [f"{lab}\n{sw[k]['per_clip']:.1f} candidates/clip" for k, lab in rules]
    bars(ax, groups, ["has a candidate", "tagged by the head"], [cand, tagged], [S2, S1], width=0.66, fmt=pct,
         label_size=6.4)
    for i, pv in enumerate(prec):                   # the tagged breaths' precision, inside their bar
        if pv is not None and tagged[i] is not None:
            ax.text(i + 0.165, tagged[i] / 2, f"{pct(pv)}\nprecise", ha="center", va="center", fontsize=5.6,
                    color="white", linespacing=1.1)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    legend_below(ax, ncol=2, pad=0.36)
    save(fig, "fig_nv_rules.png")


NV_HEAD_ROWS = [("filler", "filler"), ("speech", "speech"), ("breath", "breath"), ("laugh", "laugh"),
                ("noise", "noise / music"), ("cough", "cough"), ("sigh", "sigh"), ("sniff", "sniff"),
                ("throat clearing", "throat clearing"), ("sneeze", "sneeze")]


def fig_nv_heads():
    """F1 of each head on clips nothing trained on: PodcastFillers' test episodes and VocalSound's test speakers."""
    hs = [(enc, lab, c, J(f"nonverbal/head_{enc}.json")) for enc, lab, c in
          (("whisper", "Whisper-large-v3", S1), ("wavlm", "WavLM-large", S2), ("ast", "AST (AudioSet)", S4))]
    hs = [x for x in hs if x[3]]
    if not hs:
        return
    rows = [(k, lab) for k, lab in NV_HEAD_ROWS if any(k in h["test"]["all"]["per_class"] for *_, h in hs)]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(rows) + 1.0))
    frame(ax, better("F1, held-out clips", lower=False), None, axis="x")
    ys = np.arange(len(rows))[::-1]
    for j, (enc, lab, c, h) in enumerate(hs):
        pc = h["test"]["all"]["per_class"]
        ax.scatter([(pc.get(k) or {}).get("f1", np.nan) for k, _ in rows], ys + (j - 1) * 0.18, s=26, color=c,
                   label=lab, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=7.2)
    ax.set_xlim(0.4, 1.0)
    legend_below(ax, ncol=3, pad=0.16 + 0.6 / len(rows))
    save(fig, "fig_nv_heads.png")


def fig_nv_curves():
    d = J("nonverbal/nonverbaltts_gaps_curves.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    frame(ax, "threshold on the class's probability", better("precision (right word)", lower=False))
    for c, col in zip(NV3, (S1, S5, S4)):
        for sp, ls in (("train", "--"), ("test", "-")):
            rows = [r for r in d["curves"].get(c, {}).get(sp, []) if r[1] is not None and r[3] >= 5]
            ax.plot([r[0] for r in rows], [r[1] for r in rows], color=col, lw=1.3, ls=ls,
                    label=f"{c}, {sp}")
    ax.axhline(0.8, color=FG, lw=0.8, ls=":")
    ax.set_ylim(0, 1.02)
    legend_below(ax, ncol=3, pad=0.2)
    save(fig, "fig_nv_curves.png")


# ----------------------------------------------------------------------------------------------- comparison

PIPES = [("emilia", "Emilia-Pipe", S4), ("melati_all", "Melati, everything", S1),
         ("melati_emilia_filter", "Melati, Emilia's filter applied", S2)]


def compare_file(src):
    """The newest comparison of a set: the release model's run if it has finished, else the one before."""
    order = {"vox": ("compare/voxconverse_rel.json", "compare/voxconverse_ft.json", "compare/voxconverse.json"),
             "pil": ("compare/pilot_nvrel4.json", "compare/pilot_nvrel2.json", "compare/pilot_nvwhisper3.json",
                     "compare/pilot_nvwhisper.json",
                     "compare/pilot_rel.json", "compare/pilot_ft.json")}[src]
    return next((J(r) for r in order if J(r)), None)


def fig_vox(items, name, ylab, fmt=pct, ylim=None, src="vox", lower=True):
    d = compare_file(src)
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    frame(ax, None, better(ylab, lower), axis="y")
    vals = []
    for k, _, _ in PIPES:
        row = []
        for get, _ in items:
            row.append(get(d, k))
        vals.append(row)
    bars(ax, [lab for _, lab in items], [p[1] for p in PIPES], vals, [p[2] for p in PIPES], fmt=fmt, label_size=6.4)
    if fmt in (pct, pct1):
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if ylim:
        ax.set_ylim(*ylim)
    legend_below(ax, ncol=3, pad=0.16)
    save(fig, name)


def fig_vox_dur(src="vox", name="fig_vox_dur.png"):
    d = compare_file(src)
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    frame(ax, "segment length (s)", None, axis="x")
    for i, (k, lab, c) in enumerate(PIPES):
        q = d["pipelines"][k]["yield"]["duration_s"]
        y = len(PIPES) - 1 - i
        ax.plot([q["10"], q["90"]], [y, y], color=c, lw=5, solid_capstyle="butt", zorder=3)
        ax.plot([q["90"], q["99"]], [y, y], color=c, lw=1.4, zorder=3)
        ax.scatter([q["50"]], [y], s=40, color="white", edgecolors=c, linewidths=1.6, zorder=4)
    ax.set_yticks(range(len(PIPES))); ax.set_yticklabels([p[1] for p in PIPES][::-1], fontsize=7.4)
    ax.text(0.99, -0.42, "bar: 10th to 90th percentile; circle: median; line: to the 99th", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=6.3, color=MUTED)
    ax.set_xlim(0, 31)
    save(fig, name)


def main():
    fig_pipeline(); fig_segment(); fig_prefix(); fig_gapmethod()
    fig_ltas(); fig_fullscale(); fig_stems(); fig_dnsmos()
    fig_suppress(); fig_halluc(); fig_stop_langs(); fig_false_stop(); fig_noise(); fig_runaway(); fig_sweep()
    fig_lid(); fig_lid("failed", "fig_failed.png", "clips failed (CER >= 0.8)", str)
    fig_perlang(); fig_lid_acc(); fig_fillers(); fig_emo_heads(); fig_emo_recall(); fig_emo_mer()
    fig_emo(); fig_emo("uar", "fig_emo_uar.png", "unweighted average recall"); fig_emo_corpus(); fig_emo_prec()
    fig_nv(0, "fig_nv_precision.png", "precision, per clip")
    fig_nv(1, "fig_nv_recall.png", "recall, per clip")
    fig_nv(2, "fig_nv_place.png", "tags at the right word")
    fig_nv_pf()
    fig_nv_frames()
    fig_laugh_sweep()
    fig_audioset()
    fig_as_map()
    fig_as_train()
    fig_full_run()
    fig_dataset_tags()
    fig_nv_heads()
    fig_nv_miss()
    fig_nv_rules()
    fig_nv_curves()
    p = lambda sec, key: (lambda d, k: d["pipelines"][k][sec][key])
    fig_vox([(p("yield", "share_of_input"), "share of the input kept"),
             (p("purity", "single_speaker_speech_kept"), "single-speaker speech kept")], "fig_vox_yield.png",
            "share of hours", ylim=(0, 1.05), lower=False)
    fig_vox([(p("purity", "share_purity_lt_0_9"), "purity < 0.9"),
             (p("purity", "share_second_speaker_ge_0_5s"), "second speaker >= 0.5 s"),
             (p("purity", "share_overlap_ge_0_2s"), "overlap >= 0.2 s")], "fig_vox_purity.png",
            "share of segments", ylim=(0, 0.25))
    e = lambda key: (lambda d, k: d["edges_by_alignment"][k][key])
    fig_vox([(e("share_first_word_at_cut"), "first word at the cut"), (e("share_last_word_at_cut"), "last word at the cut"),
             (e("share_either"), "either")], "fig_vox_cut.png", "share of segments", ylim=(0, 0.5))
    fig_vox_dur()
    fig_vox([(p("yield", "share_of_input"), "share of the input kept")], "fig_pil_yield.png", "share of hours",
            ylim=(0, 1.05), src="pil", lower=False)
    fig_vox([(e("share_first_word_at_cut"), "first word at the cut"), (e("share_last_word_at_cut"), "last word at the cut"),
             (e("share_either"), "either")], "fig_pil_cut.png", "share of segments", ylim=(0, 0.5), src="pil")
    fig_vox_dur("pil", "fig_pil_dur.png")


if __name__ == "__main__":
    main()
