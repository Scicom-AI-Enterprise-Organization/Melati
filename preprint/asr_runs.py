"""The ASR checkpoints the preprint reports, shared by plot_figs.py and make_tables.py.

key -> (run directory under $MELATI_DATA/asr, label). Every number is read only for a run whose
benchmark was decoded as vLLM serves it (asr/summarise.py's `decoding` == "none").
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"

ASR_RUNS = {
    "base": ("runs/base_large_v3", "whisper-large-v3"),
    "bug": ("runs/v3_lora_r64_all_lr5e-4_s1000", "recipe as published"),
    "fix": ("runs2/v3_lora_r64_all_lr5e-4_s1000", "prefix fed"),
    "fix25": ("runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000", "+ 25% Common Voice"),
    "fix50": ("runs2/v3_lora_r64_all_cv22_r50_lr5e-4_s1000", "+ 50% Common Voice"),
    "distill": ("runs2/v3_lora_r64_all_lr5e-4_s1000_liddistill", "language KL"),
    "label": ("runs2/v3_lora_r64_all_lr5e-4_s1000_lidlabel", "language label"),
    "distill25": ("runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000_liddistill", "25% CV, language KL"),
    "label25": ("runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000_lidlabel", "25% CV, language label"),
    "d25bl": ("runs2/v3_lora_r64_all_cv22_r25_lr5e-4_s1000_liddistill_blbase", "25% CV, KL, base's blank languages"),
    "ttsbl": ("runs2/v3_lora_r64_all_cv22_r25_tts_lr5e-4_s1000_liddistill_blbase",
              "25% CV, Indic, KL, base's blank languages"),
    "tts": ("runs2/v3_lora_r64_all_cv22_r25_tts_lr5e-4_s1000_liddistill", "25% CV, Indic speech, KL"),
    "attn": ("runs2/v3_loraattn_r64_all_lr5e-4_s1000", "LoRA on attention only"),
    "lr1e-4": ("runs2/v3_lora_r64_all_lr1e-4_s1000", "LoRA, lr 1e-4"),
    "full2e-5": ("runs2/v3_full_all_lr2e-5_s1000", "full fine-tune, lr 2e-5"),
    "full5e-6": ("runs2/v3_full_all_lr5e-6_s1000", "full fine-tune, lr 5e-6"),
    # merges of the deltas (asr/train/merge_models.py); `merge` is the released model
    "merge": ("merges/tts_full", "the Indic + KL model and the full fine-tune, 50/50"),
    "m_7030": ("merges/tf73", "the same, 70/30"),
    "m_3070": ("merges/tf37", "the same, 30/70"),
    "m_sum": ("merges/tf_sum", "the same, both at full strength"),
    "m_three": ("merges/tfl", "the same plus the language-label model, a third each"),
    "m_soup": ("merges/soup25", "the four 25% replay models, a quarter each"),
    "m_label": ("merges/tts_label", "the Indic + KL and language-label models, 50/50"),
    "m_ties": ("merges/ties4", "TIES over four models"),
    "m_dare": ("merges/dare3", "DARE over three models"),
    "m_exact": ("merges/tts16", "the Indic + KL model alone, re-merged in float16"),
}
RELEASE = "merge"
_FILES, _ROWS = {}, {}


def _j(rel):
    if rel not in _FILES:
        p = RES / rel
        _FILES[rel] = json.loads(p.read_text()) if p.exists() else {}
    return _FILES[rel]


def asr(key):
    """A checkpoint's summary row with its FLEURS report (`lid`) and failure breakdown (`fail`); None unless it
    was scored as vLLM serves it, so a stale number never reaches the paper."""
    if key not in _ROWS:
        path = ASR_RUNS[key][0]
        root, name = path.split("/")
        row = _j({"runs": "asr/summary_runs1.json", "merges": "asr/summary_merges.json"}.get(root, "asr/summary.json")).get(name)
        ok = row and (row.get("config") or {}).get("decoding") == "none"
        _ROWS[key] = dict(row, lid=_j("asr/lid.json").get(path) or {},
                          fail=_j("asr/served_failures.json").get(path) or {}) if ok else None
    return _ROWS[key]
