#!/usr/bin/env python3
"""Run a model card's Use snippet as a user would: the README's first python block, in a fresh process, on one clip of
AudioSet's evaluation set saved as clip.wav, printing what it prints and the clip's own labels beside it. The README is
a local file or the one on the Hub; the snippet loads the model from the Hub (REPO_ID already filled in). RUNS ON THE
BOX, in .venv-sed.

    python audioset/check_card_use.py --repo Scicom-intl/Melati-AudioSet-EAT-Large [--readme <path>] [--clip 7]
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import classes, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--readme", type=Path, default=None, help="a local README.md (default: the one on the Hub)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--clip", type=int, default=7, help="which evaluation clip")
    a = ap.parse_args()
    if a.readme is None:
        from huggingface_hub import hf_hub_download
        a.readme = Path(hf_hub_download(a.repo, "README.md", revision=a.revision))
    code = re.search(r"```python\n(.*?)```", a.readme.read_text(), re.S).group(1)
    if a.revision:                                           # a version not yet on main: every load pinned to it
        code = code.replace("trust_remote_code=True)", f'trust_remote_code=True, revision="{a.revision}")')
        code = code.replace(f'snapshot_download("{a.repo}")', f'snapshot_download("{a.repo}", revision="{a.revision}")')
    W, ids, lens, Y = load("eval")
    names = [n for _, _, n in classes()]
    with tempfile.TemporaryDirectory() as d:
        sf.write(Path(d) / "clip.wav", np.asarray(W[a.clip][:lens[a.clip]], np.float32), 16000)
        (Path(d) / "use.py").write_text(code)
        print(f"clip {ids[a.clip]}: labelled {', '.join(names[i] for i in np.flatnonzero(Y[a.clip]))}", flush=True)
        r = subprocess.run([sys.executable, "use.py"], cwd=d, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode:
        print(r.stderr[-3000:], flush=True)
        raise SystemExit(f"the card's snippet failed ({r.returncode})")
    print("SNIPPET_OK", flush=True)


if __name__ == "__main__":
    main()
