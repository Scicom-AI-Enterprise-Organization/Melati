"""Upload a Hugging Face card's redrawn figures, and with --readme its README, and nothing else: the weights and the
data stay as they are. --delete removes figures the card no longer shows. --patch-readme takes the live README instead
of rebuilding it (a model's card is written by its release, which would rebuild the weights too) and changes only its
figures, each shown through melati.plotstyle.card_img, then uploads it. RUNS ON THE BOX.

    python scripts/push_card_figures.py --repo Scicom-intl/Melati-Whisper-Large-V3 --dir $MELATI_DATA/release/cards/asr
    python scripts/push_card_figures.py --repo Scicom-intl/Melati --repo-type dataset --dir $MELATI_DATA/podcast/hf \\
        --readme --delete figures/languages_lengths.png
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--repo-type", default="model")
ap.add_argument("--readme", action="store_true", help="upload <dir>/README.md as well")
ap.add_argument("--patch-readme", action="store_true",
                help="write <dir>/README.md from the live one, its markdown figures shown through card_img, and upload it")
ap.add_argument("--delete", nargs="*", default=[], help="paths in the repo to remove (figures no longer drawn)")
ap.add_argument("--message", default="Figures in the preprint's style, with an arrow to the better end")
a = ap.parse_args()
files = sorted(p.name for p in (a.dir / "figures").glob("*.png"))
assert files, f"no figures under {a.dir / 'figures'}"
if a.patch_readme:
    import re
    import sys
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from melati.plotstyle import card_img, hf_url
    FIG = re.compile(r"!\[([^\]]*)\]\((figures/[^)\s]+\.png)\)")
    live = Path(hf_hub_download(a.repo, "README.md", repo_type=a.repo_type)).read_text()
    (a.dir / "README.md").write_text(FIG.sub(lambda m: card_img(a.dir / m.group(2), m.group(1),
                                                                hf_url(a.repo, m.group(2), a.repo_type)), live))
    print(f"README.md: {len(FIG.findall(live))} figures now shown through card_img")
    a.readme = True
api = HfApi()
api.upload_folder(repo_id=a.repo, repo_type=a.repo_type, folder_path=str(a.dir),
                  allow_patterns=["figures/*.png"] + (["README.md"] if a.readme else []),
                  delete_patterns=a.delete or None, commit_message=a.message)
print(f"pushed {files}{' and README.md' if a.readme else ''} to {a.repo}" + (f", removed {a.delete}" if a.delete else ""))
