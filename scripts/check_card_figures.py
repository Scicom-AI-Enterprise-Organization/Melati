"""What a Hugging Face collection's cards show right now: for every repo in it, the README's figures (markdown or
<img>), each one's last commit and, given --local, whether the live file is the one drawn on the box. RUNS ON THE BOX.

    python scripts/check_card_figures.py --collection Scicom-intl/melati \\
        --local Scicom-intl/Melati-Whisper-Large-V3=$MELATI_DATA/release/cards/asr
"""
import argparse
import hashlib
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)|<img[^>]*\bsrc=\"([^\"]+)\"")


def repo_path(src, repo):
    """The repo path an image URL points at, or None for an image hosted elsewhere."""
    if "://" not in src:
        return src.lstrip("./")
    m = re.search(rf"{re.escape(repo)}/(?:resolve|blob|raw)/[^/]+/(.+?)(?:\?.*)?$", src)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, help="owner/slug, or a title to look up among the owner's")
    ap.add_argument("--local", nargs="*", default=[], metavar="REPO=DIR")
    a = ap.parse_args()
    api = HfApi()
    owner, slug = a.collection.split("/", 1)
    cols = [c for c in api.list_collections(owner=owner, limit=100) if c.slug.split("/", 1)[1].startswith(slug)]
    assert cols, f"no collection {a.collection}"
    col = api.get_collection(cols[0].slug)
    local = dict(s.split("=", 1) for s in a.local)
    print(f"{col.slug}: {col.title} ({len(col.items)} items)")
    for it in col.items:
        if it.item_type not in ("model", "dataset"):
            print(f"\n== {it.item_type} {it.item_id}")
            continue
        rid, rtype = it.item_id, it.item_type
        print(f"\n== {rtype} {rid}")
        for c in api.list_repo_commits(rid, repo_type=rtype)[:3]:
            print(f"   commit {c.created_at:%m-%d %H:%M} {c.title}")
        readme = Path(hf_hub_download(rid, "README.md", repo_type=rtype)).read_text()
        srcs = [m.group(1) or m.group(2) for m in IMG.finditer(readme)]
        paths = [repo_path(s, rid) for s in srcs]
        tags = re.findall(r"<img[^>]*>", readme)
        print(f"   README: {len(srcs)} images, {len(tags)} as <img>, {sum('width=' in t for t in tags)} with a width")
        info = {p.path: p for p in api.get_paths_info(rid, [p for p in paths if p], repo_type=rtype, expand=True)}
        for s, p in zip(srcs, paths):
            if not p:
                print(f"   external {s}")
                continue
            f = info.get(p)
            if f is None:
                print(f"   MISSING  {p}")
                continue
            when = f"{f.last_commit.date:%m-%d %H:%M}" if f.last_commit else "?"
            note = ""
            if rid in local:
                lp = Path(local[rid]) / p
                if lp.exists():
                    b = lp.read_bytes()
                    if f.lfs:                                   # an LFS file is named by its sha256, a git blob by sha1
                        same = hashlib.sha256(b).hexdigest() == f.lfs.sha256
                    else:
                        same = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == f.blob_id
                    note = "same as local" if same else "DIFFERS from local"
                else:
                    note = "no local copy"
            print(f"   {when} {p} {f.size // 1024} KB {note}")


if __name__ == "__main__":
    main()
