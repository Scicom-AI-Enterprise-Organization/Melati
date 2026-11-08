#!/usr/bin/env python3
"""PodcastFillers episodes by audio licence and split, and which training rows each licence holds. RUNS ON THE BOX.

The audio of PodcastFillers is CC BY 3.0, CC BY-SA 3.0 or CC BY-ND 3.0 per episode (LICENSE_AUDIO.csv), under
annotations licensed for non-commercial research only. Writes {episode: licence} for nonverbal/ to filter on.

    python scripts/pf_licences.py --pf $MELATI_DATA/bench/podcastfillers/PodcastFillers \\
        --manifests $MELATI_DATA/nv/manifest_rel.jsonl --out $MELATI_DATA/nv/pf_episode_licence.json
"""
import argparse, csv, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pf", type=Path, required=True)
ap.add_argument("--manifests", nargs="*", type=Path, default=[])
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
lic = {r["Episode_Name"].strip(): r["license"].strip()
       for r in csv.DictReader((args.pf / "LICENSE_AUDIO.csv").open(encoding="utf-8"))}
split = {}
for r in csv.DictReader(next(args.pf.rglob("PodcastFillers.csv")).open(encoding="utf-8")):
    split[r["podcast_filename"].strip()] = r["episode_split_subset"]
missing = sorted(set(split) - set(lic))
print(f"{len(lic)} licensed episodes, {len(split)} annotated, {len(missing)} without a licence row: {missing[:3]}")
print("by split:", dict(Counter((split[e], lic.get(e, "?")) for e in split)))
args.out.write_text(json.dumps({e: lic.get(e) for e in split}, indent=1))
for m in args.manifests:
    c = Counter()
    for line in m.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("episode"):
            c[(r["corpus"], lic.get(r["episode"], "?"))] += 1
    print(m.name, dict(sorted(c.items())))
