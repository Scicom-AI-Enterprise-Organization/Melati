#!/bin/bash
# The arXiv upload: exactly the files the paper reads, then the same compile arXiv runs.
#
#   bash preprint/arxiv.sh        # -> preprint/arxiv.zip
#
# arXiv runs pdfLaTeX only, with no bibtex (it reads our .bbl). So the bundle takes the file list from the last
# build's recorder file (.fls), and is then unpacked into an empty directory and compiled there with plain pdflatex.
# A bundle that fails here would fail on arXiv. Every font comes from TeX Live: font files sent with the source never
# reached arXiv's compile (2026-10-19).
set -eu
cd "$(dirname "$0")"
bash build.sh
here=$PWD
stage=$(mktemp -d)
check=$(mktemp -d)
trap 'rm -rf "$stage" "$check"' EXIT

# everything the build read from this folder, except LaTeX's own scratch files
grep '^INPUT \./' neurips_2023.fls | sed 's|^INPUT \./||' | sort -u \
  | grep -v '\.\(aux\|out\|toc\|lof\|lot\)$' > "$stage/.files"
cp neurips_2023.tex "$stage/"
while read -r f; do mkdir -p "$stage/$(dirname "$f")"; cp "$f" "$stage/$f"; done < "$stage/.files"
rm "$stage/.files"
# scicom.sty without its OpenType branch, which pdfLaTeX never takes: arXiv's file check reads the font names in its
# \setmainfont and \setmonofont as files and reports "file not found: Inter", "JetBrainsMono"
python3 - "$stage/scicom.sty" <<'PY'
import sys
p = sys.argv[1]
lines = open(p).read().split("\n")
i = lines.index(r"\ifsc@otf")
j = lines.index(r"\else", i)
k = lines.index(r"\fi", j)
lines = lines[:i] + ["% (the OpenType branch for LuaLaTeX is left out of the arXiv copy)"] + lines[j + 1:k] + lines[k + 1:]
open(p, "w").write("\n".join(lines))
assert "setmainfont" not in open(p).read()
PY
rm -f arxiv.zip
(cd "$stage" && zip -q -X -r "$here/arxiv.zip" .)

# arXiv's compile: unpack into an empty directory, plain pdflatex three times, nothing on the search path but TeX Live
(cd "$check" && unzip -q "$here/arxiv.zip")
for i in 1 2 3; do
  (cd "$check" && env -u TEXINPUTS pdflatex -interaction=batchmode -halt-on-error neurips_2023.tex > /dev/null 2>&1) \
    || { echo "arXiv-style compile FAILED (run $i):"; grep -A3 '^!' "$check/neurips_2023.log" | head -20; exit 1; }
done
python3 - "$check/neurips_2023.log" neurips_2023.log <<'PY'
import re, sys
bundle, build = (open(p, errors="ignore").read() for p in sys.argv[1:])
pages = lambda log: (re.findall(r"Output written on .*?\((\d+) pages", log) or ["?"])[-1]
miss = sorted(set(re.findall(r"File `([^']+)' not found", bundle)))
# a font shape LaTeX could not find is replaced without an error: arXiv set our mono text in Times that way
subs = sorted(set(re.findall(r"Font shape `([^']+)' undefined", bundle)))
print(f"arxiv.zip: {pages(bundle)} pages (the build: {pages(build)}) | "
      f"undefined {len(re.findall('undefined', bundle)) - len(subs)} | missing files {miss or 0} | "
      f"substituted fonts {subs or 0}")
PY
echo "-> preprint/arxiv.zip ($(du -h arxiv.zip | cut -f1)): upload it as is"
