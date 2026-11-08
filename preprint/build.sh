#!/bin/bash
# Build the preprint, and report the things that actually matter.
#
#   bash preprint/build.sh
#
# latexmk, not a hand-rolled pass sequence. Running pdflatex/bibtex by hand raced with itself
# more than once and produced an empty .bbl with "I found no \bibdata command", which surfaces
# as several dozen undefined citations that have nothing to do with the source.
set -u
cd "$(dirname "$0")"
# latexmk -C alone has left a stale aux behind after an interrupted run, which then throws
# "Missing \begin{document}" from inside hyperref's .out file. Remove them outright.
rm -f neurips_2023.aux neurips_2023.out neurips_2023.toc neurips_2023.bbl \
      neurips_2023.blg neurips_2023.fls neurips_2023.fdb_latexmk
latexmk -C > /dev/null 2>&1
# numbers and tables first: the prose quotes every number through \val{} from numbers.tex
../.venv/bin/python make_tables.py > /dev/null || { echo "make_tables.py FAILED: tables and numbers are stale"; exit 1; }
../.venv/bin/python plot_figs.py > /dev/null || { echo "plot_figs.py FAILED: figures are stale"; exit 1; }
# pdfLaTeX, the engine arXiv runs (it refuses LuaLaTeX and XeLaTeX), so this PDF is the one arXiv makes. Every font
# comes from TeX Live (scicom.sty), as on arXiv.
latexmk -pdf -interaction=nonstopmode -halt-on-error neurips_2023.tex > latexmk.log 2>&1
# every float must be cited before it appears, and in the same order as its citations
python3 - <<'PY'
import re
src = open("neurips_2023.tex").read().split("\n")
order, first = [], {}
for i, l in enumerate(src):
    m = re.match(r"\\begin\{(figure|table)\}", l)
    if m:
        j = i
        while j < len(src) and not src[j].startswith("\\end{" + m.group(1) + "}"):
            j += 1
        lab = next((re.search(r"\\label\{((?:fig|tab):[^}]+)\}", src[k]).group(1)
                    for k in range(i, j + 1)
                    if re.search(r"\\label\{((?:fig|tab):[^}]+)\}", src[k])), None)
        if lab:
            order.append((lab, i))
    for mm in re.finditer(r"\\ref\{((?:fig|tab):[^}]+)\}", l):
        first.setdefault(mm.group(1), i)
late = [k for k, i in order if k not in first or first[k] > i]
mis = [(order[a][0], order[a + 1][0]) for a in range(len(order) - 1)
       if order[a][0] in first and order[a + 1][0] in first
       and first[order[a][0]] > first[order[a + 1][0]]]
if late: print("  FLOATS BEFORE THEIR FIRST MENTION:", late)
if mis:  print("  FLOAT ORDER DISAGREES WITH MENTION ORDER:", mis)
PY
python3 - <<'PY'
import re
log = open("neurips_2023.log", errors="ignore").read()
pages = re.findall(r"Output written on .*?\((\d+) pages", log)
errs = re.findall(r"(?m)^! .*", log)
print(f"pages {pages[0] if pages else '?'} | errors {len(errs)} | "
      f"undefined {len(re.findall('undefined', log))} | "
      f"undefined values {len(re.findall('undefined value', log))} | "
      f"overfull {len(re.findall('Overfull', log))} | "
      f"em dashes {open('neurips_2023.tex').read().count(chr(8212))}")
for e in errs[:5]:
    print("  ", e.strip())
PY
# do not pipe this script through head: SIGPIPE kills it mid-build
echo "a clean log is not a clean page: rasterise one and look at it"
echo "  gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r62 -dFirstPage=N -dLastPage=N -sOutputFile=/tmp/p.png neurips_2023.pdf"
