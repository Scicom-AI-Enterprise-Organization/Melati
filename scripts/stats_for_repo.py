"""Copy the dataset's stats.json for the repo's results/ without its `examples`: those are rows of the dataset, i.e.
transcripts of real recordings, which results/ never holds (preprint/CLAUDE.md). The card reads them on the box.
RUNS ON THE BOX.

    python scripts/stats_for_repo.py $MELATI_DATA/results/dataset/stats.json > stats.json
"""
import json
import sys

d = json.loads(open(sys.argv[1]).read())
d.pop("examples", None)
print(json.dumps(d, indent=1))
