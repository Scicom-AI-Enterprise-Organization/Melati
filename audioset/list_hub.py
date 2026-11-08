"""Models on the Hub by an author, with their tags and downloads, to find a checkpoint to start from. RUNS ANYWHERE.

    python audioset/list_hub.py mispeech
"""
import sys

from huggingface_hub import HfApi

for m in HfApi().list_models(author=sys.argv[1], full=True, limit=200):
    print(m.id, m.downloads, m.pipeline_tag, ",".join(t for t in (m.tags or []) if not t.startswith(("region:", "endpoints"))))
