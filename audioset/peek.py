"""Print a parquet shard's schema and its first rows (audio as shape and rate, not bytes), to see a mirror's layout
before writing its reader. RUNS ON THE BOX.

    python audioset/peek.py $MELATI_DATA/audioset/strong/data/test/test-batch0000.parquet
"""
import io
import sys

import pyarrow.parquet as pq
import soundfile as sf

pf = pq.ParquetFile(sys.argv[1])
print(pf.schema_arrow)
print(pf.metadata.num_rows, "rows")
for r in pf.read_row_group(0).slice(0, int(sys.argv[2]) if len(sys.argv) > 2 else 2).to_pylist():
    a = r.pop("audio")
    b = a["bytes"] if isinstance(a, dict) else a
    x, sr = sf.read(io.BytesIO(b))
    print({k: v for k, v in r.items()}, "| audio", x.shape, sr, a.get("path") if isinstance(a, dict) else None)
