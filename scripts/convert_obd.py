#!/usr/bin/env python3
"""Stream the Open Bandit Dataset CSVs into compact Parquet files.

The raw CSVs total ~11 GB, so they are read in batches with pyarrow and never
loaded into memory at once. Affinity columns are cast to float32.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/open_bandit_dataset"
OUT = ROOT / "data/processed"
POLICIES = ["random", "bts"]
CAMPAIGNS = ["all", "men", "women"]


def cast_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    columns, fields = [], []
    for name, column in zip(batch.schema.names, batch.columns):
        if name.startswith("user-item_affinity") or name == "propensity_score":
            column = column.cast(pa.float32())
        elif name in ("item_id", "position", "click"):
            column = column.cast(pa.int16())
        columns.append(column)
        fields.append(pa.field(name, column.type))
    return pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields))


def convert(policy: str, campaign: str) -> None:
    src = RAW / policy / campaign / f"{campaign}.csv"
    dst = OUT / f"{policy}_{campaign}.parquet"
    if dst.exists():
        print(f"skip {dst.name}")
        return
    reader = pacsv.open_csv(src, read_options=pacsv.ReadOptions(block_size=64 << 20))
    writer = None
    rows = 0
    for batch in reader:
        # The first CSV column is an unnamed pandas index; drop it.
        if batch.schema.names[0] in ("", "Unnamed: 0"):
            batch = batch.drop_columns([batch.schema.names[0]])
        batch = cast_batch(batch)
        if writer is None:
            writer = pq.ParquetWriter(dst, batch.schema, compression="zstd")
        writer.write_batch(batch)
        rows += batch.num_rows
    writer.close()
    print(f"{dst.name}: {rows:,} rows")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        for campaign in CAMPAIGNS:
            convert(policy, campaign)
            ctx = RAW / policy / campaign / "item_context.csv"
            dst = OUT / f"item_context_{campaign}.csv"
            if not dst.exists():
                dst.write_bytes(ctx.read_bytes())


if __name__ == "__main__":
    main()
