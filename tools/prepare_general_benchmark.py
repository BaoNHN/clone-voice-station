#!/usr/bin/env python
"""
tools/prepare_general_benchmark.py
One-time setup: downloads a fixed general-Vietnamese-speech benchmark into
stt_general_benchmark/ -- this is voice/stt_local_train.py's Gate 2: every
Tier 2 adapter gets scored against this same fixed set before shipping, on
top of (not instead of) the guest's own held-out split.

It's a dataset the guest never touches, not just a bigger held-out slice of
their own upload -- a held-out split from the same upload batch still shares
recording session/speaker/domain characteristics with the training data and
systematically overstates real-world quality (confirmed against
tools/import_hf_stt_dataset.py's medical-consultation experiments). This
catches only the general-regression half of the problem; it doesn't replace
a genuine held-out check of the guest's own domain.

doof-ferb/vlsp2020_vinai_100h was picked because it shares nothing with any
guest's training domain (general spontaneous speech, not read-aloud). It has
no official train/test split, which would disqualify it as fine-tuning data
but doesn't matter here: this script takes one fixed deterministic slice and
never touches the dataset again.

Usage
-----
    python tools/prepare_general_benchmark.py

Re-running is safe (overwrites the same fixed slice) but only needs to run
once per deployment, or if BENCHMARK_SIZE changes.
"""
import os
import sys

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
DATASET = "doof-ferb/vlsp2020_vinai_100h"
# Offset away from row 0, arbitrary (just avoids the dataset's own row ordering).
BENCHMARK_OFFSET = 1000
# Downloaded once and reused forever, so a larger size only costs one-time
# disk/eval time, not repeated bandwidth -- buys a tighter WER estimate for
# this gate.
BENCHMARK_SIZE = 500
MAX_DURATION_SEC = 20  # keep clips on the shorter side of this dataset's own <=80s range, given BENCHMARK_SIZE=500 already multiplies total eval time
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stt_general_benchmark")


def fetch_rows(offset: int, length: int) -> list:
    rows = []
    remaining = length
    while remaining > 0:
        page_len = min(remaining, 100)  # datasets-server caps "length" at 100/call
        resp = requests.get(DATASETS_SERVER, params={
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": offset + len(rows), "length": page_len,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        page_rows = data.get("rows", [])
        if not page_rows:
            break
        for entry in page_rows:
            row = entry["row"]
            audio = row.get("audio")
            text = (row.get("transcription") or "").strip()
            if not audio or not text:
                continue
            rows.append({"audio_url": audio[0]["src"], "text": text})
        remaining -= len(page_rows)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching {BENCHMARK_SIZE} rows from {DATASET} (offset={BENCHMARK_OFFSET})...")
    # Over-fetch a bit in case some rows get skipped (missing audio/text, over duration).
    rows = fetch_rows(BENCHMARK_OFFSET, int(BENCHMARK_SIZE * 1.3) + 5)

    import soundfile as sf
    from io import BytesIO

    saved = 0
    for row in rows:
        if saved >= BENCHMARK_SIZE:
            break
        try:
            audio_bytes = requests.get(row["audio_url"], timeout=60).content
            duration = sf.info(BytesIO(audio_bytes)).frames / sf.info(BytesIO(audio_bytes)).samplerate
        except Exception as e:
            print(f"  skip (download/probe failed: {e})")
            continue
        if duration > MAX_DURATION_SEC:
            print(f"  skip ({duration:.1f}s > {MAX_DURATION_SEC}s cap)")
            continue
        stem = f"{saved:03d}"
        with open(os.path.join(OUT_DIR, f"{stem}.wav"), "wb") as f:
            f.write(audio_bytes)
        with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(row["text"])
        saved += 1
        print(f"  [{saved}/{BENCHMARK_SIZE}] {stem}.wav ({duration:.1f}s)")

    if saved < BENCHMARK_SIZE:
        print(f"WARNING: only saved {saved}/{BENCHMARK_SIZE} -- voice/stt_local_train.py "
              f"will still work with fewer, just a noisier Gate 2.", file=sys.stderr)
    print(f"General benchmark written to {OUT_DIR} ({saved} pairs).")


if __name__ == "__main__":
    main()
