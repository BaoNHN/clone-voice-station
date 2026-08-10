#!/usr/bin/env python
"""
tools/prepare_general_benchmark.py
One-time setup: downloads a fixed general-Vietnamese-speech benchmark into
stt_general_benchmark/ -- this is voice/stt_local_train.py's Gate 2 (see that module's
docstring): every Tier 2 adapter, regardless of which guest or domain trained it, gets
scored against this same fixed set before it's allowed to ship, on top of (not instead of)
the guest's own held-out test split from _split_train_val_test().

Why a *separate* dataset from the guest's own domain, not just a bigger held-out slice of
their upload: confirmed for real (see tools/import_hf_stt_dataset.py's medical-consultation
experiments) that a held-out split carved from the same upload batch -- no matter how it's
sized or split -- still systematically overstated an adapter's real-world quality, because
train and held-out samples from one guest's one upload share the same recording
session/speaker/domain characteristics. A completely disjoint dataset the guest never
touches is the only way to check "did this adapter get worse at Vietnamese speech in
general", independent of whatever narrow domain it was fine-tuned on. This does NOT
replace a genuine held-out check of the guest's own domain (that would need the guest to
supply a second, separately-collected batch -- a product/UX change out of scope here) --
it only catches the general-regression half of the problem.

doof-ferb/vlsp2020_vinai_100h (~56.4k rows, general spontaneous Vietnamese speech scraped
from real recordings, not read-aloud sentences) was picked over the medical dataset's own
"test" split specifically because it has *nothing* to do with medical consultations --
sharing a source with the training domain would reintroduce the same-distribution problem
this benchmark exists to avoid. It has no speaker metadata and only a "train" split exists
upstream (no official train/test division) -- both irrelevant disqualifiers for *fine-
tuning* data (see the writeup that ruled it out for that role) but not for this one: this
script deterministically takes a fixed slice once, saves it to disk, and that slice is
never used for anything else in this project again -- what makes it "held out" is that no
guest's training data ever comes from this dataset, not an upstream split label or speaker
disjointness.

Usage
-----
    python tools/prepare_general_benchmark.py

Re-running is safe (overwrites the same fixed slice, same offset/limit -- see BENCHMARK_
constants below) but shouldn't be necessary; this only needs to run once per deployment
(or if BENCHMARK_SIZE changes).
"""
import os
import sys

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
DATASET = "doof-ferb/vlsp2020_vinai_100h"
# Offset chosen arbitrarily away from row 0 (no particular reason row 0 would be worse,
# just avoids relying on however the upstream dataset orders its first few rows).
BENCHMARK_OFFSET = 1000
# 500, not the 50-ish size of the guest-facing eval sets elsewhere in this project --
# this set is downloaded once and reused by every training run forever, so its size only
# costs disk space and per-run eval time, not repeated download bandwidth. Larger buys a
# statistically tighter WER estimate for the one gate that's supposed to be trustworthy
# regardless of what domain a guest trained on. The real cost this trades against is
# per-run latency: two full passes over BENCHMARK_SIZE clips (once for the untrained base,
# once for the selected epoch) on top of everything _split_train_val_test() already
# evaluates -- worth watching if guest-facing training time becomes a complaint, in which
# case shrinking this constant is the one-line fix.
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
