#!/usr/bin/env python
"""
tools/rehearsal_mix_experiment.py
One-off controlled experiment: does mixing general-domain rehearsal data into the
training set stop whisper-tiny's medical-domain fine-tune from regressing on Gate 2
(the fixed VLSP2020 general benchmark, see voice/stt_local_train.py)?

Prior experiments (tools/import_hf_stt_dataset.py), all whisper-tiny/whisper-base,
medical-only data, were all rejected by Gate 2:
  - 60 medical samples:  Gate 2 44.4% -> 46.9%  (regression)
  - 400 medical samples: Gate 2 44.4% -> 52.4%  (worse regression -- more same-domain
    data made it WORSE, not better: no diversity added, just deeper overfit)
  - 400 medical samples, whisper-base: Gate 2 34.8% -> 42.0% (regression persists even
    with more model capacity -- rules out "capacity was the bottleneck")

This experiment tests the next lever in the plan: keep whisper-tiny, keep 60 medical
samples (unchanged from the first experiment, for a clean single-variable comparison),
but fill the rest of the adapter's 500-sample cap (MAX_STT_TRAIN_SAMPLES) with general-
domain VLSP2020 rehearsal data -- 60 medical + 440 general = 500 total.

Critical methodology point: the rehearsal samples MUST come from a disjoint slice of
doof-ferb/vlsp2020_vinai_100h from Gate 2's own fixed benchmark (offset=1000, length=500,
see tools/prepare_general_benchmark.py) -- training on the exact rows used to grade the
adapter would be leakage, making a "pass" meaningless. This script uses offset=5000 for
rehearsal data, comfortably clear of the benchmark's [1000, 1500) window.

Usage
-----
    python tools/rehearsal_mix_experiment.py --train --download-pack rehearsal.stt-pack.zip
"""
import argparse
import os
import re
import sys
import time
from io import BytesIO

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import MAX_STT_TRAIN_SAMPLES, MAX_STT_SAMPLE_DURATION_SEC

# Reuse the already-proven HTTP-session/upload/poll plumbing instead of duplicating it.
from tools.import_hf_stt_dataset import (
    StationSession, download_audio, audio_duration_sec, wait_for_training,
)

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
MEDICAL_DATASET = "HieuNguyen203/Vietnamese_Medical_Consultation"
GENERAL_DATASET = "doof-ferb/vlsp2020_vinai_100h"
# Disjoint from Gate 2's own fixed benchmark slice (offset=1000, length=500 -- see
# tools/prepare_general_benchmark.py) -- must never overlap, or "passing Gate 2" would
# just mean the model memorized the exact rows it's graded on.
REHEARSAL_OFFSET = 5000
MEDICAL_COUNT = 60
REHEARSAL_COUNT = MAX_STT_TRAIN_SAMPLES - MEDICAL_COUNT  # 440, hits the 500 cap exactly
HF_PAGE_SIZE = 100


def fetch_rows(dataset: str, split: str, offset: int, length: int, text_key: str) -> list:
    rows = []
    remaining = length
    while remaining > 0:
        page_len = min(remaining, HF_PAGE_SIZE)
        resp = requests.get(DATASETS_SERVER, params={
            "dataset": dataset, "config": "default", "split": split,
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
            text = (row.get(text_key) or "").strip()
            if not audio or not text:
                continue
            rows.append({"audio_url": audio[0]["src"], "text": text})
        remaining -= len(page_rows)
    return rows


def upload_rows(station: StationSession, adapter_id: int, rows: list, limit: int, label: str) -> int:
    print(f"Uploading up to {limit} {label} samples...")
    uploaded = 0
    for row in rows:
        if uploaded >= limit:
            break
        try:
            audio_bytes = download_audio(row["audio_url"])
            duration = audio_duration_sec(audio_bytes)
        except Exception as e:
            print(f"  skip (download/probe failed: {e})")
            continue
        if duration > MAX_STT_SAMPLE_DURATION_SEC:
            print(f"  skip ({duration:.1f}s > {MAX_STT_SAMPLE_DURATION_SEC}s cap)")
            continue
        station.upload_sample(adapter_id, audio_bytes, row["text"])
        uploaded += 1
        print(f"  [{uploaded}/{limit}] {label} uploaded ({duration:.1f}s)")
    return uploaded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--username", default="testUser1")
    ap.add_argument("--password", default="123456P@ss")
    ap.add_argument("--adapter-name", default="Rehearsal mix (60 medical + 440 VLSP2020)")
    ap.add_argument("--base-model", default="whisper-tiny")
    ap.add_argument("--backend", default="local", choices=("auto", "colab", "local"))
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--download-pack", default=None)
    args = ap.parse_args()

    station = StationSession(args.base_url)
    station.session.get(f"{args.base_url}/api/health", timeout=5).raise_for_status()

    print(f"Registering/logging in as '{args.username}'...")
    station.register_or_login(args.username, args.password)

    adapter_id = station.create_adapter(args.adapter_name, args.base_model)
    print(f"Created adapter #{adapter_id} ({args.base_model}).")

    print(f"\nFetching {MEDICAL_COUNT} medical rows (same first-{MEDICAL_COUNT} slice as the "
          f"original 60-sample experiment, offset=0, for a clean single-variable comparison)...")
    medical_rows = fetch_rows(MEDICAL_DATASET, "train", offset=0, length=int(MEDICAL_COUNT * 1.3) + 5, text_key="text")
    n_medical = upload_rows(station, adapter_id, medical_rows, MEDICAL_COUNT, "medical")

    print(f"\nFetching {REHEARSAL_COUNT} VLSP2020 general rehearsal rows "
          f"(offset={REHEARSAL_OFFSET}, disjoint from Gate 2's benchmark window [1000,1500))...")
    general_rows = fetch_rows(GENERAL_DATASET, "train", offset=REHEARSAL_OFFSET,
                               length=int(REHEARSAL_COUNT * 1.3) + 5, text_key="transcription")
    n_general = upload_rows(station, adapter_id, general_rows, REHEARSAL_COUNT, "VLSP2020 rehearsal")

    print(f"\nTotal uploaded: {n_medical} medical + {n_general} rehearsal = {n_medical + n_general}")

    if args.train:
        station.start_training(adapter_id, backend=args.backend)
        wait_for_training(station, adapter_id)
        if args.download_pack:
            pack_bytes = station.download_pack(adapter_id)
            with open(args.download_pack, "wb") as f:
                f.write(pack_bytes)
            print(f"Pack saved to {args.download_pack}")
    else:
        print(f"Samples uploaded. Log in at {args.base_url}/stt-lab to train adapter #{adapter_id}.")


if __name__ == "__main__":
    main()
