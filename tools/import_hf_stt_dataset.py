#!/usr/bin/env python
"""
tools/import_hf_stt_dataset.py
Imports a HuggingFace audio+text dataset (default: HieuNguyen203/Vietnamese_Medical_Consultation)
into STT Lab Tier 2 (LoRA fine-tune, voice/stt_local_train.py) as real training samples through
the actual HTTP API a browser would use (register/login, create adapter, upload samples) --
this is the answer to "can the system consume a dataset shaped like this one?": it does, end
to end, not just by construction.

Uses the public datasets-server.huggingface.co REST API (rows endpoint) instead of pulling in
the `datasets`/`pyarrow` packages -- that endpoint already returns per-row {"audio": [{"src":
<signed wav URL>}], "text": <transcript>} JSON, so a plain `requests` GET per page is enough
(both clone-voice-station's requirements.txt and this dataset's schema were checked directly
before writing this, not assumed).

What it does
------------
1. Registers (or logs into) a fresh STT Lab guest account.
2. Creates a Tier 2 adapter.
3. Pulls TRAIN_LIMIT rows from the dataset's "train" split, uploads each as a training sample
   (skipping any whose audio exceeds MAX_STT_SAMPLE_DURATION_SEC -- this dataset's own card says
   20-30s per clip, right at that cap, so some rows do get skipped).
4. Pulls EVAL_LIMIT rows from the "test" split and writes them to --eval-dir as
   tools/eval_stt_wer.py-compatible (NNN.wav, NNN.txt) sidecar pairs -- a held-out set for
   measuring the trained adapter's WER improvement (see voice-lab-example/tools/
   test_medical_lora_wer.py, the other half of this demo).
5. With --train: kicks off real training (backend=local, this machine's CPU/GPU) and polls
   until done, printing progress the same way the STT Lab web UI would show it.
6. With --download-pack PATH: once ready, downloads the .stt-pack.zip -- drop it into
   voice-lab-example/stt_pack/ (or pass straight to test_medical_lora_wer.py --pack) to run
   the LoRA-vs-base WER comparison.

Usage
-----
    python tools/import_hf_stt_dataset.py --train --download-pack medical.stt-pack.zip

Defaults to training on the system's full per-adapter cap (MAX_STT_TRAIN_SAMPLES,
database/database.py) -- a registered STT Lab account is training its own adapter on its own
uploaded data, not an anonymous drive-by request, so there's no reason to hold back to a small
subset by default the way the page's old guest-oriented cap did. Pass a smaller --train-limit
for a quick pipeline smoke test.
"""
import argparse
import os
import re
import secrets
import sys
import time
from io import BytesIO

import requests

# Progress messages come straight from the server in Vietnamese (adapter["progress_message"]) --
# on Windows, stdout defaults to the console's codepage (cp1252 here), which can't encode them
# and crashes mid-poll. Same fix rvc_local.py applies to its own subprocess's stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import (
    ALLOWED_STT_BASE_MODELS, MAX_STT_SAMPLE_DURATION_SEC, MAX_STT_TRAIN_SAMPLES, MIN_STT_TRAIN_SAMPLES,
)

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "HieuNguyen203/Vietnamese_Medical_Consultation"
CSRF_TOKEN_RE   = re.compile(r'CSRF_TOKEN\s*=\s*"([^"]+)"')
POLL_INTERVAL_SEC = 10
HF_PAGE_SIZE = 100  # datasets-server's rows endpoint hard-caps "length" at 100 per call


def fetch_rows(dataset: str, split: str, offset: int, length: int) -> list:
    """Returns a list of {"audio_url": str, "text": str} from datasets-server's row preview API
    -- no download of the underlying parquet file needed, just this JSON endpoint. Paginates
    internally in HF_PAGE_SIZE chunks since the endpoint rejects length > 100 outright."""
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
            break  # ran off the end of the split
        for entry in page_rows:
            row = entry["row"]
            audio = row.get("audio")
            text = (row.get("text") or "").strip()
            if not audio or not text:
                continue
            rows.append({"audio_url": audio[0]["src"], "text": text})
        remaining -= len(page_rows)
    return rows


def download_audio(url: str) -> bytes:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def audio_duration_sec(audio_bytes: bytes) -> float:
    import soundfile as sf
    info = sf.info(BytesIO(audio_bytes))
    return info.frames / info.samplerate


class StationSession:
    """Thin wrapper around requests.Session that reproduces what the STT Lab browser UI does:
    register/login (cookie session), read the CSRF token embedded in the /stt-lab page, then
    call the JSON API with it -- see app.py's require_stt_guest()."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.csrf_token = None

    def register_or_login(self, username: str, password: str):
        resp = self.session.post(f"{self.base_url}/stt-lab/register",
                                  data={"username": username, "password": password}, timeout=15)
        if resp.status_code == 409:  # username taken -- fall back to login
            resp = self.session.post(f"{self.base_url}/stt-lab/login",
                                      data={"username": username, "password": password}, timeout=15)
        resp.raise_for_status()

        page = self.session.get(f"{self.base_url}/stt-lab", timeout=15)
        page.raise_for_status()
        m = CSRF_TOKEN_RE.search(page.text)
        if not m:
            raise RuntimeError("Could not find CSRF_TOKEN in /stt-lab page -- login likely failed.")
        self.csrf_token = m.group(1)

    def _headers(self) -> dict:
        return {"X-CSRF-Token": self.csrf_token}

    def create_adapter(self, name: str, base_model: str) -> int:
        resp = self.session.post(f"{self.base_url}/api/stt/adapters",
                                  json={"name": name, "base_model": base_model},
                                  headers=self._headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()["adapter_id"]

    def upload_sample(self, adapter_id: int, audio_bytes: bytes, reference_text: str):
        resp = self.session.post(
            f"{self.base_url}/api/stt/adapters/{adapter_id}/samples",
            files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
            data={"reference_text": reference_text},
            headers=self._headers(), timeout=30,
        )
        resp.raise_for_status()

    def start_training(self, adapter_id: int, backend: str = "local"):
        resp = self.session.post(f"{self.base_url}/api/stt/adapters/{adapter_id}/train",
                                  json={"backend": backend}, headers=self._headers(), timeout=15)
        resp.raise_for_status()

    def get_adapter(self, adapter_id: int) -> dict:
        # A longer read timeout than the other calls: local training runs synchronous
        # CPU/GPU work on a background thread, but a burst of that work (e.g. right as a
        # new epoch's model-loading step kicks in) can still delay the event loop's
        # response to this poll well past a short timeout -- hit this for real (a 15s
        # timeout aborted a poll mid-training even though the job itself finished
        # correctly seconds later).
        resp = self.session.get(f"{self.base_url}/api/stt/adapters/{adapter_id}",
                                 headers=self._headers(), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def download_pack(self, adapter_id: int) -> bytes:
        resp = self.session.get(f"{self.base_url}/api/stt/adapters/{adapter_id}/download",
                                 headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.content


def import_training_samples(station: StationSession, adapter_id: int, dataset: str, limit: int):
    print(f"Fetching up to {limit} training rows from {dataset} (split=train)...")
    # Over-fetch a bit: some rows get skipped for exceeding MAX_STT_SAMPLE_DURATION_SEC
    # (this dataset's clips run right up to that 30s cap) or a failed download.
    rows = fetch_rows(dataset, "train", offset=0, length=int(limit * 1.3) + 5)
    uploaded = 0
    for row in rows:
        if uploaded >= limit or uploaded >= MAX_STT_TRAIN_SAMPLES:
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
        print(f"  [{uploaded}/{limit}] uploaded ({duration:.1f}s, {len(row['text'])} chars)")
    if uploaded < MIN_STT_TRAIN_SAMPLES:
        raise RuntimeError(
            f"Only {uploaded} samples uploaded, need at least {MIN_STT_TRAIN_SAMPLES} to train -- "
            f"try a higher --train-limit or check network access to datasets-server.huggingface.co."
        )
    return uploaded


def export_eval_set(dataset: str, limit: int, eval_dir: str):
    print(f"Fetching up to {limit} held-out rows from {dataset} (split=test)...")
    os.makedirs(eval_dir, exist_ok=True)
    rows = fetch_rows(dataset, "test", offset=0, length=int(limit * 1.3) + 5)
    saved = 0
    for row in rows:
        if saved >= limit:
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
        stem = f"{saved:03d}"
        with open(os.path.join(eval_dir, f"{stem}.wav"), "wb") as f:
            f.write(audio_bytes)
        with open(os.path.join(eval_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(row["text"])
        saved += 1
        print(f"  [{saved}/{limit}] {stem}.wav ({duration:.1f}s)")
    print(f"Eval set written to {eval_dir} ({saved} pairs) -- point "
          f"voice-lab-example/tools/test_medical_lora_wer.py at this directory.")
    return saved


def wait_for_training(station: StationSession, adapter_id: int):
    print("Training started (backend=local) -- polling status...")
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            adapter = station.get_adapter(adapter_id)
        except requests.exceptions.RequestException as e:
            print(f"  ...poll failed ({e}), retrying")
            continue
        status = adapter["status"]
        if status == "training":
            print(f"  ...{adapter.get('progress_message') or 'training'}")
            continue
        if status == "ready":
            print(f"Training complete (backend_used={adapter.get('backend_used')}).")
            return
        if status == "failed":
            raise RuntimeError(f"Training failed: {adapter.get('error_message')}")
        print(f"  status={status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--base-url", default=os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--username", default=f"hf-import-{secrets.token_hex(4)}",
                     help="STT Lab guest username to create/reuse (default: random)")
    ap.add_argument("--password", default=secrets.token_urlsafe(12))
    ap.add_argument("--adapter-name", default="Medical Consultation (HF import)")
    ap.add_argument("--base-model", default="whisper-tiny", choices=ALLOWED_STT_BASE_MODELS)
    ap.add_argument("--train-limit", type=int, default=MAX_STT_TRAIN_SAMPLES,
                     help=f"Training samples to upload (default: the system max, {MAX_STT_TRAIN_SAMPLES} -- "
                          f"a registered STT Lab account trains on everything it uploads, not a small guest-style subset)")
    ap.add_argument("--eval-limit", type=int, default=30, help="Held-out test samples to export (default 30)")
    ap.add_argument("--eval-dir", default=os.path.join(os.path.dirname(__file__), "medical_eval_set"))
    ap.add_argument("--train", action="store_true", help="Kick off training and wait for it to finish")
    ap.add_argument("--download-pack", default=None, help="Path to save the .stt-pack.zip once ready")
    args = ap.parse_args()

    station = StationSession(args.base_url)
    try:
        station.session.get(f"{args.base_url}/api/health", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"clone-voice-station not reachable at {args.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Registering guest '{args.username}' on {args.base_url}...")
    station.register_or_login(args.username, args.password)
    print(f"  (credentials -- username: {args.username}  password: {args.password} -- "
          f"save these to log back in and inspect the adapter at /stt-lab)")

    adapter_id = station.create_adapter(args.adapter_name, args.base_model)
    print(f"Created adapter #{adapter_id} ({args.base_model}).")

    import_training_samples(station, adapter_id, args.dataset, args.train_limit)
    export_eval_set(args.dataset, args.eval_limit, args.eval_dir)

    if args.train:
        station.start_training(adapter_id, backend="local")
        wait_for_training(station, adapter_id)

        if args.download_pack:
            pack_bytes = station.download_pack(adapter_id)
            with open(args.download_pack, "wb") as f:
                f.write(pack_bytes)
            print(f"Pack saved to {args.download_pack} "
                  f"(copy into voice-lab-example/stt_pack/, or pass --pack directly).")
    else:
        print(f"Samples uploaded. Log in at {args.base_url}/stt-lab to train adapter #{adapter_id}, "
              f"or re-run this script with --train.")


if __name__ == "__main__":
    main()
