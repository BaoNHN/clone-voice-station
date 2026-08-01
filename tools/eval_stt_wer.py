#!/usr/bin/env python
"""
tools/eval_stt_wer.py
Word Error Rate + latency evaluation harness for clone-voice-station's STT path
(POST /api/transcribe) -- this is the ASR acceptance-criterion measurement the
thesis already commits to (Section 6.1: WER <= 15% on Vietnamese legal queries)
and the model-size comparison (T1, Section 6.3): change ASR_MODEL_NAME in
colab/voice_server.ipynb (or WHISPER_MODEL for the local fallback in
voice/stt.py) between runs and compare the printed results.

Test-set convention: a directory containing pairs of files sharing the same
stem -- an audio file plus a .txt sidecar with the reference transcript:
    001.wav   001.txt
    002.mp3   002.txt
    ...
The reference transcript is plain UTF-8 text, normally one line.

WER here is computed exactly as defined in the thesis (Section 6.2.1):
WER = (S + D + I) / N, via word-level Levenshtein edit distance against a
lowercased, punctuation-stripped tokenization -- no external dependency
(e.g. jiwer) required.

Usage:
    python tools/eval_stt_wer.py path/to/testset/
    python tools/eval_stt_wer.py path/to/testset/ --language vi --out results.csv
"""
import argparse
import csv
import glob
import mimetypes
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clone_voice_client import VoiceStationClient, VoiceStationError

AUDIO_EXTS = (".wav", ".mp3", ".webm", ".ogg", ".m4a")
WER_THRESHOLD_PCT = 15.0  # thesis Section 6.1 acceptance criterion


def resolve_api_key(explicit: str) -> str:
    if explicit:
        return explicit
    if os.path.exists("voice_station_key.txt"):
        with open("voice_station_key.txt") as f:
            return f.read().strip()
    return os.getenv("VOICE_STATION_API_KEY", "")


def _normalize(text: str) -> list:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.split()


def edit_distance(reference: str, hypothesis: str) -> tuple:
    """Returns (edits, ref_word_count) via word-level Levenshtein distance."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m], max(n, 1)


def find_pairs(testset_dir: str) -> list:
    pairs = []
    for txt_path in sorted(glob.glob(os.path.join(testset_dir, "*.txt"))):
        stem = os.path.splitext(txt_path)[0]
        audio_path = next((stem + ext for ext in AUDIO_EXTS if os.path.exists(stem + ext)), None)
        if audio_path:
            pairs.append((audio_path, txt_path))
        else:
            print(f"WARNING: no matching audio for {txt_path}, skipping", file=sys.stderr)
    return pairs


def main():
    ap = argparse.ArgumentParser(description="WER + latency evaluation for clone-voice-station's STT")
    ap.add_argument("testset_dir")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", default=None, help="Optional CSV path to write per-file results")
    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("No API key found -- pass --api-key, put one in voice_station_key.txt, "
              "or set VOICE_STATION_API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = VoiceStationClient(base_url=args.base_url, api_key=api_key)
    if not client.is_available():
        print(f"clone-voice-station is not reachable at {client.base_url}", file=sys.stderr)
        sys.exit(1)

    pairs = find_pairs(args.testset_dir)
    if not pairs:
        print(f"No (audio, .txt) pairs found in {args.testset_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_edits = 0
    total_ref_words = 0

    for audio_path, txt_path in pairs:
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        with open(audio_path, "rb") as f:
            content = f.read()
        mime = mimetypes.guess_type(audio_path)[0]

        t0 = time.perf_counter()
        try:
            result = client.transcribe(os.path.basename(audio_path), content, mime=mime, language=args.language)
            hypothesis = result.get("text", "")
            engine = result.get("engine", "?")
        except VoiceStationError as e:
            hypothesis = ""
            engine = "ERROR: " + e.message
        elapsed = time.perf_counter() - t0

        edits, ref_words = edit_distance(reference, hypothesis)
        wer_pct = round(100 * edits / ref_words, 1)
        total_edits += edits
        total_ref_words += ref_words

        row = {
            "file": os.path.basename(audio_path), "reference": reference, "hypothesis": hypothesis,
            "wer_pct": wer_pct, "latency_s": round(elapsed, 2), "engine": engine,
        }
        rows.append(row)
        print(f"{row['file']:20s}  WER={wer_pct:5.1f}%  latency={row['latency_s']:5.2f}s  engine={engine}")

    corpus_wer = round(100 * total_edits / max(total_ref_words, 1), 1)
    avg_latency = round(sum(r["latency_s"] for r in rows) / len(rows), 2)
    verdict = "PASS" if corpus_wer <= WER_THRESHOLD_PCT else "FAIL"
    print(f"\n{len(rows)} files  |  corpus WER = {corpus_wer}%  "
          f"(threshold <= {WER_THRESHOLD_PCT}%: {verdict})  |  mean latency = {avg_latency}s")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "reference", "hypothesis", "wer_pct", "latency_s", "engine"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-file results written to {args.out}")


if __name__ == "__main__":
    main()
