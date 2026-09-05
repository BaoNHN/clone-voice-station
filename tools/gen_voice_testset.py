#!/usr/bin/env python
"""
tools/gen_voice_testset.py
Generates the paired audio test set that tools/eval_voice_quality.py scores, so
the RQ2 comparison in the thesis (Section 6.1, Table 8) runs on the same texts
through both voice paths.

Both systems are reached through the one /api/speak endpoint. Which path runs is
a property of the profile, not of the request:

    TTS + RVC   a cloned profile whose status is 'ready' -- edge-TTS produces the
                base audio and the trained RVC model re-voices it. The response
                carries the spoken AI-disclosure prefix (Section 5.2).

    F5-TTS      a profile whose base_tts_voice is "f5tts:default", which routes
                synthesis to F5-TTS-Vietnamese-ViVoice on the Colab endpoint. No
                RVC step, so no disclosure prefix.

Pass --list-profiles to see the profile ids available for a user and pick the two.

Input is one text per line, UTF-8. The thesis test set is the 30 Vietnamese legal
questions of Section 6, so answers -- not questions -- belong in this file: the
evaluation scores the audio the assistant speaks back.

    python tools/gen_voice_testset.py --list-profiles --external-user-id u1
    python tools/gen_voice_testset.py --texts answers.txt --external-user-id u1 \
        --profile-id 12 --out rvc_out/
    python tools/gen_voice_testset.py --texts answers.txt --external-user-id u1 \
        --profile-id 3  --out f5_out/

Files are named by line number (001.wav, 002.wav, ...) so the two directories
pair up by stem, which is what eval_voice_quality.py expects. A line that fails
is reported and skipped, leaving a gap in the numbering rather than shifting
every later file onto the wrong text.

Latency is recorded per request into latency.csv beside the audio, which is the
end-to-end speech-output measurement RQ3 needs (Section 6.2.2). Note it covers
the /api/speak leg only: ASR and the LLM are measured separately.
"""
import argparse
import csv
import mimetypes
import os
import sys
import time

# --list-profiles prints profile names, which are Vietnamese ("GiongNu"), and a
# Windows console defaults to cp1252 and cannot encode them. Same guard as
# tools/split_recording.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clone_voice_client import VoiceStationClient, VoiceStationError

# engine/voice_engine.py only prepends the AI-disclosure clip when RVC actually
# converted the audio, so its absence is how a silent fallback to plain TTS is
# detected -- the same marker eval_voice_quality.py uses to exclude a file.
WAV_MAGIC = b"RIFF"


def resolve_api_key(explicit):
    if explicit:
        return explicit
    if os.path.exists("voice_station_key.txt"):
        with open("voice_station_key.txt") as f:
            return f.read().strip()
    return os.getenv("VOICE_STATION_API_KEY", "")


def main():
    ap = argparse.ArgumentParser(
        description="Generate the paired TTS+RVC / F5-TTS audio test set")
    ap.add_argument("--texts", help="UTF-8 file, one answer text per line")
    ap.add_argument("--out", help="Output directory for the generated WAVs")
    ap.add_argument("--external-user-id", required=True,
                    help="Opaque user id the profile belongs to")
    ap.add_argument("--profile-id", type=int,
                    help="Voice profile to speak through (see --list-profiles)")
    ap.add_argument("--list-profiles", action="store_true",
                    help="List this user's profiles and exit")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only generate the first N texts (for a smoke test)")
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

    if args.list_profiles:
        profiles = client.list_voice_profiles(args.external_user_id)
        if not profiles:
            print("No profiles for that external_user_id.")
            return
        print(f"{'id':>4}  {'kind':10s} {'status':10s} {'base_tts_voice':28s} name")
        for p in profiles:
            print(f"{p.get('id', '?'):>4}  {str(p.get('kind')):10s} "
                  f"{str(p.get('status')):10s} {str(p.get('base_tts_voice')):28s} "
                  f"{p.get('name', '')}")
        print("\nFor the RQ2 pair: the TTS+RVC system is a kind='cloned', status='ready'"
              "\nprofile; the baseline is a profile whose base_tts_voice is 'f5tts:default'.")
        return

    if not args.texts or not args.out or args.profile_id is None:
        ap.error("--texts, --out and --profile-id are required unless --list-profiles")

    with open(args.texts, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if args.limit:
        texts = texts[:args.limit]
    if not texts:
        print(f"No non-empty lines in {args.texts}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    failures = 0

    for i, text in enumerate(texts, start=1):
        stem = f"{i:03d}"
        t0 = time.perf_counter()
        try:
            result = client.speak(text, args.external_user_id, profile_id=args.profile_id)
        except VoiceStationError as e:
            print(f"  {stem}  FAILED: {e.message}", file=sys.stderr)
            failures += 1
            continue
        elapsed = time.perf_counter() - t0

        audio = result["audio"]
        mime = result.get("mime", "")
        if not audio:
            print(f"  {stem}  FAILED: empty response body", file=sys.stderr)
            failures += 1
            continue

        if not audio.startswith(WAV_MAGIC):
            # speak_text() returns the base TTS mime (mp3) untouched whenever RVC
            # did not convert; only the converted path re-exports as WAV. Keep the
            # file so the run stays auditable, but flag it: eval_voice_quality.py
            # would exclude it from the aggregate anyway.
            note = f"RVC did not run, plain TTS returned as {mime or 'unknown type'}"
        else:
            note = ""

        path = os.path.join(args.out, stem + ".wav")
        with open(path, "wb") as f:
            f.write(audio)

        rows.append({"file": stem, "latency_s": round(elapsed, 2),
                     "bytes": len(audio), "note": note, "text": text})
        print(f"  {stem}  {elapsed:5.2f}s  {len(audio):>8} bytes  {note}")

    if rows:
        lat = sorted(r["latency_s"] for r in rows)
        mid = lat[len(lat) // 2] if len(lat) % 2 else (lat[len(lat)//2 - 1] + lat[len(lat)//2]) / 2
        csv_path = os.path.join(args.out, "latency.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "latency_s", "bytes", "note", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} file(s) written to {args.out}"
              + (f", {failures} failed" if failures else ""))
        print(f"/api/speak latency: median {mid:.2f}s, mean "
              f"{sum(lat)/len(lat):.2f}s, max {max(lat):.2f}s")
        print(f"Per-request latency written to {csv_path}")

        bad = [r["file"] for r in rows if r["note"]]
        if bad:
            print(f"\n{len(bad)} response(s) came back without RVC conversion: "
                  + ", ".join(bad)
                  + "\nBring the Colab RVC session up and regenerate those before scoring.")


if __name__ == "__main__":
    main()
