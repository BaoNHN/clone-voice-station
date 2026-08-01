#!/usr/bin/env python
"""
tools/test_stt.py
Quick smoke test for clone-voice-station's Speech-to-Text path (POST /api/transcribe).
Sends one audio file, prints the transcribed text, which engine actually served it
(phowhisper:<model> via Colab, or whisper-local:<model> fallback -- see the "engine"
field added in voice/stt.py and colab/voice_server.ipynb's /transcribe route), and
round-trip latency. Useful to confirm the whole path works before a demo, without
going through the browser mic UI.

Usage:
    python tools/test_stt.py path/to/audio.wav
    python tools/test_stt.py path/to/audio.wav --language vi --base-url http://127.0.0.1:8090

API key resolution order: --api-key, then voice_station_key.txt in the current
directory, then the VOICE_STATION_API_KEY env var.
"""
import argparse
import mimetypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clone_voice_client import VoiceStationClient, VoiceStationError


def resolve_api_key(explicit: str) -> str:
    if explicit:
        return explicit
    if os.path.exists("voice_station_key.txt"):
        with open("voice_station_key.txt") as f:
            return f.read().strip()
    return os.getenv("VOICE_STATION_API_KEY", "")


def main():
    ap = argparse.ArgumentParser(description="Smoke-test clone-voice-station's /api/transcribe")
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/webm/ogg/m4a)")
    ap.add_argument("--language", default="vi", help="Language hint (default: vi)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None, help="Default: VOICE_STATION_URL env var or http://127.0.0.1:8090")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print(f"File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("No API key found -- pass --api-key, put one in voice_station_key.txt, "
              "or set VOICE_STATION_API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = VoiceStationClient(base_url=args.base_url, api_key=api_key)

    if not client.is_available():
        print(f"clone-voice-station is not reachable at {client.base_url} -- is it running?", file=sys.stderr)
        sys.exit(1)

    with open(args.audio, "rb") as f:
        content = f.read()

    mime = mimetypes.guess_type(args.audio)[0]

    t0 = time.perf_counter()
    try:
        result = client.transcribe(os.path.basename(args.audio), content, mime=mime, language=args.language)
    except VoiceStationError as e:
        print(f"Transcription failed: {e.message}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print(f"Text     : {result.get('text')}")
    print(f"Language : {result.get('language')}")
    print(f"Engine   : {result.get('engine', '?')}")
    print(f"Latency  : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
