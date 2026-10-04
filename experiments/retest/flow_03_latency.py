#!/usr/bin/env python
"""
experiments/retest/flow_03_latency.py
Re-measures /api/transcribe and /api/speak latency for one path (colab-local, colab-ngrok, local).
Details: ai_change_log.txt (retest flows).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Report, add_common_args, load_questions, load_set30, make_client, summarize

UNREACHABLE = "http://127.0.0.1:9"


def timed(fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        return time.perf_counter() - t0, out, None
    except Exception as e:
        return time.perf_counter() - t0, None, e


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--path", required=True, choices=["colab-local", "colab-ngrok", "local"])
    ap.add_argument("--n-asr", type=int, default=30)
    ap.add_argument("--n-tts", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    c = make_client(args)
    if not c.is_available():
        sys.exit(f"station not reachable at {c.base_url}")
    uid = args.user_id
    profile_id = None
    for p in c.list_all_voice_profiles():
        if p.get("kind") == "cloned" and p.get("status") == "ready" and (not uid or str(p["external_user_id"]) == uid):
            uid, profile_id = str(p["external_user_id"]), p["id"]
            break
    if profile_id is None:
        sys.exit("no ready cloned profile found - pass --user-id")

    orig = c.get_rvc_endpoint().get("endpoint") or ""
    if args.path != "local" and not c.get_rvc_endpoint().get("available"):
        sys.exit("Colab endpoint is not reachable; start the Colab session first")

    rep = Report(f"flow_03_latency_{args.path.replace('-', '_')}")
    clips = load_set30()[: args.n_asr]
    questions = load_questions()[: args.n_tts]
    try:
        if args.path == "local":
            c.set_rvc_endpoint(UNREACHABLE)

        def asr(clip):
            _, wav_path, _ = clip
            with open(wav_path, "rb") as f:
                data = f.read()
            return lambda: c.transcribe(os.path.basename(wav_path), data, mime="audio/wav")

        def tts(q):
            return lambda: c.speak(q, uid, profile_id)

            cold_asr, _, e1 = timed(asr(clips[0]))
        cold_tts, _, e2 = timed(tts(questions[0]))
        rep.add("cold-asr", "INFO", "first call after start", f"{cold_asr:.2f}s", str(e1 or ""))
        rep.add("cold-tts", "INFO", "first call after start", f"{cold_tts:.2f}s", str(e2 or ""))
        for i in range(args.warmup):
            timed(asr(clips[i % len(clips)]))
            timed(tts(questions[i % len(questions)]))

        asr_t, asr_err = [], 0
        for clip in clips:
            dt, _, err = timed(asr(clip))
            asr_err += bool(err)
            if not err:
                asr_t.append(dt)
        tts_t, tts_err = [], 0
        for q in questions:
            dt, _, err = timed(tts(q))
            tts_err += bool(err)
            if not err:
                tts_t.append(dt)
    finally:
        if args.path == "local":
            c.set_rvc_endpoint(orig)

    for name, vals, errs in (("/api/transcribe", asr_t, asr_err), ("/api/speak", tts_t, tts_err)):
        s = summarize(vals)
        rep.add(name, "INFO", "mean / median / p95 (s)",
                f"n={s.get('n')} mean={s.get('mean')} median={s.get('median')} p95={s.get('p95')}",
                f"{errs} errors" if errs else "")
    rep.extra = {"path": args.path, "base_url": args.base_url, "cold_asr_s": round(cold_asr, 2),
                 "cold_tts_s": round(cold_tts, 2), "asr": summarize(asr_t), "tts": summarize(tts_t),
                 "thesis_reference_s": {
                     "colab-local": {"asr_mean": 2.99, "asr_p95": 3.59, "tts_mean": 5.15, "tts_p95": 6.07},
                     "colab-ngrok": {"asr_mean": 6.10, "asr_p95": 6.98, "tts_mean": 8.63, "tts_p95": 9.73},
                     "local": {"asr_mean": 2.07, "asr_p95": 2.46, "tts_mean": 3.43, "tts_p95": 8.41}}[args.path]}
    rep.save()


if __name__ == "__main__":
    main()
