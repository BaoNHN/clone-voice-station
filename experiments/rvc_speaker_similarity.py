#!/usr/bin/env python
"""
experiments/rvc_speaker_similarity.py
Runs the same Resemblyzer-based speaker-similarity check that
engine/realism_engine.py exposes as the manager dashboard's on-demand
"realism test" (Section 6.2.4 of the thesis), but across every currently
trained voice profile at once instead of one profile on manager request,
and prints/logs the result for the thesis's own record rather than
returning it as an HTTP response.

Why this exists. Section 6.4.2 of the thesis originally reported only a
single synthesised clip (voice_01.wav), rated subjectively by the author
at 4/5 on one listen -- one rater, one sample, explicitly flagged there as
not statistically meaningful on its own. This script extends that with an
objective, reproducible measure (cosine similarity between speaker
embeddings) run across every trained profile and every one of that
profile's own held-out original samples, giving a broader (still
small-scale) quantitative check to sit alongside the subjective one.

Uses the local RVC fallback path (voice/rvc_local.py) rather than the
Colab primary backend, since a Colab tunnel session is not assumed to be
running when this is re-run later. No submitted code is modified --
this only calls the existing convert_local()/realism_engine helpers.

Usage:
    python experiments/rvc_speaker_similarity.py
    python experiments/rvc_speaker_similarity.py > experiments/logs/rvc_similarity_check.txt
"""
import asyncio
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

_BIN_DIR = os.path.join(BASE_DIR, "bin")
if os.path.isdir(_BIN_DIR):
    os.environ["PATH"] = _BIN_DIR + os.pathsep + os.environ.get("PATH", "")

from database.database import get_conn
from voice import tts, rvc_local
from engine.realism_engine import (
    DEFAULT_TEST_TEXT, _embed_audio_bytes, _embed_audio_file,
    _cosine_similarity, verdict_for,
)

OUT_JSON = os.path.join(BASE_DIR, "experiments", "logs", "rvc_similarity_check.json")


def get_ready_profiles():
    """Every profile with status='ready' and at least one on-disk sample."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, external_user_id, name, speaker_id, base_tts_voice "
        "FROM voice_profiles WHERE status='ready' AND kind='cloned'"
    ).fetchall()
    profiles = []
    for r in rows:
        pid, ext_uid, name, speaker_id, base_voice = r
        samples = conn.execute(
            "SELECT id, script_id, file_path FROM voice_samples WHERE profile_id=?",
            (pid,),
        ).fetchall()
        samples = [{"id": s[0], "script_id": s[1], "file_path": s[2]} for s in samples
                   if os.path.exists(s[2])]
        if not samples:
            continue
        profiles.append({
            "id": pid, "external_user_id": ext_uid, "name": name,
            "speaker_id": speaker_id, "base_tts_voice": base_voice or tts.DEFAULT_VOICE,
            "samples": samples,
        })
    return profiles


async def run_one(profile, text):
    t0 = time.time()
    tts_audio, base_mime = await tts.synthesize(text, voice=profile["base_tts_voice"])
    t_tts = time.time() - t0

    t1 = time.time()
    synth_audio = await asyncio.to_thread(
        rvc_local.convert_local, tts_audio, profile["speaker_id"], mime=base_mime,
    )
    t_rvc = time.time() - t1

    if synth_audio is None:
        return {"profile": profile["name"], "profile_id": profile["id"],
                "error": "local RVC conversion returned None (no local model or conversion failure)"}

    t2 = time.time()
    synth_embed = await asyncio.to_thread(_embed_audio_bytes, synth_audio)

    sample_scores = []
    embeddings = []
    for s in profile["samples"]:
        try:
            emb = await asyncio.to_thread(_embed_audio_file, s["file_path"])
        except Exception as e:
            sample_scores.append({"sample_id": s["id"], "error": str(e)})
            continue
        embeddings.append(emb)
        pair_score = round(max(0.0, _cosine_similarity(synth_embed, emb)) * 100, 1)
        sample_scores.append({"sample_id": s["id"], "score_pct": pair_score})
    t_embed = time.time() - t2

    if not embeddings:
        return {"profile": profile["name"], "profile_id": profile["id"], "error": "no readable samples"}

    import numpy as np
    centroid = np.mean(embeddings, axis=0)
    overall = round(max(0.0, _cosine_similarity(synth_embed, centroid)) * 100, 1)

    return {
        "profile": profile["name"],
        "profile_id": profile["id"],
        "speaker_id": profile["speaker_id"],
        "score_pct": overall,
        "verdict": verdict_for(overall),
        "n_samples": len(profile["samples"]),
        "sample_scores": sample_scores,
        "timing_sec": {"tts": round(t_tts, 2), "rvc_local": round(t_rvc, 2), "embed": round(t_embed, 2)},
    }


async def main():
    profiles = get_ready_profiles()
    print(f"Found {len(profiles)} ready profile(s) with on-disk samples")
    results = []
    for p in profiles:
        print(f"--- Profile {p['id']} ({p['name']}), {len(p['samples'])} sample(s) ---")
        try:
            r = await run_one(p, DEFAULT_TEST_TEXT)
        except Exception as e:
            r = {"profile": p["name"], "profile_id": p["id"], "error": repr(e)}
        print(json.dumps(r, ensure_ascii=True, indent=2))
        results.append(r)

    scores = [r["score_pct"] for r in results if "score_pct" in r]
    if scores:
        print(f"\nMean centroid similarity across {len(scores)} profile(s): "
              f"{sum(scores) / len(scores):.1f}% (range {min(scores):.1f}-{max(scores):.1f}%)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
