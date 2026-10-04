#!/usr/bin/env python
"""
experiments/retest/flow_02_speaker_similarity.py
Re-runs the Resemblyzer speaker-similarity check across all trained profiles.
Details: ai_change_log.txt (retest flows).
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STATION_DIR, Report

SCRIPT = os.path.join(STATION_DIR, "experiments", "rvc_speaker_similarity.py")
RESULT = os.path.join(STATION_DIR, "experiments", "logs", "rvc_similarity_check.json")
THESIS = {"mean": 82.2, "range": (80.5, 84.1)}


def main():
    rep = Report("flow_02_speaker_similarity")
    if os.path.exists(RESULT):
        shutil.copy(RESULT, RESULT + ".bak_before_retest")
    proc = subprocess.run([sys.executable, SCRIPT], cwd=STATION_DIR)
    if proc.returncode != 0 or not os.path.exists(RESULT):
        rep.add("SIM", "FAIL", "script finishes", f"exit {proc.returncode}")
        rep.save()
        sys.exit(1)

    with open(RESULT, encoding="utf-8") as f:
        results = json.load(f)
    scores = []
    n_pairs = 0
    for r in results:
        if "score_pct" in r:
            scores.append(r["score_pct"])
            n_pairs += sum("score_pct" in s for s in r["sample_scores"])
            rep.add(f"P{r['profile_id']}", "INFO", "centroid similarity (khá giống >= ~80%)",
                    f"{r['score_pct']}% ({r['verdict']})", r["profile"])
        else:
            rep.add(f"P{r['profile_id']}", "FAIL", "score", r.get("error", "?"), r["profile"])
    if scores:
        mean = round(sum(scores) / len(scores), 1)
        rep.extra = {"profiles": len(scores), "sample_pairs": n_pairs, "mean_pct": mean,
                     "range": [min(scores), max(scores)], "thesis_reference": THESIS}
        rep.add("SUMMARY", "INFO", f"thesis: mean {THESIS['mean']}%, {THESIS['range']}",
                f"{len(scores)} profiles, {n_pairs} pairs, mean {mean}%, range {min(scores)}-{max(scores)}%",
                "one sentence per profile: automated embedding similarity, not perceptual naturalness")
    rep.save()


if __name__ == "__main__":
    main()
