"""
experiments/retest/common.py
Shared helpers for the retest flows (client, WER, audio, report).
Details: ai_change_log.txt (retest flows).
"""
import argparse
import io
import json
import math
import os
import re
import statistics
import sys
import wave
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STATION_DIR = os.path.dirname(os.path.dirname(HERE))
PROJECT_DIR = os.path.dirname(STATION_DIR)
LOG_DIR = os.path.join(HERE, "logs")
SET30_DIR = os.path.join(STATION_DIR, "tools", "legal_testset", "set30")
QUESTIONS_TXT = os.path.join(STATION_DIR, "tools", "legal_testset", "questions_vi.txt")

# SDK lives in a sibling repo (pip install -e ../clone-voice-client also works)
_SDK = os.path.join(PROJECT_DIR, "clone-voice-client")
if os.path.isdir(_SDK) and _SDK not in sys.path:
    sys.path.insert(0, _SDK)

# Key lookup order: --api-key, env, then the key files already used by the apps.
KEY_FILES = [
    os.path.join(STATION_DIR, "voice_station_key.txt"),
    os.path.join(PROJECT_DIR, "voice-lab-example", "voice_station_key.txt"),
    os.path.join(PROJECT_DIR, "rag-legal-assistant", "voice_station_key.txt"),
]


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--base-url", default=os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--user-id", default=None, help="external_user_id owning a ready cloned profile")
    return ap


def resolve_api_key(explicit=None) -> str:
    if explicit:
        return explicit
    if os.getenv("VOICE_STATION_API_KEY"):
        return os.environ["VOICE_STATION_API_KEY"]
    for p in KEY_FILES:
        if os.path.exists(p):
            with open(p) as f:
                k = f.read().strip()
            if k:
                return k
    return ""


def make_client(args):
    from clone_voice_client import VoiceStationClient
    return VoiceStationClient(base_url=args.base_url, api_key=resolve_api_key(args.api_key),
                              request_timeout=30, speak_timeout=120, upload_timeout=300)


# ---------------------------------------------------------------- statistics
def pctl(values, p):
    """Linear-interpolated percentile (matches numpy default)."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def summarize(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p95": round(pctl(values, 95), 2), "min": round(min(values), 2),
            "max": round(max(values), 2)}


# ----------------------------------------------------------------------- WER
def _norm(text):
    text = re.sub(r"[^\w\s]", "", text.lower().strip(), flags=re.UNICODE)
    return text.split()


def edit_counts(ref, hyp):
    """Word-level Levenshtein -> (S, D, I, N). Same definition as Section 6.2.1."""
    r, h = _norm(ref), _norm(hyp)
    n, m = len(r), len(h)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]  # (cost, S, D, I)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c, s, d, ins = dp[i - 1][j - 1]; sub = (c + 1, s + 1, d, ins)
            c, s, d, ins = dp[i - 1][j];     dele = (c + 1, s, d + 1, ins)
            c, s, d, ins = dp[i][j - 1];     inse = (c + 1, s, d, ins + 1)
            dp[i][j] = min(sub, dele, inse)
    _, S, D, I = dp[n][m]
    return S, D, I, max(n, 1)


def corpus_wer(pairs):
    """pairs: iterable of (reference, hypothesis) -> dict like wer_legal_set30_summary.json"""
    S = D = I = N = 0
    exact = 0
    for ref, hyp in pairs:
        s, d, i, n = edit_counts(ref, hyp)
        S += s; D += d; I += i; N += n
        exact += (s + d + i == 0)
    errs = S + D + I
    return {"corpus_wer_pct": round(100 * errs / max(N, 1), 2), "total_errors": errs,
            "total_ref_words": N, "substitutions": S, "deletions": D, "insertions": I,
            "clips_exact": exact}


def load_set30(limit=None):
    """[(stem, wav_path, reference_text)] for the 30 legal-speech clips."""
    out = []
    for name in sorted(os.listdir(SET30_DIR)):
        if name.endswith(".wav"):
            stem = name[:-4]
            txt = os.path.join(SET30_DIR, stem + ".txt")
            if os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    out.append((stem, os.path.join(SET30_DIR, name), f.read().strip()))
    return out[:limit] if limit else out


def load_questions():
    with open(QUESTIONS_TXT, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# --------------------------------------------------------------------- audio
def silent_wav(seconds, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


def concat_wavs(paths, min_seconds):
    """Concatenate clips (cycling) until >= min_seconds. Returns wav bytes."""
    frames, params, total, i = [], None, 0.0, 0
    while total < min_seconds:
        with wave.open(paths[i % len(paths)], "rb") as w:
            if params is None:
                params = w.getparams()
            elif (w.getframerate(), w.getnchannels(), w.getsampwidth()) != \
                    (params.framerate, params.nchannels, params.sampwidth):
                i += 1
                continue
            frames.append(w.readframes(w.getnframes()))
            total += w.getnframes() / w.getframerate()
        i += 1
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def wav_info(content: bytes):
    """(sample_rate, seconds, comment tag or None) from raw WAV bytes."""
    with wave.open(io.BytesIO(content), "rb") as w:
        sr, secs = w.getframerate(), w.getnframes() / w.getframerate()
    m = re.search(rb"ICMT.{4}(.*?)\x00", content, re.S)
    return sr, secs, (m.group(1).decode("utf-8", "ignore") if m else None)


# -------------------------------------------------------------------- report
class Report:
    """Collects rows and writes logs/<flow>_<timestamp>.json + a .md table."""

    def __init__(self, flow):
        self.flow, self.rows, self.extra = flow, [], {}
        self.started = datetime.now()
        os.makedirs(LOG_DIR, exist_ok=True)

    def add(self, id_, status, expected, actual, note=""):
        assert status in ("PASS", "FAIL", "SKIP", "INFO")
        self.rows.append({"id": id_, "status": status, "expected": expected,
                          "actual": actual, "note": note})
        print(f"[{status:4}] {id_:8} {actual}" + (f"  ({note})" if note else ""), flush=True)

    def save(self):
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(LOG_DIR, f"{self.flow}_{stamp}")
        counts = {s: sum(r["status"] == s for r in self.rows) for s in ("PASS", "FAIL", "SKIP", "INFO")}
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"flow": self.flow, "run_at": self.started.isoformat(timespec="seconds"),
                       "counts": counts, "rows": self.rows, "extra": self.extra},
                      f, ensure_ascii=False, indent=2)

        def esc(s):
            return str(s).replace("|", "\\|").replace("\n", " ")

        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(f"# {self.flow} - {self.started:%Y-%m-%d %H:%M}\n\n{counts}\n\n")
            f.write("| ID | Status | Expected | Actual | Note |\n|---|---|---|---|---|\n")
            for r in self.rows:
                f.write(f"| {r['id']} | {r['status']} | {esc(r['expected'])} | {esc(r['actual'])} | {esc(r['note'])} |\n")
        print(f"\nsaved {base}.json / .md   {counts}")
        return counts
