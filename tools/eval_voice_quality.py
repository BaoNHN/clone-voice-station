#!/usr/bin/env python
"""
tools/eval_voice_quality.py
Objective output-quality evaluation for the TTS + RVC voice path -- the RVC-output
half of the thesis acceptance criteria (Section 6.1, Table 8) and the metric
definitions in Section 6.2.3.

Four measurements, one per criterion in Table 8:

  SNR (dB)          Artefact screen only. Active speech level per ITU-T P.56 [23]
                    against the RMS of the inactive regions. Threshold >= 20 dB is
                    this thesis's own screening bound, not a standard's -- see
                    Section 6.2.3 for why no standard defines one for synthesised
                    speech. Pure numpy, no extra dependency, always runs.

  UTMOS (1-5)       Reference-free naturalness predictor [25], VoiceMOS Challenge
                    2022 top system. Relative criterion: must exceed the
                    F5-TTS-Vietnamese-ViVoice baseline on the same test set.

  NISQA (1-5)       Second reference-free naturalness predictor [26], used as a
                    cross-check on UTMOS rather than as an independent criterion.

  ECAPA cosine      Target-speaker similarity: cosine distance between ECAPA-TDNN
                    speaker embeddings [27] of the converted output and of held-out
                    reference recordings of the same speaker. Relative criterion,
                    same as UTMOS.

The three neural metrics are optional and load lazily. A missing dependency
degrades that one metric to "skipped" and the rest of the run continues, the same
degrade-gracefully contract the station itself uses for Colab outages. Run with
--check to see what is available before committing to a full pass.

SNR needs only numpy and scipy, so it runs under the station's own interpreter
with nothing installed. UTMOS and ECAPA pull torchaudio and speechbrain, which
would change eight shared packages in the machine's base environment (including a
huggingface_hub upgrade) and pair torchaudio against a torch it was not built
for. They are therefore installed into a throwaway virtualenv beside the repo,
which leaves the interpreter that serves the station untouched:

    python -m venv .venv-eval
    .venv-eval/Scripts/python -m pip install numpy scipy speechbrain
    .venv-eval/Scripts/python -m pip install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
    .venv-eval/Scripts/python tools/eval_voice_quality.py --check

    # NISQA is a source checkout rather than a package:
    git clone https://github.com/gabrielmittag/NISQA   # then pass --nisqa-dir

venv writes its own .gitignore, so .venv-eval stays out of the repository.

DISCLOSURE PREFIX -- read this before trusting the numbers
    engine/voice_engine.py prepends a spoken AI-disclosure clip plus 350 ms of
    silence to every genuinely RVC-converted response (Section 5.2). That clip is
    synthesised with the *base* TTS voice and is NOT voice-converted, so measuring
    a whole /api/speak WAV would score edge-TTS audio as if it were RVC output and
    would drag the speaker-similarity score toward the base voice. This script
    detects the 350 ms separator and trims everything before it by default.

    That same prefix doubles as the marker for whether RVC actually ran: /api/speak
    returns plain TTS with no disclosure whenever the Colab RVC session is offline.
    A --system file with no detectable prefix is reported as NO-RVC and excluded
    from the aggregate, so a silent fallback cannot inflate the results.

    Baseline (F5-TTS) files carry no disclosure -- pass --baseline-no-trim, which
    is the default for that directory.

Test-set convention, following tools/eval_stt_wer.py: files are paired across
directories by filename stem.

    rvc_out/001.wav   f5_out/001.wav      speaker_refs/*.wav
    rvc_out/002.wav   f5_out/002.wav

Usage:
    python tools/eval_voice_quality.py --check
    python tools/eval_voice_quality.py --system rvc_out/
    python tools/eval_voice_quality.py --system rvc_out/ --baseline f5_out/ \
        --speaker-refs speaker_refs/ --out results.csv
"""
import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave

# This machine's shared Python environment links two OpenMP runtimes, so importing
# torch aborts the process with "OMP: Error #15" before any of this script runs.
# Same class of environment defect as the conda-forge ffmpeg conflict that
# clone_voice_client's local STT path already works around. setdefault so an
# operator who has deliberately set the variable keeps their value.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Thesis Section 6.1, Table 8. SNR is an absolute screening bound; UTMOS and the
# ECAPA cosine are relative criteria, judged against the baseline on the same set.
SNR_SCREEN_DB = 20.0

# ITU-T P.56 method B constants.
P56_TIME_CONSTANT_S = 0.03
P56_HANGOVER_S = 0.2
P56_MARGIN_DB = 15.9

# Noise-floor estimation (see noise_floor_power for why P.56's own activity mask
# is not used here). A clip whose activity factor exceeds HIGH_ACTIVITY_WARN has
# too little silence for the percentile to land on the floor, so its SNR is a
# lower bound and is flagged in the per-file output.
NOISE_FRAME_MS = 20
NOISE_HOP_MS = 10
NOISE_PERCENTILE = 10
HIGH_ACTIVITY_WARN = 0.95

# A run of near-digital-silence at least this long marks the disclosure separator
# (engine/voice_engine.py inserts exactly 350 ms). The disclosure clip itself runs
# several seconds, so a separator found before MIN_ or after MAX_ is not it.
SILENCE_AMPLITUDE = 1e-4
SEPARATOR_MIN_S = 0.25
DISCLOSURE_MIN_S = 0.5
DISCLOSURE_MAX_S = 15.0


# --------------------------------------------------------------------------- io

def read_wav(path):
    """Returns (mono float64 in [-1, 1], sample_rate). PCM WAV only."""
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype="u1").astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width} bytes in {path}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def write_wav(path, x, sr):
    """16-bit PCM mono, used to hand NISQA the same trimmed audio the other
    metrics see."""
    data = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def resample_to(x, sr, target_sr):
    if sr == target_sr:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    return resample_poly(x, int(target_sr) // g, int(sr) // g)


# ------------------------------------------------------------ disclosure prefix

def find_disclosure_end(x, sr):
    """Seconds to trim so the returned audio starts at the RVC-converted content.

    Looks for the first >= SEPARATOR_MIN_S run of near-zero samples occurring
    between DISCLOSURE_MIN_S and DISCLOSURE_MAX_S. Returns None when no such run
    exists, which means the file carries no disclosure -- for a --system file that
    is the signal that RVC did not run.
    """
    quiet = np.abs(x) < SILENCE_AMPLITUDE
    if not quiet.any():
        return None

    lo = int(DISCLOSURE_MIN_S * sr)
    hi = min(len(x), int(DISCLOSURE_MAX_S * sr))
    need = int(SEPARATOR_MIN_S * sr)

    run_start = None
    for i in range(lo, hi):
        if quiet[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= need:
                return i / sr
            run_start = None

    if run_start is not None and hi - run_start >= need:
        return hi / sr
    return None


# -------------------------------------------------------------------- ITU-T P.56

def p56_active_level(x, sr):
    """ITU-T P.56 method B.

    Returns (active_level_dbov, activity_factor, activity_mask) or None when the
    signal carries no energy. The active level is referenced to full scale, so it
    is negative for any signal that does not clip.
    """
    from scipy.signal import lfilter

    n = len(x)
    if n == 0:
        return None
    sq = float(np.dot(x, x))
    if sq <= 0.0:
        return None

    # Two cascaded first-order smoothers give the P.56 envelope q(i).
    g = math.exp(-1.0 / (sr * P56_TIME_CONSTANT_S))
    b, a = [1.0 - g], [1.0, -g]
    p = lfilter(b, a, np.abs(x))
    q = lfilter(b, a, p)

    hangover = int(round(P56_HANGOVER_S * sr))
    idx = np.arange(n)

    best = None
    for j in range(1, 17):
        c = 2.0 ** (-j)
        above = q >= c
        if not above.any():
            continue
        # Union of [i, i+hangover) over every i where the envelope crosses c.
        last_true = np.maximum.accumulate(np.where(above, idx, -1))
        active = (last_true >= 0) & ((idx - last_true) < hangover)
        a_count = int(active.sum())
        if a_count == 0:
            continue

        level_db = 10.0 * math.log10(sq / a_count)
        margin = level_db - 20.0 * math.log10(c)
        # Walk thresholds downward until the margin closes on 15.9 dB, then keep
        # the threshold whose margin sits nearest to it.
        if best is None or abs(margin - P56_MARGIN_DB) < abs(best[0] - P56_MARGIN_DB):
            best = (margin, level_db, a_count / n, active)

    if best is None:
        return None
    _, level_db, activity, active = best
    return level_db, activity, active


def noise_floor_power(x, sr, frame_ms=NOISE_FRAME_MS, hop_ms=NOISE_HOP_MS,
                      percentile=NOISE_PERCENTILE):
    """Noise-floor power as a low percentile of short-term frame power.

    The P.56 activity mask is deliberately NOT used for this. That mask carries a
    200 ms hangover, so its complement still contains speech decay, and measuring
    the residual there yields a figure driven by speech leakage rather than by the
    noise floor -- on synthetic signals it stayed pinned near 12 dB while the true
    SNR was swept from 60 dB to 20 dB. A low percentile of frame power is the
    standard minimum-statistics estimate and tracks the true floor to within about
    1.2 dB across that same sweep.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame <= 0 or hop <= 0 or len(x) < frame:
        return None
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    power = (x[idx] ** 2).mean(axis=1)
    return float(np.percentile(power, percentile))


def snr_db(x, sr):
    """Active speech level (ITU-T P.56) over the estimated noise floor, in dB.

    P.56 standardises the active-level measurement only; this ratio is the
    thesis's own construction and Section 6.2.3 says so. Returns
    (snr_db, activity_factor); either element may be None.

    The estimate carries a small negative bias of roughly 1 dB and it degrades
    when a clip has no pauses at all, because the percentile then lands on the
    quietest speech frame instead of on silence. The activity factor is reported
    per file so that case is visible: above HIGH_ACTIVITY_WARN the SNR column
    should be read as a lower bound.
    """
    res = p56_active_level(x, sr)
    if res is None:
        return None, None
    level_db, activity, _ = res

    noise_power = noise_floor_power(x, sr)
    if noise_power is None:
        return None, activity
    if noise_power <= 0.0:
        return float("inf"), activity
    return level_db - 10.0 * math.log10(noise_power), activity


# ------------------------------------------------------------------- UTMOS [25]

class Utmos:
    name = "UTMOS"

    def __init__(self):
        self.model = None
        self.error = None

    def load(self):
        if self.model is not None or self.error is not None:
            return
        try:
            import torch
            self.torch = torch
            self.model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            self.model.eval()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def score(self, x, sr):
        self.load()
        if self.model is None:
            return None
        wav = resample_to(x, sr, 16000)
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
            return float(self.model(t, 16000).item())


# ------------------------------------------------------------- ECAPA-TDNN [27]

class Ecapa:
    name = "ECAPA cosine"

    def __init__(self):
        self.encoder = None
        self.error = None
        self.reference = None

    def load(self):
        if self.encoder is not None or self.error is not None:
            return
        try:
            import torch
            self.torch = torch
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier

            kwargs = dict(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(tempfile.gettempdir(), "spkrec-ecapa-voxceleb"),
            )
            # SpeechBrain >= 1.0 fetches by symlinking out of the HF cache, which on
            # Windows needs Developer Mode or an elevated shell and otherwise fails
            # with WinError 1314. Copying costs a few hundred MB of disk once and
            # needs no privilege. Older releases have no such parameter.
            try:
                from speechbrain.utils.fetching import LocalStrategy
                kwargs["local_strategy"] = LocalStrategy.COPY
            except ImportError:
                pass

            self.encoder = EncoderClassifier.from_hparams(**kwargs)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def embed(self, x, sr):
        self.load()
        if self.encoder is None:
            return None
        wav = resample_to(x, sr, 16000)
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
            emb = self.encoder.encode_batch(t).squeeze()
        v = emb.cpu().numpy().reshape(-1)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else None

    def fit_reference(self, ref_paths):
        """Mean of the held-out reference recordings of the target speaker."""
        vecs = []
        for path in ref_paths:
            try:
                x, sr = read_wav(path)
            except Exception as e:
                print(f"  reference {os.path.basename(path)}: unreadable ({e})", file=sys.stderr)
                continue
            v = self.embed(x, sr)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return False
        mean = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            return False
        self.reference = mean / norm
        return True

    def score(self, x, sr):
        if self.reference is None:
            return None
        v = self.embed(x, sr)
        return None if v is None else float(np.dot(v, self.reference))


# ------------------------------------------------------------------- NISQA [26]

class Nisqa:
    name = "NISQA"

    def __init__(self, nisqa_dir):
        # Absolute: run_predict.py is launched with cwd set to the NISQA checkout
        # (it imports its own `nisqa` package by relative path), so every path
        # handed to the subprocess has to survive that change of directory.
        self.dir = os.path.abspath(nisqa_dir) if nisqa_dir else None
        self.error = None
        if not nisqa_dir:
            self.error = "not requested (pass --nisqa-dir)"
        elif not os.path.isdir(self.dir):
            self.error = f"directory not found: {self.dir}"

    def score_dir(self, wav_dir):
        """Runs NISQA's own run_predict.py over a directory. Returns {stem: mos}."""
        if self.error:
            return {}
        weights = os.path.join(self.dir, "weights", "nisqa_tts.tar")
        if not os.path.exists(weights):
            weights = os.path.join(self.dir, "weights", "nisqa.tar")
        if not os.path.exists(weights):
            self.error = "no weights found under NISQA/weights/"
            return {}

        out_dir = tempfile.mkdtemp(prefix="nisqa_")
        cmd = [
            sys.executable, os.path.join(self.dir, "run_predict.py"),
            "--mode", "predict_dir", "--pretrained_model", weights,
            "--data_dir", os.path.abspath(wav_dir), "--num_workers", "0", "--bs", "1",
            "--output_dir", out_dir,
        ]
        try:
            proc = subprocess.run(cmd, cwd=self.dir, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or e.stdout or "").strip().splitlines()[-3:]
            self.error = "run_predict.py exited %d: %s" % (e.returncode, " | ".join(tail))
            return {}
        except Exception as e:
            self.error = f"could not launch run_predict.py: {type(e).__name__}: {e}"
            return {}

        scores = {}
        produced = glob.glob(os.path.join(out_dir, "*.csv"))
        if not produced:
            self.error = f"run_predict.py wrote no CSV into {out_dir}"
            return {}
        for csv_path in produced:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    deg = row.get("deg") or row.get("filename") or ""
                    mos = row.get("mos_pred") or row.get("mos")
                    if deg and mos:
                        try:
                            scores[os.path.splitext(os.path.basename(deg))[0]] = float(mos)
                        except ValueError:
                            pass
        return scores


# ------------------------------------------------------------------ measurement

def measure_dir(label, directory, utmos, ecapa, nisqa, trim):
    """Scores every WAV in a directory.

    NISQA runs over a temporary copy of the *trimmed* audio rather than over the
    source directory, so all four metrics see exactly the same samples. Scoring
    the source directory instead would feed NISQA the AI-disclosure prefix that
    UTMOS and ECAPA never see.
    """
    rows = []
    paths = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if not paths:
        print(f"No .wav files in {directory}", file=sys.stderr)
        return rows

    trimmed_dir = tempfile.mkdtemp(prefix=f"evalq_{label}_")
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            x, sr = read_wav(path)
        except Exception as e:
            print(f"  {stem:16s} unreadable: {e}", file=sys.stderr)
            continue

        duration = len(x) / sr
        trimmed = 0.0
        rvc_ran = ""

        if trim:
            cut = find_disclosure_end(x, sr)
            if cut is None:
                # No disclosure prefix: /api/speak fell back to plain TTS.
                rvc_ran = "NO-RVC"
            else:
                trimmed = cut
                x = x[int(cut * sr):]

        if len(x) < sr * 0.2:
            print(f"  {stem:16s} too short after trimming, skipped", file=sys.stderr)
            continue

        snr, activity = snr_db(x, sr)

        flags = [rvc_ran] if rvc_ran else []
        if activity is not None and activity > HIGH_ACTIVITY_WARN:
            # Too little silence for the percentile to find the floor; the SNR
            # below is a lower bound rather than an estimate.
            flags.append("SNR-LOWER-BOUND")

        row = {
            "file": stem, "system": label,
            "duration_s": round(duration, 2), "trimmed_s": round(trimmed, 2),
            "flag": " ".join(flags),
            "snr_db": None if snr is None else round(snr, 1),
            "activity_pct": None if activity is None else round(100 * activity, 1),
            "utmos": None, "nisqa": None, "ecapa_cos": None,
        }
        write_wav(os.path.join(trimmed_dir, stem + ".wav"), x, sr)
        u = utmos.score(x, sr)
        if u is not None:
            row["utmos"] = round(u, 3)
        c = ecapa.score(x, sr)
        if c is not None:
            row["ecapa_cos"] = round(c, 4)

        rows.append(row)

    nisqa_scores = nisqa.score_dir(trimmed_dir)
    for row in rows:
        if row["file"] in nisqa_scores:
            row["nisqa"] = round(nisqa_scores[row["file"]], 3)
    shutil.rmtree(trimmed_dir, ignore_errors=True)

    for row in rows:
        print(f"  {row['file']:16s} SNR={_fmt(row['snr_db'],'5.1f')} dB  "
              f"UTMOS={_fmt(row['utmos'],'5.3f')}  "
              f"NISQA={_fmt(row['nisqa'],'5.3f')}  "
              f"ECAPA={_fmt(row['ecapa_cos'],'6.4f')}  {row['flag']}")
    return rows


def _fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "  n/a"


def mean_of(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and math.isfinite(r[key])]
    return (sum(vals) / len(vals)) if vals else None


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Objective RVC output-quality evaluation (thesis Table 8)")
    ap.add_argument("--system", help="Directory of TTS + RVC output WAVs")
    ap.add_argument("--baseline", help="Directory of F5-TTS-Vietnamese-ViVoice baseline WAVs")
    ap.add_argument("--speaker-refs", help="Directory of held-out target-speaker WAVs for ECAPA")
    ap.add_argument("--nisqa-dir", help="Path to a clone of gabrielmittag/NISQA")
    ap.add_argument("--no-trim", action="store_true",
                    help="Do not trim the AI-disclosure prefix from --system files")
    ap.add_argument("--baseline-trim", action="store_true",
                    help="Also trim a disclosure prefix from --baseline files")
    ap.add_argument("--out", help="CSV path for per-file results")
    ap.add_argument("--check", action="store_true",
                    help="Report which metrics are available, then exit")
    args = ap.parse_args()

    utmos = Utmos()
    ecapa = Ecapa()
    nisqa = Nisqa(args.nisqa_dir)

    if args.check:
        print("SNR (ITU-T P.56)   available (numpy + scipy, no download)")
        utmos.load()
        print(f"UTMOS              {'available' if utmos.model else 'UNAVAILABLE -- ' + utmos.error}")
        ecapa.load()
        print(f"ECAPA-TDNN         {'available' if ecapa.encoder else 'UNAVAILABLE -- ' + ecapa.error}")
        print(f"NISQA              {'configured at ' + str(args.nisqa_dir) if not nisqa.error else 'UNAVAILABLE -- ' + nisqa.error}")
        return

    if not args.system:
        ap.error("--system is required (or use --check)")

    if args.speaker_refs:
        refs = sorted(glob.glob(os.path.join(args.speaker_refs, "*.wav")))
        print(f"Fitting ECAPA reference from {len(refs)} recording(s)...")
        if not ecapa.fit_reference(refs):
            print("  speaker reference unavailable; ECAPA cosine will be skipped"
                  + (f" ({ecapa.error})" if ecapa.error else ""), file=sys.stderr)
    else:
        print("No --speaker-refs given; ECAPA cosine will be skipped.")

    print(f"\nSystem (TTS + RVC): {args.system}")
    sys_rows = measure_dir("system", args.system, utmos, ecapa, nisqa,
                           trim=not args.no_trim)

    base_rows = []
    if args.baseline:
        print(f"\nBaseline (F5-TTS): {args.baseline}")
        base_rows = measure_dir("baseline", args.baseline, utmos, ecapa, nisqa,
                                trim=args.baseline_trim)

    if utmos.error:
        print(f"\nUTMOS skipped: {utmos.error}", file=sys.stderr)
    if ecapa.error:
        print(f"ECAPA skipped: {ecapa.error}", file=sys.stderr)
    if nisqa.error:
        print(f"NISQA skipped: {nisqa.error}", file=sys.stderr)

    # Files where RVC silently fell back to plain TTS are not RVC measurements.
    no_rvc = [r for r in sys_rows if "NO-RVC" in r["flag"]]
    scored = [r for r in sys_rows if "NO-RVC" not in r["flag"]]
    lower_bound = [r for r in scored if "SNR-LOWER-BOUND" in r["flag"]]

    print("\n" + "=" * 72)
    print(f"System: {len(scored)} RVC file(s) scored" +
          (f", {len(no_rvc)} EXCLUDED with no disclosure prefix (RVC did not run)" if no_rvc else ""))
    if no_rvc:
        print("  excluded: " + ", ".join(r["file"] for r in no_rvc))
        print("  Those responses came back as plain TTS -- check the Colab RVC session"
              " was up for the whole generation pass, then regenerate them.")

    snr_mean = mean_of(scored, "snr_db")
    if snr_mean is not None:
        verdict = "PASS" if snr_mean >= SNR_SCREEN_DB else "FAIL"
        print(f"\nSNR              mean {snr_mean:6.1f} dB   "
              f"(artefact screen >= {SNR_SCREEN_DB:.0f} dB: {verdict})")
        if lower_bound:
            print(f"  note: {len(lower_bound)} file(s) have almost no silence, so their SNR is a"
                  " lower bound: " + ", ".join(r["file"] for r in lower_bound))

    for key, label, fmt in [("utmos", "UTMOS", "6.3f"),
                            ("nisqa", "NISQA", "6.3f"),
                            ("ecapa_cos", "ECAPA cosine", "6.4f")]:
        s = mean_of(scored, key)
        if s is None:
            continue
        b = mean_of(base_rows, key) if base_rows else None
        line = f"{label:16s} mean {format(s, fmt)}"
        if b is not None:
            verdict = "PASS" if s > b else "FAIL"
            line += f"   baseline {format(b, fmt)}   (must exceed baseline: {verdict})"
        else:
            line += "   (no baseline given; relative criterion not evaluated)"
        print(line)

    if not base_rows and (mean_of(scored, "utmos") is not None
                          or mean_of(scored, "ecapa_cos") is not None):
        print("\nTable 8 states UTMOS and ECAPA cosine as relative criteria, so RQ2 needs"
              "\n--baseline pointing at the F5-TTS-Vietnamese-ViVoice output for the same texts.")

    if args.out:
        fields = ["file", "system", "duration_s", "trimmed_s", "flag", "snr_db",
                  "activity_pct", "utmos", "nisqa", "ecapa_cos"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sys_rows + base_rows)
        print(f"\nPer-file results written to {args.out}")


if __name__ == "__main__":
    main()
