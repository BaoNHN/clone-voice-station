#!/usr/bin/env python
"""
tools/split_recording.py
Cuts one long recording, in which a speaker read a numbered list of sentences
with a pause between each, into one audio file per sentence plus a matching
.txt transcript sidecar.

That output shape is deliberately the one two other tools already expect:

    001.wav  001.txt      <- tools/eval_stt_wer.py test-set convention
    002.wav  002.txt
    ...

So a single reading session produces both of the recorded inputs the evaluation
needs (Section 6.5 of the thesis):

  1. A legal-domain WER test set. tools/eval_stt_wer.py scores the transcripts
     against these files directly. The thesis currently reports WER from one
     medical-domain pilot utterance only, which it flags as not standing in for
     the formal legal-domain evaluation.

  2. Speaker references for the ECAPA cosine similarity in
     tools/eval_voice_quality.py --speaker-refs. Those references must be real
     recordings of the target speaker that were NOT part of that speaker's RVC
     training data. voice/scripts.py is deliberately general-topic (weather,
     food, travel, family), so legal sentences share no text with it.

There are two ways to find the boundaries, and --align-asr is the one to use.

  default (pause length)  Cuts at the N-1 longest pauses. Works only when pauses
      between sentences are clearly longer than pauses inside them. On a natural
      reading they usually are not: in the recording this was developed against,
      the shortest pause kept as a boundary was 0.60s and the longest one rejected
      was 0.57s, a separation of 0.03s, which is a coin flip rather than a
      decision. The script measures that separation and refuses to write files
      when it is too small, instead of handing back confidently mislabelled clips.

  --align-asr (content)   Over-segments at every plausible pause, transcribes each
      span, then assigns consecutive spans to the known sentences by maximising
      word-level similarity. Because the sentence texts are already known, the
      boundaries can be recovered from what was said rather than from how long the
      speaker paused, which removes the ambiguity entirely. On that same recording
      it aligned all 30 sentences at 0.96 mean similarity.

      Transcription goes to the endpoint in colab/voice_server.ipynb, which serves
      PhoWhisper (Vietnamese-tuned). Alignment quality depends on Vietnamese
      recognition quality, so do not point this at a stock multilingual Whisper.

A spoken separator between sentences ("tách câu", "hết câu") is supported through
--separator, but only together with --align-asr, and it is worth understanding why
the combination matters. The marker is speech that exists in the audio and not in
the transcript, so any part of it left inside a clip is scored as inserted words:
a two-word marker on a fifteen-word sentence is about 13% WER of pure artefact,
the same order as the 15% threshold being measured. --separator removes that risk
by transcribing every span and discarding the ones that are the marker, so the
marker audio never reaches a clip. It cannot rescue a marker spoken without a
pause around it, because a span containing both the marker and the sentence has no
timing inside it to cut on; those spans are reported as a warning instead.

Choosing a marker: two or three syllables, clearly separated by a real pause on
both sides, and not a phrase that opens any sentence in the list. Matching is done
on characters rather than words so that one misheard tone still identifies it.

Note that a marker is a convenience for the reader, not a requirement: --align-asr
alone recovered all 30 boundaries from a natural reading whose sentence pauses
were no longer than its phrase pauses.

Usage:
    # look at the segmentation before writing anything
    python tools/split_recording.py recording.m4a \
        --sentences tools/legal_testset/questions_vi.txt --dry-run

    # write the test set
    python tools/split_recording.py recording.m4a \
        --sentences tools/legal_testset/questions_vi.txt --out legal_testset/

If the segment count does not match the sentence count, the report says which
way it went and --min-silence is the knob: raise it when one sentence was split
in two (a mid-sentence breath was read as a gap), lower it when two sentences
were merged (the pause between them was too short).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

# This script prints the Vietnamese sentences it matched to each segment, and a
# Windows console defaults to cp1252, which cannot encode them: without this the
# report dies on the first accented character. errors="replace" keeps a console
# stuck on a legacy code page readable instead of aborting.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

# Frame analysis, matching the SNR estimator in tools/eval_voice_quality.py.
FRAME_MS = 20
HOP_MS = 10
NOISE_PERCENTILE = 10
# Speech is this far above the estimated noise floor. 8 dB clears room tone and
# breath without cutting into the quiet tail of a sentence.
SPEECH_MARGIN_DB = 8.0
# Padding kept around each segment so a clip never starts on a clipped consonant.
PAD_S = 0.15

# The conda-forge ffmpeg in this machine's shared environment fails to launch
# (a DLL conflict), which is why the repository bundles a working static build;
# voice/rvc_local.py and voice/stt.py apply the same fix.
BUNDLED_FFMPEG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "ffmpeg.exe")


def decode_to_wav(path):
    """Returns a path to a mono PCM WAV. Non-WAV input goes through ffmpeg."""
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() == 2:
                return path, None
    except (wave.Error, EOFError):
        pass

    ffmpeg = BUNDLED_FFMPEG if os.path.isfile(BUNDLED_FFMPEG) else "ffmpeg"
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [ffmpeg, "-y", "-i", path, "-ac", "1", "-c:a", "pcm_s16le", tmp.name]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(f"Could not find ffmpeg (looked for {BUNDLED_FFMPEG}). Convert the "
              f"recording to WAV yourself and pass that instead.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        print("ffmpeg could not decode the recording:\n  " + "\n  ".join(tail), file=sys.stderr)
        sys.exit(1)
    return tmp.name, tmp.name


def read_wav(path):
    with wave.open(path, "rb") as w:
        channels, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        print(f"Expected 16-bit PCM, got {width * 8}-bit.", file=sys.stderr)
        sys.exit(1)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def write_wav(path, x, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def frame_rms(x, sr):
    frame, hop = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    if len(x) < frame:
        return np.array([]), hop
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((x[idx] ** 2).mean(axis=1)), hop


def voiced_mask(x, sr, margin_db):
    rms, hop = frame_rms(x, sr)
    if rms.size == 0:
        return None, None, None
    floor = max(float(np.percentile(rms, NOISE_PERCENTILE)), 1e-6)
    return rms > floor * (10.0 ** (margin_db / 20.0)), hop, floor


def interior_gaps(voiced, min_gap_s=0.15):
    """Silence runs strictly between the first and last speech frame.

    Returns [(duration_s, start_frame, end_frame)], longest first.
    """
    if not voiced.any():
        return [], 0, 0
    first = int(np.argmax(voiced))
    last = len(voiced) - int(np.argmax(voiced[::-1]))
    gaps, run = [], None
    for i in range(first, last):
        if not voiced[i]:
            if run is None:
                run = i
        elif run is not None:
            dur = (i - run) * HOP_MS / 1000.0
            if dur >= min_gap_s:
                gaps.append((dur, run, i))
            run = None
    gaps.sort(key=lambda g: -g[0])
    return gaps, first, last


def segment_by_count(x, sr, n_expected, margin_db):
    """Cuts into exactly n_expected pieces using the n_expected-1 longest pauses.

    A fixed silence threshold cannot separate sentence boundaries from the pauses
    a reader takes between phrases when the two overlap in length, and picking one
    threshold then hoping the count lands right hides that. Using the known
    sentence count instead makes the ambiguity measurable: `separation` below is
    the distance between the shortest pause accepted as a boundary and the longest
    one rejected. When that is small, the cut points are a guess, and the caller
    is told so rather than handed 30 confidently mislabelled files.

    Returns (segments, separation_s, chosen_min_s).
    """
    voiced, hop, _ = voiced_mask(x, sr, margin_db)
    if voiced is None:
        return [], 0.0, 0.0
    gaps, first, last = interior_gaps(voiced)
    if len(gaps) < n_expected - 1:
        return [], 0.0, 0.0

    chosen = sorted(gaps[:n_expected - 1], key=lambda g: g[1])
    rejected = gaps[n_expected - 1:]
    chosen_min = min(g[0] for g in chosen)
    separation = chosen_min - (rejected[0][0] if rejected else 0.0)

    cuts = [first] + [(a + b) // 2 for _, a, b in chosen] + [last]
    pad = int(PAD_S * sr)
    segs = []
    for i in range(len(cuts) - 1):
        s = cuts[i] * hop
        e = cuts[i + 1] * hop + int(sr * FRAME_MS / 1000)
        segs.append((max(0, s - pad), min(len(x), e + pad)))
    return segs, separation, chosen_min


def _norm_words(text):
    """Lowercased, punctuation-free word list. Matches tools/eval_stt_wer.py."""
    import re
    return re.sub(r"[^\w\s]", "", text.lower().strip(), flags=re.UNICODE).split()


def _word_similarity(a, b):
    """1.0 for identical word sequences, 0.0 for nothing in common."""
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return max(0.0, 1.0 - prev[m] / max(n, m))


def _char_similarity(a, b):
    """Character-level version of _word_similarity, for matching the separator.

    Word-level matching is too brittle on a two-word marker: one misheard syllable
    ("tach cau" for "tach cau" with a different tone) halves the score and the
    marker survives into a clip, which is exactly the failure the marker was meant
    to avoid. Comparing characters degrades gracefully instead.
    """
    a, b = "".join(a), "".join(b)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return max(0.0, 1.0 - prev[m] / max(n, m))


def candidate_chunks(x, sr, margin_db, min_gap_s=0.25):
    """Speech spans between every pause long enough to *possibly* end a sentence.

    Deliberately over-segments: the ASR alignment below decides which of these
    boundaries are real, so missing a true boundary here is unrecoverable while
    an extra one costs nothing.
    """
    voiced, hop, _ = voiced_mask(x, sr, margin_db)
    if voiced is None:
        return []
    gaps, first, last = interior_gaps(voiced, min_gap_s)
    cuts = [first] + sorted((a + b) // 2 for _, a, b in gaps) + [last]
    return [(cuts[i] * hop, cuts[i + 1] * hop) for i in range(len(cuts) - 1)]


def transcribe_span(endpoint, x, sr, a, b, language="vi", timeout=180):
    import io as _io
    import requests
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x[a:b], -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    resp = requests.post(f"{endpoint.rstrip('/')}/transcribe",
                         files={"audio": ("chunk.wav", buf.getvalue(), "audio/wav")},
                         data={"language": language}, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()


def align_chunks(chunk_words, sentences):
    """Monotonic alignment of consecutive chunks onto the known sentences.

    Silence length alone cannot tell a pause between sentences from a pause
    inside one when the two overlap in duration. The sentence texts are known,
    though, so the boundaries can be recovered from *content* instead: assign
    consecutive runs of chunks to sentences in order, maximising the word-level
    similarity between each run's transcript and its sentence. Every chunk is
    used and order is preserved, which is what makes this a forced alignment
    rather than a search.

    Returns [(first_chunk, last_chunk)] per sentence, and the mean similarity.
    """
    M, N = len(chunk_words), len(sentences)
    if M < N:
        return None, 0.0
    sent_words = [_norm_words(s) for s in sentences]

    NEG = float("-inf")
    dp = [[NEG] * (N + 1) for _ in range(M + 1)]
    back = [[0] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0
    for j in range(1, N + 1):
        for i in range(j, M - (N - j) + 1):
            best, best_k = NEG, j - 1
            for k in range(j - 1, i):
                if dp[k][j - 1] == NEG:
                    continue
                run = []
                for t in range(k, i):
                    run.extend(chunk_words[t])
                score = dp[k][j - 1] + _word_similarity(run, sent_words[j - 1])
                if score > best:
                    best, best_k = score, k
            dp[i][j], back[i][j] = best, best_k

    if dp[M][N] == NEG:
        return None, 0.0
    spans, i = [], M
    for j in range(N, 0, -1):
        k = back[i][j]
        spans.append((k, i - 1))
        i = k
    spans.reverse()
    return spans, dp[M][N] / N


def segment_chunks(x, sr, margin_db, chunk_s, min_speech_s):
    """Continuous speech cut into fixed-length chunks, silence dropped.

    For speaker references the text is irrelevant -- ECAPA embeds timbre, not
    words -- so boundaries do not have to fall between sentences. This path
    therefore works on a recording whose pauses are too short to segment by
    sentence, which is exactly when it is needed.
    """
    voiced, hop, _ = voiced_mask(x, sr, margin_db)
    if voiced is None:
        return []
    keep = np.repeat(voiced, hop)[:len(x)]
    speech = x[keep[:len(x)]] if keep.size else x
    n = int(chunk_s * sr)
    out = []
    for i in range(0, len(speech) - int(min_speech_s * sr) + 1, n):
        out.append(speech[i:i + n])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Split one long reading into per-sentence clips plus transcripts")
    ap.add_argument("recording", help="The single long recording (wav, mp3, m4a, ...)")
    ap.add_argument("--mode", choices=["sentences", "refs"], default="sentences",
                    help="'sentences' writes NNN.wav + NNN.txt aligned to --sentences "
                         "(WER test set). 'refs' writes fixed-length speaker-reference "
                         "clips only, which needs no sentence alignment at all.")
    ap.add_argument("--sentences",
                    help="UTF-8 file, one sentence per line, in the order they were read "
                         "(required for --mode sentences)")
    ap.add_argument("--out", help="Output directory (required unless --dry-run)")
    ap.add_argument("--margin-db", type=float, default=SPEECH_MARGIN_DB,
                    help=f"How far above the noise floor counts as speech "
                         f"(default {SPEECH_MARGIN_DB:g})")
    ap.add_argument("--chunk", type=float, default=8.0,
                    help="Clip length in seconds for --mode refs (default 8)")
    ap.add_argument("--min-speech", type=float, default=0.6,
                    help="Ignore anything shorter than this (default 0.6)")
    ap.add_argument("--align-asr", action="store_true",
                    help="Find the boundaries by transcribing and matching against "
                         "--sentences instead of by pause length. Needed whenever "
                         "pauses between sentences are no longer than pauses inside "
                         "them. Requires a reachable ASR endpoint.")
    ap.add_argument("--asr-endpoint",
                    help="ASR base URL for --align-asr (default: the station's stored "
                         "rvc_endpoint)")
    ap.add_argument("--separator",
                    help="A short phrase spoken between sentences, e.g. \"tach cau\". "
                         "Spans whose transcript is that phrase are dropped before "
                         "alignment, so the marker never lands inside a clip. Requires "
                         "--align-asr: without transcription the marker cannot be "
                         "located, and it would be scored as speech the transcript does "
                         "not contain.")
    ap.add_argument("--force", action="store_true",
                    help="Write the sentence split even when the pause separation is "
                         "too small to trust the boundaries")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the segmentation without writing files")
    args = ap.parse_args()

    wav_path, tmp = decode_to_wav(args.recording)
    x, sr = read_wav(wav_path)
    if tmp and os.path.exists(tmp):
        os.unlink(tmp)  # samples are in memory now; the decode scratch file is done

    print(f"Recording : {args.recording}")
    print(f"Duration  : {len(x)/sr:.1f}s at {sr} Hz")

    # ---------------------------------------------------------------- refs mode
    if args.mode == "refs":
        chunks = segment_chunks(x, sr, args.margin_db, args.chunk, args.min_speech)
        speech_s = sum(len(c) for c in chunks) / sr
        print(f"Mode      : speaker references, {args.chunk:g}s per clip")
        print(f"Produced  : {len(chunks)} clip(s), {speech_s:.1f}s of speech "
              f"(silence removed)\n")
        if not chunks:
            print("No speech found. Try lowering --margin-db.", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print("Re-run without --dry-run and with --out to write the clips.")
            return
        if not args.out:
            print("--out is required when not doing a dry run.", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.out, exist_ok=True)
        for i, c in enumerate(chunks, start=1):
            write_wav(os.path.join(args.out, f"ref{i:03d}.wav"), c, sr)
        print(f"Wrote {len(chunks)} clip(s) to {args.out}.")
        print("\nNo transcripts are written: these are speaker references, not a WER"
              "\ntest set. Use them with:")
        print(f"  python tools/eval_voice_quality.py --system rvc_out/ "
              f"--speaker-refs {args.out}")
        print("\nThese must be the target speaker's real voice and must not be audio"
              "\nthat the RVC model was trained on.")
        return

    # ----------------------------------------------------------- sentences mode
    if not args.sentences:
        ap.error("--sentences is required for --mode sentences")
    with open(args.sentences, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    if not sentences:
        print(f"No sentences in {args.sentences}", file=sys.stderr)
        sys.exit(1)

    print(f"Sentences : {len(sentences)}")

    if args.align_asr:
        endpoint = args.asr_endpoint
        if not endpoint:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from database.database import get_setting
                endpoint = get_setting("rvc_endpoint")
            except Exception:
                endpoint = None
        if not endpoint:
            print("No ASR endpoint. Pass --asr-endpoint, or set the Colab endpoint in "
                  "the manager dashboard.", file=sys.stderr)
            sys.exit(1)

        cands = candidate_chunks(x, sr, args.margin_db)
        print(f"Mode      : ASR alignment against {endpoint}")
        print(f"Candidates: {len(cands)} speech span(s) to transcribe\n")
        if len(cands) < len(sentences):
            print(f"Only {len(cands)} spans for {len(sentences)} sentences: some "
                  "sentences have no pause\nbetween them at all. Re-record with clearer "
                  "pauses.", file=sys.stderr)
            sys.exit(1)

        sep_words = _norm_words(args.separator) if args.separator else None
        chunk_words, kept, dropped, impure = [], [], 0, []
        for i, (a, b) in enumerate(cands, start=1):
            try:
                text = transcribe_span(endpoint, x, sr, a, b)
            except Exception as e:
                print(f"\nASR failed on span {i}: {type(e).__name__}: {e}", file=sys.stderr)
                sys.exit(1)
            words = _norm_words(text)
            tag = ""
            if sep_words:
                # A span that is only the marker can be discarded outright. A span
                # that merely contains it was not surrounded by pauses, so removing
                # the marker would need sub-span timing the ASR does not give us.
                if words and _char_similarity(words, sep_words) >= 0.75:
                    dropped += 1
                    tag = "  [separator, dropped]"
                elif len(words) > len(sep_words) and _char_similarity(
                        words[:len(sep_words)], sep_words) >= 0.8:
                    impure.append(i)
                    tag = "  [separator ran into the sentence]"
            print(f"  span {i:03d}/{len(cands)}  {(b-a)/sr:5.2f}s  {text[:56]}{tag}")
            if tag.startswith("  [separator, dropped]"):
                continue
            chunk_words.append(words)
            kept.append((a, b))

        if sep_words:
            print(f"\nSeparator : dropped {dropped} marker span(s)")
            if impure:
                print(f"  WARNING: on span(s) {impure} the marker ran straight into the "
                      "sentence with no\n  pause between them, so it stays in the audio "
                      "and will count as inserted words.\n  Leave a clear pause on BOTH "
                      "sides of the marker when recording.")
            if dropped and dropped < len(sentences) - 1:
                print(f"  Note: {dropped} marker(s) found but {len(sentences)-1} sentence "
                      "boundaries are needed;\n  the alignment still has to infer the rest.")
            cands = kept

        spans, mean_sim = align_chunks(chunk_words, sentences)
        if spans is None:
            print("\nAlignment failed.", file=sys.stderr)
            sys.exit(1)

        pad = int(PAD_S * sr)
        segs = []
        for first_c, last_c in spans:
            a = max(0, cands[first_c][0] - pad)
            b = min(len(x), cands[last_c][1] + pad)
            segs.append((a, b))

        durations = [(b - a) / sr for a, b in segs]
        print(f"\nAlignment : mean word similarity {mean_sim:.2f} "
              f"(1.00 = transcript matches the sentence exactly)\n")
        for i, (a, b) in enumerate(segs, start=1):
            n_chunks = spans[i-1][1] - spans[i-1][0] + 1
            print(f"  {i:03d}  {a/sr:7.2f}s  {durations[i-1]:5.2f}s  "
                  f"[{n_chunks} span]  {sentences[i-1][:48]}")

        # Similarity is the honest check here: PhoWhisper mis-hearing a word costs a
        # little, but a boundary in the wrong place costs a lot, and only the latter
        # drags the mean down hard.
        if mean_sim < 0.55:
            print(f"\nALIGNMENT WEAK (mean similarity {mean_sim:.2f}). The transcripts do "
                  "not match the\nsentence list well enough to trust these cuts. Check "
                  "that the recording really\ncontains all these sentences in this order.")
            if not args.force:
                sys.exit(1)
        else:
            print(f"\nAlignment looks sound. Still listen to two or three clips before "
                  "trusting the\nnumbers they produce.")

        if args.dry_run:
            print("\nRe-run without --dry-run and with --out to write the files.")
            return
        if not args.out:
            print("\n--out is required when not doing a dry run.", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.out, exist_ok=True)
        for i, ((a, b), text) in enumerate(zip(segs, sentences), start=1):
            stem = os.path.join(args.out, f"{i:03d}")
            write_wav(stem + ".wav", x[a:b], sr)
            with open(stem + ".txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")
        print(f"\nWrote {len(segs)} clip(s) and transcript(s) to {args.out} "
              f"({sum(durations):.1f}s of speech).")
        return

    segs, separation, chosen_min = segment_by_count(x, sr, len(sentences), args.margin_db)
    if not segs:
        print("\nNot enough distinct pauses to cut this many sentences. The reader "
              "\nprobably ran sentences together. Re-record with a clear pause.",
              file=sys.stderr)
        sys.exit(1)

    durations = [(b - a) / sr for a, b in segs]
    med = float(np.median(durations))
    print(f"Boundaries: the {len(sentences)-1} longest pauses, shortest of them "
          f"{chosen_min:.2f}s\n")

    for i, (a, b) in enumerate(segs, start=1):
        flag = "  <-- unusually long, likely two sentences" if durations[i-1] > 1.8 * med else ""
        print(f"  {i:03d}  {a/sr:7.2f}s  {durations[i-1]:5.2f}s  {sentences[i-1][:52]}{flag}")

    outliers = sum(1 for d in durations if d > 1.8 * med)
    print(f"\nPause separation: {separation:.2f}s between the shortest pause kept and "
          f"the longest\n  pause rejected.")

    trustworthy = separation >= 0.15 and outliers == 0
    if not trustworthy:
        print("\nBOUNDARIES NOT RELIABLE.")
        if separation < 0.15:
            print(f"  The gap between a kept pause ({chosen_min:.2f}s) and a rejected one "
                  f"is only {separation:.2f}s,\n  so pauses between sentences are not "
                  "distinguishable from pauses within them.")
        if outliers:
            print(f"  {outliers} segment(s) run over 1.8x the median length, which means "
                  "sentences were merged.")
        print("\n  A WER test set needs each clip to match its transcript exactly, so "
              "writing\n  these would silently corrupt the measurement. Two ways forward:")
        print("    1. Re-record leaving a full 2-second pause between sentences, and none"
              "\n       longer than about half a second inside a sentence.")
        print(f"    2. Use this recording for speaker references instead, which needs no"
              f"\n       alignment at all:  --mode refs --out speaker_refs/")
        print("\n  --force writes the split anyway; only do that after listening to the "
              "clips.")
        if not args.force:
            sys.exit(1)

    if args.dry_run:
        print("\nRe-run without --dry-run and with --out to write the files.")
        return

    if not args.out:
        print("\n--out is required when not doing a dry run.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    for i, ((a, b), text) in enumerate(zip(segs, sentences), start=1):
        stem = os.path.join(args.out, f"{i:03d}")
        write_wav(stem + ".wav", x[a:b], sr)
        with open(stem + ".txt", "w", encoding="utf-8") as f:
            f.write(text + "\n")

    total = sum((b - a) for a, b in segs) / sr
    print(f"\nWrote {len(segs)} clip(s) and transcript(s) to {args.out} "
          f"({total:.1f}s of speech).")
    print("\nThat directory is now usable as both:")
    print(f"  python tools/eval_stt_wer.py {args.out} --language vi --out wer.csv")
    print(f"  python tools/eval_voice_quality.py --system rvc_out/ --speaker-refs {args.out}")
    print("\nListen to a couple of clips before trusting the numbers: a clip that "
          "starts\nor ends mid-word means --min-silence needs adjusting.")



if __name__ == "__main__":
    main()
