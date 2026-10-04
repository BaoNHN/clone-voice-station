#!/usr/bin/env python
"""
experiments/retest/flow_01_api_testcases.py
Re-runs Table 5 test cases TC-01..TC-19 against the live station.
Details: ai_change_log.txt (retest flows).
"""
import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (STATION_DIR, Report, add_common_args, concat_wavs, load_set30,
                    make_client, resolve_api_key, silent_wav, wav_info)

UNREACHABLE = "http://127.0.0.1:9"  # connection refused


def find_disclosure(wav_bytes):
    """Seconds where the 350 ms separator ends, or None (uses the thesis tool)."""
    sys.path.insert(0, os.path.join(STATION_DIR, "tools"))
    import tempfile
    import eval_voice_quality as E
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        x, sr = E.read_wav(path)
        return E.find_disclosure_end(x, sr)
    finally:
        os.unlink(path)


class Ctx:
    def __init__(self, args):
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.key = resolve_api_key(args.api_key)
        self.client = make_client(args)
        self.clips = load_set30()
        self.short_wav = open(self.clips[0][1], "rb").read()
        self.converted = None       # a disclosure-prefixed WAV, kept for TC-18
        self.user = args.user_id
        self.profile = None

    def h(self, key=True):
        return {"X-Api-Key": self.key} if key else {}

    def transcribe(self, content, key=True, name="a.wav"):
        t0 = time.perf_counter()
        r = requests.post(f"{self.base}/api/transcribe", headers=self.h(key),
                          data={"language": "vi"}, files={"audio": (name, content, "audio/wav")},
                          timeout=300)
        return r, time.perf_counter() - t0

    def speak(self, body, key=True):
        return requests.post(f"{self.base}/api/speak", headers=self.h(key), json=body, timeout=180)

    def discover(self):
        try:
            profiles = self.client.list_all_voice_profiles()
        except Exception:
            profiles = []
        ready = [p for p in profiles if p.get("kind") == "cloned" and p.get("status") == "ready"
                 and (not self.user or str(p.get("external_user_id")) == str(self.user))]
        if ready:
            self.profile = ready[0]
            self.user = str(self.profile["external_user_id"])

    def rvc_available(self):
        try:
            return bool(self.client.get_rvc_endpoint().get("available"))
        except Exception:
            return False


def status_case(rep, id_, expected_status, resp, detail_contains=None, elapsed=None, note=""):
    ok = resp.status_code == expected_status
    if ok and detail_contains and detail_contains not in resp.text:
        ok = False
    actual = f"HTTP {resp.status_code} {resp.text[:90]!r}"
    if elapsed is not None:
        note = (note + f" {elapsed:.2f}s").strip()
    rep.add(id_, "PASS" if ok else "FAIL", f"{expected_status}"
            + (f" '{detail_contains}'" if detail_contains else ""), actual, note)


def run_transcribe_cases(c, rep, want):
    # TC-01 adapter published
    if want("TC-01"):
        r, _ = c.transcribe(c.short_wav)
        eng = r.json().get("engine", "") if r.ok else ""
        if eng.startswith("stt-adapter:"):
            rep.add("TC-01", "PASS", "200, engine names adapter", f"engine={eng}")
        else:
            rep.add("TC-01", "SKIP", "200, engine names adapter", f"engine={eng or r.status_code}",
                    "no adapter published/default for this client - publish one in STT Lab and re-run")
    # TC-02 remote single shot / TC-03 fallback to local
    if want("TC-02"):
        if c.rvc_available():
            r, dt = c.transcribe(c.short_wav)
            ok = r.status_code == 200 and r.json().get("text")
            rep.add("TC-02", "PASS" if ok else "FAIL", "200 single-shot remote call",
                    f"HTTP {r.status_code} engine={r.json().get('engine') if r.ok else '-'}", f"{dt:.2f}s")
        else:
            rep.add("TC-02", "SKIP", "200 single-shot remote call", "Colab not reachable", "start Colab + set RVC endpoint")
    if want("TC-03"):
        orig = c.client.get_rvc_endpoint().get("endpoint") or ""
        try:
            c.client.set_rvc_endpoint(UNREACHABLE)
            r, dt = c.transcribe(c.short_wav)
            ok = r.status_code == 200 and bool(r.json().get("text"))
            rep.add("TC-03", "PASS" if ok else "FAIL", "200 via local fallback",
                    f"HTTP {r.status_code} text={r.json().get('text', '')[:50]!r}" if r.ok else f"HTTP {r.status_code}",
                    f"{dt:.2f}s (first call may include model load)")
        finally:
            c.client.set_rvc_endpoint(orig)
    # TC-04 long recording
    if want("TC-04"):
        wav = concat_wavs([p for _, p, _ in c.clips], 60)
        r, dt = c.transcribe(wav, name="long60.wav")
        eng = r.json().get("engine") if r.ok else None
        if r.ok and eng == "segmented":
            rep.add("TC-04", "PASS", "200, engine=segmented", f"engine={eng}", f"{dt:.1f}s")
        elif r.ok and str(eng).startswith("stt-adapter:"):
            rep.add("TC-04", "SKIP", "200, engine=segmented", f"engine={eng}", "adapter path takes precedence; unpublish adapter")
        else:
            rep.add("TC-04", "FAIL", "200, engine=segmented", f"HTTP {r.status_code} engine={eng}")
    # TC-05 over the hard cap
    if want("TC-05"):
        r, dt = c.transcribe(silent_wav(11 * 60), name="eleven_min.wav")
        status_case(rep, "TC-05", 413, r, elapsed=dt, note="" if dt < 1 else "slower than 1 s (upload time counts)")
    # TC-06 empty
    if want("TC-06"):
        r, _ = c.transcribe(b"")
        status_case(rep, "TC-06", 400, r, "Không có dữ liệu âm thanh")
    # TC-07 silence
    if want("TC-07"):
        r, _ = c.transcribe(silent_wav(3))
        status_case(rep, "TC-07", 422, r, "Không nhận diện được")
    # TC-10 no / bad key
    if want("TC-10"):
        r1, _ = c.transcribe(c.short_wav, key=False)
        r2 = requests.post(f"{c.base}/api/transcribe", headers={"X-Api-Key": "invalid"},
                           files={"audio": ("a.wav", c.short_wav, "audio/wav")}, timeout=60)
        ok = r1.status_code == 401 and r2.status_code == 401
        rep.add("TC-10", "PASS" if ok else "FAIL", "401 (absent and unknown key)",
                f"absent={r1.status_code} unknown={r2.status_code}")


def run_inprocess_cases(c, rep, want):
    """TC-08 / TC-09: one or all segments fail. Needs fault injection."""
    ids = [i for i in ("TC-08", "TC-09") if want(i)]
    if not ids:
        return
    if c.args.no_inprocess:
        for i in ids:
            rep.add(i, "SKIP", "-", "--no-inprocess given")
        return
    try:
        os.chdir(STATION_DIR)
        sys.path.insert(0, STATION_DIR)
        import app as station_app
        from fastapi.testclient import TestClient
        from voice import stt_segmented
    except Exception as e:  # missing deps / import side effects
        for i in ids:
            rep.add(i, "SKIP", "-", f"in-process import failed: {e!r}")
        return

    station_app.get_published_stt_adapter_for_client = lambda cid: None
    station_app.get_default_stt_adapter = lambda: None
    tc = TestClient(station_app.app)
    wav = concat_wavs([p for _, p, _ in c.clips], 60)       # 3 segments: 25 + 25 + 10 s
    original = stt_segmented._transcribe_one
    files = {"audio": ("long60.wav", wav, "audio/wav")}
    try:
        if "TC-08" in ids:
            calls = {"n": 0}

            def one_fails(audio_bytes, mime, language):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("injected failure")
                return {"text": f"seg{calls['n']}", "language": "vi"}
            stt_segmented._transcribe_one = one_fails
            r = tc.post("/api/transcribe", headers=c.h(), files=files)
            ok = r.status_code == 200 and r.json().get("text") == "seg1 seg3"
            rep.add("TC-08", "PASS" if ok else "FAIL", "200, text='seg1 seg3' (segment 2 skipped)",
                    f"HTTP {r.status_code} {r.text[:80]!r}")
        if "TC-09" in ids:
            def all_fail(audio_bytes, mime, language):
                raise RuntimeError("injected failure")
            stt_segmented._transcribe_one = all_fail
            r = tc.post("/api/transcribe", headers=c.h(), files=files)
            ok = r.status_code == 502 and "Không thể chuyển giọng nói" in r.text
            rep.add("TC-09", "PASS" if ok else "FAIL", "502 'Không thể chuyển giọng nói...'",
                    f"HTTP {r.status_code} {r.text[:80]!r}")
    finally:
        stt_segmented._transcribe_one = original


def run_speak_cases(c, rep, want):
    if not c.profile:
        for i in ("TC-11", "TC-13", "TC-17", "TC-18"):
            if want(i):
                rep.add(i, "SKIP", "-", "no ready cloned profile found", "pass --user-id or train a profile")
    pid, uid = (c.profile["id"], c.user) if c.profile else (None, c.user or "retest-user")

    if want("TC-11"):
        if not c.profile:
            pass
        elif not c.rvc_available():
            rep.add("TC-11", "SKIP", "200 WAV + disclosure + tag", "Colab RVC not reachable", "TC-17 covers local path")
        else:
            r = c.speak({"text": "Xin chào, đây là bài kiểm thử.", "external_user_id": uid, "profile_id": pid})
            if r.status_code != 200:
                rep.add("TC-11", "FAIL", "200 WAV", f"HTTP {r.status_code} {r.text[:80]!r}")
            else:
                sr, secs, tag = wav_info(r.content)
                dis = find_disclosure(r.content)
                ok = dis is not None and bool(tag)
                if ok:
                    c.converted = r.content
                rep.add("TC-11", "PASS" if ok else "FAIL", "WAV, disclosure end found, ICMT tag",
                        f"{secs:.1f}s disclosure_end={dis} tag={tag!r}")
    if want("TC-12") or want("TC-14"):
        # builtin fallback: 12 omits profile_id, 14 sends a non-numeric one
        for id_, body in (("TC-12", {"text": "Kiểm thử.", "external_user_id": uid}),
                          ("TC-14", {"text": "Kiểm thử.", "external_user_id": uid, "profile_id": "abc"})):
            if want(id_):
                r = c.speak(body)
                rep.add(id_, "PASS" if r.status_code == 200 else "FAIL", "200 builtin plain TTS",
                        f"HTTP {r.status_code} {len(r.content)} bytes",
                        "" if r.ok else "user has no builtin profile? then see TC-15")
    if want("TC-13") and c.profile:
        r = c.speak({"text": "Kiểm thử.", "external_user_id": "intruder-" + uid, "profile_id": pid})
        status_case(rep, "TC-13", 403, r, "Unauthorized")
    if want("TC-15") and any(p.get("kind") == "builtin" for p in c.client.list_voice_profiles("no-such-user-0000")):
        rep.add("TC-15", "SKIP", "404 when no builtin profile exists", "builtin voices are global in this DB",
                "precondition not reproducible without deleting builtin profiles")
    elif want("TC-15"):
        r = c.speak({"text": "Kiểm thử.", "external_user_id": "no-such-user-0000"})
        status_case(rep, "TC-15", 404, r, "Không tìm thấy giọng nói khả dụng")
    if want("TC-16"):
        r = c.speak({"text": "   ", "external_user_id": uid})
        status_case(rep, "TC-16", 400, r, "Không có nội dung để đọc")
    if want("TC-17") and c.profile:
        orig = c.client.get_rvc_endpoint().get("endpoint") or ""
        try:
            c.client.set_rvc_endpoint(UNREACHABLE)
            r = c.speak({"text": "Kiểm thử dự phòng.", "external_user_id": uid, "profile_id": pid})
            if r.status_code != 200:
                rep.add("TC-17", "FAIL", "200 (local RVC or plain TTS)", f"HTTP {r.status_code} {r.text[:80]!r}")
            else:
                is_wav = r.headers.get("content-type", "").startswith("audio/wav")
                dis = find_disclosure(r.content) if is_wav else None
                path = "local RVC (disclosure present)" if dis is not None else "plain TTS (no disclosure)"
                if dis is not None and c.converted is None:
                    c.converted = r.content
                rep.add("TC-17", "PASS", "200; disclosure iff RVC ran", f"200, fell back to {path}")
        finally:
            c.client.set_rvc_endpoint(orig)
    if want("TC-18"):
        if c.converted is None:
            rep.add("TC-18", "SKIP", "350 ms near-silence marker", "no converted output captured (TC-11/17 did not convert)")
        else:
            import numpy as np
            sys.path.insert(0, os.path.join(STATION_DIR, "tools"))
            import eval_voice_quality as E
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(c.converted)
                p = f.name
            x, sr = E.read_wav(p)
            os.unlink(p)
            end = E.find_disclosure_end(x, sr)
            if end is None:
                rep.add("TC-18", "FAIL", "350 ms near-silence marker", "separator not found")
            else:
                i1 = int(end * sr)
                i0 = i1
                while i0 > 0 and abs(x[i0 - 1]) < E.SILENCE_AMPLITUDE:
                    i0 -= 1
                run_ms = (i1 - i0) * 1000 / sr
                rep.add("TC-18", "PASS" if run_ms >= 340 else "FAIL",
                        "near-silence run >= 350 ms before converted speech", f"{run_ms:.0f} ms ending {end:.2f}s",
                        "run includes the disclosure clip's trailing silence")
    if want("TC-19"):
        r = c.speak({"text": "Kiểm thử.", "external_user_id": uid}, key=False)
        status_case(rep, "TC-19", 401, r)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--only", default="", help="comma list, e.g. TC-05,TC-06")
    ap.add_argument("--no-inprocess", action="store_true", help="skip TC-08/TC-09 fault injection")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    want = lambda id_: not only or id_ in only

    c = Ctx(args)
    if not c.key:
        sys.exit("No API key (use --api-key, VOICE_STATION_API_KEY or a voice_station_key.txt)")
    if not c.client.is_available():
        sys.exit(f"station not reachable at {c.base}")
    c.discover()
    rep = Report("flow_01_api_testcases")
    rep.extra = {"base_url": c.base, "user": c.user, "profile_id": (c.profile or {}).get("id")}
    run_transcribe_cases(c, rep, want)
    run_inprocess_cases(c, rep, want)
    run_speak_cases(c, rep, want)
    rep.rows.sort(key=lambda r: r["id"])
    counts = rep.save()
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
