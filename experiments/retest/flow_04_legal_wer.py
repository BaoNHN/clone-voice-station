#!/usr/bin/env python
"""
experiments/retest/flow_04_legal_wer.py
Re-runs legal-domain WER on the 30 questions for remote, base, hotwords and adapter conditions.
Details: ai_change_log.txt (retest flows).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CLONE_VOICE_LOCAL_MODEL", "tiny")   # must precede local_stt import
from common import HERE, PROJECT_DIR, Report, add_common_args, corpus_wer, load_set30, make_client

PACK_DIR = os.path.join(PROJECT_DIR, "voice-lab-example", "stt_pack")
THESIS = {"remote": 1.9, "base": 41.6, "hotwords": 38.2, "medical": 51.5, "vlsp": 40.8}


def find_pack(name_fragment):
    with open(os.path.join(PACK_DIR, "index.json"), encoding="utf-8") as f:
        packs = json.load(f)["packs"]
    for p in packs:
        manifest = os.path.join(p["zip_path"] + "_extracted", "manifest.json")
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as f:
                if name_fragment.lower() in json.load(f).get("name", "").lower():
                    return p["zip_path"]
    return None


def run_condition(name, clips, args, client, local_stt):
    hyps, lat = [], []
    pack = prompt = None
    if name == "hotwords":
        with open(os.path.join(HERE, "hotwords_legal.json"), encoding="utf-8") as f:
            prompt = ", ".join(json.load(f))
    elif name in ("medical", "vlsp"):
        zip_path = (args.medical_pack if name == "medical" else args.vlsp_pack) or \
            find_pack("Medical Consultation (holdout gate" if name == "medical" else "VLSP2020")
        if not zip_path:
            return None, f"adapter pack for '{name}' not found in {PACK_DIR}"
        pack = local_stt.load_pack(zip_path)
    for stem, wav_path, ref in clips:
        with open(wav_path, "rb") as f:
            data = f.read()
        t0 = time.perf_counter()
        if name == "remote":
            text = client.transcribe(os.path.basename(wav_path), data, mime="audio/wav").get("text", "")
        elif pack:
            text = local_stt.transcribe_with_lora(data, pack["base_model"], pack["adapter_dir"],
                                                   mime="audio/wav", language="vi")["text"]
        else:
            text = local_stt.transcribe(data, mime="audio/wav", language="vi", initial_prompt=prompt)["text"]
        lat.append(time.perf_counter() - t0)
        hyps.append((ref, text))
        print(f"  {name} {stem}: {text[:70]}", flush=True)
    return (hyps, lat), None


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--conditions", default="remote,base,hotwords,medical,vlsp")
    ap.add_argument("--medical-pack", default=None)
    ap.add_argument("--vlsp-pack", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N clips (smoke test)")
    args = ap.parse_args()

    clips = load_set30(args.limit)
    rep = Report("flow_04_legal_wer")
    local_stt = None
    all_hyps = {}
    for name in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        print(f"== {name}", flush=True)
        if name != "remote" and local_stt is None:
            try:
                from clone_voice_client import local_stt as mod
                local_stt = mod
            except Exception as e:
                rep.add(name, "SKIP", "-", f"local STT unavailable: {e!r}", "pip install clone-voice-client[local]")
                continue
        client = make_client(args) if name == "remote" else None
        if client and not client.is_available():
            rep.add(name, "SKIP", "-", f"station not reachable at {client.base_url}")
            continue
        try:
            out, err = run_condition(name, clips, args, client, local_stt)
        except Exception as e:
            rep.add(name, "FAIL", f"thesis {THESIS[name]}%", repr(e))
            continue
        if err:
            rep.add(name, "SKIP", "-", err)
            continue
        hyps, lat = out
        w = corpus_wer(hyps)
        all_hyps[name] = [h for _, h in hyps]
        rep.add(name, "INFO", f"thesis {THESIS[name]}%",
                f"WER {w['corpus_wer_pct']}% ({w['total_errors']}/{w['total_ref_words']} words; "
                f"S{w['substitutions']} D{w['deletions']} I{w['insertions']}; exact {w['clips_exact']}/{len(hyps)})",
                f"mean latency {sum(lat) / len(lat):.2f}s")
        rep.extra[name] = {**w, "mean_latency_s": round(sum(lat) / len(lat), 2)}
    rep.extra["hypotheses"] = all_hyps
    rep.save()


if __name__ == "__main__":
    main()
