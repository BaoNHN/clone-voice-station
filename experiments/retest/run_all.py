#!/usr/bin/env python
"""
experiments/retest/run_all.py
Runs the retest flows in order and prints a one-line verdict per flow.
Details: ai_change_log.txt (retest flows).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FLOWS = {
    "api": ["flow_01_api_testcases.py"],
    "sim": ["flow_02_speaker_similarity.py"],
    "latency": ["flow_03_latency.py", "--path", "colab-local"],
    "wer": ["flow_04_legal_wer.py"],
    "snr": ["flow_05_snr.py"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--only", default=",".join(FLOWS), help="comma list of: " + ", ".join(FLOWS))
    ap.add_argument("extra", nargs="*", help="extra args passed to every flow (e.g. --user-id U)")
    args = ap.parse_args()

    summary = {}
    for name in [n.strip() for n in args.only.split(",") if n.strip()]:
        print(f"\n######## {name}", flush=True)
        cmd = [sys.executable, os.path.join(HERE, FLOWS[name][0])] + FLOWS[name][1:]
        if name in ("api", "latency"):
            cmd += args.extra
        summary[name] = subprocess.run(cmd).returncode
    print("\n==== summary")
    for name, code in summary.items():
        print(f"{name:8} {'OK' if code == 0 else 'FAILED (exit %d)' % code}")
    sys.exit(1 if any(summary.values()) else 0)


if __name__ == "__main__":
    main()
