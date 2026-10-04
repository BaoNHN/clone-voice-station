#!/usr/bin/env python
"""
experiments/retest/flow_05_snr.py
Re-runs the SNR estimator validation and compares it with Table 11.
Details: ai_change_log.txt (retest flows).
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STATION_DIR, Report

TABLE_11 = {60.0: 58.8, 40.0: 39.0, 20.0: 18.8}


def main():
    rep = Report("flow_05_snr")
    p = subprocess.run([sys.executable, os.path.join(STATION_DIR, "experiments", "validate_snr_estimator.py")],
                       cwd=STATION_DIR, capture_output=True, text=True, encoding="utf-8")
    print(p.stdout)
    if p.returncode != 0:
        rep.add("SNR", "FAIL", "script finishes", p.stderr[-300:])
        rep.save()
        sys.exit(1)
    rep.extra["stdout"] = p.stdout
    found = 0
    seen = set()
    for line in p.stdout.splitlines():
        nums = re.findall(r"-?\d+\.\d+", line)
        if len(nums) >= 3 and float(nums[0]) in TABLE_11 and "dB" in line and float(nums[0]) not in seen:
            seen.add(float(nums[0]))
            ref, meas = float(nums[0]), float(nums[1])
            found += 1
            ok = abs(meas - TABLE_11[ref]) <= 0.1
            rep.add(f"{ref:g}dB", "PASS" if ok else "FAIL", f"{TABLE_11[ref]} dB (Table 11)",
                    f"{meas} dB", f"error {meas - ref:+.1f} dB")
    if not found:
        rep.add("SNR", "FAIL", "table rows parsed", "no 'ref measured' rows found - check script output format")
    rep.save()


if __name__ == "__main__":
    main()
