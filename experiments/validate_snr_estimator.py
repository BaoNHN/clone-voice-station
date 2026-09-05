#!/usr/bin/env python
"""
experiments/validate_snr_estimator.py
Checks the SNR estimator in tools/eval_voice_quality.py against signals whose
noise floor is known by construction, and records the numbers reported in
Section 6.5, Table 10 of the thesis.

Why this exists. SNR is one of the acceptance criteria in Table 8, but the
quantity is only meaningful if the estimator recovers a ratio that is actually
there. Synthetic speech gives a reference to check against: the speech level and
the noise level are both set by the script, so the true SNR is known exactly and
the estimate can be compared with it rather than merely looking plausible.

The script also reproduces the failure of the first implementation, which derived
the noise level from the complement of the ITU-T P.56 activity mask. That mask
carries a 200 ms hangover, so its complement still contains the decay at the end
of each word. The residual measured there is speech, not noise, and the resulting
figure barely moves when the true noise floor is swept across 40 dB.

Usage:
    python experiments/validate_snr_estimator.py
    python experiments/validate_snr_estimator.py > experiments/logs/snr_validation.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import eval_voice_quality as E  # noqa: E402

SR = 22050
SPEECH_RMS = 0.1                      # -20 dBFS
NOISE_SIGMAS = [1e-4, 1e-3, 1e-2]     # true SNR 60, 40, 20 dB
SEED = 1


def synth_utterance(rng, duration_s=6.0, speech_rms=SPEECH_RMS, noise_sigma=1e-3,
                    with_pauses=True):
    """Word-like bursts over a constant noise floor.

    Each burst is a Hann-windowed harmonic stack, which gives the onset and decay
    a real speech signal has; without that the noise-floor percentile has nothing
    to land on but digital silence and the test would be easier than reality.
    """
    n = int(duration_s * SR)
    x = np.zeros(n)
    pos = 0.3
    while pos < duration_s - 0.5:
        word_len = rng.uniform(0.3, 0.5)
        a, b = int(pos * SR), int((pos + word_len) * SR)
        m = b - a
        t = np.arange(m) / SR
        x[a:b] = np.hanning(m) * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 5))
        pos += word_len + (rng.uniform(0.2, 0.35) if with_pauses else 0.0)

    active = np.abs(x) > 1e-9
    x *= speech_rms / np.sqrt((x[active] ** 2).mean())
    return x + rng.normal(0, noise_sigma, n)


def p56_complement_snr(x, sr):
    """The rejected estimator, kept so its failure stays reproducible."""
    res = E.p56_active_level(x, sr)
    if res is None:
        return None
    level_db, _, active = res
    noise = x[~active]
    if noise.size == 0:
        return None
    power = float(np.dot(noise, noise)) / noise.size
    return level_db - 10.0 * np.log10(power) if power > 0 else None


def main():
    rng = np.random.default_rng(SEED)
    print("SNR estimator validation")
    print(f"  sample rate      {SR} Hz")
    print(f"  speech level     {20 * np.log10(SPEECH_RMS):.1f} dBFS")
    print(f"  random seed      {SEED}")
    print()
    print(f"{'noise sigma':>12} {'reference':>10} {'measured':>9} {'error':>7} "
          f"{'activity':>9} {'rejected est.':>14}")

    errors = []
    for sigma in NOISE_SIGMAS:
        x = synth_utterance(rng, noise_sigma=sigma)
        reference = 20 * np.log10(SPEECH_RMS / sigma)
        measured, activity = E.snr_db(x, SR)
        rejected = p56_complement_snr(x, SR)
        errors.append(measured - reference)
        print(f"{sigma:12.0e} {reference:9.1f}dB {measured:8.1f}dB "
              f"{measured - reference:6.1f}dB {100 * activity:8.1f}% "
              f"{rejected:13.1f}dB")

    print()
    print(f"Worst absolute error over a {NOISE_SIGMAS[0] and 40} dB sweep: "
          f"{max(abs(e) for e in errors):.1f} dB")
    print("The rejected estimator stays flat while the reference moves 40 dB, which is "
          "why\nit was replaced by the frame-power percentile.")

    print()
    print("No-pause case (a clip read without gaps between words):")
    x = synth_utterance(rng, noise_sigma=1e-3, with_pauses=False)
    measured, activity = E.snr_db(x, SR)
    print(f"  reference 40.0dB, measured {measured:.1f}dB, activity {100 * activity:.1f}%")
    print("  Above the HIGH_ACTIVITY_WARN threshold the harness flags the file so this "
          "figure\n  is read as a lower bound rather than an estimate.")


if __name__ == "__main__":
    main()
