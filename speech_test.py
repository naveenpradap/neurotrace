"""
speech_test.py

Records a real microphone sample and extracts genuine speech biomarkers
with librosa: voice-activity ratio, pause durations (via silence
splitting), pitch (F0) stability, and a clarity proxy from spectral
flatness. No speech-to-text is used here (that's an optional upgrade -
see README), so "speech rate" is a voiced-segment-count proxy, not a
true words-per-minute figure.

Run directly:
    python speech_test.py --name yourname --out data/speech_metrics.json
"""

import argparse
import json
import os
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

SENTENCE = "Today I am completing a short speech assessment for concussion screening."
SAMPLE_RATE = 22050
DURATION_SEC = 12
MIN_RMS_FRAMES_VOICED = 8   # validity guard: need at least this many voiced frames


def record_audio(duration=DURATION_SEC, sr=SAMPLE_RATE):
    print("\a")  # terminal bell - alerts you to switch to this window
    print("=" * 60)
    print(f'READ THIS SENTENCE ALOUD:\n\n  "{SENTENCE}"')
    print("=" * 60)
    for i in (3, 2, 1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)
    print(">>> RECORDING NOW - SPEAK <<<", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), sr


def run_speech_test(subject_name="subject", save_wav_path=None):
    y, sr = record_audio()

    if save_wav_path:
        os.makedirs(os.path.dirname(save_wav_path) or ".", exist_ok=True)
        sf.write(save_wav_path, y, sr)

    # ---- absolute silence floor (independent of top_db, which is only
    # relative to the clip's own peak - a fully silent/ambient-noise clip
    # would otherwise pass as "one long voiced interval") ----
    peak_amp = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak_amp < 0.02:
        return {
            "valid": False,
            "reason": ("no audio detected above the noise floor - the mic may be muted, "
                       "set to the wrong input device, or the recording ran before you were "
                       "ready. Check your terminal for the '3...2...1...Recording now' prompt "
                       "and speak right after it, then retry."),
        }

    # ---- voice activity / pauses via silence splitting ----
    intervals = librosa.effects.split(y, top_db=30)  # returns [[start,end], ...] in samples

    if len(intervals) == 0:
        return {"valid": False, "reason": "no speech detected - retry closer to the mic, in a quiet room"}

    voiced_samples = sum((end - start) for start, end in intervals)
    voiced_ratio = float(voiced_samples / len(y))

    pause_durations_ms = []
    for i in range(1, len(intervals)):
        gap_samples = intervals[i][0] - intervals[i - 1][1]
        if gap_samples > 0:
            pause_durations_ms.append((gap_samples / sr) * 1000)
    avg_pause_ms = float(np.mean(pause_durations_ms)) if pause_durations_ms else 200.0

    # crude segment-count-based speech rate proxy (NOT true words/sec - see README)
    speech_rate_proxy = round(len(intervals) / (len(y) / sr), 2)

    total_voiced_sec = voiced_samples / sr
    # validity is based on how much real voiced audio we captured, not on
    # pause count - a fluent sentence with no mid-pause is a single interval
    # and that is NOT invalid.
    if voiced_ratio < 0.03 or total_voiced_sec < 1.0:
        return {"valid": False, "reason": "insufficient speech captured - retry and speak clearly for the full duration"}

    # ---- pitch (F0) via pYIN ----
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) >= MIN_RMS_FRAMES_VOICED:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        # lower relative variability -> higher stability score
        pitch_stability = float(max(0.0, min(100.0, 100 - (f0_std / max(f0_mean, 1)) * 300)))
    else:
        f0_mean, f0_std = 0.0, 0.0
        pitch_stability = 40.0

    # ---- clarity proxy via spectral flatness (lower flatness = more tonal/clear speech) ----
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))
    clarity = float(max(0.0, min(100.0, (1 - mean_flatness * 6) * 100)))

    metrics = {
        "valid": True,
        "subject": subject_name,
        "voiced_ratio": round(voiced_ratio, 3),
        "avg_pause_ms": round(avg_pause_ms, 1),
        "speech_rate_proxy_segments_per_sec": speech_rate_proxy,
        "pitch_mean_hz": round(f0_mean, 1),
        "pitch_std_hz": round(f0_std, 1),
        "pitch_stability": round(pitch_stability, 1),
        "clarity": round(clarity, 1),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="subject")
    parser.add_argument("--out", default="data/speech_metrics.json")
    parser.add_argument("--save-wav", default=None, help="optional path to also save the raw recording")
    args = parser.parse_args()

    result = run_speech_test(args.name, save_wav_path=args.save_wav)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if not result.get("valid"):
        print(f"Speech test INVALID: {result.get('reason')}")
    else:
        print(f"Saved speech metrics to {args.out}")
        print(json.dumps(result, indent=2))
