"""
reaction_test.py

Reaction-time test, two variants:
  --mode visual  : a dot appears on screen; press SPACE as fast as possible.
                    Used for standard / deaf / speech-impaired profiles.
  --mode audio    : a tone plays through the speakers; press SPACE as fast
                    as possible. Used for the blind profile.

Opens with a spoken+beeped "instructions, then 3-2-1" lead-in (test_kit.py)
instead of jumping straight into trials, and the tone is louder/longer and
volume-controllable via --volume.

Run directly:
    python reaction_test.py --mode visual --name yourname --out data/reaction_metrics.json
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from test_kit import (
    BG, TEXT, ACCENT, DANGER, blank_canvas, intro_and_countdown, play_tone, speak,
)

WIN = "NeuroTrace - Reaction Test"
W, H = 900, 500
N_TRIALS = 6
BASE_TIMEOUT_MS = 2000


def _canvas(msg=""):
    c = blank_canvas(W, H)
    if msg:
        cv2.putText(c, msg, (30, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2)
    return c


def run_reaction_test(mode="visual", subject_name="subject", extended_time=1.0,
                       reduce_motion=False, tts_enabled=False, volume=1.0, tts_rate=170):
    timeout_ms = BASE_TIMEOUT_MS * extended_time

    if mode == "visual":
        instructions = [
            "A green dot will appear on screen at a random moment.",
            "Press the SPACE bar as fast as you can when you see it.",
            f"There will be {N_TRIALS} rounds.",
        ]
    else:
        instructions = [
            "You will hear a tone at a random moment.",
            "Press the SPACE bar as fast as you can when you hear it.",
            f"There will be {N_TRIALS} rounds.",
        ]

    ok = intro_and_countdown(WIN, W, H, "⚡ Reaction-Time Test", instructions,
                              tts_enabled=tts_enabled, volume=volume, tts_rate=tts_rate)
    if not ok:
        return None

    reaction_times = []
    missed = 0
    aborted = False

    for trial in range(N_TRIALS):
        wait_s = random.uniform(1.2, 3.0)
        t_wait_start = time.time()
        while time.time() - t_wait_start < wait_s:
            canvas = _canvas(f"Round {trial+1}/{N_TRIALS} — wait for it... (ESC/Q to cancel)")
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                aborted = True
                break
        if aborted:
            break

        stim_time = time.time()
        if mode == "audio":
            play_tone(freq=880, duration=0.28, volume=volume)
        responded = False
        rt_ms = None
        while (time.time() - stim_time) * 1000 < timeout_ms:
            canvas = _canvas("Round in progress — ESC/Q to cancel")
            if mode == "visual":
                cv2.circle(canvas, (W // 2, H // 2), 60, ACCENT, -1)
                cv2.putText(canvas, "PRESS SPACE", (W // 2 - 110, H // 2 + 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2)
            else:
                cv2.putText(canvas, "(tone played) PRESS SPACE", (W // 2 - 220, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(10) & 0xFF
            if key == 32:
                rt_ms = (time.time() - stim_time) * 1000
                responded = True
                break
            if key in (27, ord("q")):
                aborted = True
                break
        if aborted:
            break

        if responded:
            reaction_times.append(rt_ms)
            fb = _canvas("")
            cv2.putText(fb, f"{rt_ms:.0f} ms", (W // 2 - 80, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, ACCENT, 3)
        else:
            missed += 1
            fb = _canvas("")
            cv2.putText(fb, "MISSED", (W // 2 - 90, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, DANGER, 3)
            if tts_enabled:
                speak("Missed", enabled=True, rate=tts_rate, volume=volume)
        cv2.imshow(WIN, fb)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    if aborted:
        return None

    total_trials = len(reaction_times) + missed
    if total_trials < N_TRIALS or len(reaction_times) == 0:
        return {"valid": False, "reason": "test cancelled or too few completed trials"}

    mean_rt = float(np.mean(reaction_times))
    std_rt = float(np.std(reaction_times)) if len(reaction_times) > 1 else 0.0
    consistency_score = float(max(0.0, min(100.0, 100 - (std_rt / 4))))

    return {
        "valid": True,
        "subject": subject_name,
        "mode": mode,
        "mean_reaction_ms": round(mean_rt, 1),
        "std_reaction_ms": round(std_rt, 1),
        "consistency_score": round(consistency_score, 1),
        "missed_trials": missed,
        "total_trials": N_TRIALS,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["visual", "audio"], default="visual")
    parser.add_argument("--name", default="subject")
    parser.add_argument("--out", default="data/reaction_metrics.json")
    parser.add_argument("--extended-time", type=float, default=1.0)
    parser.add_argument("--reduce-motion", action="store_true")
    parser.add_argument("--tts", action="store_true", help="speak instructions/countdown aloud")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--tts-rate", type=int, default=170)
    args = parser.parse_args()

    result = run_reaction_test(args.mode, args.name, args.extended_time, args.reduce_motion,
                                tts_enabled=args.tts, volume=args.volume, tts_rate=args.tts_rate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if result is None:
        print("Reaction test cancelled - no file written.")
    else:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
