"""
cognitive_test.py

Two variants:
  --mode visual : shape-selection test (pick the named shape, 4 trials) +
                  word recognition-memory test (5 yes/no questions).
  --mode audio  : spoken digit-recall test via TTS - the person hears a
                  short number sequence, then types it back. Used for the
                  blind profile.

Both now open with a spoken+beeped "instructions, then 3-2-1" lead-in via
test_kit.py, and in audio mode ALL instructions are spoken aloud, not just
the digit sequences - the earlier version only spoke the numbers, which
meant a blind user couldn't actually read the on-screen setup text.

Known limitation (still true, said plainly): the audio mode still opens an
on-screen window to capture keyboard input, which needs OS focus - it is
NOT a fully screen-reader / mouse-free input path. A production version
would need an OS-level key hook or an accessible GUI toolkit instead of cv2.

Run directly:
    python cognitive_test.py --mode visual --name yourname --out data/cognitive_metrics.json
"""

import argparse
import json
import os
import random
import time

import cv2
import numpy as np

from test_kit import (
    BG, TEXT, ACCENT, DANGER, blank_canvas, intro_and_countdown, speak,
    record_audio_array, transcribe, words_to_digits,
)

try:
    import librosa
except Exception:
    librosa = None

WIN_VISUAL = "NeuroTrace - Cognitive Test"
WIN_AUDIO = "NeuroTrace - Cognitive Test (Audio)"
W, H = 900, 560


def _canvas(title=""):
    c = blank_canvas(W, H)
    if title:
        cv2.putText(c, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2)
    return c


def draw_shape(canvas, shape, center, size, color):
    x, y = center
    if shape == "circle":
        cv2.circle(canvas, (x, y), size, color, -1)
    elif shape == "square":
        cv2.rectangle(canvas, (x - size, y - size), (x + size, y + size), color, -1)
    elif shape == "triangle":
        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
        cv2.fillPoly(canvas, [pts], color)
    elif shape == "star":
        pts = []
        for i in range(10):
            r = size if i % 2 == 0 else size * 0.45
            ang = -np.pi / 2 + i * np.pi / 5
            pts.append([x + int(r * np.cos(ang)), y + int(r * np.sin(ang))])
        cv2.fillPoly(canvas, [np.array(pts, np.int32)], color)


# ------------------- VISUAL: shape-selection test -------------------
def run_selection_test(reduce_motion=False, extended_time=1.0):
    shapes = ["circle", "square", "triangle", "star"]
    positions = [(200, 350), (400, 350), (600, 350), (800, 350)]
    timeout_ms = 4000 * extended_time
    correct = 0
    rts = []

    for trial in range(4):
        target = random.choice(shapes)
        order = shapes[:]
        random.shuffle(order)

        t0 = time.time()
        answered = None
        while (time.time() - t0) * 1000 < timeout_ms:
            canvas = _canvas(f"Trial {trial+1}/4 — SELECT THE {target.upper()}   (ESC/Q cancel)")
            for i, shp in enumerate(order):
                cx = positions[i][0]
                draw_shape(canvas, shp, (cx, 350), 55, ACCENT)
                cv2.putText(canvas, str(i + 1), (cx - 8, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2)
            cv2.imshow(WIN_VISUAL, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                return None
            if ord("1") <= key <= ord("4"):
                idx = key - ord("1")
                answered = order[idx]
                rt = (time.time() - t0) * 1000
                break
        else:
            answered, rt = None, timeout_ms

        rts.append(rt)
        if answered == target:
            correct += 1
        fb = _canvas("")
        msg, color = ("CORRECT", ACCENT) if answered == target else ("INCORRECT" if answered else "NO RESPONSE", DANGER)
        cv2.putText(fb, msg, (W // 2 - 140, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow(WIN_VISUAL, fb)
        cv2.waitKey(400)

    return {"accuracy": correct / 4, "avg_rt_ms": float(np.mean(rts))}


# ------------------- VISUAL: recognition-memory test -------------------
WORD_POOL = ["APPLE", "HOUSE", "BLUE", "RIVER", "CHAIR", "MUSIC", "CLOUD", "TIGER", "BRIDGE", "LANTERN"]


def run_memory_test(reduce_motion=False, extended_time=1.0):
    targets = random.sample(WORD_POOL, 3)
    foils = random.sample([w for w in WORD_POOL if w not in targets], 2)

    study = _canvas("MEMORIZE THESE WORDS")
    cv2.putText(study, "  —  ".join(targets), (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ACCENT, 2)
    cv2.imshow(WIN_VISUAL, study)
    cv2.waitKey(int(3500 * extended_time))

    hide = _canvas("...")
    cv2.imshow(WIN_VISUAL, hide)
    cv2.waitKey(1000)

    questions = [(w, True) for w in targets] + [(w, False) for w in foils]
    random.shuffle(questions)

    correct = 0
    rts = []
    timeout_ms = 4000 * extended_time
    for word, is_target in questions:
        t0 = time.time()
        answered = None
        while (time.time() - t0) * 1000 < timeout_ms:
            canvas = _canvas("Was this word in the list?   Y = yes   N = no   (ESC/Q cancel)")
            cv2.putText(canvas, word, (W // 2 - 100, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, TEXT, 3)
            cv2.imshow(WIN_VISUAL, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                return None
            if key in (ord("y"), ord("n")):
                answered = (key == ord("y"))
                rts.append((time.time() - t0) * 1000)
                break
        if answered == is_target:
            correct += 1

    return {"accuracy": correct / len(questions), "avg_rt_ms": float(np.mean(rts)) if rts else timeout_ms}


def run_visual_cognitive(subject_name, extended_time=1.0, reduce_motion=False, tts_enabled=False, volume=1.0, tts_rate=170):
    instructions = [
        "Part 1: four shapes appear. Press the number below the shape you're asked for.",
        "Part 2: memorize three words, then answer yes/no if a word was in the list.",
    ]
    ok = intro_and_countdown(WIN_VISUAL, W, H, "🧠 Cognitive Test", instructions,
                              tts_enabled=tts_enabled, volume=volume, tts_rate=tts_rate)
    if not ok:
        return None
    try:
        sel = run_selection_test(reduce_motion, extended_time)
        if sel is None:
            return None
        mem = run_memory_test(reduce_motion, extended_time)
        if mem is None:
            return None
    finally:
        cv2.destroyAllWindows()

    accuracy = (sel["accuracy"] + mem["accuracy"]) / 2
    avg_rt = (sel["avg_rt_ms"] + mem["avg_rt_ms"]) / 2
    accuracy_score = accuracy * 100
    speed_score = float(max(0.0, min(100.0, 100 - (avg_rt / 40))))
    cognitive_score = accuracy_score * 0.65 + speed_score * 0.35

    return {
        "valid": True,
        "subject": subject_name,
        "mode": "visual",
        "selection_accuracy": round(sel["accuracy"] * 100, 1),
        "memory_accuracy": round(mem["accuracy"] * 100, 1),
        "avg_response_ms": round(avg_rt, 1),
        "cognitive_score": round(cognitive_score, 1),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------- AUDIO: spoken digit-recall + speech biomarkers combined (blind mode) -------------------
def _speech_biomarkers_from_clip(y, sr_rate):
    """Same DSP approach as speech_test.py's standalone speech test, run on the
    spoken digit-recall responses instead of a separate recording, so blind
    users get real speech biomarkers without a second, unreadable test."""
    if librosa is None or y is None or len(y) == 0:
        return None
    peak_amp = float(np.max(np.abs(y)))
    if peak_amp < 0.02:
        return None
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) == 0:
        return None
    voiced_samples = sum((end - start) for start, end in intervals)
    voiced_ratio = float(voiced_samples / len(y))
    pause_ms = []
    for i in range(1, len(intervals)):
        gap = intervals[i][0] - intervals[i - 1][1]
        if gap > 0:
            pause_ms.append((gap / sr_rate) * 1000)
    avg_pause_ms = float(np.mean(pause_ms)) if pause_ms else 200.0
    try:
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr_rate)
        f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        f0v = np.array([])
    if len(f0v) >= 5:
        f0_mean, f0_std = float(np.mean(f0v)), float(np.std(f0v))
        pitch_stability = float(max(0.0, min(100.0, 100 - (f0_std / max(f0_mean, 1)) * 300)))
    else:
        pitch_stability = 40.0
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    clarity = float(max(0.0, min(100.0, (1 - flatness * 6) * 100)))
    return {"voiced_ratio": round(voiced_ratio, 3), "avg_pause_ms": round(avg_pause_ms, 1),
            "pitch_stability": round(pitch_stability, 1), "clarity": round(clarity, 1)}


def run_audio_cognitive(subject_name, extended_time=1.0, tts_enabled=True, volume=1.0, tts_rate=170):
    """Returns (cognitive_metrics, speech_metrics) - the same spoken responses
    are used for both, since typing was not a real option for this profile."""
    instructions = [
        "You will hear a short sequence of numbers.",
        "After the tone, say the numbers out loud in order.",
        "There will be three rounds, getting slightly longer each time.",
    ]
    ok = intro_and_countdown(WIN_AUDIO, W, 260, "🧠 Cognitive + Speech Test (Audio)", instructions,
                              tts_enabled=tts_enabled, volume=volume, tts_rate=tts_rate)
    if not ok:
        return None, None

    rounds = [3, 4, 5]
    correct_rounds = 0
    response_times = []
    clips = []
    aborted = False

    for length in rounds:
        digits = [random.randint(0, 9) for _ in range(length)]
        canvas = _canvas(f"Listen carefully — {length} digits incoming...")
        cv2.imshow(WIN_AUDIO, canvas)
        cv2.waitKey(300)
        speak("Listen carefully. I will say some numbers. Remember them in order.",
              enabled=tts_enabled, rate=tts_rate, volume=volume)
        speak(" ... ".join(str(d) for d in digits), enabled=tts_enabled, rate=tts_rate, volume=volume)
        speak("Now say the numbers out loud, in order.", enabled=tts_enabled, rate=tts_rate, volume=volume)

        rec_seconds = max(3.0, length * 1.2) * extended_time
        canvas = _canvas(f"🎙️ Speak now — recording for {rec_seconds:.0f}s")
        cv2.imshow(WIN_AUDIO, canvas)
        cv2.waitKey(1)
        t0 = time.time()
        y, sr_rate = record_audio_array(duration=rec_seconds, samplerate=22050)
        response_times.append((time.time() - t0) * 1000)
        if y is not None:
            clips.append((y, sr_rate))

        text, err = transcribe(y, sr_rate) if y is not None else (None, "no recording")
        spoken_digits = words_to_digits(text) if text else ""
        target = "".join(str(d) for d in digits)

        fb = _canvas("")
        if spoken_digits == target:
            correct_rounds += 1
            cv2.putText(fb, "CORRECT", (W // 2 - 110, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, ACCENT, 3)
            speak("Correct.", enabled=tts_enabled, rate=tts_rate, volume=volume)
        else:
            heard = f"heard: {text}" if text else f"couldn't transcribe ({err})"
            cv2.putText(fb, "NOTED", (W // 2 - 90, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, DANGER, 3)
            cv2.putText(fb, heard, (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1)
            speak("Noted.", enabled=tts_enabled, rate=tts_rate, volume=volume)
        cv2.imshow(WIN_AUDIO, fb)
        key = cv2.waitKey(900) & 0xFF
        if key in (27, ord("q")):
            aborted = True
            break

    cv2.destroyAllWindows()
    if aborted or not clips:
        return None, None

    accuracy = correct_rounds / len(rounds)
    avg_rt = float(np.mean(response_times))
    accuracy_score = accuracy * 100
    speed_score = float(max(0.0, min(100.0, 100 - (avg_rt / 100))))
    cognitive_score = accuracy_score * 0.7 + speed_score * 0.3

    cognitive_metrics = {
        "valid": True,
        "subject": subject_name,
        "mode": "audio",
        "digit_recall_accuracy": round(accuracy * 100, 1),
        "avg_response_ms": round(avg_rt, 1),
        "cognitive_score": round(cognitive_score, 1),
        "note": "Digit recall was spoken and transcribed (needs internet for transcription), "
                "not typed - see README.",
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # combine biomarkers across all three spoken-response clips into one speech_metrics result
    combined_y = np.concatenate([c[0] for c in clips]) if clips else None
    sr_rate = clips[0][1] if clips else 22050
    sb = _speech_biomarkers_from_clip(combined_y, sr_rate)
    if sb is None:
        speech_metrics = {"valid": False, "reason": "insufficient voiced audio across responses to score speech biomarkers"}
    else:
        speech_metrics = {"valid": True, "subject": subject_name, "source": "derived from spoken digit-recall responses",
                           **sb, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    return cognitive_metrics, speech_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["visual", "audio"], default="visual")
    parser.add_argument("--name", default="subject")
    parser.add_argument("--out", default="data/cognitive_metrics.json")
    parser.add_argument("--speech-out", default=None,
                         help="audio mode only: also write derived speech biomarkers here")
    parser.add_argument("--extended-time", type=float, default=1.0)
    parser.add_argument("--reduce-motion", action="store_true")
    parser.add_argument("--tts", action="store_true", help="speak instructions aloud")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--tts-rate", type=int, default=170)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "visual":
        result = run_visual_cognitive(args.name, args.extended_time, args.reduce_motion,
                                       tts_enabled=args.tts, volume=args.volume, tts_rate=args.tts_rate)
        if result is None:
            print("Cognitive test cancelled - no file written.")
        else:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
    else:
        cognitive_metrics, speech_metrics = run_audio_cognitive(
            args.name, args.extended_time, tts_enabled=True, volume=args.volume, tts_rate=args.tts_rate)
        if cognitive_metrics is None:
            print("Cognitive+speech test cancelled - no file written.")
        else:
            with open(args.out, "w") as f:
                json.dump(cognitive_metrics, f, indent=2)
            print(json.dumps(cognitive_metrics, indent=2))
            if args.speech_out:
                os.makedirs(os.path.dirname(args.speech_out) or ".", exist_ok=True)
                with open(args.speech_out, "w") as f:
                    json.dump(speech_metrics, f, indent=2)
                print(json.dumps(speech_metrics, indent=2))
