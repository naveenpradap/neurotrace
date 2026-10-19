"""
test_kit.py

Shared building blocks for every native test window (eye_test.py,
reaction_test.py, cognitive_test.py): consistent colors/typography, a real
"instructions -> 3,2,1 countdown -> begin" opening sequence (spoken AND
audible-beeped, not just shown on screen, so it works for the blind
profile), and a louder/longer/volume-aware tone generator.

This directly answers three problems that were found in testing:
  - blind-mode tests only spoke the digit sequence, not the instructions
  - tests jumped straight into trials with no lead-in
  - the reaction-test tone was too quiet/short to reliably hear
"""

import time

import cv2
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

import numpy as _np

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "ate": "8", "nine": "9",
}


def words_to_digits(text: str) -> str:
    """'four seven two' -> '472'. Also passes through literal digit characters."""
    if not text:
        return ""
    out = []
    for tok in text.lower().replace("-", " ").split():
        tok = tok.strip(".,!?")
        if tok.isdigit():
            out.append(tok)
        elif tok in DIGIT_WORDS:
            out.append(DIGIT_WORDS[tok])
    return "".join(out)


def record_audio_array(duration=5.0, samplerate=22050):
    if sd is None:
        return None, samplerate
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), samplerate


def transcribe(audio_float, samplerate):
    """Uses SpeechRecognition's free Google Web Speech recognizer - requires
    internet at runtime, no API key needed. Returns (text_or_None, error_or_None)."""
    if sr is None:
        return None, "speech_recognition not installed"
    if audio_float is None or len(audio_float) == 0:
        return None, "no audio captured"
    try:
        pcm16 = (audio_float * 32767).astype(_np.int16).tobytes()
        audio_data = sr.AudioData(pcm16, samplerate, 2)
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(audio_data)
        return text, None
    except Exception as e:
        return None, f"transcription failed ({type(e).__name__}) - check internet connection"

# ---- shared palette (BGR, matches the app's CSS accent/bg colors) ----
BG = (26, 21, 15)
TEXT = (231, 237, 241)
ACCENT = (73, 224, 179)
DANGER = (255, 107, 82)
MUTED = (153, 136, 116)


def speak(text, enabled=True, rate=170, volume=1.0):
    """Blocking TTS. Silently no-ops (and prints) if pyttsx3/engine unavailable."""
    if not enabled:
        return
    if pyttsx3 is None:
        print(f"[TTS unavailable — printing instead] {text}")
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print(f"[TTS failed — printing instead] {text}")


def play_tone(freq=880, duration=0.28, volume=1.0, sr=22050):
    """Louder/longer default beep than earlier versions, scaled by the volume setting."""
    if sd is None:
        return
    t = np.linspace(0, duration, int(sr * duration), False)
    # short fade in/out so it doesn't click, otherwise full amplitude for audibility
    fade = min(300, len(t) // 6)
    env = np.ones_like(t)
    if fade > 0:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    amp = max(0.0, min(1.0, volume)) * 0.8
    tone = amp * env * np.sin(2 * np.pi * freq * t)
    try:
        sd.play(tone.astype(np.float32), sr)
        sd.wait()
    except Exception:
        pass


def wrap_text(text, max_chars=54):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def blank_canvas(w, h, color=BG):
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[:] = color
    return c


def intro_and_countdown(window_name, w, h, title, instruction_lines,
                         tts_enabled=False, volume=1.0, tts_rate=170,
                         min_read_seconds=3.5, countdown_from=3):
    """
    Standard opening for every test:
      1. Shows a title + wrapped instruction text for at least
         `min_read_seconds` (skippable early with any key once that time
         has passed), reading it aloud first if tts_enabled.
      2. Real 3-2-1 countdown - shown on screen AND spoken/beeped, so a
         blind user gets the same lead-in a sighted user does.
    Returns False if the user cancelled (ESC/Q), True otherwise.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    spoken = title + ". " + " ".join(instruction_lines)
    if tts_enabled:
        # show the screen once before speaking so sighted+blind users both have context
        canvas = blank_canvas(w, h)
        cv2.putText(canvas, title, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.95, ACCENT, 2)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)
        speak(spoken, enabled=True, rate=tts_rate, volume=volume)

    t0 = time.time()
    while True:
        canvas = blank_canvas(w, h)
        cv2.putText(canvas, title, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.95, ACCENT, 2)
        y = 110
        for line in instruction_lines:
            cv2.putText(canvas, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT, 1)
            y += 32
        remaining = max(0.0, min_read_seconds - (time.time() - t0))
        hint = "Press any key to continue..." if remaining <= 0 else f"Starting in {remaining:0.0f}s (or press any key)"
        cv2.putText(canvas, hint, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            return False
        if key != 255 and (time.time() - t0) >= min_read_seconds:
            break
        if (time.time() - t0) >= min_read_seconds + 2.0:  # don't block forever
            break

    # ---- 3, 2, 1 countdown: visible + beeped + (optionally) spoken ----
    for n in (countdown_from, countdown_from - 1, countdown_from - 2):
        if n <= 0:
            continue
        canvas = blank_canvas(w, h)
        cv2.putText(canvas, str(n), (w // 2 - 25, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 3.0, ACCENT, 6)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)
        play_tone(freq=520, duration=0.12, volume=volume)
        if tts_enabled:
            speak(str(n), enabled=True, rate=tts_rate + 40, volume=volume)
        else:
            time.sleep(0.55)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False

    canvas = blank_canvas(w, h)
    cv2.putText(canvas, "GO", (w // 2 - 70, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.6, ACCENT, 6)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)
    play_tone(freq=1050, duration=0.3, volume=volume)
    if tts_enabled:
        speak("Go", enabled=True, rate=tts_rate, volume=volume)
    cv2.waitKey(200)
    return True
