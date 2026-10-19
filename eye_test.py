"""
eye_test.py

Real-time eye-movement biomarker capture using a standard webcam and
MediaPipe's FaceLandmarker (Tasks API - the current, non-deprecated API;
the old `mediapipe.solutions.face_mesh` API used in older tutorials has
been removed from recent mediapipe releases).

The FaceLandmarker model outputs 478 landmarks per face, including iris
centers (index 468 = left iris, 473 = right iris) - same indices as the
old refine_landmarks=True output, so the biomarker math below is unchanged
in spirit, just re-plumbed onto the current API.

No screen-gaze calibration is performed, so this measures RELATIVE
eye-movement biomarkers (fixation jitter, pursuit correlation, saccade
reaction latency, blink rate, per-eye asymmetry) rather than exact
on-screen gaze coordinates. Be upfront about that in your pitch.

First run downloads the ~4MB face_landmarker.task model once.

Run directly:
    python eye_test.py --name yourname --out data/eye_metrics.json
Controls: ESC or Q at any time to abort.
"""

import argparse
import json
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_CORNERS = (33, 133)      # (outer, inner)
RIGHT_EYE_CORNERS = (362, 263)    # (inner, outer)
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

EAR_BLINK_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2
W, H = 960, 600

MIN_FRAMES_WITH_FACE = 60   # validity guard: need at least this many good frames total


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading FaceLandmarker model (one-time, ~4MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Could not download the FaceLandmarker model automatically ({e}). "
                f"Manually download it from:\n{MODEL_URL}\nand save it as:\n{MODEL_PATH}"
            )


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def build_landmarker():
    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def run_eye_test(subject_name="subject", tts_enabled=False, volume=1.0, tts_rate=170):
    from test_kit import intro_and_countdown

    instructions = [
        "Part 1 — Fixation: hold your gaze on the center dot.",
        "Part 2 — Smooth Pursuit: trace the dot as it moves in a circle.",
        "Part 3 — Saccade: look LEFT / CENTER / RIGHT as cued.",
    ]
    ok = intro_and_countdown("NeuroTrace - Eye Test", W, H, "👁 Eye-Movement Test", instructions,
                              tts_enabled=tts_enabled, volume=volume, tts_rate=tts_rate)
    if not ok:
        return None

    landmarker = build_landmarker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions.")

    cv2.namedWindow("NeuroTrace - Eye Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NeuroTrace - Eye Test", W, H)

    blink_events = []
    ear_below_count = 0
    frames_with_face = 0
    fixation_samples = []
    pursuit_samples = []
    saccade_cues = []
    left_eye_activity, right_eye_activity = [], []  # for asymmetry check

    def normalized_gaze(landmarks, w, h):
        lx, ly = get_xy(landmarks, LEFT_IRIS, w, h)
        rx, ry = get_xy(landmarks, RIGHT_IRIS, w, h)
        l_outer = get_xy(landmarks, LEFT_EYE_CORNERS[0], w, h)
        l_inner = get_xy(landmarks, LEFT_EYE_CORNERS[1], w, h)
        r_inner = get_xy(landmarks, RIGHT_EYE_CORNERS[0], w, h)
        r_outer = get_xy(landmarks, RIGHT_EYE_CORNERS[1], w, h)

        l_width = (l_inner[0] - l_outer[0]) or 1e-6
        r_width = (r_outer[0] - r_inner[0]) or 1e-6
        l_ratio = (lx - l_outer[0]) / l_width
        r_ratio = (rx - r_inner[0]) / r_width
        gaze_x = (l_ratio + r_ratio) / 2.0
        gaze_y = ((ly - l_outer[1]) + (ry - r_inner[1])) / 2.0 / ((l_width + r_width) / 2)
        return gaze_x, gaze_y, l_ratio, r_ratio

    def get_ear(landmarks, idx_set, w, h):
        pts = [get_xy(landmarks, i, w, h) for i in idx_set]
        return eye_aspect_ratio(pts)

    start_time = time.time()
    phase = "settle"
    phase_start = start_time
    saccade_labels = ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER", "LEFT", "RIGHT", "CENTER"]
    saccade_idx = 0
    saccade_interval = 1.4
    last_cue_change = 0
    aborted = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            canvas = cv2.resize(frame, (W, H))
            sx, sy = W / fw, H / fh

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000) + 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now = time.time()
            t_phase = now - phase_start
            face_found = bool(result.face_landmarks)

            if face_found:
                frames_with_face += 1
                landmarks = result.face_landmarks[0]
                gaze_x, gaze_y, l_ratio, r_ratio = normalized_gaze(landmarks, fw, fh)
                left_eye_activity.append(l_ratio)
                right_eye_activity.append(r_ratio)

                left_ear = get_ear(landmarks, LEFT_EYE_EAR, fw, fh)
                right_ear = get_ear(landmarks, RIGHT_EYE_EAR, fw, fh)
                ear = (left_ear + right_ear) / 2.0
                if ear < EAR_BLINK_THRESHOLD:
                    ear_below_count += 1
                else:
                    if ear_below_count >= EAR_CONSEC_FRAMES:
                        blink_events.append(now)
                    ear_below_count = 0

                for idx in (LEFT_IRIS, RIGHT_IRIS):
                    x, y = get_xy(landmarks, idx, fw, fh)
                    cv2.circle(canvas, (int(x * sx), int(y * sy)), 4, (73, 224, 179), -1)
            else:
                gaze_x, gaze_y = None, None

            if phase == "settle":
                cv2.putText(canvas, "Starting — look at the camera...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 237, 241), 2)
                if t_phase > 1.0:
                    phase, phase_start = "fixation", now

            elif phase == "fixation":
                cx, cy = W // 2, H // 2
                cv2.circle(canvas, (cx, cy), 12, (73, 224, 179), -1)
                cv2.putText(canvas, "FIXATION: hold your gaze on the dot", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 237, 241), 2)
                if face_found and t_phase > 1.0:
                    fixation_samples.append((t_phase, gaze_x, gaze_y))
                if t_phase > 4.0:
                    phase, phase_start = "pursuit", now

            elif phase == "pursuit":
                pe = t_phase
                angle = (pe / 8.0) * math.pi * 2 * 1.5
                tx = W // 2 + int(math.cos(angle) * 220)
                ty = H // 2 + int(math.sin(angle) * 160)
                cv2.circle(canvas, (tx, ty), 12, (73, 224, 179), -1)
                cv2.putText(canvas, "SMOOTH PURSUIT: trace the moving dot", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 237, 241), 2)
                if face_found:
                    pursuit_samples.append((pe, gaze_x, gaze_y, tx / W, ty / H))
                if t_phase > 8.0:
                    phase, phase_start = "saccade", now
                    saccade_idx = 0
                    last_cue_change = now
                    saccade_cues.append({"label": saccade_labels[0], "cue_time": now, "resolved_latency": None})

            elif phase == "saccade":
                if now - last_cue_change > saccade_interval and saccade_idx < len(saccade_labels) - 1:
                    saccade_idx += 1
                    last_cue_change = now
                    saccade_cues.append({"label": saccade_labels[saccade_idx], "cue_time": now, "resolved_latency": None})

                label = saccade_labels[saccade_idx]
                cv2.putText(canvas, f"LOOK {label}", (W // 2 - 140, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (73, 224, 179), 3)
                cv2.putText(canvas, "SACCADE TEST: follow the on-screen cue with your eyes", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (231, 237, 241), 2)

                if face_found:
                    cue = saccade_cues[-1]
                    if cue["resolved_latency"] is None:
                        target_dir = {"LEFT": -1, "RIGHT": 1, "CENTER": 0}[cue["label"]]
                        deviation = (gaze_x - 0.5) * 2
                        if target_dir != 0 and (deviation * target_dir) > 0.25:
                            cue["resolved_latency"] = (now - cue["cue_time"]) * 1000
                        elif target_dir == 0 and abs(deviation) < 0.25:
                            cue["resolved_latency"] = (now - cue["cue_time"]) * 1000

                if t_phase > saccade_interval * len(saccade_labels) + 0.5:
                    phase = "done"

            if not face_found:
                cv2.putText(canvas, "Face not detected - center yourself in frame", (30, H - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 107, 82), 2)

            cv2.putText(canvas, "ESC / Q to cancel", (W - 220, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (116, 136, 153), 1)
            cv2.imshow("NeuroTrace - Eye Test", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                aborted = True
                break
            if phase == "done":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    if aborted:
        return {"valid": False, "reason": "cancelled"}

    # ---- validity guard ----
    if frames_with_face < MIN_FRAMES_WITH_FACE:
        return {"valid": False, "reason": "face not reliably detected - retry with better lighting/framing"}

    # ---- compute metrics ----
    if len(fixation_samples) >= 5:
        xs = np.array([s[1] for s in fixation_samples])
        ys = np.array([s[2] for s in fixation_samples])
        jitter = float(np.std(xs) + np.std(ys))
        fixation_stability = max(0.0, min(100.0, 100 - jitter * 800))
    else:
        fixation_stability = 40.0

    if len(pursuit_samples) >= 10:
        gx = np.array([s[1] for s in pursuit_samples])
        gy = np.array([s[2] for s in pursuit_samples])
        tx = np.array([s[3] for s in pursuit_samples])
        ty = np.array([s[4] for s in pursuit_samples])
        try:
            corr_x = np.corrcoef(gx, tx)[0, 1]
            corr_y = np.corrcoef(gy, ty)[0, 1]
            corr_x = 0.0 if np.isnan(corr_x) else corr_x
            corr_y = 0.0 if np.isnan(corr_y) else corr_y
            pursuit_accuracy = max(0.0, min(100.0, ((abs(corr_x) + abs(corr_y)) / 2) * 100))
        except Exception:
            pursuit_accuracy = 40.0
    else:
        pursuit_accuracy = 40.0

    resolved = [c["resolved_latency"] for c in saccade_cues[1:] if c["resolved_latency"] is not None]
    saccade_latency_ms = float(np.mean(resolved)) if resolved else 700.0

    duration_min = max(1e-6, (time.time() - start_time) / 60.0)
    blink_rate_per_min = len(blink_events) / duration_min

    # per-eye asymmetry (real, calibration-free biomarker)
    asymmetry = 0.0
    if len(left_eye_activity) > 10 and len(right_eye_activity) > 10:
        l_std = np.std(left_eye_activity)
        r_std = np.std(right_eye_activity)
        denom = (l_std + r_std) or 1e-6
        asymmetry = round(float(abs(l_std - r_std) / denom) * 100, 1)

    metrics = {
        "valid": True,
        "subject": subject_name,
        "fixation_stability": round(fixation_stability, 1),
        "pursuit_accuracy": round(pursuit_accuracy, 1),
        "saccade_latency_ms": round(saccade_latency_ms, 1),
        "blink_count": len(blink_events),
        "blink_rate_per_min": round(blink_rate_per_min, 1),
        "interocular_asymmetry_pct": asymmetry,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="subject")
    parser.add_argument("--out", default="data/eye_metrics.json")
    parser.add_argument("--tts", action="store_true", help="speak instructions/countdown aloud")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--tts-rate", type=int, default=170)
    args = parser.parse_args()

    result = run_eye_test(args.name, tts_enabled=args.tts, volume=args.volume, tts_rate=args.tts_rate)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if result is None:
        print("Eye test cancelled - no file written.")
    else:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        if not result.get("valid"):
            print(f"Eye test INVALID: {result.get('reason')}")
        else:
            print(f"Saved eye metrics to {args.out}")
            print(json.dumps(result, indent=2))
