"""
fusion.py

Combines whichever biomarkers were actually collected into a single
screening score. Only biomarkers marked "available" by the accessibility
engine are scored - "not_applicable" and "skipped" biomarkers are excluded
entirely and never default to a number, never penalize the score, and
never appear as abnormal.
"""

from accessibility.adaptive_engine import resolve_available_biomarkers, normalized_weights


def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


def score_eye(m: dict) -> float:
    fixation = m.get("fixation_stability", 50)
    pursuit = m.get("pursuit_accuracy", 50)
    latency_ms = m.get("saccade_latency_ms", 500)
    asymmetry = m.get("interocular_asymmetry_pct", 0)
    latency_score = clamp(100 - (latency_ms / 6))
    asymmetry_penalty = clamp(asymmetry) * 0.3
    return round(clamp(
        fixation * 0.28 + pursuit * 0.42 + latency_score * 0.20 + clamp(100 - asymmetry_penalty) * 0.10
    ), 1)


def score_speech(m: dict) -> float:
    voiced_ratio = m.get("voiced_ratio", 0.5)
    avg_pause_ms = m.get("avg_pause_ms", 400)
    pitch_stability = m.get("pitch_stability", 50)
    clarity = m.get("clarity", 50)
    activity_score = clamp(voiced_ratio * 140)
    pause_score = clamp(100 - (avg_pause_ms / 8))
    return round(clamp(
        activity_score * 0.30 + pause_score * 0.25 + pitch_stability * 0.20 + clarity * 0.25
    ), 1)


def score_reaction(m: dict) -> float:
    mean_rt = m.get("mean_reaction_ms", 400)
    consistency = m.get("consistency_score", 50)
    missed = m.get("missed_trials", 0)
    total = m.get("total_trials", 6)
    speed_score = clamp(100 - ((mean_rt - 200) / 6))
    miss_penalty = clamp((missed / max(total, 1)) * 100)
    return round(clamp(
        speed_score * 0.55 + consistency * 0.30 + clamp(100 - miss_penalty) * 0.15
    ), 1)


def score_cognitive(m: dict) -> float:
    # both the visual and audio cognitive_test.py variants already compute
    # and return this directly, weighting their own accuracy/speed mix
    return round(clamp(m.get("cognitive_score", 50)), 1)


SCORERS = {
    "eye": score_eye,
    "speech": score_speech,
    "reaction": score_reaction,
    "cognitive": score_cognitive,
}


def fuse_adaptive(profile: str, captured: dict) -> dict:
    """
    profile: one of the accessibility.profile constants
    captured: {"eye": result_or_None, "speech": ..., "reaction": ..., "cognitive": ...}
              each result dict must have "valid": True/False (as produced by
              eye_test.py / speech_test.py / reaction_test.py / cognitive_test.py)
    """
    status = resolve_available_biomarkers(profile, captured)
    available = [b for b, s in status.items() if s == "available"]

    if not available:
        return {"valid": False, "status": status, "reason": "no valid biomarkers were captured"}

    scores = {b: SCORERS[b](captured[b]) for b in available}
    weights = normalized_weights(available)
    performance = sum(scores[b] * weights[b] for b in available)
    risk_score = round(clamp(100 - performance), 1)

    if risk_score <= 30:
        band, color = "LOWER CONCERN", "green"
    elif risk_score <= 60:
        band, color = "NEEDS ATTENTION", "amber"
    else:
        band, color = "HIGHER CONCERN", "red"

    return {
        "valid": True,
        "status": status,          # per-biomarker: available / not_applicable / skipped / invalid
        "scores": scores,          # per-biomarker score, only for "available" ones
        "weights": weights,        # renormalized weights actually used
        "risk_score": risk_score,
        "band": band,
        "color": color,
    }
