"""
ml/predict.py

Loads the model trained by train_model.py (on SYNTHETIC data) and scores
a real captured feature set. Only usable when all 12 underlying features
are present (i.e. eye + speech + reaction + cognitive all captured) -
for accessibility profiles with Not Applicable biomarkers, the rule-based
fusion.py path is used instead, since the model was trained on the full
feature set and has no honest way to handle missing biomarkers.

This score is shown in the UI as a clearly-labeled secondary comparison,
never as the primary screening result.
"""

import os

import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))  # works whether run standalone or imported as ml.predict
from generate_synthetic_data import FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            return None, None
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def model_available() -> bool:
    model, scaler = _load()
    return model is not None


def build_feature_vector(captured: dict):
    """captured: {'eye':..., 'speech':..., 'reaction':..., 'cognitive':...} — all must be present & valid."""
    eye, speech, reaction, cognitive = captured["eye"], captured["speech"], captured["reaction"], captured["cognitive"]
    if not all([eye, speech, reaction, cognitive]):
        return None
    row = {
        "fixation_stability": eye.get("fixation_stability"),
        "pursuit_accuracy": eye.get("pursuit_accuracy"),
        "saccade_latency_ms": eye.get("saccade_latency_ms"),
        "interocular_asymmetry_pct": eye.get("interocular_asymmetry_pct", 0),
        "voiced_ratio": speech.get("voiced_ratio"),
        "avg_pause_ms": speech.get("avg_pause_ms"),
        "pitch_stability": speech.get("pitch_stability"),
        "clarity": speech.get("clarity"),
        "mean_reaction_ms": reaction.get("mean_reaction_ms"),
        "consistency_score": reaction.get("consistency_score"),
        "missed_trials": reaction.get("missed_trials"),
        "cognitive_score": cognitive.get("cognitive_score"),
    }
    if any(v is None for v in row.values()):
        return None
    return np.array([[row[f] for f in FEATURES]])


def predict_risk(captured: dict):
    """Returns (risk_probability_pct, label) or (None, reason) if unavailable."""
    model, scaler = _load()
    if model is None:
        return None, "no trained model found — run ml/train_model.py first"
    x = build_feature_vector(captured)
    if x is None:
        return None, "requires all four biomarkers (eye, speech, reaction, cognitive) to be present"
    x_s = scaler.transform(x)
    proba_atypical = float(model.predict_proba(x_s)[0, 1])
    label = "atypical_pattern" if proba_atypical >= 0.5 else "typical_pattern"
    return round(proba_atypical * 100, 1), label
