"""
ml/generate_synthetic_data.py

Generates a SYNTHETIC labeled dataset for demonstrating a real train/val/
test ML pipeline. This is NOT real patient data - there is no clinical
dataset behind this project. Every value here is drawn from a distribution
I picked based on the plausible direction of each biomarker (e.g. slower
reaction time and lower tracking accuracy skew toward the "atypical"
class), with deliberate class overlap so it isn't trivially separable.

Label meaning:
  0 = "typical_pattern"   (biomarkers broadly in the expected range)
  1 = "atypical_pattern"  (biomarkers broadly deviate from expected range)

This label is a stand-in for "screening flag", not a concussion diagnosis
label - there is no real-world ground truth being modeled here.

Run:
    python ml/generate_synthetic_data.py --n 3000 --out data/synthetic_dataset.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

FEATURES = [
    "fixation_stability", "pursuit_accuracy", "saccade_latency_ms", "interocular_asymmetry_pct",
    "voiced_ratio", "avg_pause_ms", "pitch_stability", "clarity",
    "mean_reaction_ms", "consistency_score", "missed_trials",
    "cognitive_score",
]

# (mean, std) per class, per feature - picked to reflect plausible direction only
PARAMS = {
    "fixation_stability":        {0: (85, 8),  1: (60, 15)},
    "pursuit_accuracy":          {0: (82, 9),  1: (55, 16)},
    "saccade_latency_ms":        {0: (280, 50), 1: (480, 110)},
    "interocular_asymmetry_pct": {0: (4, 3),   1: (14, 8)},
    "voiced_ratio":              {0: (0.60, 0.10), 1: (0.42, 0.15)},
    "avg_pause_ms":              {0: (230, 60), 1: (420, 130)},
    "pitch_stability":           {0: (78, 10), 1: (55, 16)},
    "clarity":                   {0: (76, 10), 1: (52, 16)},
    "mean_reaction_ms":          {0: (320, 45), 1: (480, 100)},
    "consistency_score":         {0: (80, 9),  1: (55, 15)},
    "missed_trials":             {0: (0.3, 0.6), 1: (1.6, 1.2)},
    "cognitive_score":           {0: (82, 8),  1: (58, 15)},
}

CLIP = {
    "fixation_stability": (0, 100), "pursuit_accuracy": (0, 100),
    "saccade_latency_ms": (100, 1200), "interocular_asymmetry_pct": (0, 60),
    "voiced_ratio": (0, 1), "avg_pause_ms": (50, 1500),
    "pitch_stability": (0, 100), "clarity": (0, 100),
    "mean_reaction_ms": (150, 1200), "consistency_score": (0, 100),
    "missed_trials": (0, 6), "cognitive_score": (0, 100),
}


def generate(n=3000, positive_ratio=0.35, seed=42):
    rng = np.random.default_rng(seed)
    n_pos = int(n * positive_ratio)
    n_neg = n - n_pos
    labels = np.array([0] * n_neg + [1] * n_pos)
    rng.shuffle(labels)

    data = {}
    for feat in FEATURES:
        vals = np.empty(n)
        for cls in (0, 1):
            mask = labels == cls
            mean, std = PARAMS[feat][cls]
            vals[mask] = rng.normal(mean, std, size=mask.sum())
        lo, hi = CLIP[feat]
        vals = np.clip(vals, lo, hi)
        if feat == "missed_trials":
            vals = np.round(vals)
        data[feat] = vals

    df = pd.DataFrame(data)
    df["label"] = labels
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out", default="data/synthetic_dataset.csv")
    args = parser.parse_args()

    df = generate(args.n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"⚠️  SYNTHETIC DATA — NOT REAL PATIENT DATA — {len(df)} rows saved to {args.out}")
    print(df["label"].value_counts(normalize=True).rename("class balance"))
