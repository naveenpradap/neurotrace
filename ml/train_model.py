"""
ml/train_model.py

A real train/val/test ML pipeline (stratified split, k-fold cross-
validation, held-out test evaluation, saved metrics + plots) - run on the
SYNTHETIC dataset from generate_synthetic_data.py. This demonstrates the
ML engineering practice; it does NOT establish clinical accuracy, because
the labels aren't real. Every printed/saved metric is tagged accordingly.

Run:
    python ml/generate_synthetic_data.py
    python ml/train_model.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from generate_synthetic_data import FEATURES

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "model")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found — run generate_synthetic_data.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df["label"].values

    # 70% train, 15% val (used only for model selection via CV), 15% held-out test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_trainval)
    X_trainval_s = scaler.transform(X_trainval)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_report = {}
    best_name, best_f1 = None, -1

    print("⚠️  All metrics below are on SYNTHETIC data — not clinically meaningful.\n")
    print("=== 5-fold cross-validation on training set ===")
    for name, model in candidates.items():
        scores = cross_validate(model, X_trainval_s, y_trainval, cv=cv,
                                 scoring=["accuracy", "f1", "roc_auc"])
        mean_f1 = float(np.mean(scores["test_f1"]))
        cv_report[name] = {
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_f1_mean": mean_f1,
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        }
        print(f"{name}: acc={cv_report[name]['cv_accuracy_mean']:.3f}  "
              f"f1={mean_f1:.3f}  roc_auc={cv_report[name]['cv_roc_auc_mean']:.3f}")
        if mean_f1 > best_f1:
            best_f1, best_name = mean_f1, name

    print(f"\nSelected model (best CV F1): {best_name}")
    best_model = candidates[best_name]
    best_model.fit(X_trainval_s, y_trainval)

    # ---- held-out test evaluation ----
    y_pred = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]

    test_report = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    print("\n=== Held-out test set ===")
    for k, v in test_report.items():
        print(f"{k}: {v:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["typical", "atypical"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix (synthetic test set)")
    axes[1].plot(fpr, tpr, label=f"ROC-AUC={test_report['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (synthetic test set)")
    axes[1].legend()
    fig.suptitle("SYNTHETIC DATA — not a clinical validation", fontsize=10, color="crimson")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "evaluation.png"), dpi=140)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    metrics = {
        "disclaimer": "Trained and evaluated on SYNTHETIC data. Not a clinical validation.",
        "model_selected": best_name,
        "features": FEATURES,
        "cv_report": cv_report,
        "test_report": test_report,
        "n_train": len(X_trainval),
        "n_test": len(X_test),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, scaler, metrics.json, and evaluation.png to {MODEL_DIR}")


if __name__ == "__main__":
    main()
