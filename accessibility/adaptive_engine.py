"""
accessibility/adaptive_engine.py

USER PROFILE -> Accessibility Engine -> valid tests -> collected biomarkers
-> AI analysis -> multimodal fusion -> screening result.

The one rule this module exists to enforce: a test that is not applicable
to a profile is never scored, never defaults to a number, and never shows
up as "abnormal." It's simply excluded, and remaining biomarker weights
are renormalized over what's actually available.
"""

from .profile import TEST_PLANS

# base relative weights when ALL FOUR biomarkers are available
BASE_WEIGHTS = {
    "eye": 0.30,
    "speech": 0.25,
    "reaction": 0.20,
    "cognitive": 0.25,
}


def get_test_plan(profile: str) -> dict:
    """Returns {'eye': 'required'|'optional'|'not_applicable', ...} for a profile."""
    return TEST_PLANS.get(profile, TEST_PLANS["standard"])


def resolve_available_biomarkers(profile: str, captured: dict) -> dict:
    """
    captured: dict of biomarker_name -> result dict (or None if skipped/not run)
              each result dict should carry "valid": True/False like the
              existing eye_test.py / speech_test.py outputs.
    Returns: {biomarker: "available" | "not_applicable" | "skipped" | "invalid"}
    """
    plan = get_test_plan(profile)
    status = {}
    for biomarker, requirement in plan.items():
        if requirement == "not_applicable":
            status[biomarker] = "not_applicable"
            continue
        result = captured.get(biomarker)
        if result is None:
            status[biomarker] = "skipped" if requirement == "optional" else "not_applicable"
        elif not result.get("valid", False):
            status[biomarker] = "invalid"
        else:
            status[biomarker] = "available"
    return status


def normalized_weights(available_biomarkers: list) -> dict:
    """Renormalize BASE_WEIGHTS over only the biomarkers actually available."""
    total = sum(BASE_WEIGHTS[b] for b in available_biomarkers if b in BASE_WEIGHTS)
    if total <= 0:
        return {}
    return {b: BASE_WEIGHTS[b] / total for b in available_biomarkers if b in BASE_WEIGHTS}
