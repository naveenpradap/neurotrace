"""
accessibility/profile.py

Defines the accessibility profiles and the per-profile test plan - i.e.
which biomarkers are collected, which are skipped as "Not Applicable",
and which are optional. Selecting a profile is never used to infer a
medical condition; it only changes which tests the person is offered.
"""

STANDARD = "standard"
DEAF = "deaf"
BLIND = "blind"
SPEECH_IMPAIRED = "speech_impaired"
UNSPECIFIED = "unspecified"

PROFILE_LABELS = {
    STANDARD: "🧑 Standard Assessment",
    DEAF: "🦻 Deaf / Hard of Hearing",
    BLIND: "👁️ Blind / Visually Impaired",
    SPEECH_IMPAIRED: "🗣️ Speech-Impaired / Non-Speaking",
    UNSPECIFIED: "Prefer not to say",
}

# valid values per biomarker: "required" | "optional" | "not_applicable"
TEST_PLANS = {
    STANDARD: {
        "eye": "required",
        "speech": "required",
        "reaction": "required",
        "cognitive": "required",
    },
    DEAF: {
        # eye/reaction/cognitive are entirely visual already - no change needed.
        # speech is left optional: many Deaf/HoH people speak; it is never
        # required and a skip is never scored as abnormal.
        "eye": "required",
        "speech": "optional",
        "reaction": "required",
        "cognitive": "required",
    },
    BLIND: {
        # standard eye-tracking assumes the person can see and follow a
        # target - not a valid test here, so it's dropped entirely rather
        # than attempted and scored.
        "eye": "not_applicable",
        "speech": "optional",
        "reaction": "required",       # auditory reaction-time variant
        "cognitive": "required",      # audio digit-recall variant
    },
    SPEECH_IMPAIRED: {
        "eye": "required",
        "speech": "not_applicable",   # never scored, never treated as abnormal
        "reaction": "required",
        "cognitive": "required",
    },
    UNSPECIFIED: {
      # Every test is optional when the participant prefers not to say.
      "eye": "optional",
      "speech": "optional",
      "reaction": "optional",
      "cognitive": "optional",
    },
}

CONSENT_TEXT = """
**Before you begin**

- This is an unvalidated hackathon prototype, not a medical device. It does
  not diagnose concussion.
- Your camera/microphone are used only during an active test. Raw video is
  never saved. Raw audio is saved only if you explicitly enable it - off by
  default.
- Selecting an accessibility profile only changes which tests are offered.
  It is never used to infer or record a medical condition, and is stored
  only as a label for which tests to run.
- You can skip any test that isn't suitable for you. A skipped test is
  marked **Not Applicable** and is never counted as an abnormal result.
"""

EMERGENCY_TEXT = """
**Seek emergency care immediately if, after a head injury, you or someone
you're with experiences:** repeated vomiting, worsening headache, seizures,
slurred speech, weakness/numbness, unequal pupils, a brief loss of
consciousness, increasing confusion, or difficulty staying awake. In the
US, call 911 (or your local emergency number). This prototype is not a
substitute for emergency medical evaluation.
"""

RESULT_DISCLAIMER = (
    "This prototype provides an AI-assisted screening result and does not "
    "diagnose concussion. A qualified healthcare professional should "
    "perform clinical evaluation, especially after a suspected head injury."
)
