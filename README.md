# NeuroTrace — local prototype with Accessibility Mode

## ⚠️ Python version
Still applies: use **Python 3.11 or 3.12** in a dedicated venv. `mediapipe`
has no Python 3.14 wheels yet.

```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Linux only:** `pyttsx3` needs an OS speech engine — `sudo apt install espeak`.
On Windows/Mac it uses the built-in OS TTS voice, no extra install.

## What's new: Accessibility Mode
One system, four adaptive pathways, chosen up front and never inferred:

| Profile | Eye | Speech | Reaction | Cognitive |
|---|---|---|---|---|
| Standard | required | required | required (visual) | required (visual) |
| Deaf / HoH | required | optional | required (visual) | required (visual) |
| Blind / Visually Impaired | **not applicable** | optional | required (**audio**) | required (**audio**, digit-recall) |
| Speech-Impaired | required | **not applicable** | required (visual) | required (visual) |

The core rule, enforced in `fusion.py`: a **not_applicable** or **skipped**
biomarker is excluded entirely from scoring — never defaulted to a number,
never shown as "abnormal." The fusion weights are renormalized across
whatever biomarkers were actually available for that person.

### New files
- `accessibility/profile.py` — profile definitions, consent text, emergency text
- `accessibility/adaptive_engine.py` — resolves which tests are valid + renormalizes fusion weights
- `reaction_test.py` — visual or audio (tone-based) reaction-time test
- `cognitive_test.py` — visual (shape-selection + word-recognition) or audio (TTS digit-recall) cognitive test
- `app.py` — now includes profile selection, settings, consent/safety, adaptive routing, demo mode, accessible dashboard

### Honest limitations — please read before a demo
1. **No full screen-reader/ARIA support.** Streamlit is a fixed React
   shell with very limited accessibility hooks. What's real here: high
   contrast, large text, reduced motion, keyboard-navigable widgets, no
   color-only signaling. Full WCAG/assistive-tech compliance would need a
   different frontend framework — don't claim more than this in front of
   judges.
2. **The audio (blind-mode) cognitive/reaction tests still open a
   graphical window that needs OS focus to receive key presses.** That's
   not a fully vision-free, mouse-free input path — a real one would need
   an OS-level key hook or a proper accessible GUI toolkit. It's flagged
   in the test's own JSON output too (`"note"` field).
3. **Sign Language Assistance is a UI placeholder only** (not built into
   this Python version yet — see below) — there is no sign-language
   recognition or generation model in this prototype.
4. **Speech rate is still a segment-count proxy**, not true words-per-minute.
5. **No screen-gaze calibration** on the eye test — relative eye-movement
   biomarkers only (see earlier README notes, unchanged).

### Demo Mode
From the home screen → "🎬 Demo Mode (judges)" → pick one of 4 canned
scenarios (Standard / Deaf / Blind / Speech-Impaired). Loads pre-recorded
sample metrics straight into the results dashboard, clearly labeled
**DEMO / SIMULATED DATA**, so you can show all four pathways without four
different people in the room.

### Baselines
Baselines are now keyed by **name + profile**, since a Blind-profile run
and a Standard-profile run for the same person don't share the same valid
biomarkers. Comparison against baseline only uses biomarkers common to
both runs.

### Save Report
Results screen has a real "⬇ Save Report (.json)" download button — the
full captured metrics + fusion result for that run, saved locally.

## Not yet built (say the word)
- Sign Language Assistance module (even as a working placeholder video/UI area)
- Screen-gaze calibration
- Head-pose compensation
- Real word-level speech rate via offline Whisper
- A proper accessible (non-Streamlit) frontend for true screen-reader support

## Files
- `app.py`, `fusion.py`, `eye_test.py`, `speech_test.py`, `reaction_test.py`, `cognitive_test.py`
- `accessibility/profile.py`, `accessibility/adaptive_engine.py`
- `data/` — created automatically: baselines.json, history.csv, per-subject metric JSONs

## UI/UX upgrade
- Step-progress bar across the assessment flow (Profile → Settings → Consent → Register → Plan → tests → Results)
- Sidebar with quick nav (Home/History), current subject/profile, and ML-model load status
- Card/badge styling consistent with high-contrast mode, a live bar chart of biomarker scores on the results screen
- "DEMO / SIMULATED DATA" badge styling made visually distinct (red) from the standard prototype badge (amber)

## ML pipeline (train/val/test, cross-validation, evaluation) — on SYNTHETIC data
**Read this before presenting it:** there is no real concussion-labeled dataset
behind this project. "Improve accuracy" isn't meaningful yet because there's no
real ground truth to be accurate against. What's here instead is a properly
built ML pipeline — stratified train/val/test split, 5-fold cross-validation,
held-out test evaluation (accuracy/precision/recall/F1/ROC-AUC), a confusion
matrix + ROC plot — run against a synthetic dataset (`ml/generate_synthetic_data.py`)
whose class separation was hand-designed by me based on plausible biomarker
direction, not real patient data. It's genuinely good to demonstrate ML
engineering maturity in a hackathon; it is not a clinical accuracy claim,
and every screen that shows its output says so.

```bash
python ml/generate_synthetic_data.py     # writes data/synthetic_dataset.csv
python ml/train_model.py                 # trains, evaluates, saves model + metrics.json + evaluation.png
streamlit run app.py                     # "🤖 ML Model Info" on the home screen shows the report
```

The trained model only scores a real assessment when **all four** biomarkers
(eye, speech, reaction, cognitive) were captured — i.e. Standard/Unspecified
profile, nothing skipped — since it was trained on that full feature set.
When available, it appears on the results screen as a clearly separate
"ML model comparison," never replacing the accessibility-aware rule-based
score, which remains the primary result for every profile.

### What would actually move the needle on real accuracy
Since it's worth saying plainly: the only way to make this system more
accurate in a real sense is a real labeled dataset — actual assessments from
people with confirmed concussion and matched controls, collected with
consent under an IRB-approved protocol, ideally alongside a clinical gold
standard (e.g. SCAT6, neuropsych battery). That's a research undertaking,
not a code change — worth knowing if this project continues past the
hackathon.

## Blind mode: Speech is now genuinely spoken, not typed
Earlier versions had blind-profile users type digits back on a keyboard for
the cognitive test, and separately read an on-screen sentence for the
optional speech test — neither is realistic if someone can't see the
screen. Fixed:

- **Cognitive + Speech are now one combined test** for the Blind profile.
  TTS speaks a number sequence, the person says it back out loud, and the
  response is transcribed (`SpeechRecognition` + Google's free Web Speech
  API — **needs an internet connection**, no API key required) and scored
  for digit-recall accuracy.
- **That same spoken audio is reused to compute real speech biomarkers**
  (voiced ratio, pause duration, pitch stability, clarity) via the same
  librosa pipeline `speech_test.py` uses — so Blind-profile users get a
  real Speech score with no separate, unreadable test required.
- The standalone Speech screen is skipped entirely for the Blind profile.

Still an honest limitation: if there's no internet at runtime, transcription
fails and the cognitive/speech scores for that round are marked accordingly
rather than guessed — check your connection before a live demo.

## Every native test now opens with instructions + a real 3-2-1 countdown
Previously tests jumped straight into trials. `test_kit.py` now provides a
shared opening sequence (used by `eye_test.py`, `reaction_test.py`,
`cognitive_test.py`): on-screen instructions (read aloud too, in Blind
mode) for a few seconds, then a countdown that's visible, audibly beeped,
and (in Blind mode) spoken - not just shown on screen. The reaction-test
tone is also louder/longer than before, and both the tone and TTS speech
are scaled by the new **Volume** and **Speech speed** sliders in Settings.
