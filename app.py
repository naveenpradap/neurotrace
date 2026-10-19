"""
app.py — NeuroTrace prototype UI (Streamlit), accessibility-enabled.

Flow: Home -> (Demo Mode | Start Screening -> Profile -> Settings ->
Consent/Safety -> Register -> Adaptive Instructions -> [Eye]/[Speech]/
Reaction/Cognitive tests, adapted to profile -> Fusion -> Results) ->
History.

Honest limitation up front: this is a Streamlit + subprocess/OpenCV
architecture. Streamlit itself has very limited native screen-reader/ARIA
support (it's a fixed React shell), so "screen reader compatibility" here
means: large text, high contrast, keyboard-navigable Streamlit widgets,
and no color-only signaling - not full WCAG/AT compliance. Said plainly
in the Settings screen too.
"""

import json
import os
import subprocess
import sys
import time

import streamlit as st
import pandas as pd

from fusion import fuse_adaptive
from accessibility.profile import (
    STANDARD, DEAF, BLIND, SPEECH_IMPAIRED, UNSPECIFIED,
    PROFILE_LABELS, TEST_PLANS, CONSENT_TEXT, EMERGENCY_TEXT, RESULT_DISCLAIMER,
)
from accessibility.adaptive_engine import get_test_plan
from accessibility.deaf_mode import (
    VISUAL_INSTRUCTIONS,
    render_sign_language_panel,
    render_visual_alert,
)

try:
    from ml.predict import predict_risk, model_available
    ML_IMPORT_OK = True
except Exception:
    ML_IMPORT_OK = False
    def model_available(): return False
    def predict_risk(captured): return None, "ml dependencies not installed"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASELINE_PATH = os.path.join(DATA_DIR, "baselines.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="NeuroTrace", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


def slug(name):
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-") or "anon"


# ---------------- session defaults ----------------
DEFAULTS = {
    "screen": "home",
    "subject": {"name": "", "age": ""},
    "profile": STANDARD,
    "settings": {
        "high_contrast": False,
        "large_text": False,
        "reduce_motion": False,
        "tts_on": True,
        "volume": 1.0,
        "tts_rate": 170,
        "extended_time": 1.0,
        "consent_ack": False,
    },
    "captured": {"eye": None, "speech": None, "reaction": None, "cognitive": None},
    "fusion_result": None,
    "is_demo": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def goto(screen):
    st.session_state.screen = screen
    st.rerun()


def render_test_notice(message, kind):
    if st.session_state.profile == DEAF:
        render_visual_alert(message, kind)
    elif kind == "error":
        st.error(message)
    else:
        st.success(message)


# ---------------- CSS (accessibility-aware) ----------------
s = st.session_state.settings
base_bg = "#ffffff" if s["high_contrast"] else "#0a0e14"
base_text = "#000000" if s["high_contrast"] else "#e7edf1"
accent = "#008060" if s["high_contrast"] else "#49e0b3"
font_scale = "1.35" if s["large_text"] else "1.0"

panel_bg = "#f2f2f2" if s["high_contrast"] else "#141c26"
border_col = "#c8c8c8" if s["high_contrast"] else "#22303c"
muted = "#333333" if s["high_contrast"] else "#748899"

st.markdown(f"""
<style>
.stApp {{ background-color: {base_bg}; color: {base_text}; }}
html, body, [class*="css"] {{ font-size: calc(1rem * {font_scale}); }}
h1, h2, h3 {{ font-family: 'JetBrains Mono', monospace; letter-spacing: 0; }}
h1 {{ font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: 1.02; margin-bottom: 0.4rem; }}
h2 {{ margin-top: 1.5rem; }}
[data-testid="stSidebar"] {{ background: {panel_bg}; border-right: 1px solid {border_col}; }}
[data-testid="stSidebar"] .stButton>button {{ background: transparent; border: 1px solid transparent; text-align:left; }}
[data-testid="stSidebar"] .stButton>button:hover {{ border-color:{border_col}; background:rgba(73,224,179,0.08); }}
.eyebrow {{ color:{accent}; font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700;
            letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.8rem; }}
.hero {{ background: linear-gradient(135deg, {panel_bg} 0%, rgba(73,224,179,0.09) 100%);
         border:1px solid {border_col}; border-radius:16px; padding:clamp(24px, 5vw, 56px);
         margin:1rem 0 1.5rem; position:relative; overflow:hidden; }}
.hero:after {{ content:' '; position:absolute; width:220px; height:220px; right:-80px; top:-90px;
              border:1px solid rgba(73,224,179,0.28); border-radius:50%; box-shadow:0 0 0 24px rgba(73,224,179,0.04), 0 0 0 48px rgba(73,224,179,0.03); }}
.hero-copy {{ max-width:720px; position:relative; z-index:1; }}
.hero-copy p {{ color:{muted}; font-size:1.08rem; line-height:1.7; max-width:650px; }}
.section-label {{ color:{muted}; font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.12em;
                 text-transform:uppercase; margin:1.7rem 0 0.7rem; }}
.feature-card {{ background:{panel_bg}; border:1px solid {border_col}; border-radius:10px; padding:18px; min-height:145px; }}
.feature-card strong {{ display:block; font-size:1.03rem; margin:7px 0; }}
.feature-card span {{ color:{muted}; font-size:0.9rem; line-height:1.5; }}
.profile-panel {{ background:{panel_bg}; border:1px solid {border_col}; border-radius:12px; padding:22px 24px; }}
.profile-panel h3 {{ margin:0 0 8px; font-size:1.15rem; }}
.profile-panel p {{ color:{muted}; line-height:1.6; margin:0; }}
.profile-note {{ border-left:3px solid {accent}; padding:10px 14px; margin-top:18px; color:{muted};
                 background:rgba(73,224,179,0.06); font-size:0.9rem; line-height:1.5; }}
.selection-summary {{ color:{accent}; font-family:'JetBrains Mono',monospace; font-size:0.78rem;
                     letter-spacing:0.08em; text-transform:uppercase; margin:8px 0 16px; }}
.stButton>button {{ min-height: 3em; font-size: 1.05em; border-radius: 8px; transition: transform .08s, border-color .15s, background .15s; }}
.stButton>button:hover {{ border-color:{accent}; }}
.badge {{ display:inline-block; font-size:12px; color:#f2b84b; border:1px solid rgba(242,184,75,0.35);
         background:rgba(242,184,75,0.08); padding:3px 10px; border-radius:20px; margin-bottom:8px; }}
.badge-demo {{ display:inline-block; font-size:12px; color:#ff6b52; border:1px solid rgba(255,107,82,0.4);
         background:rgba(255,107,82,0.1); padding:3px 10px; border-radius:20px; margin-bottom:8px; }}
.na-tag {{ color:{muted}; font-style:italic; }}
.avail-tag {{ color:{accent}; font-weight:700; }}
.stButton>button:active {{ transform: scale(0.98); }}
div[data-testid="stMetric"] {{ background:{panel_bg}; border:1px solid {border_col}; padding:12px 14px; border-radius:10px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {border_col}; border-radius:10px; overflow:hidden; }}
div[data-testid="stExpander"] {{ background-color: {panel_bg}; border:1px solid {border_col}; border-radius:8px; }}

/* step progress bar */
.steps-wrap {{ display:flex; gap:4px; margin-bottom:22px; }}
.step-seg {{ flex:1; height:5px; border-radius:3px; background:{border_col}; }}
.step-seg.done {{ background:{accent}; }}
.step-caption {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:{muted};
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px; }}

/* card container */
.nt-card {{ background:{panel_bg}; border:1px solid {border_col}; border-radius:10px; padding:18px 20px; margin-bottom:14px; }}

/* biomarker status rows */
.bm-row {{ display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid {border_col}; }}
.bm-name {{ font-weight:600; }}
.bm-available {{ color:{accent}; font-family:'JetBrains Mono',monospace; font-size:13px; }}
.bm-na {{ color:{muted}; font-style:italic; font-family:'JetBrains Mono',monospace; font-size:13px; }}
.bm-invalid {{ color:#ff6b52; font-family:'JetBrains Mono',monospace; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

FLOW_STEPS = ["profile", "settings", "consent", "register", "plan",
              "run_eye", "run_speech", "run_reaction", "run_cognitive", "results"]


def render_step_bar(current_screen):
    """Thin segmented progress bar shown across the assessment flow (not on home/demo/history)."""
    if current_screen not in FLOW_STEPS:
        return
    idx = FLOW_STEPS.index(current_screen)
    segs = "".join(
        f'<div class="step-seg {"done" if i <= idx else ""}"></div>' for i in range(len(FLOW_STEPS))
    )
    st.markdown(f'<div class="step-caption">Step {idx+1} of {len(FLOW_STEPS)}</div>'
                f'<div class="steps-wrap">{segs}</div>', unsafe_allow_html=True)


def load_baselines():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {}


def save_baseline(key, profile, fusion_result, captured):
    baselines = load_baselines()
    baselines[key] = {
        "profile": profile,
        "fusion": fusion_result,
        "captured": captured,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baselines, f, indent=2)


def append_history(name, profile, fusion_result, is_demo=False):
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "profile": profile,
        "risk_score": fusion_result["risk_score"],
        "band": fusion_result["band"],
        "demo": is_demo,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORY_PATH):
        df_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_PATH, index=False)


def accessibility_args():
    """Common --tts/--volume/--tts-rate flags, shared by eye/reaction/cognitive test launches."""
    args = ["--volume", str(s["volume"]), "--tts-rate", str(s["tts_rate"])]
    if st.session_state.profile == BLIND and s["tts_on"]:
        args.append("--tts")
    return args


def run_subprocess_test(script_name, out_path, extra_args=None):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), script_name),
           "--name", st.session_state.subject["name"], "--out", out_path]
    if extra_args:
        cmd += extra_args
    with st.spinner(f"Running {script_name} — a native window will open. Complete it there, then return here."):
        subprocess.run(cmd, check=False)
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return None


DEMO_SCENARIOS = {
    "Demo 1 — Standard User": {
        "profile": STANDARD,
        "captured": {
            "eye": {"valid": True, "fixation_stability": 88, "pursuit_accuracy": 84, "saccade_latency_ms": 260, "interocular_asymmetry_pct": 3},
            "speech": {"valid": True, "voiced_ratio": 0.62, "avg_pause_ms": 220, "pitch_stability": 78, "clarity": 80},
            "reaction": {"valid": True, "mean_reaction_ms": 310, "consistency_score": 82, "missed_trials": 0, "total_trials": 6},
            "cognitive": {"valid": True, "cognitive_score": 86},
        },
    },
    "Demo 2 — Deaf User": {
        "profile": DEAF,
        "captured": {
            "eye": {"valid": True, "fixation_stability": 80, "pursuit_accuracy": 76, "saccade_latency_ms": 300, "interocular_asymmetry_pct": 5},
            "speech": None,  # optional, skipped
            "reaction": {"valid": True, "mean_reaction_ms": 340, "consistency_score": 75, "missed_trials": 0, "total_trials": 6},
            "cognitive": {"valid": True, "cognitive_score": 79},
        },
    },
    "Demo 3 — Blind User": {
        "profile": BLIND,
        "captured": {
            "eye": None,  # not applicable
            "speech": None,  # optional, skipped
            "reaction": {"valid": True, "mode": "audio", "mean_reaction_ms": 380, "consistency_score": 70, "missed_trials": 1, "total_trials": 6},
            "cognitive": {"valid": True, "mode": "audio", "digit_recall_accuracy": 66.7, "cognitive_score": 68},
        },
    },
    "Demo 4 — Speech-Impaired User": {
        "profile": SPEECH_IMPAIRED,
        "captured": {
            "eye": {"valid": True, "fixation_stability": 70, "pursuit_accuracy": 62, "saccade_latency_ms": 420, "interocular_asymmetry_pct": 9},
            "speech": None,  # not applicable
            "reaction": {"valid": True, "mean_reaction_ms": 430, "consistency_score": 60, "missed_trials": 1, "total_trials": 6},
            "cognitive": {"valid": True, "cognitive_score": 58},
        },
    },
}


render_step_bar(st.session_state.screen)

with st.sidebar:
    st.markdown("<div class='eyebrow'>🧠 NeuroTrace</div>", unsafe_allow_html=True)
    st.caption("Accessible cognitive screening prototype")
    if st.session_state.subject.get("name"):
        st.markdown(f"**{st.session_state.subject['name']}**")
        st.caption(PROFILE_LABELS.get(st.session_state.profile, "—"))
        st.divider()
    if st.button("🏠 Home", use_container_width=True):
        goto("home")
    if st.button("📊 History", use_container_width=True):
        goto("history")
    st.divider()
    model_state = "Ready" if model_available() else "Not trained"
    st.caption(f"🤖 ML model · {model_state}")
    st.caption("Rule-based screening remains primary.")

# ==================== HOME ====================
if st.session_state.screen == "home":
    st.markdown('<span class="badge">PROTOTYPE · NOT A MEDICAL DEVICE</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero"><div class="hero-copy">
      <div class="eyebrow">Adaptive screening, designed around the person</div>
      <h1>Understand the signal.<br>Respect the individual.</h1>
      <p>NeuroTrace brings together eye movement, speech, reaction time, and cognitive performance into one explainable screening view. Every pathway adapts to the accessibility profile selected by the participant.</p>
    </div></div>
    """, unsafe_allow_html=True)
    primary, secondary = st.columns([1, 1], gap="small")
    if primary.button("▶  Start Screening", type="primary", use_container_width=True):
        goto("profile")
    if secondary.button("🎬  Explore Demo Scenarios", use_container_width=True):
        goto("demo")

    st.markdown('<div class="section-label">What the experience includes</div>', unsafe_allow_html=True)
    cards = st.columns(3, gap="small")
    cards[0].markdown('<div class="feature-card"><div style="font-size:1.5rem">♿</div><strong>Accessibility first</strong><span>Choose a pathway for Standard, Deaf/HoH, Blind/VI, or Speech-Impaired participants.</span></div>', unsafe_allow_html=True)
    cards[1].markdown('<div class="feature-card"><div style="font-size:1.5rem">◌</div><strong>Explainable results</strong><span>See which biomarkers were available, skipped, or not applicable to the assessment.</span></div>', unsafe_allow_html=True)
    cards[2].markdown('<div class="feature-card"><div style="font-size:1.5rem">↗</div><strong>Built for follow-up</strong><span>Save a report, compare against a same-profile baseline, or review previous sessions.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Quick access</div>', unsafe_allow_html=True)
    quick1, quick2, quick3 = st.columns(3, gap="small")
    if quick1.button("📄  How it works", use_container_width=True):
        goto("instructions")
    if quick2.button("📊  Previous results", use_container_width=True):
        goto("history")
    if quick3.button("🤖  ML model info", use_container_width=True):
        goto("ml_info")
    st.info(RESULT_DISCLAIMER)

# ==================== DEMO MODE ====================
elif st.session_state.screen == "demo":
    st.header("🎬 Demo Mode")
    st.info("Loads pre-recorded sample data so you can show all four pathways without four people. "
            "Results from here are always labeled DEMO / SIMULATED DATA.")
    for label, scenario in DEMO_SCENARIOS.items():
        if st.button(label, use_container_width=True):
            st.session_state.subject = {"name": label.split("—")[-1].strip(), "age": ""}
            st.session_state.profile = scenario["profile"]
            st.session_state.captured = scenario["captured"]
            st.session_state.fusion_result = fuse_adaptive(scenario["profile"], scenario["captured"])
            st.session_state.is_demo = True
            goto("results")
    if st.button("← Back"):
        goto("home")

# ==================== INSTRUCTIONS ====================
elif st.session_state.screen == "instructions":
    st.header("How It Works")
    st.markdown("""
1. **Choose an accessibility profile** — this only changes which tests are offered, never assumed from anything else.
2. **Adjust settings** — contrast, text size, motion, text-to-speech, response time.
3. **Consent & safety notice** — what's recorded (and what isn't), and emergency guidance.
4. **Tests** — only the tests valid for your profile run; anything not applicable is skipped and never scored.
5. **Result** — a screening score with a plain-language explanation of what each available biomarker showed.
""")
    if st.button("Got it", type="primary"):
        goto("home")

# ==================== PROFILE ====================
elif st.session_state.screen == "profile":
    st.markdown('<div class="eyebrow">Step 1 · Personalise the pathway</div>', unsafe_allow_html=True)
    st.header("Choose your assessment mode")
    st.caption("Select the pathway that best matches how you want to interact with the screening. This choice only changes which tests are offered; it never infers or records a medical condition.")
    profile_descriptions = {
        STANDARD: "Full visual and spoken pathway with all four biomarker channels.",
        DEAF: "Visual-first pathway with no audio required; speech remains optional.",
        BLIND: "Audio-guided pathway with eye movement excluded and spoken responses supported.",
        SPEECH_IMPAIRED: "Visual pathway with speech excluded from scoring.",
        UNSPECIFIED: "Use when you prefer not to specify an accessibility pathway.",
    }
    profile_col, detail_col = st.columns([1.25, 0.75], gap="large")
    with profile_col:
        st.markdown('<div class="selection-summary">Available pathways</div>', unsafe_allow_html=True)
        choice = st.radio("Profile", list(PROFILE_LABELS.values()), label_visibility="collapsed")
    selected_profile = {label: key for key, label in PROFILE_LABELS.items()}[choice]
    with detail_col:
        st.markdown(f'<div class="profile-panel"><h3>{PROFILE_LABELS[selected_profile]}</h3><p>{profile_descriptions[selected_profile]}</p><div class="profile-note">Your answers are used to adapt the experience, not to make a diagnosis.</div></div>', unsafe_allow_html=True)
    inv_map = {v: k for k, v in PROFILE_LABELS.items()}
    st.markdown('<div class="section-label">Next</div>', unsafe_allow_html=True)
    next_col, back_col = st.columns([1, 1], gap="small")
    if next_col.button("Continue to settings →", type="primary", use_container_width=True):
        st.session_state.profile = inv_map[choice]
        goto("settings")
    if back_col.button("← Back to home", use_container_width=True):
        goto("home")

# ==================== SETTINGS ====================
elif st.session_state.screen == "settings":
    st.header("Accessibility Settings")
    st.caption("Note on scope: this Streamlit interface supports high contrast, large text, "
               "reduced motion, text-to-speech, and extended response time, plus standard "
               "keyboard-navigable widgets. It does not currently have full screen-reader/ARIA "
               "support — Streamlit's rendering layer has limited hooks for that.")

    st.subheader("Visual")
    st.session_state.settings["high_contrast"] = st.checkbox("High contrast", value=s["high_contrast"])
    st.session_state.settings["large_text"] = st.checkbox("Large text", value=s["large_text"])
    st.session_state.settings["reduce_motion"] = st.checkbox("Reduce animation in native test windows", value=s["reduce_motion"])

    st.subheader("Audio")
    st.session_state.settings["tts_on"] = st.checkbox("Text-to-speech instructions (used in Blind mode)", value=s["tts_on"])
    st.session_state.settings["volume"] = st.slider("Volume (tones + speech)", 0.1, 1.0, value=s["volume"], step=0.1)
    st.session_state.settings["tts_rate"] = st.slider("Speech speed (words/min)", 110, 240, value=s["tts_rate"], step=10)

    st.subheader("Interaction")
    st.session_state.settings["extended_time"] = st.slider(
        "Extended response time multiplier", 1.0, 3.0, value=s["extended_time"], step=0.25
    )

    if st.button("Continue →", type="primary"):
        goto("consent")
    if st.button("← Back"):
        goto("profile")

# ==================== CONSENT + SAFETY ====================
elif st.session_state.screen == "consent":
    st.header("Consent & Safety Notice")
    st.markdown(CONSENT_TEXT)
    with st.expander("⚠️ Emergency guidance — read this"):
        st.markdown(EMERGENCY_TEXT)
    ack = st.checkbox("I have read and understood the above.", value=s["consent_ack"])
    st.session_state.settings["consent_ack"] = ack
    if st.button("Continue →", type="primary", disabled=not ack):
        goto("register")
    if st.button("← Back"):
        goto("settings")

# ==================== REGISTER ====================
elif st.session_state.screen == "register":
    st.header("Subject Intake")
    name = st.text_input("Name", value=st.session_state.subject.get("name", ""))
    age = st.text_input("Age", value=st.session_state.subject.get("age", ""))

    baseline_key = None
    if name.strip():
        baselines = load_baselines()
        baseline_key = f"{slug(name)}::{st.session_state.profile}"
        if baseline_key in baselines:
            st.info(f"Baseline on file for this profile from {baselines[baseline_key]['saved_at']}.")
        else:
            st.info("No baseline on file for this profile yet — you can save this run as one.")

    if st.button("Continue →", type="primary", disabled=not name.strip()):
        st.session_state.subject = {"name": name.strip(), "age": age.strip()}
        st.session_state.captured = {"eye": None, "speech": None, "reaction": None, "cognitive": None}
        st.session_state.fusion_result = None
        st.session_state.is_demo = False
        goto("plan")
    if st.button("← Back"):
        goto("consent")

# ==================== ADAPTIVE PLAN OVERVIEW ====================
elif st.session_state.screen == "plan":
    profile = st.session_state.profile
    plan = get_test_plan(profile)
    st.header("Your Adaptive Test Plan")
    st.caption(f"Profile: {PROFILE_LABELS[profile]}")
    for biomarker, req in plan.items():
        icon = {"required": "✅", "optional": "➕", "not_applicable": "🚫"}[req]
        label = {"required": "Required", "optional": "Optional", "not_applicable": "Not Applicable"}[req]
        st.write(f"{icon} **{biomarker.capitalize()}** — {label}")
    if profile == BLIND:
        st.info("Instructions for Eye/Speech-only content will be read aloud. Keep your speakers on.")
    if profile == DEAF:
        st.info("Every test here is already fully visual — no audio is required.")
        render_sign_language_panel(VISUAL_INSTRUCTIONS["plan"])
    if profile == SPEECH_IMPAIRED:
        st.info("Speech is never tested or scored for this profile.")
    if st.button("Begin Tests →", type="primary"):
        goto("run_eye")
    if st.button("← Back"):
        goto("register")

# ==================== EYE TEST ====================
elif st.session_state.screen == "run_eye":
    plan = get_test_plan(st.session_state.profile)
    req = plan["eye"]
    if req == "not_applicable":
        st.session_state.captured["eye"] = None
        goto("run_speech")
    else:
        st.header("👁 Eye-Movement Test")
        st.write("A native webcam window will open. Follow the dot, then the LOOK cues.")
        if st.session_state.profile == DEAF:
            render_sign_language_panel(VISUAL_INSTRUCTIONS["eye"])
        if st.button("Launch Eye Test", type="primary"):
            out = os.path.join(DATA_DIR, f"{slug(st.session_state.subject['name'])}_eye.json")
            result = run_subprocess_test("eye_test.py", out, accessibility_args())
            if result is None:
                render_test_notice("No output produced — try again.", "error")
            elif not result.get("valid", False):
                render_test_notice(f"Invalid capture: {result.get('reason') or result.get('message')}. Retry.", "error")
            else:
                st.session_state.captured["eye"] = result
                render_test_notice("✓ TEST COMPLETE", "success")
        if st.session_state.captured["eye"]:
            st.json(st.session_state.captured["eye"])
            if st.button("Continue →", type="primary"):
                goto("run_speech")
        if req == "optional" and st.button("Skip (not suitable for me)"):
            st.session_state.captured["eye"] = None
            goto("run_speech")

# ==================== SPEECH TEST ====================
elif st.session_state.screen == "run_speech":
    plan = get_test_plan(st.session_state.profile)
    req = plan["speech"]
    if req == "not_applicable":
        st.session_state.captured["speech"] = None
        goto("run_reaction")
    elif st.session_state.profile == BLIND:
        # Blind profile never gets a standalone speech screen - typing/reading
        # a sentence isn't a real option here. Speech biomarkers are instead
        # derived from the spoken responses in the Cognitive test (see below).
        goto("run_reaction")
    else:
        st.header("🗣 Speech Test")
        st.warning("⚠️ Switch to your terminal right after clicking — the countdown and 'RECORDING NOW' prompt appear there.")
        if st.button("Launch Speech Test", type="primary"):
            out = os.path.join(DATA_DIR, f"{slug(st.session_state.subject['name'])}_speech.json")
            result = run_subprocess_test("speech_test.py", out)
            if result is None:
                render_test_notice("No output produced — try again.", "error")
            elif not result.get("valid", False):
                render_test_notice(f"Invalid capture: {result.get('reason')}. Retry.", "error")
            else:
                st.session_state.captured["speech"] = result
                render_test_notice("✓ TEST COMPLETE", "success")
        if st.session_state.captured["speech"]:
            st.json(st.session_state.captured["speech"])
        col1, col2 = st.columns(2)
        if col1.button("Continue →", type="primary"):
            goto("run_reaction")
        if req == "optional" and col2.button("Skip (not applicable for me)"):
            st.session_state.captured["speech"] = None
            goto("run_reaction")

# ==================== REACTION TEST ====================
elif st.session_state.screen == "run_reaction":
    st.header("⚡ Reaction-Time Test")
    mode = "audio" if st.session_state.profile == BLIND else "visual"
    st.write(f"Mode: **{mode}**. " + ("Listen for the tone, then press SPACE." if mode == "audio"
             else "A dot will appear — press SPACE as fast as you can."))
    if st.session_state.profile == DEAF:
        render_sign_language_panel(VISUAL_INSTRUCTIONS["reaction"])
    if st.button("Launch Reaction Test", type="primary"):
        out = os.path.join(DATA_DIR, f"{slug(st.session_state.subject['name'])}_reaction.json")
        extra = ["--mode", mode, "--extended-time", str(s["extended_time"])] + accessibility_args()
        if s["reduce_motion"]:
            extra.append("--reduce-motion")
        result = run_subprocess_test("reaction_test.py", out, extra)
        if result is None:
            render_test_notice("No output produced — try again.", "error")
        elif not result.get("valid", False):
            render_test_notice(f"Invalid capture: {result.get('reason')}. Retry.", "error")
        else:
            st.session_state.captured["reaction"] = result
            render_test_notice("✓ TEST COMPLETE", "success")
    if st.session_state.captured["reaction"]:
        st.json(st.session_state.captured["reaction"])
        if st.button("Continue →", type="primary"):
            goto("run_cognitive")
    if get_test_plan(st.session_state.profile)["reaction"] == "optional" and st.button("Skip reaction test"):
        st.session_state.captured["reaction"] = None
        goto("run_cognitive")

# ==================== COGNITIVE TEST ====================
elif st.session_state.screen == "run_cognitive":
    is_blind = st.session_state.profile == BLIND
    st.header("🧠 Cognitive Test" + (" + Speech (combined)" if is_blind else ""))
    mode = "audio" if is_blind else "visual"
    if is_blind:
        st.write("You'll hear number sequences via text-to-speech, then say them back out loud. "
                 "Your spoken responses also produce your speech biomarkers — no separate "
                 "typing or sentence-reading step, since neither was a real option here.")
        st.caption("Transcription needs an internet connection (free Google Web Speech API, no key required).")
    else:
        st.write("Mode: **visual**. Shape-selection and word-recognition tasks.")
        if st.session_state.profile == DEAF:
            render_sign_language_panel(VISUAL_INSTRUCTIONS["cognitive"])
    if mode == "audio" and not s["tts_on"]:
        st.warning("Text-to-speech is turned off in Settings — turn it on for this test to give spoken instructions.")
    if st.button("Launch Cognitive Test", type="primary"):
        out = os.path.join(DATA_DIR, f"{slug(st.session_state.subject['name'])}_cognitive.json")
        extra = ["--mode", mode, "--extended-time", str(s["extended_time"])] + accessibility_args()
        if s["reduce_motion"]:
            extra.append("--reduce-motion")
        speech_out = None
        if is_blind:
            speech_out = os.path.join(DATA_DIR, f"{slug(st.session_state.subject['name'])}_speech.json")
            extra += ["--speech-out", speech_out]
        result = run_subprocess_test("cognitive_test.py", out, extra)
        if result is None:
            render_test_notice("No output produced — try again.", "error")
        elif not result.get("valid", False):
            render_test_notice("Invalid/cancelled capture. Retry.", "error")
        else:
            st.session_state.captured["cognitive"] = result
            render_test_notice("✓ TEST COMPLETE", "success")
            if is_blind and speech_out and os.path.exists(speech_out):
                with open(speech_out) as f:
                    speech_result = json.load(f)
                if speech_result.get("valid"):
                    st.session_state.captured["speech"] = speech_result
                    st.success("Speech biomarkers also derived from your spoken responses.")
                else:
                    st.info(f"Speech biomarkers not derived: {speech_result.get('reason')}. "
                            "(Cognitive score above is unaffected.)")
    if st.session_state.captured["cognitive"]:
        st.json(st.session_state.captured["cognitive"])
        if st.button("Analyze Biomarkers →", type="primary"):
            fr = fuse_adaptive(st.session_state.profile, st.session_state.captured)
            st.session_state.fusion_result = fr
            if fr.get("valid"):
                append_history(st.session_state.subject["name"], st.session_state.profile, fr, is_demo=False)
            goto("results")
    if get_test_plan(st.session_state.profile)["cognitive"] == "optional" and st.button("Skip cognitive test"):
        fr = fuse_adaptive(st.session_state.profile, st.session_state.captured)
        st.session_state.fusion_result = fr
        if fr.get("valid"):
            append_history(st.session_state.subject["name"], st.session_state.profile, fr, is_demo=False)
        goto("results")

# ==================== RESULTS ====================
elif st.session_state.screen == "results":
    if st.session_state.is_demo:
        st.markdown('<span class="badge">DEMO / SIMULATED DATA</span>', unsafe_allow_html=True)
    st.header(f"Result — {st.session_state.subject['name'] or 'Demo Subject'}")
    st.caption(f"Assessment Mode: {PROFILE_LABELS[st.session_state.profile]}")
    fr = st.session_state.fusion_result
    captured = st.session_state.captured

    if not fr or not fr.get("valid"):
        st.error(f"Could not compute a result: {fr.get('reason') if fr else 'no data'}")
    else:
        st.subheader("Biomarker Availability")
        status_icon = {"available": "✅ Available", "not_applicable": "🚫 Not Applicable — Accessibility Mode",
                       "skipped": "⏭ Skipped", "invalid": "⚠️ Invalid capture"}
        for b, stt in fr["status"].items():
            line = f"**{b.capitalize()}**: {status_icon[stt]}"
            if stt == "available":
                line += f" — score {fr['scores'][b]}"
            st.markdown(line)

        band_emoji = {"LOWER CONCERN": "🟢", "NEEDS ATTENTION": "🟡", "HIGHER CONCERN": "🔴"}[fr["band"]]
        st.subheader(f"{band_emoji} Screening Status: {fr['band']}")
        st.progress(min(1.0, fr["risk_score"] / 100))
        st.caption(f"Risk score: {fr['risk_score']} / 100 (computed only from available biomarkers)")

        if fr["scores"]:
            st.subheader("Biomarker Scores")
            for biomarker, score in fr["scores"].items():
                st.progress(
                    max(0.0, min(1.0, float(score) / 100)),
                    text=f"{biomarker.capitalize()}: {score}/100",
                )

        with st.expander("Biomarker Summary (explainable)"):
            for b, stt in fr["status"].items():
                if stt == "not_applicable":
                    st.write(f"**{b.capitalize()}**: Not evaluated due to accessibility profile.")
                elif stt == "skipped":
                    st.write(f"**{b.capitalize()}**: Skipped by choice — not counted.")
                elif stt == "invalid":
                    st.write(f"**{b.capitalize()}**: Capture was invalid — excluded from scoring.")
                else:
                    st.write(f"**{b.capitalize()}**: score {fr['scores'][b]}/100 — see raw metrics below.")
            st.json(captured)

        baselines = load_baselines()
        baseline_key = f"{slug(st.session_state.subject['name'])}::{st.session_state.profile}"
        if baseline_key in baselines:
            b = baselines[baseline_key]["fusion"]
            common = set(fr["scores"]) & set(b.get("scores", {}))
            if common:
                st.caption("vs. baseline (same profile, same available biomarkers only): " +
                           ", ".join(f"{k} {fr['scores'][k]-b['scores'][k]:+.1f}" for k in common) +
                           f", risk {fr['risk_score']-b['risk_score']:+.1f}")

        if not st.session_state.is_demo and model_available():
            ml_risk, ml_label = predict_risk(captured)
            if ml_risk is not None:
                st.markdown("---")
                st.caption("🤖 **ML model comparison** — trained on SYNTHETIC data only. "
                           "Shown for reference, not used as the primary screening result.")
                mc1, mc2 = st.columns(2)
                mc1.metric("Model-predicted risk", f"{ml_risk}%")
                mc2.metric("Model label", ml_label.replace("_", " ").title())

        st.info(RESULT_DISCLAIMER)
        with st.expander("⚠️ Emergency guidance"):
            st.markdown(EMERGENCY_TEXT)

        c1, c2 = st.columns(2)
        if c1.button("💾 Save as Baseline") and not st.session_state.is_demo:
            save_baseline(baseline_key, st.session_state.profile, fr, captured)
            st.success("Saved.")
        report = json.dumps({"subject": st.session_state.subject, "profile": st.session_state.profile,
                              "result": fr, "captured": captured}, indent=2)
        c2.download_button("⬇ Save Report (.json)", report, file_name="neurotrace_report.json")

    colA, colB = st.columns(2)
    if colA.button("📊 View History"):
        goto("history")
    if colB.button("🔁 New Assessment"):
        goto("profile")

# ==================== ML MODEL INFO ====================
elif st.session_state.screen == "ml_info":
    st.header("🤖 ML Model Info")
    st.warning("Trained and evaluated on **synthetic data only** — there is no real labeled "
               "concussion dataset behind this project. These metrics show the ML pipeline "
               "(train/val/test split, cross-validation, held-out evaluation) is real and "
               "correctly built — they are not evidence of clinical accuracy.")
    metrics_path = os.path.join(DATA_DIR, "model", "metrics.json")
    eval_img = os.path.join(DATA_DIR, "model", "evaluation.png")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.subheader(f"Selected model: {metrics['model_selected']}")
        st.markdown("**5-fold cross-validation (training set)**")
        st.dataframe(pd.DataFrame(metrics["cv_report"]).T)
        st.markdown(f"**Held-out test set** ({metrics['n_test']} samples)")
        st.dataframe(pd.DataFrame([metrics["test_report"]]))
        if os.path.exists(eval_img):
            st.image(eval_img, caption="Confusion matrix + ROC curve (synthetic test set)")
    else:
        st.info("No trained model yet. From the project folder, run:\n\n"
                "```\npython ml/generate_synthetic_data.py\npython ml/train_model.py\n```")
    if st.button("← Back"):
        goto("home")

# ==================== HISTORY ====================
elif st.session_state.screen == "history":
    st.header("Previous Results")
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No assessments recorded yet.")
    if st.button("← Back to Home"):
        goto("home")
