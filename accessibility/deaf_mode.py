"""
accessibility/deaf_mode.py

Deaf / Hard-of-Hearing mode support: visual-only phrasing for every test's
instructions, a "Sign Language Instruction" panel, and a visual-alert
helper to replace audio notifications.

IMPORTANT — read before demoing this:
There is no real sign-language recognition or generation model here, and
building one is out of scope for this prototype (it needs a trained
video/pose model and a labeled ASL dataset, neither of which exist in this
project). This file implements a rule-based TEXT-TO-ICON-CUE mapper: it
looks up keywords in an instruction sentence and shows a short sequence of
icon+caption "cue cards" as a stand-in for where a real sign-language
avatar/video would eventually go. This is the UI area for a future AI
sign-language module. It does not sign real ASL.

Usage from app.py:
    from accessibility.deaf_mode import (
        render_sign_language_panel, render_visual_alert, VISUAL_INSTRUCTIONS
    )
    render_sign_language_panel("Follow the dot with your eyes, then press space.")
"""

import html
import time

import streamlit as st

# ---------------------------------------------------------------------
# 1. Visual-only phrasing for each test phase (no audio/spoken assumptions)
# ---------------------------------------------------------------------
VISUAL_INSTRUCTIONS = {
    "eye": "Follow the dot with your eyes. Then look LEFT, CENTER, or RIGHT when it appears on screen.",
    "reaction": "A green dot will flash on screen. Press SPACE as fast as you can when you see it.",
    "cognitive": "Pick the shape you're shown, then remember the words and answer yes or no.",
    "plan": "Every test in this profile is fully visual already — no audio is required at any step.",
}

# ---------------------------------------------------------------------
# 2. Keyword -> (icon, caption) lookup for the placeholder sign-cue generator
# ---------------------------------------------------------------------
SIGN_LEXICON = {
    "look": ("👀", "LOOK"), "eyes": ("👀", "EYES"), "follow": ("👁️➡️", "FOLLOW"),
    "dot": ("🔵", "DOT"), "left": ("👈", "LEFT"), "right": ("👉", "RIGHT"),
    "center": ("☝️", "CENTER"), "up": ("👆", "UP"), "down": ("👇", "DOWN"),
    "press": ("👇", "PRESS"), "space": ("⎵", "SPACE BAR"), "button": ("🔘", "BUTTON"),
    "wait": ("⏳", "WAIT"), "start": ("✅", "START"), "stop": ("✋", "STOP"),
    "fast": ("⚡", "FAST"), "flash": ("💡", "FLASH"), "green": ("🟢", "GREEN"),
    "shape": ("🔺", "SHAPE"), "select": ("☝️", "SELECT"), "pick": ("☝️", "PICK"),
    "memorize": ("🧠", "MEMORIZE"), "remember": ("🧠", "REMEMBER"), "word": ("🔤", "WORD"),
    "words": ("🔤", "WORDS"), "yes": ("👍", "YES"), "no": ("👎", "NO"),
    "correct": ("✅", "CORRECT"), "complete": ("🎉", "DONE"), "done": ("🎉", "DONE"),
    "test": ("📋", "TEST"), "round": ("🔁", "ROUND"), "screen": ("🖥️", "SCREEN"),
}
STOPWORDS = {"the", "a", "an", "to", "your", "you", "will", "then", "when", "on", "in",
             "at", "is", "it", "as", "or", "with", "each", "this", "for", "of", "be", "are"}


def text_to_sign_sequence(instruction_text: str):
    """Turns an instruction sentence into a list of (icon, caption) cue cards.
    Unmapped meaningful words fall back to a generic pointing icon + the
    word itself, so the sequence still reacts to the actual instruction
    rather than looping the same fixed animation every time."""
    tokens = [t.strip(".,!?").lower() for t in instruction_text.split()]
    seq = []
    for tok in tokens:
        if not tok or tok in STOPWORDS:
            continue
        if tok in SIGN_LEXICON:
            seq.append(SIGN_LEXICON[tok])
        elif len(tok) > 2:
            seq.append(("🤟", tok.upper()))
    return seq or [("🤟", "READY")]


def render_sign_language_panel(instruction_text: str, cycle_seconds: float = 1.4):
    """Renders the placeholder Sign Language Instruction panel, cycling
    through cue cards generated from instruction_text."""
    sequence = text_to_sign_sequence(instruction_text)
    idx = int(time.time() / cycle_seconds) % len(sequence)
    icon, caption = sequence[idx]

    st.markdown(f"""
    <div style="background:#141c26; border:1px solid #22303c; border-radius:10px;
                padding:18px 20px; margin-bottom:14px; text-align:center;">
      <div style="font-size:11px; color:#748899; text-transform:uppercase;
                  letter-spacing:0.08em; margin-bottom:8px;">
        🤟 Sign Language Instruction — placeholder, not real ASL interpretation
      </div>
      <div style="font-size:52px; margin:4px 0;">{icon}</div>
      <div style="font-size:15px; font-weight:600; letter-spacing:0.04em;">{html.escape(caption)}</div>
      <div style="font-size:11px; color:#4d5c6b; margin-top:10px;">
        cue {idx+1} / {len(sequence)}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# 3. Visual-alert helper - replaces audio notifications (spec section 3)
# ---------------------------------------------------------------------
def render_visual_alert(message: str, kind: str = "success"):
    """A large, hard-to-miss visual notification with an animated border,
    used in place of an audio chime/notification sound."""
    colors = {"success": "#49e0b3", "error": "#ff6b52", "info": "#f2b84b"}
    color = colors.get(kind, "#49e0b3")
    st.markdown(f"""
    <style>
    @keyframes nt-flash {{
        0%   {{ box-shadow: 0 0 0 0 {color}66; }}
        70%  {{ box-shadow: 0 0 0 14px {color}00; }}
        100% {{ box-shadow: 0 0 0 0 {color}00; }}
    }}
    .nt-visual-alert {{
        border: 2px solid {color}; border-radius: 10px; padding: 16px 20px;
        text-align:center; font-weight:700; font-size:16px; color:{color};
        animation: nt-flash 1s ease-out 2; margin-bottom:14px;
    }}
    </style>
    <div class="nt-visual-alert">{html.escape(message)}</div>
    """, unsafe_allow_html=True)