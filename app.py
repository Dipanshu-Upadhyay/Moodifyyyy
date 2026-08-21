"""
app.py
------
MoodSync: Mood-Based Music, Movie & Web Series Recommender.

Run with:
    streamlit run app.py
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.emotion_detector import MOOD_LABELS, analyze_emotion
from src.recommender import get_song_recommendations
from src.tmdb_service import get_recommendations as get_movie_recommendations

# ---------------------------------------------------------------------------
# Page config + custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MoodSync",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOOD_EMOJI = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐",
    "Surprise": "😲",
    "Fear": "😨",
}

MOOD_COLOR = {
    "Happy": "#f5b700",
    "Sad": "#3a86ff",
    "Angry": "#e63946",
    "Neutral": "#8d99ae",
    "Surprise": "#ff6fb5",
    "Fear": "#6a4c93",
}

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .mood-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 22px;
        border-radius: 999px;
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        margin: 10px 0 18px 0;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #eaeaea;
        margin: 4px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.08);
    }
    .song-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .song-title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #f2f2f2;
        margin-bottom: 2px;
    }
    .song-artist {
        font-size: 0.85rem;
        color: #a8a8b3;
        margin-bottom: 10px;
    }
    .movie-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .movie-body {
        padding: 12px 14px 14px 14px;
    }
    .movie-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f2f2f2;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #a8a8b3;
        margin: 2px 0 8px 0;
    }
    .movie-overview {
        font-size: 0.85rem;
        color: #d0d0d8;
        line-height: 1.4;
    }
    .rating-chip {
        display: inline-block;
        background: rgba(255,215,0,0.15);
        color: #ffd700;
        border-radius: 6px;
        padding: 1px 8px;
        font-weight: 700;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    tmdb_api_key = st.text_input(
        "TMDb API Key (optional)",
        value=os.environ.get("TMDB_API_KEY", ""),
        type="password",
        help="Leave blank to use the built-in offline movie/TV library instead.",
    )
    if tmdb_api_key:
        st.success("Using live TMDb API for recommendations.")
    else:
        st.info("No API key set — using the offline movie/TV library.")

    media_type = st.radio("Recommend", ["Movies", "Web Series (TV)"], horizontal=False)
    media_type_code = "tv" if media_type.startswith("Web") else "movie"

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Capture or upload a photo (or pick a mood manually).\n"
        "2. MoodSync detects your dominant emotion.\n"
        "3. Get a matching, playable music mix and movie/TV picks."
    )
    st.markdown("---")
    st.caption("MoodSync · Mood-based recommender demo")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "mood" not in st.session_state:
    st.session_state.mood = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎭 MoodSync")
st.caption("Mood-Based Music, Movie & Web Series Recommender")

# ---------------------------------------------------------------------------
# Input methods
# ---------------------------------------------------------------------------

input_tab, upload_tab, manual_tab = st.tabs(["📷 Webcam", "🖼️ Upload Photo", "🎚️ Manual Mood"])

captured_image_bgr = None

with input_tab:
    cam_image = st.camera_input("Take a photo to detect your mood")
    if cam_image is not None:
        pil_img = Image.open(cam_image).convert("RGB")
        captured_image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

with upload_tab:
    uploaded_file = st.file_uploader("Upload a JPG/PNG/JPEG photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        captured_image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

with manual_tab:
    manual_choice = st.selectbox(
        "Pick your mood directly",
        ["— Select —"] + MOOD_LABELS,
    )
    if manual_choice != "— Select —":
        if st.button("Use this mood", type="primary"):
            st.session_state.mood = manual_choice
            st.session_state.confidence = 100.0
            st.session_state.annotated_image = None
            st.rerun()

# Run emotion detection whenever a new image comes in from webcam or upload.
if captured_image_bgr is not None:
    with st.spinner("Analyzing your expression..."):
        result = analyze_emotion(captured_image_bgr)
    if result.success:
        st.session_state.mood = result.mood
        st.session_state.confidence = result.confidence
        st.session_state.annotated_image = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
    else:
        st.warning(result.message)

# ---------------------------------------------------------------------------
# Mood display
# ---------------------------------------------------------------------------

mood = st.session_state.mood

if mood:
    color = MOOD_COLOR.get(mood, "#888")
    emoji = MOOD_EMOJI.get(mood, "")
    confidence = st.session_state.confidence or 0

    badge_col, image_col = st.columns([2, 1])
    with badge_col:
        st.markdown(
            f'<div class="mood-badge" style="background:{color};">'
            f'{emoji} {mood} &nbsp;·&nbsp; {confidence:.0f}% confidence</div>',
            unsafe_allow_html=True,
        )
    with image_col:
        if st.session_state.annotated_image is not None:
            st.image(st.session_state.annotated_image, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Two-column recommendation layout
    # -----------------------------------------------------------------------
    music_col, screen_col = st.columns(2, gap="large")

    with music_col:
        st.markdown('<div class="section-header">🎵 Music for your mood</div>', unsafe_allow_html=True)
        songs = get_song_recommendations(mood, count=6)
        if not songs:
            st.info("No songs found for this mood yet.")
        for song in songs:
            st.markdown(
                f'<div class="song-card">'
                f'<div class="song-title">{song["title"]}</div>'
                f'<div class="song-artist">{song["artist"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.components.v1.iframe(song["embed_url"], height=200)

    with screen_col:
        label = "Movies" if media_type_code == "movie" else "Web Series"
        st.markdown(f'<div class="section-header">🎬 {label} for your mood</div>', unsafe_allow_html=True)
        titles = get_movie_recommendations(
            mood, api_key=tmdb_api_key or None, media_type=media_type_code, count=6
        )
        if not titles:
            st.info("No recommendations available right now.")
        for t in titles:
            poster_html = (
                f'<img src="{t["poster_url"]}" style="width:100%;display:block;" />'
                if t.get("poster_url")
                else '<div style="height:180px;background:#2a2a3d;'
                'display:flex;align-items:center;justify-content:center;'
                'color:#888;font-size:0.8rem;">No poster available</div>'
            )
            st.markdown(
                f'<div class="movie-card">'
                f"{poster_html}"
                f'<div class="movie-body">'
                f'<div class="movie-title">{t["title"]} '
                f'<span style="font-weight:400;color:#a8a8b3;">({t["year"]})</span></div>'
                f'<div class="movie-meta">'
                f'<span class="rating-chip">⭐ {t["rating"]}</span>'
                f'{"  ·  " + t["genres"] if t.get("genres") else ""}'
                f"</div>"
                f'<div class="movie-overview">{t["overview"]}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )
else:
    st.info(
        "👋 Get started: take a webcam photo, upload one, or pick your mood "
        "manually from the tabs above."
    )
