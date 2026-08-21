"""
src/recommender.py
-------------------
Song data loading, filtering, and sampling for the music panel.

Combines up to three CSV sources (only the ones present are used, so the
app still runs with just the small curated data/songs.csv if the larger
processed libraries haven't been built):

    1. data/songs.csv                        - 15-song curated sample,
                                                 real verified YouTube IDs.
    2. data/processed/songs_bollywood.csv     - larger curated Bollywood
                                                 library, real verified
                                                 YouTube IDs (built from the
                                                 user-supplied Bollywood +
                                                 Spotify datasets).
    3. data/processed/songs_global.csv        - large global library, mood-
                                                 labelled via audio features
                                                 (valence/energy/tempo), NO
                                                 YouTube ID -> played via a
                                                 YouTube *search* embed.

Public API:
    get_song_recommendations(mood: str, count: int = 6) -> list[dict]
    build_embed_url(song: dict) -> str
"""

import csv
import os
import random
import urllib.parse
from functools import lru_cache

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SONGS_SAMPLE_PATH = os.path.join(_HERE, "data", "songs.csv")
SONGS_BOLLYWOOD_PATH = os.path.join(_HERE, "data", "processed", "songs_bollywood.csv")
SONGS_GLOBAL_PATH = os.path.join(_HERE, "data", "processed", "songs_global.csv")

MOODS = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]


def _read_csv_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _load_library() -> dict:
    """Load and merge all available song sources, indexed by mood.
    Cached for the life of the process; the underlying CSVs are static."""
    by_mood = {m: [] for m in MOODS}

    for row in _read_csv_rows(SONGS_SAMPLE_PATH):
        mood = row.get("mood")
        if mood in by_mood and row.get("youtube_id"):
            by_mood[mood].append(
                {
                    "title": row["title"],
                    "artist": row["artist"],
                    "youtube_id": row["youtube_id"].strip(),
                    "tier": "curated_sample",
                }
            )

    for row in _read_csv_rows(SONGS_BOLLYWOOD_PATH):
        mood = row.get("mood")
        if mood in by_mood and row.get("youtube_id"):
            by_mood[mood].append(
                {
                    "title": row["title"],
                    "artist": row["artist"],
                    "youtube_id": row["youtube_id"].strip(),
                    "tier": "bollywood",
                }
            )

    for row in _read_csv_rows(SONGS_GLOBAL_PATH):
        mood = row.get("mood")
        if mood in by_mood:
            by_mood[mood].append(
                {
                    "title": row["title"],
                    "artist": row["artist"],
                    "youtube_id": (row.get("youtube_id") or "").strip() or None,
                    "tier": "global_search",
                }
            )

    return by_mood


def build_embed_url(song: dict) -> str:
    """
    Build a playable YouTube embed URL for a song.

    - If we have a real, verified video ID -> direct embed (best experience).
    - Otherwise -> a YouTube *search* embed (no API key required). YouTube
      supports embedding a live search-results "playlist" via
      `listType=search`, which lets us play songs we only know the title/
      artist for (e.g. the large audio-feature-classified global library).
    """
    if song.get("youtube_id"):
        return f"https://www.youtube.com/embed/{song['youtube_id']}"

    query = f"{song['title']} {song['artist']} official audio"
    encoded = urllib.parse.quote(query)
    return f"https://www.youtube.com/embed?listType=search&list={encoded}"


def get_song_recommendations(mood: str, count: int = 6) -> list:
    """
    Return up to `count` songs for `mood`, sampled across all available
    sources. Prefers songs with a direct (verified) YouTube ID over
    search-embed-only entries, for the most reliable playback experience,
    while still mixing in variety from the larger global library.
    """
    if mood not in MOODS:
        return []

    library = _load_library()
    candidates = list(library.get(mood, []))
    if not candidates:
        return []

    direct = [s for s in candidates if s["youtube_id"]]
    search_only = [s for s in candidates if not s["youtube_id"]]

    random.shuffle(direct)
    random.shuffle(search_only)

    # Aim for a good mix: prioritize direct-embed songs, fill any remaining
    # slots from the larger search-embed library.
    picks = direct[:count]
    if len(picks) < count:
        picks += search_only[: count - len(picks)]

    for song in picks:
        song["embed_url"] = build_embed_url(song)

    return picks
