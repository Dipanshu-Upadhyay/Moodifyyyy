"""
src/tmdb_service.py
--------------------
Mood -> genre mapping plus a TMDb Discover API client, with a two-tier
fallback chain when no TMDB_API_KEY is configured (or a live call fails):

    1. Live TMDb API           (if TMDB_API_KEY is set and the request succeeds)
    2. Offline CSV library     (data/processed/movies_offline.csv, built by
                                 scripts/build_movie_library.py from the
                                 1M-row IMDB/TMDB metadata dataset)
    3. Tiny built-in fallback  (a handful of hardcoded well-known titles,
                                 the ultimate safety net so the app never
                                 shows an empty recommendation panel)

Public API:
    get_recommendations(mood: str, api_key: str | None, media_type: str,
                         count: int = 6) -> list[dict]
"""

import csv
import os
import random
from functools import lru_cache
from typing import Optional

import requests

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_MOVIES_PATH = os.path.join(_HERE, "data", "processed", "movies_offline.csv")

# --- Mood -> TMDb genre ID mapping ------------------------------------------
# https://developer.themoviedb.org/reference/genre-movie-list
# Kept in sync with MOOD_GENRES in scripts/build_movie_library.py (by name).
MOOD_GENRE_IDS = {
    "Happy": [35, 16, 10751],       # Comedy, Animation, Family
    "Sad": [18, 10749],             # Drama, Romance (biased toward "uplifting" below)
    "Angry": [28, 53],              # Action, Thriller
    "Neutral": [99, 12, 18],        # Documentary, Adventure, Drama
    "Fear": [27, 9648],             # Horror, Mystery
    "Surprise": [9648, 14, 878],    # Mystery, Fantasy, Science Fiction
}

# TV genre IDs differ slightly from movie genre IDs on TMDb.
MOOD_TV_GENRE_IDS = {
    "Happy": [35, 16, 10751],
    "Sad": [18],
    "Angry": [10759, 80],           # Action & Adventure, Crime
    "Neutral": [99, 18],
    "Fear": [9648, 10765],          # Mystery, Sci-Fi & Fantasy
    "Surprise": [9648, 10765],
}

# Ultimate hardcoded fallback if even the offline CSV is missing/corrupted.
_TINY_FALLBACK = {
    "Happy": [
        {"title": "The Grand Budapest Hotel", "year": "2014", "rating": 8.1,
         "genres": "Comedy, Adventure", "overview": "A concierge and his protégé "
         "become entangled in a whirlwind comic caper across Europe.", "poster_path": None},
        {"title": "Paddington", "year": "2014", "rating": 7.2,
         "genres": "Comedy, Family", "overview": "A young Peruvian bear finds a new "
         "home (and family) in London.", "poster_path": None},
    ],
    "Sad": [
        {"title": "The Pursuit of Happyness", "year": "2006", "rating": 8.0,
         "genres": "Drama", "overview": "A struggling salesman fights for a better "
         "life for himself and his son.", "poster_path": None},
    ],
    "Angry": [
        {"title": "John Wick", "year": "2014", "rating": 7.4,
         "genres": "Action, Thriller", "overview": "A retired hitman seeks vengeance "
         "for a senseless act of cruelty.", "poster_path": None},
    ],
    "Neutral": [
        {"title": "The Shawshank Redemption", "year": "1994", "rating": 8.7,
         "genres": "Drama", "overview": "Two imprisoned men bond over years, finding "
         "solace and eventual redemption.", "poster_path": None},
    ],
    "Fear": [
        {"title": "A Quiet Place", "year": "2018", "rating": 7.5,
         "genres": "Horror, Thriller", "overview": "A family must live in silence "
         "to avoid creatures that hunt by sound.", "poster_path": None},
    ],
    "Surprise": [
        {"title": "Inception", "year": "2010", "rating": 8.4,
         "genres": "Sci-Fi, Thriller", "overview": "A thief who enters dreams to "
         "steal secrets is offered one last impossible job.", "poster_path": None},
    ],
}


def _poster_url(poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


def _two_line_synopsis(overview: str, max_chars: int = 220) -> str:
    overview = (overview or "").strip()
    if len(overview) <= max_chars:
        return overview
    truncated = overview[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"


# --- Tier 1: Live TMDb API ---------------------------------------------------

def _fetch_from_tmdb_api(mood: str, api_key: str, media_type: str, count: int) -> list:
    genre_map = MOOD_TV_GENRE_IDS if media_type == "tv" else MOOD_GENRE_IDS
    genre_ids = genre_map.get(mood, [])
    if not genre_ids:
        return []

    endpoint = f"{TMDB_BASE_URL}/discover/{'tv' if media_type == 'tv' else 'movie'}"
    params = {
        "api_key": api_key,
        "with_genres": ",".join(str(g) for g in genre_ids),
        "sort_by": "vote_average.desc",
        "vote_count.gte": 300,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    resp = requests.get(endpoint, params=params, timeout=8)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    # For Sad, bias toward gentler / more uplifting picks: TMDb doesn't expose
    # sentiment, so we approximate by filtering to the highest-rated, most
    # broadly loved titles rather than every low-rated tragedy.
    if mood == "Sad":
        results = [r for r in results if r.get("vote_average", 0) >= 6.5]

    random.shuffle(results)  # avoid always showing the exact same top-N
    picks = results[:count]

    out = []
    for r in picks:
        title = r.get("title") or r.get("name") or "Untitled"
        date = r.get("release_date") or r.get("first_air_date") or ""
        out.append(
            {
                "title": title,
                "year": date[:4] if date else "—",
                "rating": round(r.get("vote_average", 0), 1),
                "genres": "",  # genre names need a second lookup; omitted for speed
                "overview": _two_line_synopsis(r.get("overview", "")),
                "poster_url": _poster_url(r.get("poster_path")),
                "source": "tmdb_api",
            }
        )
    return out


# --- Tier 2: Offline CSV library --------------------------------------------

@lru_cache(maxsize=1)
def _load_offline_library() -> dict:
    """Load and index the offline movie library by mood. Cached for the
    life of the process (the file is static, generated at build time)."""
    by_mood = {m: [] for m in MOOD_GENRE_IDS}
    if not os.path.exists(OFFLINE_MOVIES_PATH):
        return by_mood

    with open(OFFLINE_MOVIES_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mood = row.get("mood")
            if mood in by_mood:
                by_mood[mood].append(row)
    return by_mood


def _fetch_from_offline_csv(mood: str, count: int) -> list:
    library = _load_offline_library()
    rows = library.get(mood, [])
    if not rows:
        return []

    sample = random.sample(rows, k=min(count, len(rows)))
    out = []
    for r in sample:
        out.append(
            {
                "title": r["title"],
                "year": r["year"],
                "rating": float(r["rating"]),
                "genres": r["genres"],
                "overview": _two_line_synopsis(r["overview"]),
                "poster_url": _poster_url(r["poster_path"]),
                "source": "offline_csv",
            }
        )
    return out


# --- Tier 3: Tiny hardcoded fallback ----------------------------------------

def _fetch_from_tiny_fallback(mood: str, count: int) -> list:
    rows = _TINY_FALLBACK.get(mood, [])
    out = []
    for r in rows[:count]:
        out.append(
            {
                "title": r["title"],
                "year": r["year"],
                "rating": r["rating"],
                "genres": r["genres"],
                "overview": r["overview"],
                "poster_url": _poster_url(r["poster_path"]),
                "source": "tiny_fallback",
            }
        )
    return out


# --- Public entry point ------------------------------------------------------

def get_recommendations(
    mood: str,
    api_key: Optional[str] = None,
    media_type: str = "movie",
    count: int = 6,
) -> list:
    """
    Return up to `count` movie/TV recommendations for `mood`.

    media_type: "movie" or "tv"
    Falls back automatically: live API -> offline CSV -> tiny hardcoded list.
    Each returned dict always has: title, year, rating, genres, overview,
    poster_url (may be None), source (which tier produced it).
    """
    if mood not in MOOD_GENRE_IDS:
        return []

    if api_key:
        try:
            results = _fetch_from_tmdb_api(mood, api_key, media_type, count)
            if results:
                return results
        except (requests.RequestException, ValueError):
            pass  # fall through to offline tiers

    results = _fetch_from_offline_csv(mood, count)
    if results:
        return results

    return _fetch_from_tiny_fallback(mood, count)
