"""
build_movie_library.py
-----------------------
One-time ETL script (run by the developer, NOT at Streamlit runtime).

Consumes data/raw/imdb/IMDB TMDB Movie Metadata Big Dataset (1M).csv
(~1M rows, ~950MB) and produces a compact, mood-tagged offline fallback
library at data/processed/movies_offline.csv, used by src/tmdb_service.py
whenever no TMDB_API_KEY is configured (or the live API call fails).

Mood -> genre mapping mirrors the *live* mapping in src/tmdb_service.py so
offline and online behaviour stay consistent. Ranking blends TMDb's
vote_average, vote_count (via an IMDb-style weighted rating / Bayesian
average) and, where available, the dataset's precomputed overview_sentiment
score -- which is especially useful for surfacing genuinely *uplifting*
dramas for the Sad mood rather than just any high-rated tragedy.

Run once:
    python scripts/build_movie_library.py
"""

import ast
import csv
import math
import os
import sys

csv.field_size_limit(sys.maxsize)

RAW_PATH = "data/raw/imdb/IMDB TMDB Movie Metadata Big Dataset (1M).csv"
OUT_PATH = "data/processed/movies_offline.csv"

MOODS = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]

# Keep in sync with MOOD_GENRE_MAP in src/tmdb_service.py.
MOOD_GENRES = {
    "Happy": {"Comedy", "Animation", "Family", "Music"},
    "Sad": {"Drama", "Romance"},  # filtered for an "uplifting" bias below
    "Angry": {"Action", "Thriller"},
    "Neutral": {"Documentary", "Adventure", "Drama"},
    "Fear": {"Horror", "Mystery"},
    "Surprise": {"Mystery", "Fantasy", "Science Fiction"},
}

MIN_VOTE_COUNT = 300
CAP_PER_MOOD = 250
GLOBAL_MEAN_VOTE = 6.5  # rough prior used in the Bayesian-average ranking


def weighted_rating(vote_average, vote_count, m=MIN_VOTE_COUNT, c=GLOBAL_MEAN_VOTE):
    """IMDb-style Bayesian weighted rating: pulls low-vote-count entries
    toward the global mean so a 9.5/5-votes title doesn't outrank a
    genuinely well-loved 8.0/50000-votes title."""
    v, r = vote_count, vote_average
    return (v / (v + m)) * r + (m / (v + m)) * c


def build():
    print("Streaming IMDB/TMDB metadata (this covers ~1M rows, please wait)...")
    buckets = {m: [] for m in MOODS}
    total, kept = 0, 0

    with open(RAW_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 200000 == 0:
                print(f"  ...{total:,} rows scanned")

            try:
                lang = row.get("original_language", "")
                if lang != "en":
                    continue
                title = row.get("title", "").strip()
                overview = row.get("overview", "").strip()
                poster_path = row.get("poster_path", "").strip()
                if not title or not overview or not poster_path:
                    continue

                vote_average = float(row.get("vote_average") or 0)
                vote_count = int(float(row.get("vote_count") or 0))
                if vote_count < MIN_VOTE_COUNT or vote_average <= 0:
                    continue

                release_date = row.get("release_date", "") or ""
                year = release_date[:4] if len(release_date) >= 4 else ""
                if not year.isdigit():
                    continue

                genres_raw = row.get("genres_list", "[]")
                try:
                    genres = set(ast.literal_eval(genres_raw))
                except (ValueError, SyntaxError):
                    genres = set()
                if not genres:
                    continue

                try:
                    sentiment = float(row.get("overview_sentiment") or 0)
                except ValueError:
                    sentiment = 0.0

                score = weighted_rating(vote_average, vote_count)
            except (ValueError, TypeError):
                continue

            for mood, mood_genres in MOOD_GENRES.items():
                if not (genres & mood_genres):
                    continue

                mood_score = score
                if mood == "Sad":
                    # Bias toward *uplifting* dramas: reward positive
                    # overview sentiment, penalize bleak ones, and require
                    # a slightly higher quality bar than other moods.
                    if vote_average < 6.8:
                        continue
                    mood_score = score + (sentiment * 2.0)

                kept += 1
                buckets[mood].append(
                    {
                        "tmdb_id": row.get("id", ""),
                        "imdb_id": row.get("imdb_id", ""),
                        "title": title,
                        "year": year,
                        "rating": round(vote_average, 1),
                        "genres": ", ".join(sorted(genres)),
                        "overview": overview,
                        "poster_path": poster_path,
                        "mood": mood,
                        "_score": mood_score,
                    }
                )

    print(f"Scanned {total:,} rows, {kept:,} mood-genre matches before ranking/capping.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tmdb_id", "imdb_id", "title", "year", "rating",
                "genres", "overview", "poster_path", "mood",
            ],
        )
        writer.writeheader()
        for mood in MOODS:
            bucket = buckets[mood]
            bucket.sort(key=lambda r: r["_score"], reverse=True)
            seen_titles = set()
            written = 0
            for r in bucket:
                key = (r["title"].lower(), r["year"])
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                r.pop("_score")
                writer.writerow(r)
                written += 1
                if written >= CAP_PER_MOOD:
                    break
            print(f"  {mood}: {written} titles")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build()
