"""
build_song_library.py
----------------------
One-time ETL script (run by the developer, NOT at Streamlit runtime).

Consumes two raw data sources supplied by the user:
  1. data/raw/spotify/tracks_features.csv
     A ~1.2M row Spotify audio-features export (valence, energy, danceability,
     tempo, loudness, etc). No YouTube IDs. Used as the authoritative signal
     for mood classification via a valence/energy quadrant model.
  2. data/raw/bollywood/data/filtered/rating_4.3_plus_songs/*.csv
     Curated Bollywood/Hindi song metadata WITH real, verified YouTube video
     IDs and community ratings, but no mood labels.

Produces two compact, versioned artifacts consumed by the app at runtime:
  - data/processed/songs_global.csv   (mood-labelled, no youtube_id -> the
    app plays these via a YouTube *search* embed, so no direct ID is needed)
  - data/processed/songs_bollywood.csv (mood-labelled via fuzzy match against
    the Spotify mood map, WITH a verified youtube_id -> the app plays these
    via a direct YouTube embed)

Run once:
    python scripts/build_song_library.py
"""

import csv
import glob
import os
import random
import re
import sys

csv.field_size_limit(sys.maxsize)

RAW_SPOTIFY = "data/raw/spotify/tracks_features.csv"
RAW_BOLLYWOOD_GLOBS = [
    "data/raw/bollywood/data/filtered/4.3_plus_songs_final_with_url_id_extracted/*.csv",
    "data/raw/bollywood/data/filtered/rating_4.0_plus_songs/*.csv",
]

# Lightweight keyword lexicon for mood inference on titles that have no
# acoustic-feature match. Bollywood titles are frequently Hindi/Urdu words
# transliterated into English; this is a pragmatic, transparent fallback
# (industry-standard "cold start" technique for content with no audio
# features), NOT a claim of linguistic precision.
MOOD_KEYWORDS = {
    "Happy": [
        "khushi", "khush", "masti", "jashn", "celebration", "party", "nachan",
        "badhai", "happy", "jhoom", "rangeela", "holi", "mast", "dhoom",
        "shaadi", "wedding", "yaari", "dosti", "josh mein",
    ],
    "Sad": [
        "judaai", "dard", "tanha", "tanhai", "bewafa", "gum", "alvida",
        "yaad", "bichad", "akela", "rona", "tears", "sad", "dukh", "bichhड़ना",
        "channa mereya", "dooriyan", "kabira", "vida",
    ],
    "Angry": [
        "josh", "jung", "krodh", "aatank", "fight", "veer", "sherni",
        "hunkar", "inquilab", "vardi", "danger", "sultan", "baaghi",
    ],
    "Fear": [
        "darr", "bhoot", "khauf", "dahshat", "andhera", "horror", "chudail",
        "raaz", "saaya", "bhayanak",
    ],
    "Surprise": [
        "hairat", "chaunk", "ajab", "gajab", "kamal", "wow", "chamatkar",
        "anokha", "ajeeb",
    ],
    # Romance/longing words default to Neutral (Bollywood love songs span
    # the whole emotional spectrum, so we don't force them into Happy/Sad).
    "Neutral": [
        "pyaar", "ishq", "mohabbat", "dil", "tera", "zindagi", "safar",
        "dua", "sapna", "chahat",
    ],
}
OUT_GLOBAL = "data/processed/songs_global.csv"
OUT_BOLLYWOOD = "data/processed/songs_bollywood.csv"

MOODS = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]

# Per-mood cap for the global (search-embed) library. Keeps the shipped CSV
# small while still giving the recommender plenty of variety.
PER_MOOD_CAP = 400

# The source Spotify export (tracks_features.csv) is heavily skewed toward
# 1990s-2000s catalog tracks (only ~6% of rows are from the 2020s), so a
# uniform reservoir sample ends up looking "dated". We split each mood's
# quota into two independently-sampled reservoirs -- recent vs. classic --
# so the shipped library is weighted toward modern tracks while still
# keeping some older/classic variety.
RECENT_YEAR_CUTOFF = 2012
RECENT_CAP = 300   # ~75% of each mood's slots reserved for 2012+ tracks
CLASSIC_CAP = 100  # remaining ~25% for pre-2012 tracks

random.seed(42)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)  # drop parenthetical subtitles
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_mood(valence: float, energy: float, tempo: float, danceability: float) -> str:
    """
    Heuristic mood classifier built on Russell's circumplex model of affect,
    extended with tempo/danceability to separate the six target moods.

    valence   -> pleasantness (low = negative, high = positive)
    energy    -> arousal/intensity
    tempo     -> beats per minute
    danceability -> rhythmic regularity / groove

    Quadrants of (valence, energy):
        high valence, high energy -> Happy / Surprise
        high valence, low  energy -> Neutral (content/calm)
        low  valence, high energy -> Angry / Fear
        low  valence, low  energy -> Sad
    """
    if valence >= 0.55 and energy >= 0.55:
        # Split the "excited" quadrant: fast, low-groove, high-tempo tracks
        # read as startling/unexpected (Surprise); the rest are Happy.
        if tempo >= 135 and danceability < 0.55:
            return "Surprise"
        return "Happy"

    if valence < 0.45 and energy >= 0.55:
        # Split the "tense" quadrant: very low valence + high loudness/energy
        # with low danceability reads as Fear (tense/anxious); otherwise Angry.
        if valence < 0.25 and danceability < 0.45:
            return "Fear"
        return "Angry"

    if valence < 0.45 and energy < 0.45:
        return "Sad"

    # Remaining middle-ground (moderate valence and/or moderate energy)
    return "Neutral"


def build_global_library():
    print("Reading Spotify audio-features export (streaming, chunked)...")
    # Each mood gets two independent reservoirs: "recent" (>= RECENT_YEAR_CUTOFF)
    # and "classic" (older). Combined at the end -> recency-weighted sample.
    recent_buckets = {m: [] for m in MOODS}
    classic_buckets = {m: [] for m in MOODS}
    recent_seen_count = {m: 0 for m in MOODS}
    classic_seen_count = {m: 0 for m in MOODS}
    seen_names = set()
    total_rows = 0
    # Full (title, artist) -> mood lookup built from EVERY row scanned, used
    # later to mood-match the (much smaller) Bollywood catalog. Kept separate
    # from `buckets`, which is capped per mood for the shipped CSV.
    full_lookup = {}

    with open(RAW_SPOTIFY, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                valence = float(row["valence"])
                energy = float(row["energy"])
                tempo = float(row["tempo"])
                danceability = float(row["danceability"])
                name = row["name"].strip()
                artists_raw = row["artists"]
                year = row.get("year", "")
            except (ValueError, KeyError):
                continue

            if not name or not artists_raw:
                continue

            # artists column is stored as a Python-list-literal string, e.g. "['Adele']"
            artist = re.sub(r"[\[\]'\"]", "", artists_raw).split(",")[0].strip()
            if not artist:
                continue

            mood = classify_mood(valence, energy, tempo, danceability)
            nname = normalize(name)
            full_lookup[(nname, normalize(artist))] = mood
            # Title-only fallback is noisy for short/generic titles ("Intro",
            # "Theme Music"), so only register it for reasonably specific
            # (3+ word) titles.
            if len(nname.split()) >= 3:
                full_lookup.setdefault(("title_only", nname), mood)

            dedupe_key = (name.lower(), artist.lower())
            if dedupe_key in seen_names:
                continue

            is_recent = year.isdigit() and int(year) >= RECENT_YEAR_CUTOFF
            bucket = recent_buckets[mood] if is_recent else classic_buckets[mood]
            cap = RECENT_CAP if is_recent else CLASSIC_CAP
            seen_count = recent_seen_count if is_recent else classic_seen_count

            record = {
                "song_id": row["id"],
                "title": name,
                "artist": artist,
                "mood": mood,
                "year": year,
                "valence": round(valence, 3),
                "energy": round(energy, 3),
                "youtube_id": "",  # unknown -> app falls back to search-embed
            }

            # Reservoir sampling within each (mood, recency) pool: keeps the
            # list bounded without biasing toward whatever appears first.
            seen_count[mood] += 1
            if len(bucket) < cap:
                seen_names.add(dedupe_key)
                bucket.append(record)
            else:
                j = random.randint(0, seen_count[mood] - 1)
                if j < cap:
                    seen_names.add(dedupe_key)
                    bucket[j] = record

            if total_rows % 200000 == 0:
                print(f"  ...{total_rows:,} rows scanned")

    print(f"Finished scanning {total_rows:,} rows.")

    # Merge the two reservoirs per mood into the final bucket used for output.
    buckets = {}
    for m in MOODS:
        buckets[m] = recent_buckets[m] + classic_buckets[m]

    os.makedirs(os.path.dirname(OUT_GLOBAL), exist_ok=True)
    with open(OUT_GLOBAL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["song_id", "title", "artist", "mood", "year", "valence", "energy", "youtube_id"],
        )
        writer.writeheader()
        for mood in MOODS:
            for r in buckets[mood]:
                writer.writerow(r)
            n_recent = len(recent_buckets[mood])
            n_classic = len(classic_buckets[mood])
            print(f"  {mood}: {len(buckets[mood])} tracks ({n_recent} recent, {n_classic} classic)")

    print(f"Wrote {OUT_GLOBAL}")
    return full_lookup


def keyword_mood(title: str):
    """Return a mood guessed from title keywords, or None if no keyword hits."""
    nt = f" {normalize(title)} "
    for mood, words in MOOD_KEYWORDS.items():
        for w in words:
            if f" {w} " in nt or nt.startswith(f"{w} ") or nt.endswith(f" {w}"):
                return mood
    return None


def build_bollywood_library(full_lookup):
    print("Reading Bollywood CSVs with verified YouTube IDs...")
    rows = []
    seen_titles_raw = set()
    for pattern in RAW_BOLLYWOOD_GLOBS:
        for path in sorted(glob.glob(pattern)):
            # Filenames encode a year range, e.g. "songs_2015_2025_final.csv" ->
            # use the start year to bias sampling toward modern tracks later
            # (the raw catalog is heavily skewed toward pre-2005 classics).
            year_match = re.search(r"(\d{4})_(\d{4})", os.path.basename(path))
            year_min = int(year_match.group(1)) if year_match else 0

            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title = (row.get("song_title") or "").strip()
                    singers = (row.get("song_singers") or "").strip()
                    yt_id = (row.get("youtubeurlid") or "").strip()
                    if not yt_id:
                        # Fall back to extracting the 11-char video ID from a
                        # full YouTube URL when the pre-extracted column is absent.
                        url = row.get("youtube_url") or row.get("music_yt_url_1") or ""
                        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
                        if m:
                            yt_id = m.group(1)
                    rating = row.get("song_rating") or "0"
                    if not title or not yt_id:
                        continue
                    dedupe = title.lower()
                    if dedupe in seen_titles_raw:
                        continue
                    if normalize(title) in ("theme music", "intro", "outro", "instrumental"):
                        continue
                    seen_titles_raw.add(dedupe)
                    first_singer = singers.split(",")[0].strip() or "Unknown Artist"
                    rows.append(
                        {
                            "title": title,
                            "year_min": year_min,
                            "artist": first_singer or singers,
                            "youtube_id": yt_id,
                            "rating": float(rating) if rating else 0.0,
                        }
                    )

    print(f"  Loaded {len(rows):,} unique candidate Bollywood tracks (rating >= 4.0).")

    # Tier 1: acoustic-feature match against the Spotify-derived mood map.
    # Tier 2: keyword/lexicon match on the title (cold-start fallback).
    # Tier 3: round-robin distribution by rating rank, so every mood still
    #         gets a reasonable playlist even with zero textual/acoustic signal.
    acoustic_hits, keyword_hits, fallback_hits = 0, 0, 0
    for r in rows:
        nt, na = normalize(r["title"]), normalize(r["artist"])
        mood = full_lookup.get((nt, na)) or full_lookup.get(("title_only", nt))
        if mood:
            r["mood"], r["source"] = mood, "acoustic"
            acoustic_hits += 1
        else:
            mood = keyword_mood(r["title"])
            if mood:
                r["mood"], r["source"] = mood, "keyword"
                keyword_hits += 1
            else:
                r["mood"], r["source"] = None, "fallback"

    rows.sort(key=lambda r: r["rating"], reverse=True)
    round_robin = MOODS * ((len(rows) // len(MOODS)) + 1)
    rr_i = 0
    for r in rows:
        if r["mood"] is None:
            r["mood"] = round_robin[rr_i]
            rr_i += 1
            fallback_hits += 1

    print(f"  Mood source -> acoustic match: {acoustic_hits}, keyword match: {keyword_hits}, "
          f"rating-ranked fallback: {fallback_hits}")

    # Fill each mood's playlist in two passes: modern tracks (year_min >= 2005)
    # first, then classics -- so the shipped Bollywood library isn't dominated
    # by pre-2005 catalog the way a straight rating/confidence sort would
    # produce (older tracks tend to have inflated nostalgia ratings). Within
    # each recency pool, still prefer higher mood-confidence sources, then rating.
    RECENT_YEAR_MIN = 2005
    RECENT_SHARE = 0.7  # ~70% of each mood's slots reserved for modern tracks
    source_rank = {"acoustic": 0, "keyword": 1, "fallback": 2}

    recent_rows = [r for r in rows if r["year_min"] >= RECENT_YEAR_MIN]
    classic_rows = [r for r in rows if r["year_min"] < RECENT_YEAR_MIN]
    recent_rows.sort(key=lambda r: (source_rank[r["source"]], -r["rating"]))
    classic_rows.sort(key=lambda r: (source_rank[r["source"]], -r["rating"]))

    CAP = 60
    recent_cap = int(CAP * RECENT_SHARE)
    seen_titles = set()
    per_mood = {m: [] for m in MOODS}

    def fill(pool, cap):
        for r in pool:
            key = r["title"].lower()
            if key in seen_titles:
                continue
            if len(per_mood[r["mood"]]) >= cap:
                continue
            seen_titles.add(key)
            per_mood[r["mood"]].append(r)

    fill(recent_rows, recent_cap)
    # Second pass allows classics (and any overflow recent tracks, since a
    # mood may not have enough modern candidates) to top each mood up to CAP.
    combined_rest = recent_rows + classic_rows
    combined_rest.sort(key=lambda r: (source_rank[r["source"]], -r["rating"]))
    fill(combined_rest, CAP)

    os.makedirs(os.path.dirname(OUT_BOLLYWOOD), exist_ok=True)
    with open(OUT_BOLLYWOOD, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["song_id", "title", "artist", "mood", "rating", "youtube_id", "mood_source"]
        )
        writer.writeheader()
        song_id = 1
        for mood in MOODS:
            for r in per_mood[mood]:
                writer.writerow(
                    {
                        "song_id": f"bw_{song_id:04d}",
                        "title": r["title"],
                        "artist": r["artist"],
                        "mood": mood,
                        "rating": r["rating"],
                        "youtube_id": r["youtube_id"],
                        "mood_source": r["source"],
                    }
                )
                song_id += 1
            src_counts = {}
            for r in per_mood[mood]:
                src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
            n_recent = sum(1 for r in per_mood[mood] if r["year_min"] >= RECENT_YEAR_MIN)
            print(f"  {mood}: {len(per_mood[mood])} tracks {src_counts} ({n_recent} modern/2005+)")

    print(f"Wrote {OUT_BOLLYWOOD}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lookup = build_global_library()
    build_bollywood_library(lookup)
    print("Done.")
