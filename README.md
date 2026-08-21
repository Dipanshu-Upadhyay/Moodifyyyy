# 🎭 MoodSync — Mood-Based Music, Movie & Web Series Recommender

MoodSync detects your facial mood (webcam, upload, or manual override) and
recommends **playable** music and **movie/web-series** picks to match it.

---

## 1. Project structure

```
moodsync/
├── app.py                        # Streamlit UI (entry point)
├── requirements.txt
├── .env.example                  # copy to .env for your TMDB_API_KEY
├── data/
│   ├── songs.csv                 # 15-song curated sample (real YouTube IDs)
│   └── processed/                # compact, pre-built libraries (shipped)
│       ├── songs_bollywood.csv   # 360 songs, real YouTube IDs, mood-tagged
│       ├── songs_global.csv      # 2,400 songs, mood-tagged via audio features
│       └── movies_offline.csv    # 1,500 movies, mood-tagged offline TMDb fallback
├── scripts/                      # one-time ETL scripts (see §5)
│   ├── build_song_library.py
│   └── build_movie_library.py
├── src/
│   ├── emotion_detector.py       # OpenCV face detection + DeepFace analysis
│   ├── tmdb_service.py           # mood→genre mapping + TMDb API + fallback
│   └── recommender.py            # song filtering / sampling / YouTube embeds
└── README.md
```

---

## 2. A note on Python version

You asked for **Python 3.14**. Being fully transparent: DeepFace's default
backend is **TensorFlow**, and as of this writing TensorFlow does not yet
publish wheels for Python 3.14 (it typically lags a version or two behind
new CPython releases). If you try to `pip install tensorflow` on 3.14 today
it will most likely fail to resolve.

**Good news:** TensorFlow 2.21.0 *does* now officially support **Python
3.13** (Windows/Linux/macOS wheels shipped March 2026), which is what
`requirements.txt` is pinned against. If you have 3.13 installed alongside
3.14 (check with `py -0` on Windows, or `ls /usr/bin/python3.*` on
macOS/Linux), just build the venv with that instead of hunting down 3.12:

```bash
py -3.13 -m venv venv        # Windows
python3.13 -m venv venv      # macOS/Linux
```

All other parts of the app (Streamlit, OpenCV, pandas, requests) are
compatible with 3.14 already — it's specifically the DeepFace/TensorFlow
emotion-detection piece that isn't yet. Once TensorFlow ships 3.14 wheels,
this project will work there unmodified. Check current status any time:

```bash
pip index versions tensorflow
```

---

## 3. Setup

```bash
# 1. Create and activate a virtual environment (Python 3.10-3.12)
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) configure your TMDb API key
cp .env.example .env
# then edit .env and paste your key, OR just paste it into the app's
# sidebar "TMDb API Key" field at runtime — either works.
```

### Getting a free TMDb API key

1. Create a free account at https://www.themoviedb.org/signup
2. Go to **Settings → API** (https://www.themoviedb.org/settings/api)
3. Click **Create** under "Request an API Key" → choose **Developer**
4. Fill in the short form (any personal/non-commercial use is fine)
5. Copy the **API Key (v3 auth)** value — that's your `TMDB_API_KEY`

No key? No problem — MoodSync automatically falls back to a **1,500-title
offline movie library** (`data/processed/movies_offline.csv`) that's
already mood-tagged and shipped with the project, so movie/TV
recommendations work out of the box either way.

---

## 4. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

1. Pick an input tab: **Webcam**, **Upload Photo**, or **Manual Mood**.
2. MoodSync detects your face and dominant emotion (or you pick one manually).
3. The left column shows playable songs (embedded YouTube players); the
   right column shows movie/TV cards with poster, rating, and synopsis.

---

## 5. Where the data comes from (and how to rebuild it)

Rather than shipping tiny hardcoded sample data for everything, three large
raw datasets were used to build richer, mood-labelled libraries offline —
**you only need to re-run these if you want to rebuild/customize the
libraries**; the compact outputs in `data/processed/` are already committed
and are all `app.py` needs at runtime.

| Script | Raw input (not shipped — too large) | Output |
|---|---|---|
| `scripts/build_song_library.py` | Spotify audio-features export (~1.2M tracks: valence/energy/tempo/danceability) + a Bollywood song catalog with real YouTube IDs | `songs_global.csv`, `songs_bollywood.csv` |
| `scripts/build_movie_library.py` | IMDB/TMDB metadata export (~1M titles: overview, genres, ratings, posters) | `movies_offline.csv` |

**Song mood classification** uses a valence/energy quadrant model (an
established approach in music-emotion research, per Russell's circumplex
model of affect), refined with tempo/danceability to separate all 6 target
moods. The source Spotify export is heavily skewed toward 1990s-2000s
catalog tracks, so sampling is split into two reservoirs per mood (~75%
from 2012+, ~25% older) to keep the library feeling current rather than
dated. Bollywood tracks (no audio features available) are mood-tagged via a
three-tier confidence chain: (1) exact match against the audio-feature map
where the same song appears in both datasets, (2) a title-keyword lexicon,
(3) a rating-ranked round-robin fallback — each row's `mood_source` column
tells you which tier assigned it — with the same recency weighting
(~70% from 2005+) applied for the same reason.

**Movie mood classification** maps TMDb genres to moods (see
`MOOD_GENRE_IDS` in `src/tmdb_service.py`, kept in sync with the build
script), ranks by an IMDb-style Bayesian weighted rating, and — for the
"Sad" mood specifically — biases toward genuinely *uplifting* dramas using
the dataset's `overview_sentiment` score, rather than just surfacing any
highly-rated tragedy.

To rebuild from your own raw files, place them at:
```
data/raw/spotify/tracks_features.csv
data/raw/bollywood/data/filtered/...
data/raw/imdb/IMDB TMDB Movie Metadata Big Dataset (1M).csv
```
then run:
```bash
python scripts/build_song_library.py
python scripts/build_movie_library.py
```

---

## 6. Design notes

- **Emotion set**: DeepFace natively detects 7 emotions (adds "disgust").
  Since the spec calls for 6, "disgust" is folded into "Angry" — its
  closest neighbour when a reduced label set is required.
- **No-face handling**: `emotion_detector.py` returns a clear, recoverable
  `EmotionResult(success=False, message=...)` rather than raising, so the
  UI can show a friendly warning instead of crashing.
- **Playback without a YouTube API key**: songs with a verified video ID
  play via a direct `youtube.com/embed/{id}` iframe. Songs from the large
  audio-feature-derived library (which has no video ID) play via YouTube's
  `listType=search` embed — a live search-results "playlist" that needs no
  API key at all.
- **Three-tier movie fallback**: live TMDb API → offline CSV library → a
  tiny hardcoded list, so the movie panel is never empty even with no
  internet access and a missing/corrupted offline file.
