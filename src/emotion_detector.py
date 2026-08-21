"""
src/emotion_detector.py
------------------------
Face detection (OpenCV Haar cascade) + emotion classification (DeepFace).

Public API:
    analyze_emotion(image: np.ndarray) -> EmotionResult

EmotionResult is a small dataclass carrying:
    - success: bool
    - mood: str | None            (one of MOOD_LABELS, or None on failure)
    - confidence: float | None    (0-100)
    - all_scores: dict | None     (DeepFace's full per-emotion breakdown)
    - annotated_image: np.ndarray | None  (BGR image with a face box + label)
    - message: str                (human-readable status, used for UI toasts)

Design notes:
    - DeepFace natively classifies 7 emotions: angry, disgust, fear, happy,
      sad, surprise, neutral. The app's spec only wants 6 (no "disgust"), so
      we fold "disgust" into "Angry" (its closest neighbour on the valence/
      arousal circumplex, and the standard mapping used in most affect
      literature when a reduced label set is required).
    - DeepFace lazily downloads model weights on first use, which can take
      a while and fail without internet access. We catch that explicitly and
      surface a friendly message rather than an ugly stack trace.
    - Face detection failures (no face, image too dark, etc.) are treated as
      an expected, recoverable outcome, not an exception.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# --- Public constants -------------------------------------------------------

MOOD_LABELS = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]

# DeepFace's raw emotion keys -> our 6-mood label set.
_DEEPFACE_TO_MOOD = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "disgust": "Angry",  # closest neighbour when collapsing to 6 moods
    "neutral": "Neutral",
    "surprise": "Surprise",
    "fear": "Fear",
}

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@dataclass
class EmotionResult:
    success: bool
    mood: Optional[str] = None
    confidence: Optional[float] = None
    all_scores: Optional[dict] = None
    annotated_image: Optional[np.ndarray] = None
    message: str = ""


def _load_face_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            "Could not load OpenCV's Haar cascade for face detection. "
            "Check your opencv-python installation."
        )
    return cascade


# Loaded once per process; cv2.CascadeClassifier is cheap and thread-safe to reuse.
_face_cascade = _load_face_cascade()


def detect_faces(image_bgr: np.ndarray):
    """Return a list of (x, y, w, h) bounding boxes for faces found in the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection under uneven lighting
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return list(faces)


def _draw_annotation(image_bgr: np.ndarray, box, mood: str, confidence: float) -> np.ndarray:
    """Draw a bounding box + mood/confidence badge on a copy of the image."""
    annotated = image_bgr.copy()
    x, y, w, h = box

    color = (46, 204, 113)  # BGR, a pleasant green
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

    label = f"{mood} ({confidence:.0f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    badge_y1 = max(0, y - text_h - baseline - 12)
    cv2.rectangle(annotated, (x, badge_y1), (x + text_w + 16, y), color, -1)
    cv2.putText(
        annotated,
        label,
        (x + 8, y - baseline - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def analyze_emotion(image_bgr: np.ndarray) -> EmotionResult:
    """
    Detect the largest face in `image_bgr` (OpenCV BGR ndarray) and classify
    its dominant emotion with DeepFace.

    Returns an EmotionResult. Never raises for "expected" failure modes
    (no face found, model unavailable) -- those are reported via
    `result.success = False` and `result.message`.
    """
    if image_bgr is None or image_bgr.size == 0:
        return EmotionResult(success=False, message="No image was provided.")

    faces = detect_faces(image_bgr)
    if not faces:
        return EmotionResult(
            success=False,
            message=(
                "No face detected. Try better lighting, move closer to the "
                "camera, or upload a clearer photo."
            ),
        )

    # If multiple faces are found, analyze the largest (closest / most prominent) one.
    faces.sort(key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = faces[0]

    # DeepFace does its own internal face alignment/detection too, but we
    # crop first so it (a) runs faster and (b) focuses on the same face we
    # draw the annotation box around.
    pad = int(0.15 * max(w, h))
    y0, y1 = max(0, y - pad), min(image_bgr.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(image_bgr.shape[1], x + w + pad)
    face_crop = image_bgr[y0:y1, x0:x1]

    try:
        from deepface import DeepFace  # imported lazily: heavy (TF) import
    except ImportError:
        return EmotionResult(
            success=False,
            message=(
                "DeepFace is not installed. Run `pip install deepface` "
                "(see requirements.txt)."
            ),
        )

    try:
        analysis = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            detector_backend="skip",  # we already cropped the face ourselves
            enforce_detection=False,
            silent=True,
        )
    except Exception as exc:  # DeepFace can raise many different exception types
        return EmotionResult(
            success=False,
            message=f"Emotion analysis failed: {exc}",
        )

    # DeepFace.analyze returns a list (one entry per detected face) as of v0.0.79+.
    result_dict = analysis[0] if isinstance(analysis, list) else analysis
    raw_scores: dict = result_dict.get("emotion", {})
    dominant_raw = result_dict.get("dominant_emotion")

    if not raw_scores or dominant_raw is None:
        return EmotionResult(success=False, message="Could not read a clear emotion from the face.")

    # Fold DeepFace's 7 emotions down into our 6-mood label set by summing
    # scores that map to the same mood (currently just fear/disgust -> Angry
    # for 'disgust'), then picking the max.
    mood_scores = {m: 0.0 for m in MOOD_LABELS}
    for raw_label, score in raw_scores.items():
        mood = _DEEPFACE_TO_MOOD.get(raw_label.lower())
        if mood:
            mood_scores[mood] += float(score)

    mood = max(mood_scores, key=mood_scores.get)
    confidence = mood_scores[mood]  # DeepFace scores are already 0-100 percentages

    annotated = _draw_annotation(image_bgr, (x, y, w, h), mood, confidence)

    return EmotionResult(
        success=True,
        mood=mood,
        confidence=round(confidence, 1),
        all_scores={k: round(v, 1) for k, v in mood_scores.items()},
        annotated_image=annotated,
        message=f"Detected mood: {mood} ({confidence:.0f}% confidence).",
    )
