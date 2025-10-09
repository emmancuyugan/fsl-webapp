import random
import re

# Focus phrases
SUPPORTED_PHRASES = [
    "see you tomorrow",
    "nice to meet you",
    "how are you",
    "good morning",
    "good evening",
    "good afternoon",
]

# Optional per-phrase teaching resources (replace with real videos)
TEACHING_MAP = {
    "see you tomorrow": {
        "video": "/static/video/see_you_tomorrow.mp4",
        "steps": [
            "Point to your eyes (SEE).",
            "Gesture towards the other person (YOU).",
            "Circle near temple/forward motion for TOMORROW."
        ],
    },
    "nice to meet you": {
        "video": "/static/video/nice_to_meet_you.mp4",
        "steps": [
            "Flat palm over other hand for NICE.",
            "Index fingers meet together (MEET).",
            "Point to the other person (YOU)."
        ],
    },
    "how are you": {
        "video": "/static/video/how_are_you.mp4",
        "steps": [
            "Rotate hands outward for HOW.",
            "Point to person (YOU)."
        ],
    },
    "good morning": {
        "video": "/static/video/good_morning.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun rising motion under arm (MORNING)."
        ],
    },
    "good evening": {
        "video": "/static/video/good_evening.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun setting motion (EVENING)."
        ],
    },
    "good afternoon": {
        "video": "/static/video/good_afternoon.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun overhead/forward tilt (AFTERNOON)."
        ],
    },
}

def teaching_for(phrase: str):
    phrase = (phrase or "").lower()
    default = {
        "video": "/static/video/sample.mp4",
        "steps": [
            "Watch the instructional video carefully.",
            "Mimic the handshape and motion.",
            "Hold each pose briefly before transitioning."
        ],
    }
    data = TEACHING_MAP.get(phrase, default)
    return {"phrase": phrase or "unknown", **data}

def random_phrase():
    return random.choice(SUPPORTED_PHRASES)

# ---------- Simple evaluation / correctness ----------

def _tokens(s: str):
    return re.findall(r"[a-zA-Z]+", s.lower())

def _normalize(s: str) -> str:
    return " ".join(_tokens(s))

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def evaluate_attempt(target: str, attempt: str):
    t = _normalize(target)
    a = _normalize(attempt)
    # snap to the nearest supported phrase (in case of typos)
    if t not in SUPPORTED_PHRASES:
        best = min(SUPPORTED_PHRASES, key=lambda p: _levenshtein(t, p))
        t = best

    dist = _levenshtein(t, a)
    correct = dist == 0
    score = max(0.0, 1.0 - dist / max(len(t), 1))

    hints = []
    if not correct:
        hints += [
            "Slow down movement and keep hands centered in frame.",
            "Hold each sign for a brief moment before moving.",
        ]

    return {
        "target": t,
        "attempt": a,
        "correct": correct,
        "score": round(score, 3),
        "hints": hints,
    }

# ---------- Stub recognizer ----------
def recognize_attempt_stub(frames):
    """
    Demo recognizer: pick a random phrase with a bias toward correctness
    if many frames were provided. Replace with real model later.
    """
    if not frames:
        return random.choice(SUPPORTED_PHRASES)
    # slightly bias: 60% chance to return the first word of a real phrase
    if random.random() < 0.6:
        return random.choice(SUPPORTED_PHRASES)
    return random.choice(SUPPORTED_PHRASES)
