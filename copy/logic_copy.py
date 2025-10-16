import random
import re
import torch
import os

# =============================================
# CONFIGURATION SECTION
# =============================================
MODEL_TYPE = "lstm_gru"
MODEL_FILE = "best_lstmgru_1.pth"

# Model hyperparameters
INPUT_SIZE = 128
NUM_CLASSES = 5

# Determine whether to use GPU (CUDA) or CPU for model computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================
# MODEL SELECTION & LOADING
# =============================================
if MODEL_TYPE == "lstm_gru":
    from model_definition.lstm_gru_model import LSTMGRUHybrid
    model = LSTMGRUHybrid(INPUT_SIZE, NUM_CLASSES, dropout_p=0.20)
elif MODEL_TYPE == "modified_lstm":
    from model_definition.modified_lstm_model import ModifiedLSTM
    model = ModifiedLSTM(input_size=INPUT_SIZE, hidden_size=256,
                         num_layers=2, num_classes=NUM_CLASSES,
                         dropout=0.30, use_layernorm=True)
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# Load the trained weights (.pth file)
model_path = os.path.join("models", MODEL_FILE)
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded {MODEL_TYPE} model from {MODEL_FILE}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
else:
    print(f"Model file not found at {model_path}")


# =============================================
# PHRASES, TEACHING, & EVALUATION
# =============================================
SUPPORTED_PHRASES = [
    "hello",
    "don't understand",
    "good morning",
    "good evening",
    "good afternoon",
]

TEACHING_MAP = {
    "hello": {
        "video": "/static/video/hello/Hello.mp4",
        "steps": [
            "Open hand with fingers together.",
            "Wave hand side to side in front of body.",
            "Smile and maintain eye contact."
        ],
    },
    "don't understand": {
        "video": "/static/video/dontunderstand/dont understand.mp4",
        "steps": [
            "Shake head side to side (DON'T).",
            "Shrug shoulders with palms up (UNDERSTAND)."
        ],
    },
    "good morning": {
        "video": "/static/video/goodmorning/goodmorning.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun rising motion under arm (MORNING)."
        ],
    },
    "good evening": {
        "video": "/static/video/goodevening/goodevening.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun setting motion (EVENING)."
        ],
    },
    "good afternoon": {
        "video": "/static/video/goodafternoon/goodafternoon.mp4",
        "steps": [
            "Flat hand from mouth to palm (GOOD).",
            "Sun overhead/forward tilt (AFTERNOON)."
        ],
    },
}

def teaching_for(phrase: str):
    phrase = (phrase or "").lower()
    default = {
        "video": "",
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


# =============================================
# MODEL INFERENCE WRAPPER
# =============================================
def recognize_attempt(frames):
    """
    Predict the sign/phrase using the currently loaded model.
    """
    if not frames:
        return random.choice(SUPPORTED_PHRASES)

    # TODO: add the preprocessing logic here
    # Example (pseudo):
    # frames_tensor = preprocess_frames(frames).to(device)
    # with torch.no_grad():
    #     outputs = model(frames_tensor)
    #     pred_idx = torch.argmax(outputs, dim=1).item()
    # return SUPPORTED_PHRASES[pred_idx]

    # Temporary placeholder until preprocessing is defined
    return random.choice(SUPPORTED_PHRASES)
