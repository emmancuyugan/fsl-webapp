from flask import Flask, render_template, request, jsonify
from backend import (
    SUPPORTED_PHRASES,
    random_phrase,
    evaluate_attempt,
    teaching_for,
    recognize_attempt_stub,
)

app = Flask(__name__)  # uses templates/ and static/ by default

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/tutor")
def tutor():
    return render_template("tutor.html")

@app.route("/activity")
def activity():
    return render_template("activity.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ---------- APIs ----------

@app.route("/api/phrases")
def api_phrases():
    return jsonify({"phrases": SUPPORTED_PHRASES})

@app.route("/api/random", methods=["GET"])
def api_random():
    return jsonify({"phrase": random_phrase()})

@app.route("/api/teach", methods=["GET"])
def api_teach():
    phrase = (request.args.get("phrase") or "").strip().lower()
    return jsonify(teaching_for(phrase))

@app.route("/api/assess", methods=["POST"])
def api_assess():
    """
    Body:
      {
        "target": "good morning",
        "attempt_text": "good evening"  # optional
        "frames": ["data:image/png;base64,..."]  # optional
      }
    """
    data = request.get_json(force=True) or {}
    target = (data.get("target") or "").strip()
    attempt_text = (data.get("attempt_text") or "").strip()

    # If no client guess, use a stub recognizer to guess from frames (demo only)
    if not attempt_text:
        frames = data.get("frames") or []
        attempt_text = recognize_attempt_stub(frames)

    result = evaluate_attempt(target, attempt_text)
    # attach teaching if incorrect
    if not result["correct"]:
        result["teach"] = teaching_for(target)
    return jsonify(result)



@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
