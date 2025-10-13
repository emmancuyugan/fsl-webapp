from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from logic import SUPPORTED_PHRASES, random_phrase, evaluate_attempt, teaching_for, recognize_attempt_stub
import os

app = Flask(__name__)

# 🗃 Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://fsl_admin:admin123@localhost:5432/fsl_database"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# User model (optional for now)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    progress = db.Column(db.Float, default=0.0)


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
    data = request.get_json(force=True) or {}
    target = (data.get("target") or "").strip()
    attempt_text = (data.get("attempt_text") or "").strip()

    if not attempt_text:
        frames = data.get("frames") or []
        attempt_text = recognize_attempt_stub(frames)

    result = evaluate_attempt(target, attempt_text)
    if not result["correct"]:
        result["teach"] = teaching_for(target)
    return jsonify(result)

@app.route("/")
def health_check():
    return jsonify({"status": "Backend API running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
