from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from logic import SUPPORTED_PHRASES, random_phrase, evaluate_attempt, teaching_for, recognize_attempt_stub
import os

app = Flask(__name__)

# =============================================
# DATABASE CONFIGURATION
# =============================================
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    # Fallback (local)
    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://fsl_admin:admin123@localhost:5432/fsl_database"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# =============================================
# DATABASE MODEL
# =============================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    progress = db.Column(db.Float, default=0.0)


# =============================================
# AUTHENTICATION ROUTES (Signup / Login)
# =============================================
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not all([username, password, confirm_password]):
        return jsonify({"success": False, "message": "All fields are required."})
    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match."})

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"success": False, "message": "Username already exists."})

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True, "message": "User registered successfully."})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Missing username or password."})

    user = User.query.filter_by(username=username, password=password).first()
    if user:
        return jsonify({"success": True, "message": "Login successful."})
    else:
        return jsonify({"success": False, "message": "Invalid credentials."})


# =============================================
# FSL PHRASE / ASSESSMENT APIs
# =============================================
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
    Request body:
    {
        "target": "good morning",
        "attempt_text": "good evening",  # optional
        "frames": ["data:image/png;base64,..."]  # optional
    }
    """
    data = request.get_json(force=True) or {}
    target = (data.get("target") or "").strip()
    attempt_text = (data.get("attempt_text") or "").strip()

    # Fallback: use stub recognizer if no attempt_text provided
    if not attempt_text:
        frames = data.get("frames") or []
        attempt_text = recognize_attempt_stub(frames)

    result = evaluate_attempt(target, attempt_text)
    if not result["correct"]:
        result["teach"] = teaching_for(target)
    return jsonify(result)


# =============================================
# HEALTH CHECK
# =============================================
@app.route("/")
def health_check():
    return jsonify({"status": "Backend API running", "database_connected": bool(DATABASE_URL)})


# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
