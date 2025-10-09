from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from backend import (
    SUPPORTED_PHRASES,
    random_phrase,
    evaluate_attempt,
    teaching_for,
    recognize_attempt_stub,
)
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# PostgreSQL connection URI format:
# postgresql://username:password@host:port/databasename
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://fsl_admin:admin123@localhost:5432/fsl_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    progress = db.Column(db.Float, default=0.0)
app.secret_key = os.urandom(24)  # Required for session management

# User management functions using database
def register_user(username, password):
    """Register a new user in database"""
    # Check if user already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, "Username already exists"

    # Create new user (in production, hash the password)
    new_user = User(
        username=username,
        progress=0.0
    )
    # Store password in a new field (in production, hash this)
    new_user.password = password

    try:
        db.session.add(new_user)
        db.session.commit()
        return True, "Registration successful"
    except Exception as e:
        db.session.rollback()
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate a user from database"""
    # Check demo admin account
    if username == "admin" and password == "fsl2025":
        return True

    # Check database users
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        return True

    return False

# Authentication decorator - must be defined before routes that use it
def login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            flash("Please login to access this page", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/tutor")
@login_required
def tutor():
    return render_template("tutor.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if authenticate_user(username, password):
            session["user"] = username
            session["authenticated"] = True
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # Validation
        if not all([username, password, confirm_password]):
            flash("All fields are required", "error")
        elif password != confirm_password:
            flash("Passwords do not match", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters long", "error")
        else:
            success, message = register_user(username, password)
            if success:
                flash("Registration successful! Please login with your credentials.", "success")
                return redirect(url_for("login"))
            else:
                flash(message, "error")

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("home"))

# Protect sensitive routes
@app.route("/activity")
@login_required
def activity():
    return render_template("activity.html")



@app.route("/results")
@login_required
def results():
    return render_template("results.html")

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
