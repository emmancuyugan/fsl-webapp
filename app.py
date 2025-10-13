from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from logic import (
    SUPPORTED_PHRASES,
    random_phrase,
    evaluate_attempt,
    teaching_for,
    recognize_attempt_stub,
)
import os
from flask_sqlalchemy import SQLAlchemy
import requests  # Added for possible backend API calls

app = Flask(__name__)

# ---------------------------------------------
# Backend URL configuration
# ---------------------------------------------
# Local fallback (127.0.0.1) for testing
# When deployed, set BACKEND_URL in Render environment variables
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
BACKEND_URL = os.getenv("BACKEND_URL", "https://fsl-webapp.onrender.com/")

# ---------------------------------------------
# Database Configuration (local use)
# ---------------------------------------------
# Your local DB setup for user login/register
# Render backend (api.py) will have its own DATABASE_URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://fsl_admin:admin123@localhost:5432/fsl_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------------------------------------
# Database Models
# ---------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    progress = db.Column(db.Float, default=0.0)

app.secret_key = os.urandom(24)  # Required for session management

# ---------------------------------------------
# User Management Functions
# ---------------------------------------------
def register_user(username, password):
    """Register a new user in database"""
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, "Username already exists"

    new_user = User(username=username, progress=0.0, password=password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return True, "Registration successful"
    except Exception as e:
        db.session.rollback()
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate a user from database"""
    # Demo admin
    if username == "admin" and password == "fsl2025":
        return True

    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        return True

    return False

# ---------------------------------------------
# Authentication Decorator
# ---------------------------------------------
def login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            flash("Please login to access this page", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ---------------------------------------------
# Web Routes (Frontend)
# ---------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

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

# ---------------------------------------------
# Protected Pages
# ---------------------------------------------
@app.route("/activity")
@login_required
def activity():
    return render_template("activity.html")

@app.route("/results")
@login_required
def results():
    return render_template("results.html")

# ---------------------------------------------
# Example connection to backend API (Render)
# ---------------------------------------------
@app.route("/get_phrases_demo")
def get_phrases_demo():
    """Example route to show how frontend connects to backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/phrases")
        data = response.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

# ---------------------------------------------
# Run the local frontend
# ---------------------------------------------
if __name__ == "__main__":
    # Local frontend app, separate from Render backend
    app.run(host="0.0.0.0", port=8000, debug=True)
