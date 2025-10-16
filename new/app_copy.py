from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import requests

app = Flask(__name__)

# =============================================
# BACKEND CONFIGURATION
# =============================================
BACKEND_URL = os.getenv("BACKEND_URL", "https://your-backend-name.onrender.com")

# Secret key for session management
app.secret_key = os.urandom(24)


# =============================================
# AUTHENTICATION DECORATOR
# =============================================
def login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            flash("Please login to access this page", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


# =============================================
# WEB ROUTES (Frontend)
# =============================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/tutor")
@login_required
def tutor():
    return render_template("tutor.html")


@app.route("/activity")
@login_required
def activity():
    return render_template("activity.html")


@app.route("/results")
@login_required
def results():
    return render_template("results.html")


# =============================================
# LOGIN & SIGNUP (via Cloud Backend)
# =============================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            response = requests.post(f"{BACKEND_URL}/api/login", json={
                "username": username,
                "password": password
            })
            data = response.json()
        except Exception as e:
            flash(f"Error connecting to backend: {str(e)}", "error")
            return render_template("login.html")

        if data.get("success"):
            session["user"] = username
            session["authenticated"] = True
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash(data.get("message", "Login failed."), "error")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        try:
            response = requests.post(f"{BACKEND_URL}/api/signup", json={
                "username": username,
                "password": password,
                "confirm_password": confirm_password
            })
            data = response.json()
        except Exception as e:
            flash(f"Error connecting to backend: {str(e)}", "error")
            return render_template("signup.html")

        if data.get("success"):
            flash("Registration successful! Please login with your credentials.", "success")
            return redirect(url_for("login"))
        else:
            flash(data.get("message", "Signup failed."), "error")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# =============================================
# DEMO CONNECTION TO BACKEND API
# =============================================
@app.route("/get_phrases_demo")
def get_phrases_demo():
    """Demo route to test backend connectivity"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/phrases")
        data = response.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# =============================================
# ENTRY POINT
# =============================================
if __name__ == "__main__":
    # Local frontend app (connects to Render backend)
    print(f"Using backend at: {BACKEND_URL}")
    app.run(host="0.0.0.0", port=8000, debug=True)