import os
import json
import sqlite3
from functools import wraps
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

# Import our ML helper
import model_utils

# ----------------------
# App config
# ----------------------
app = Flask(__name__)
app.config.from_object("config.Config")

# Auto logout after 30 minutes of inactivity
app.permanent_session_lifetime = timedelta(minutes=30)

# Database path
DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "site.db")

# ----------------------
# DB Functions
# ----------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        # Users table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        # Predictions table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_json TEXT,
            result_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        ''')
        conn.commit()

# Initialize DB
init_db()

# Load ML models once at startup
MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")
model_utils.load_models(models_dir=MODEL_DIR)

# ----------------------
# Auth Decorator
# ----------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ----------------------
# Routes
# ----------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username").strip()
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")
        password2 = request.form.get("password2")

        if not username or not email or not password:
            flash("Please fill all fields.", "danger")
            return render_template("register.html")

        if password != password2:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        hashed = generate_password_hash(password)
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, hashed)
                )
                conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username_or_email = request.form.get("username_or_email").strip()
        password = request.form.get("password")

        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (username_or_email, username_or_email)
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.", "danger")

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session.get("user_id")
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

    predictions = []
    for r in rows:
        # input_json
        try:
            input_data = json.loads(r["input_json"]) if r["input_json"] else {}
        except Exception:
            input_data = {"raw": r["input_json"]}

        # result_json (normalize to {'predictions': {...}})
        try:
            result_data = json.loads(r["result_json"]) if r["result_json"] else {}
        except Exception:
            result_data = {"raw": r["result_json"]}

        # NORMALIZATION: ensure result_data has a top-level "predictions" key
        # - if result_data already has 'predictions', keep it
        # - if result_data top-level keys look like disease keys (diabetes/heart/kidney),
        #   wrap them under 'predictions'
        # - otherwise, if result_data is not a dict, wrap it
        if isinstance(result_data, dict):
            if "predictions" not in result_data:
                # if top-level contains the disease keys, wrap them
                disease_keys = set(model_utils.FEATURES.keys())
                if any(k in result_data for k in disease_keys):
                    result_data = {"predictions": result_data}
                else:
                    # if it's an empty dict or unknown structure, keep as-is but ensure predictions exists
                    result_data = {"predictions": result_data}
        else:
            # result_data not a dict (e.g., string), wrap it
            result_data = {"predictions": result_data}

        predictions.append({
            "id": r["id"],
            "input": input_data,
            "result": result_data,
            "created_at": r["created_at"]
        })

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        predictions=predictions
    )

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        input_data = {}

        # Single Age input reused for all models:
        age_value = request.form.get("Age")
        if age_value is not None and age_value != "":
            input_data["Age"] = age_value   # for diabetes model feature name
            input_data["age"] = age_value   # for heart & kidney models

        # Grab remaining expected features (skip age since handled)
        for key, feature_list in model_utils.FEATURES.items():
            for feature in feature_list:
                if feature.lower() == "age":
                    continue
                if feature not in input_data:
                    value = request.form.get(feature)
                    if value is not None:
                        input_data[feature] = value

        # Run predictions
        results = model_utils.predict_all(input_data)

        # Save into DB
        result_data = {"predictions": results}
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO predictions (user_id, input_json, result_json) VALUES (?, ?, ?)",
                (session["user_id"], json.dumps(input_data), json.dumps(result_data))
            )
            conn.commit()

        flash("Prediction completed and saved!", "success")
        return render_template("predict.html", results=results)

    return render_template("predict.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
