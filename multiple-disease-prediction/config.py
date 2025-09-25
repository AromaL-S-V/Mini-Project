import os

# Base directory of the project
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Secret key for session management
    SECRET_KEY = os.environ.get("SECRET_KEY") or "supersecretkey123"

    # SQLite database
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or \
        "sqlite:///" + os.path.join(basedir, "site.db")

    SQLALCHEMY_TRACK_MODIFICATIONS = False
