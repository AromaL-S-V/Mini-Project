from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin
from datetime import datetime

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()

# ----------------- User Model -----------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password_hash = db.Column(db.String(128), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        """Hashes password and stores it securely"""
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        """Checks hashed password against user input"""
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    diabetes = db.Column(db.Boolean, nullable=False)
    diabetes_confidence = db.Column(db.Float)

    heart = db.Column(db.Boolean, nullable=False)
    heart_confidence = db.Column(db.Float)

    kidney = db.Column(db.Boolean, nullable=False)
    kidney_confidence = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)