from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "admin_user"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    # Relationships for sent and received shared links
    sent_links = db.relationship("SharedLink", foreign_keys="SharedLink.sender_id", backref="sender", lazy=True)
    received_links = db.relationship("SharedLink", foreign_keys="SharedLink.recipient_id", backref="recipient", lazy=True)

    def __repr__(self):
        return f"<User {self.email}>"


class SharedLink(db.Model):
    __tablename__ = "shared_links"

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2048), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("admin_user.id"), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey("admin_user.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SharedLink {self.url} -> {self.prediction}>"
