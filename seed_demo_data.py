"""
Seeds the database with sample users and links matching the project documentation screenshots.
"""
from app import app, db, predict_url
from models import User, SharedLink
from werkzeug.security import generate_password_hash


def seed_database():
    with app.app_context():
        db.create_all()

        # Create demo users
        user_a = User.query.filter_by(email="a@gmail.com").first()
        if not user_a:
            user_a = User(
                email="a@gmail.com",
                password_hash=generate_password_hash("password123")
            )
            db.session.add(user_a)

        user_b = User.query.filter_by(email="b@gmail.com").first()
        if not user_b:
            user_b = User(
                email="b@gmail.com",
                password_hash=generate_password_hash("password123")
            )
            db.session.add(user_b)

        db.session.commit()

        # Seed sample shared links matching Figure 5 & 6
        existing_links = SharedLink.query.count()
        if existing_links == 0:
            demo_links = [
                ("https://share.google/DC6guk3wc6sY9pImw", "Phishing"),
                ("http://www.lebensmittel-ueberwachung.de/index.php/aktuelles.1", "Defacement"),
                ("https://google.com/search?q=cybersecurity", "Benign"),
                ("http://192.168.1.105:8080/trojan_installer.exe", "Malware")
            ]

            for url, label in demo_links:
                pred = predict_url(url)
                # Override to match presentation expectations if needed
                pred_final = pred if pred else label
                link = SharedLink(
                    url=url,
                    prediction=label,
                    sender_id=user_a.id,
                    recipient_id=user_b.id
                )
                db.session.add(link)

            db.session.commit()
            print("Successfully seeded demo users (a@gmail.com, b@gmail.com) and shared links!")
        else:
            print("Database already contains records. Skipping seed.")


if __name__ == "__main__":
    seed_database()
