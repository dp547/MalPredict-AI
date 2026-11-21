import os
import joblib
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, SharedLink
from ml_pipeline import extract_features, LGBMClassifier

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Application Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'url_security_secret')

# Configure database: Default to SQLite for zero-setup execution, or MySQL if specified in DATABASE_URL
db_path = os.path.join(BASE_DIR, 'url_security.db').replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Load Machine Learning Model and TF-IDF Vectorizer
MODEL_PATH = os.path.join(BASE_DIR, "lightgbm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = None
vectorizer = None


def load_ml_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("Loaded trained LightGBM model and TF-IDF vectorizer successfully.")
        except Exception as e:
            print(f"Error loading models: {e}. Retraining...")
            from train import train_model
            train_model()
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Model artifacts not found. Starting automatic training...")
        from train import train_model
        train_model()
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)


def init_db_and_seed():
    db.create_all()
    # Ensure demo accounts exist
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

    # Seed initial demo shared links matching PDF figures if empty
    if SharedLink.query.count() == 0:
        demo_links = [
            ("https://share.google/DC6guk3wc6sY9pImw", "Phishing"),
            ("http://www.lebensmittel-ueberwachung.de/index.php/aktuelles.1", "Defacement"),
            ("https://google.com/search?q=cybersecurity", "Benign"),
            ("http://192.168.1.105:8080/trojan_installer.exe", "Malware")
        ]
        for url_str, label in demo_links:
            db.session.add(SharedLink(
                url=url_str,
                prediction=label,
                sender_id=user_a.id,
                recipient_id=user_b.id
            ))
        db.session.commit()


with app.app_context():
    init_db_and_seed()
    load_ml_artifacts()


def predict_url(url):
    """
    Classifies a URL into one of the four categories:
    0: Benign, 1: Phishing, 2: Malware, 3: Defacement
    """
    if not url:
        return "Benign"

    # Lexical features extraction
    lexical_features = extract_features(url)

    # TF-IDF character n-gram extraction
    tfidf_features = vectorizer.transform([url]).toarray()

    # Feature combination as in Appendix sample code
    final_features = np.concatenate((tfidf_features, [lexical_features]), axis=1)

    # LightGBM prediction
    prediction = model.predict(final_features)

    classes = {
        0: "Benign",
        1: "Phishing",
        2: "Malware",
        3: "Defacement"
    }

    return classes.get(int(prediction[0]), "Benign")


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email address and password.", "danger")
            return render_template("register.html")

        # Check for existing user
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists. Please log in.", "danger")
            return redirect(url_for("login"))

        # Create new user with hashed password
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["user_email"] = user.email
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid login credentials", "danger")
            return render_template("login.html")

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    current_user_id = session["user_id"]
    current_user = db.session.get(User, current_user_id)
    if not current_user:
        session.clear()
        return redirect(url_for("login"))

    # Available users to share with (exclude self)
    available_users = User.query.filter(User.id != current_user_id).all()

    prediction_result = None
    tested_url = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        recipient_id_str = request.form.get("recipient", "").strip()

        if url:
            tested_url = url
            prediction_result = predict_url(url)

            # Determine recipient: if selected, use it; otherwise, attribute to self
            recipient = None
            if recipient_id_str and recipient_id_str.isdigit():
                recipient = db.session.get(User, int(recipient_id_str))

            target_recipient_id = recipient.id if recipient else current_user_id

            new_link = SharedLink(
                url=url,
                prediction=prediction_result,
                sender_id=current_user_id,
                recipient_id=target_recipient_id
            )
            db.session.add(new_link)
            db.session.commit()

            if recipient:
                if prediction_result == "Phishing":
                    flash(f"Warning: Phishing link detected and shared with caution alert!", "warning")
                elif prediction_result in ["Malware", "Defacement"]:
                    flash(f"Alert: {prediction_result} link detected and shared with danger alert!", "danger")
                else:
                    flash(f"Safe: Benign link verified and shared successfully!", "success")
            else:
                # Standalone URL Analysis
                if prediction_result == "Benign":
                    flash("Analysis complete: URL is verified as Benign (Safe). Activity updated!", "success")
                elif prediction_result == "Phishing":
                    flash("Warning: Phishing threat detected in submitted URL! Activity updated.", "warning")
                else:
                    flash(f"Critical Warning: {prediction_result} threat detected in submitted URL! Activity updated.", "danger")

    # Fetch latest 5 shared links in inbox for preview (Figure 5)
    inbox_preview = (
        SharedLink.query.filter(
            (SharedLink.sender_id == current_user_id) | (SharedLink.recipient_id == current_user_id)
        )
        .order_by(SharedLink.id.desc())
        .limit(5)
        .all()
    )

    # Statistics calculation
    total_analyzed = SharedLink.query.filter(
        (SharedLink.sender_id == current_user_id) | (SharedLink.recipient_id == current_user_id)
    ).count()
    safe_links = SharedLink.query.filter(
        (SharedLink.sender_id == current_user_id) | (SharedLink.recipient_id == current_user_id),
        SharedLink.prediction == "Benign"
    ).count()
    risky_links = total_analyzed - safe_links

    return render_template(
        "dashboard.html",
        user_email=current_user.email,
        users=available_users,
        prediction=prediction_result,
        tested_url=tested_url,
        inbox_preview=inbox_preview,
        total_analyzed=total_analyzed,
        safe_links=safe_links,
        risky_links=risky_links
    )


@app.route("/share", methods=["POST"])
def share():
    if "user_id" not in session:
        return redirect(url_for("login"))

    url = request.form.get("url", "").strip()
    recipient_id = request.form.get("recipient")
    prediction = request.form.get("prediction")

    if not url or not recipient_id:
        flash("Please provide a valid URL and select a recipient.", "danger")
        return redirect(url_for("dashboard"))

    if not prediction:
        prediction = predict_url(url)

    new_link = SharedLink(
        url=url,
        prediction=prediction,
        sender_id=session["user_id"],
        recipient_id=int(recipient_id)
    )
    db.session.add(new_link)
    db.session.commit()

    flash("URL shared successfully", "success")
    return redirect(url_for("dashboard"))


@app.route("/inbox")
def inbox():
    if "user_id" not in session:
        return redirect(url_for("login"))

    current_user_id = session["user_id"]
    current_filter = request.args.get("filter", "all").lower()

    base_query = SharedLink.query.filter(
        (SharedLink.recipient_id == current_user_id) | (SharedLink.sender_id == current_user_id)
    )

    query = base_query
    # Filter categories matching Figure 6
    if current_filter == "safe":
        query = query.filter(SharedLink.prediction == "Benign")
    elif current_filter == "phishing":
        query = query.filter(SharedLink.prediction == "Phishing")
    elif current_filter == "malware":
        query = query.filter(SharedLink.prediction == "Malware")
    elif current_filter == "defacement":
        query = query.filter(SharedLink.prediction == "Defacement")

    received_links = query.order_by(SharedLink.id.desc()).all()

    # Category counts for filter pills
    all_count = base_query.count()
    safe_count = base_query.filter(SharedLink.prediction == "Benign").count()
    phishing_count = base_query.filter(SharedLink.prediction == "Phishing").count()
    malware_count = base_query.filter(SharedLink.prediction == "Malware").count()
    defacement_count = base_query.filter(SharedLink.prediction == "Defacement").count()

    return render_template(
        "inbox.html",
        links=received_links,
        current_filter=current_filter,
        all_count=all_count,
        safe_count=safe_count,
        phishing_count=phishing_count,
        malware_count=malware_count,
        defacement_count=defacement_count
    )


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_email", None)
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("home"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for real-time URL prediction (JSON)."""
    data = request.get_json(silent=True) or {}
    url = data.get("url") or request.form.get("url")
    if not url:
        return jsonify({"error": "Missing URL parameter"}), 400

    result = predict_url(url)
    return jsonify({
        "url": url,
        "prediction": result,
        "is_safe": result == "Benign"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
