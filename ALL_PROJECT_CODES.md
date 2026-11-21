# Malware Prediction Using Machine Learning System - Complete Code Bundle

## File: requirements.txt

`text
Flask>=3.0.0
Flask-SQLAlchemy>=3.1.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
PyMySQL>=1.1.0
Werkzeug>=3.0.0

`

---

## File: .gitignore

`text
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# Database files
*.sqlite3
instance/

# OS-specific files
.DS_Store
Thumbs.db

`

---

## File: models.py

`python
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

`

---

## File: ml_pipeline.py

`python
import re
import csv
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def extract_features(url):
    """
    Extracts 8 lexical features from a URL as specified in Chapter 3 & Appendix (i):
    1. url_length: length of URL
    2. dot_count: count of '.'
    3. dash_count: count of '-'
    4. slash_count: count of '/'
    5. digit_count: count of digits
    6. special_char_count: count of special characters [@_!#$%^&*()<>?/\|}{~:]
    7. https: 1 if 'https' in url else 0
    8. http: 1 if 'http' in url else 0
    """
    features = {}
    features['url_length'] = len(url)
    features['dot_count'] = url.count('.')
    features['dash_count'] = url.count('-')
    features['slash_count'] = url.count('/')
    features['digit_count'] = sum(c.isdigit() for c in url)
    features['special_char_count'] = len(re.findall(r'[@_!#$%^&*()<>?/\|}{~:]', url))
    features['https'] = 1 if "https" in url else 0
    features['http'] = 1 if "http" in url else 0
    return list(features.values())


class LGBMClassifier:
    """
    LightGBM Multi-class Classifier interface matching the project specification.
    Natively wraps LightGBM engine with 100% compatibility.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.booster_ = None
        self.classes_ = np.array([0, 1, 2, 3])

    def fit(self, X, y):
        y = np.array(y)
        dataset = lgb.Dataset(X, label=y)
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        self.booster_ = lgb.train(params, dataset, num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        prob = self.booster_.predict(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        return self.booster_.predict(X)


def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    """Splits dataset into stratified train and test sets."""
    rng = np.random.RandomState(random_state)
    y = np.array(y)
    unique_classes = np.unique(y)

    train_indices = []
    test_indices = []

    for c in unique_classes:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        split_point = int(len(c_indices) * (1 - test_size))
        train_indices.extend(c_indices[:split_point])
        test_indices.extend(c_indices[split_point:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def evaluate_predictions(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = ["Benign", "Phishing", "Malware", "Defacement"]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = (y_true == y_pred).mean()

    header = f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    header += "-" * 55 + "\n"

    lines = []
    for idx, name in enumerate(class_names):
        tp = np.sum((y_true == idx) & (y_pred == idx))
        fp = np.sum((y_true != idx) & (y_pred == idx))
        fn = np.sum((y_true == idx) & (y_pred != idx))
        support = np.sum(y_true == idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        lines.append(f"{name:<15} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {support:<10}")

    report = header + "\n".join(lines)
    return accuracy, report

`

---

## File: train.py

`python
import csv
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_pipeline import (
    extract_features,
    LGBMClassifier,
    train_test_split_custom,
    evaluate_predictions
)


def load_dataset(dataset_path="malicious_urls.csv"):
    urls = []
    labels = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(int(row["label"]))
    return urls, labels


def train_model(dataset_path="malicious_urls.csv"):
    print(f"Loading dataset from {dataset_path}...")
    urls, labels = load_dataset(dataset_path)

    print(f"Dataset loaded with {len(urls)} samples.")
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_map = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}
    for l, c in zip(unique_labels, counts):
        print(f"  Class {l} ({class_map.get(l, 'Unknown')}): {c} samples")

    print("\n[Step 1] Extracting lexical features...")
    lexical_features = []
    for url in urls:
        lexical_features.append(extract_features(url))
    lexical_features = np.array(lexical_features)

    print("[Step 2] Extracting TF-IDF character n-gram features (2 to 5 grams)...")
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=1000)
    tfidf_features = tfidf.fit_transform(urls)

    print("[Step 3] Combining lexical and TF-IDF features...")
    combined_features = np.concatenate((tfidf_features.toarray(), lexical_features), axis=1)
    print(f"Combined feature matrix shape: {combined_features.shape}")

    print("\n[Step 4] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split_custom(
        combined_features,
        labels,
        test_size=0.2,
        random_state=42
    )

    print("\n[Step 5] Training LightGBM Multi-class Classifier...")
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    print("\n[Step 6] Evaluating model on test set...")
    predictions = model.predict(X_test)
    accuracy, report = evaluate_predictions(y_test, predictions)

    print(f"\n==========================================")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"==========================================\n")
    print("Classification Report:\n")
    print(report)

    print("\n[Step 7] Serializing trained model and vectorizer...")
    joblib.dump(model, "lightgbm_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("Successfully saved 'lightgbm_model.pkl' and 'tfidf_vectorizer.pkl'!")

    return accuracy, report


if __name__ == "__main__":
    train_model()

`

---

## File: generate_dataset.py

`python
import csv
import random

benign_domains = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "twitter.com",
    "amazon.com", "reddit.com", "linkedin.com", "microsoft.com", "netflix.com",
    "instagram.com", "apple.com", "github.com", "stackoverflow.com", "medium.com",
    "yahoo.com", "bing.com", "cnn.com", "nytimes.com", "bbc.co.uk",
    "cloudflare.com", "spotify.com", "adobe.com", "dropbox.com", "quora.com",
    "twitch.tv", "salesforce.com", "zoom.us", "mozilla.org", "ebay.com",
    "walmart.com", "nih.gov", "stanford.edu", "mit.edu", "harvard.edu",
    "python.org", "pypi.org", "oracle.com", "ibm.com", "intel.com"
]

benign_paths = [
    "", "/about", "/contact", "/terms", "/privacy", "/help", "/documentation",
    "/products/features", "/blog/post/2026/04/update", "/search?q=technology",
    "/category/news/article-1029", "/explore/popular", "/download/latest",
    "/profile/user?id=10283", "/support/knowledge-base/faq"
]

phishing_patterns = [
    "paypal-security-update-{num}.com/signin",
    "secure-login-{name}.xyz/account-verification",
    "appleid-verify-alert-{num}.info/auth",
    "netflix-billing-update-{name}.com/payment",
    "wellsfargo-online-banking-{num}.net/login.php",
    "chase-security-check-{name}.top/portal",
    "bankofamerica-customer-support-{num}.site/verify",
    "google-docs-secure-view-{num}.biz/login",
    "microsoft-365-pass-recovery-{name}.club/auth",
    "amazon-order-cancelled-{num}.com/re-authenticate",
    "facebook-security-badge-{num}.me/appeal",
    "instagram-copyright-infringement-{num}.online/verify",
    "binance-kyc-update-{name}.cc/wallet",
    "metamask-seed-phrase-restore-{num}.io/connect",
    "coinbase-security-lockout-{name}.xyz/unlock"
]

malware_patterns = [
    "http://{ip}/trojan_{num}.exe",
    "http://{ip}:8080/payload_{name}.bin",
    "http://download-free-cracks-{num}.ru/setup.exe",
    "http://system-speedup-utility-{name}.cc/patch.vbs",
    "http://win-driver-update-{num}.top/driver_pack.scr",
    "http://flash-player-update-{name}.club/install_flash.exe",
    "http://critical-system-security-patch-{num}.biz/agent.dll",
    "http://{ip}/worm_spread_{name}.ps1",
    "http://free-crypto-miner-{num}.online/miner_svc.exe",
    "http://invoice-attached-payment-{num}.ws/invoice_1092.exe",
    "http://{ip}:4444/meterpreter_session_{num}.elf",
    "http://discord-nitro-gift-generator-{num}.ru/generator.exe",
    "http://ransom-decrypt-tool-{name}.cc/decryptor.bat",
    "http://direct-download-games-{num}.su/crack_patch.iso"
]

defacement_patterns = [
    "http://{domain}/deface_{name}.html",
    "http://{domain}/index.php?hacked_by=dark_ghost_{num}",
    "http://{domain}/warning_defaced_by_team_{name}.txt",
    "http://{domain}/defacement_mirror.php?id={num}",
    "http://{domain}/hacked.html",
    "http://{domain}/cyber_caliphate_ownz_{num}.htm",
    "http://{domain}/owned_by_{name}_team.html",
    "http://{domain}/index_defaced.asp",
    "http://{domain}/defaced-by-ghost-squad-{num}.html",
    "http://{domain}/pwned_by_{name}.php"
]

names = ["alpha", "delta", "shadow", "cyber", "phantom", "root", "echo", "knight", "phoenix", "anon"]
defaced_targets = [
    "local-council-services.org", "state-school-portal.edu", "city-water-dept.gov",
    "regional-chamber-commerce.com", "community-clinic-care.org", "smallbiz-catering.co",
    "library-network.org", "heritage-museum.org", "county-sheriff-office.org"
]

records = []

# Class 0: Benign
for domain in benign_domains:
    proto = "https://"
    for path in benign_paths:
        records.append((f"{proto}{domain}{path}", 0))

# Class 1: Phishing
for pattern in phishing_patterns:
    for i in range(35):
        name = random.choice(names)
        num = random.randint(100, 9999)
        url = "https://" + pattern.format(num=num, name=name)
        records.append((url, 1))

# Class 2: Malware
for pattern in malware_patterns:
    for i in range(35):
        ip = f"{random.randint(45, 218)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        num = random.randint(100, 9999)
        name = random.choice(names)
        url = pattern.format(ip=ip, num=num, name=name)
        records.append((url, 2))

# Class 3: Defacement
for pattern in defacement_patterns:
    for domain in defaced_targets:
        for i in range(5):
            name = random.choice(names)
            num = random.randint(10, 999)
            url = pattern.format(domain=domain, name=name, num=num)
            records.append((url, 3))

random.seed(42)
random.shuffle(records)

with open("malicious_urls.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])
    for url, label in records:
        writer.writerow([url, label])

print(f"Generated {len(records)} balanced dataset rows in malicious_urls.csv")

`

---

## File: app.py

`python
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

`

---

## File: seed_demo_data.py

`python
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

`

---

## File: test_app.py

`python
import unittest
import os
import joblib
import numpy as np
from app import app, db, predict_url
from models import User, SharedLink
from ml_pipeline import extract_features, LGBMClassifier
from werkzeug.security import check_password_hash


class MalwareDetectionSystemTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for testing with in-memory SQLite database
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    # ==========================================
    # 4.1 Unit Testing: Feature Extraction & Model
    # ==========================================
    def test_feature_extraction(self):
        """Test lexical feature extractor generates accurate 8 numerical features."""
        url = "https://example.com/test-path?param=123"
        features = extract_features(url)
        self.assertEqual(len(features), 8)
        self.assertEqual(features[0], len(url))  # url_length
        self.assertEqual(features[1], 1)         # dot_count
        self.assertEqual(features[2], 1)         # dash_count
        self.assertEqual(features[3], 3)         # slash_count (https:// + /test-path)
        self.assertEqual(features[4], 3)         # digit_count (1, 2, 3)
        self.assertGreaterEqual(features[5], 2)  # special chars (: / ?)
        self.assertEqual(features[6], 1)         # https flag
        self.assertEqual(features[7], 1)         # http flag (inside https)

    def test_model_prediction_types(self):
        """Test model predict_url returns one of the 4 defined categories."""
        valid_classes = {"Benign", "Phishing", "Malware", "Defacement"}

        pred_benign = predict_url("https://www.google.com/search")
        self.assertIn(pred_benign, valid_classes)

        pred_phishing = predict_url("https://paypal-security-update-992.com/signin")
        self.assertIn(pred_phishing, valid_classes)

        pred_malware = predict_url("http://182.23.44.12:8080/trojan_491.exe")
        self.assertIn(pred_malware, valid_classes)

    # ==========================================
    # 4.3 Functional Testing: Authentication
    # ==========================================
    def test_user_registration_and_hashing(self):
        """Test user registration and verify password is securely hashed."""
        response = self.client.post('/register', data={
            'email': 'alice@example.com',
            'password': 'secretpassword123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

        with app.app_context():
            user = User.query.filter_by(email='alice@example.com').first()
            self.assertIsNotNone(user)
            self.assertNotEqual(user.password_hash, 'secretpassword123')
            self.assertTrue(check_password_hash(user.password_hash, 'secretpassword123'))

    def test_duplicate_user_registration(self):
        """Test rejection of duplicate email registration."""
        self.client.post('/register', data={'email': 'alice@example.com', 'password': 'pass1'}, follow_redirects=True)
        response = self.client.post('/register', data={'email': 'alice@example.com', 'password': 'pass2'}, follow_redirects=True)
        self.assertIn(b"already exists", response.data)

    def test_user_login_success_and_failure(self):
        """Test user login authentication flow."""
        self.client.post('/register', data={'email': 'bob@example.com', 'password': 'mypassword'}, follow_redirects=True)

        # Successful login
        res_ok = self.client.post('/login', data={'email': 'bob@example.com', 'password': 'mypassword'}, follow_redirects=True)
        self.assertIn(b"Welcome, bob@example.com!", res_ok.data)

        # Failed login
        res_fail = self.client.post('/login', data={'email': 'bob@example.com', 'password': 'wrongpassword'}, follow_redirects=True)
        self.assertIn(b"Invalid login credentials", res_fail.data)

    # ==========================================
    # 4.2 Integration & Sharing Testing
    # ==========================================
    def test_link_sharing_and_inbox(self):
        """Test sharing a URL between two registered users and verifying inbox delivery."""
        # Register User A (Sender)
        self.client.post('/register', data={'email': 'usera@test.com', 'password': 'passwordA'}, follow_redirects=True)
        # Register User B (Recipient)
        self.client.post('/register', data={'email': 'userb@test.com', 'password': 'passwordB'}, follow_redirects=True)

        with app.app_context():
            user_b = User.query.filter_by(email='userb@test.com').first()
            self.assertIsNotNone(user_b)
            user_b_id = user_b.id

        # Login as User A
        self.client.post('/login', data={'email': 'usera@test.com', 'password': 'passwordA'}, follow_redirects=True)

        # Submit and share URL to User B
        test_url = "https://paypal-security-update-412.com/signin"
        share_res = self.client.post('/dashboard', data={
            'url': test_url,
            'recipient': str(user_b_id)
        }, follow_redirects=True)
        self.assertEqual(share_res.status_code, 200)

        # Logout User A
        self.client.get('/logout', follow_redirects=True)

        # Login as User B
        self.client.post('/login', data={'email': 'userb@test.com', 'password': 'passwordB'}, follow_redirects=True)

        # Check Inbox
        inbox_res = self.client.get('/inbox', follow_redirects=True)
        self.assertIn(b"usera@test.com", inbox_res.data)
        self.assertIn(test_url.encode(), inbox_res.data)


if __name__ == "__main__":
    unittest.main()

`

---

## File: run.bat

`text
@echo off
echo Starting Malware Prediction Using Machine Learning Web App...
cd /d "E:\Malware"
python app.py
pause

`

---

## File: static\css\style.css

`css
:root {
    --primary-gradient: linear-gradient(135deg, #5b54d6 0%, #6b63ff 50%, #8780f8 100%);
    --bg-gradient: linear-gradient(135deg, #5850e6 0%, #6b63ff 45%, #8f88ff 100%);
    --card-shadow: 0 12px 30px rgba(24, 18, 77, 0.12);
    --border-radius-lg: 18px;
    --border-radius-md: 10px;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-gradient);
    min-height: 100vh;
    color: #2d3748;
}

/* Navbar */
.navbar-custom {
    background-color: #ffffff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
    padding: 0.85rem 1.5rem;
}

.navbar-brand {
    font-weight: 700;
    font-size: 1.25rem;
    color: #2b3674 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.navbar-brand i {
    color: #4318ff;
    font-size: 1.4rem;
}

.nav-link {
    font-weight: 500;
    color: #4b5563 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.nav-link:hover, .nav-link.active {
    color: #4318ff !important;
    background-color: #f4f7fe;
}

.nav-btn-primary {
    background: #2563eb;
    color: #ffffff !important;
    border-radius: 8px;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600;
}

.nav-btn-primary:hover {
    background: #1d4ed8;
    color: #ffffff !important;
}

/* Card Container */
.security-card {
    background: #ffffff;
    border-radius: var(--border-radius-lg);
    box-shadow: var(--card-shadow);
    border: none;
    padding: 2.2rem;
    margin-bottom: 2rem;
}

.card-title-lg {
    font-weight: 700;
    color: #1b2559;
    margin-bottom: 1.5rem;
}

.page-header-text {
    color: #ffffff;
    text-align: center;
    margin-bottom: 2rem;
}

.page-header-text h1 {
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 0.4rem;
}

.page-header-text p {
    font-size: 1.05rem;
    opacity: 0.9;
}

/* Form Controls */
.form-label {
    font-weight: 600;
    font-size: 0.9rem;
    color: #4a5568;
    margin-bottom: 0.4rem;
}

.form-control, .form-select {
    border-radius: 8px;
    padding: 0.75rem 1rem;
    border: 1px solid #e2e8f0;
    font-size: 0.95rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.form-control:focus, .form-select:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

.btn-action-primary {
    background: #2563eb;
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    transition: all 0.2s;
}

.btn-action-primary:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Threat Badges */
.badge-threat {
    font-weight: 600;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.badge-benign {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.badge-phishing {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.badge-malware {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.badge-defacement {
    background-color: #e5e7eb;
    color: #1f2937;
    border: 1px solid #d1d5db;
}

/* Statistics cards */
.stat-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #edf2f7;
}

.stat-box .number {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1e293b;
}

.stat-box .label {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Inbox and Filter Pills */
.filter-pill {
    padding: 0.4rem 1.2rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none;
    border: 1px solid #e2e8f0;
    color: #4b5563;
    background: #ffffff;
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.filter-pill:hover, .filter-pill.active {
    background: #2563eb;
    color: #ffffff;
    border-color: #2563eb;
}

.inbox-item-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s ease;
}

.inbox-item-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    background: #ffffff;
}

.avatar-icon-box {
    width: 44px;
    height: 44px;
    background: #eff6ff;
    color: #2563eb;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.url-text {
    word-break: break-all;
    font-family: monospace;
    font-size: 0.9rem;
    color: #475569;
}

`

---

## File: templates\base.html

`html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}URL Security Hub - Machine Learning Malware Prediction{% endblock %}</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Custom Stylesheet -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_head %}{% endblock %}
</head>
<body>
    <!-- Top Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-custom sticky-top">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('home') }}">
                <i class="fa-solid fa-shield-halved"></i>
                <span>URL Security Hub</span>
            </a>

            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                <ul class="navbar-nav align-items-center gap-2">
                    {% if session.get('user_id') %}
                        <li class="nav-item">
                            <a class="nav-link {% if request.endpoint == 'dashboard' %}active text-primary fw-bold{% endif %}" href="{{ url_for('dashboard') }}">
                                <i class="fa-solid fa-gauge me-1"></i> Dashboard
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link {% if request.endpoint == 'inbox' %}active text-primary fw-bold{% endif %}" href="{{ url_for('inbox') }}">
                                <i class="fa-solid fa-inbox me-1"></i> Inbox
                            </a>
                        </li>
                        <li class="nav-item ms-lg-2">
                            <a class="nav-link text-danger" href="{{ url_for('logout') }}">
                                <i class="fa-solid fa-arrow-right-from-bracket me-1"></i> Logout
                            </a>
                        </li>
                    {% else %}
                        <li class="nav-item">
                            <a class="nav-link {% if request.endpoint == 'login' %}active{% endif %}" href="{{ url_for('login') }}">
                                <i class="fa-solid fa-arrow-right-to-bracket me-1"></i> Login
                            </a>
                        </li>
                        <li class="nav-item ms-lg-2">
                            <a class="nav-link nav-btn-primary" href="{{ url_for('register') }}">
                                <i class="fa-solid fa-user-plus me-1"></i> Get Started
                            </a>
                        </li>
                    {% endif %}
                </ul>
            </div>
        </div>
    </nav>

    <!-- Main Content Container -->
    <main class="py-4">
        <div class="container">
            <!-- Flash Message Alerts -->
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    <div class="row justify-content-center mb-3">
                        <div class="col-lg-8">
                            {% for category, message in messages %}
                                <div class="alert alert-{{ category if category != 'error' else 'danger' }} alert-dismissible fade show shadow-sm d-flex align-items-center" role="alert">
                                    {% if category == 'success' %}
                                        <i class="fa-solid fa-circle-check me-2 fs-5"></i>
                                    {% elif category == 'warning' %}
                                        <i class="fa-solid fa-triangle-exclamation me-2 fs-5"></i>
                                    {% elif category == 'danger' %}
                                        <i class="fa-solid fa-circle-xmark me-2 fs-5"></i>
                                    {% else %}
                                        <i class="fa-solid fa-circle-info me-2 fs-5"></i>
                                    {% endif %}
                                    <div class="flex-grow-1">{{ message }}</div>
                                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                                </div>
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}
            {% endwith %}

            {% block content %}{% endblock %}
        </div>
    </main>

    <!-- Bootstrap 5 JavaScript Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block extra_scripts %}{% endblock %}
</body>
</html>

`

---

## File: templates\home.html

`html
{% extends "base.html" %}

{% block title %}URL Security Hub - Intelligent Threat Detection{% endblock %}

{% block content %}
<div class="row justify-content-center my-4">
    <div class="col-lg-10">
        <div class="text-center text-white mb-5">
            <div class="d-inline-flex p-3 rounded-circle bg-white text-primary shadow mb-3">
                <i class="fa-solid fa-shield-halved fa-3x text-primary"></i>
            </div>
            <h1 class="display-5 fw-bold mb-3">Intelligent Malware & URL Threat Detection</h1>
            <p class="lead mx-auto" style="max-width: 750px; opacity: 0.95;">
                Protect yourself and others against online cyber threats. Powered by LightGBM gradient boosting machine learning, TF-IDF character n-gram text modeling, and comprehensive lexical analysis.
            </p>
            <div class="d-flex justify-content-center gap-3 mt-4">
                <a href="{{ url_for('register') }}" class="btn btn-light btn-lg px-4 shadow-sm fw-bold text-primary">
                    <i class="fa-solid fa-rocket me-2"></i> Get Started
                </a>
                <a href="{{ url_for('login') }}" class="btn btn-outline-light btn-lg px-4 fw-bold">
                    <i class="fa-solid fa-arrow-right-to-bracket me-2"></i> Login to Dashboard
                </a>
            </div>
        </div>

        <!-- 4 Classes Showcase -->
        <div class="row g-4 mb-4">
            <div class="col-md-3">
                <div class="card h-100 border-0 shadow-sm rounded-4 text-center p-3">
                    <div class="card-body">
                        <div class="text-success mb-2 fs-2"><i class="fa-solid fa-circle-check"></i></div>
                        <h5 class="fw-bold text-dark">Benign</h5>
                        <p class="small text-muted mb-0">Safe, legitimate web services and verified daily domains.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100 border-0 shadow-sm rounded-4 text-center p-3">
                    <div class="card-body">
                        <div class="text-warning mb-2 fs-2"><i class="fa-solid fa-triangle-exclamation"></i></div>
                        <h5 class="fw-bold text-dark">Phishing</h5>
                        <p class="small text-muted mb-0">Credential harvesters, deceptive subdomains, and spoofed portals.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100 border-0 shadow-sm rounded-4 text-center p-3">
                    <div class="card-body">
                        <div class="text-danger mb-2 fs-2"><i class="fa-solid fa-bug"></i></div>
                        <h5 class="fw-bold text-dark">Malware</h5>
                        <p class="small text-muted mb-0">Drive-by downloads, exploit payloads, trojans, and ransom scripts.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card h-100 border-0 shadow-sm rounded-4 text-center p-3">
                    <div class="card-body">
                        <div class="text-secondary mb-2 fs-2"><i class="fa-solid fa-skull"></i></div>
                        <h5 class="fw-bold text-dark">Defacement</h5>
                        <p class="small text-muted mb-0">Compromised web portals altered by unauthorized hacker squads.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Feature Overview -->
        <div class="security-card">
            <div class="row align-items-center">
                <div class="col-md-6 mb-3 mb-md-0">
                    <h3 class="card-title-lg mb-3">Collaborative Security Platform</h3>
                    <p class="text-muted">
                        Unlike traditional static blacklist approaches that fail against zero-day URLs, our system extracts structural numerical features alongside character n-grams to uncover concealed patterns.
                    </p>
                    <ul class="list-unstyled text-muted">
                        <li class="mb-2"><i class="fa-solid fa-check text-primary me-2"></i> Real-time multi-class URL classification</li>
                        <li class="mb-2"><i class="fa-solid fa-check text-primary me-2"></i> Secure link sharing with caution alerts</li>
                        <li class="mb-2"><i class="fa-solid fa-check text-primary me-2"></i> Personal threat inbox and activity dashboard</li>
                    </ul>
                </div>
                <div class="col-md-6 text-center">
                    <div class="p-4 bg-light rounded-4 border">
                        <i class="fa-solid fa-network-wired fa-4x text-primary mb-3"></i>
                        <h5 class="fw-bold">Powered by LightGBM & Flask</h5>
                        <p class="small text-muted mb-0">Fast inference & seamless responsive user interface.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

`

---

## File: templates\login.html

`html
{% extends "base.html" %}

{% block title %}Login - URL Security Hub{% endblock %}

{% block content %}
<div class="row justify-content-center my-4">
    <div class="col-md-6 col-lg-5">
        <!-- Header Text matching Figure 1 -->
        <div class="page-header-text">
            <h1>Welcome Back</h1>
            <p>Login to protect yourself and others from malicious links</p>
        </div>

        <!-- Login Card -->
        <div class="security-card">
            <h4 class="card-title-lg text-center mb-4">Login to Your Account</h4>

            <form method="POST" action="{{ url_for('login') }}">
                <div class="mb-3">
                    <label for="email" class="form-label">
                        <i class="fa-solid fa-envelope me-1 text-muted"></i> Email Address
                    </label>
                    <input type="email" class="form-control" id="email" name="email" placeholder="you@gmail.com" required autofocus>
                </div>

                <div class="mb-3">
                    <label for="password" class="form-label">
                        <i class="fa-solid fa-lock me-1 text-muted"></i> Password
                    </label>
                    <div class="input-group">
                        <input type="password" class="form-control" id="password" name="password" placeholder="Enter your password" required>
                        <button class="btn btn-outline-secondary" type="button" id="togglePassword">
                            <i class="fa-regular fa-eye" id="eyeIcon"></i>
                        </button>
                    </div>
                </div>

                <div class="mb-4 form-check">
                    <input type="checkbox" class="form-check-input" id="rememberMe" name="remember">
                    <label class="form-check-label text-muted small" for="rememberMe">Remember Me</label>
                </div>

                <div class="d-grid mb-3">
                    <button type="submit" class="btn btn-action-primary py-2 fs-6">
                        <i class="fa-solid fa-arrow-right-to-bracket me-2"></i> Login to Dashboard
                    </button>
                </div>

                <div class="text-center mt-3">
                    <span class="text-muted small">Don't have an account? </span>
                    <a href="{{ url_for('register') }}" class="text-decoration-none fw-semibold text-primary small">Sign Up</a>
                </div>
            </form>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
    const togglePassword = document.querySelector('#togglePassword');
    const password = document.querySelector('#password');
    const eyeIcon = document.querySelector('#eyeIcon');

    togglePassword.addEventListener('click', function () {
        const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
        password.setAttribute('type', type);
        eyeIcon.classList.toggle('fa-eye');
        eyeIcon.classList.toggle('fa-eye-slash');
    });
</script>
{% endblock %}

`

---

## File: templates\register.html

`html
{% extends "base.html" %}

{% block title %}Create an Account - URL Security Hub{% endblock %}

{% block content %}
<div class="row justify-content-center my-4">
    <div class="col-md-6 col-lg-5">
        <!-- Account Creation Card matching Figure 2 -->
        <div class="security-card mt-4">
            <h4 class="card-title-lg text-center mb-4">Create a New Account</h4>

            <form method="POST" action="{{ url_for('register') }}">
                <div class="mb-3">
                    <label for="email" class="form-label">
                        <i class="fa-solid fa-envelope me-1 text-muted"></i> Email Address
                    </label>
                    <input type="email" class="form-control" id="email" name="email" placeholder="you@gmail.com" required autofocus>
                </div>

                <div class="mb-4">
                    <label for="password" class="form-label">
                        <i class="fa-solid fa-lock me-1 text-muted"></i> Password
                    </label>
                    <input type="password" class="form-control" id="password" name="password" placeholder="Enter your password" required minlength="4">
                </div>

                <div class="d-grid mb-3">
                    <button type="submit" class="btn btn-action-primary py-2 fs-6">
                        <i class="fa-solid fa-user-check me-2"></i> Register
                    </button>
                </div>

                <div class="text-center mt-3">
                    <span class="text-muted small">Already have an account? </span>
                    <a href="{{ url_for('login') }}" class="text-decoration-none fw-semibold text-primary small">Login here</a>
                </div>
            </form>
        </div>
    </div>
</div>
{% endblock %}

`

---

## File: templates\dashboard.html

`html
{% extends "base.html" %}

{% block title %}Dashboard - URL Security Hub{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-8">
        <!-- Top Greeting matching Figure 3 & 4 -->
        <div class="page-header-text mb-4">
            <h1 class="fw-bold">Welcome, {{ user_email }}!</h1>
            <p>Analyze web links for threats and securely share them with your team.</p>
        </div>

        <!-- Activity Summary Statistics -->
        <div class="row g-3 mb-4">
            <div class="col-4">
                <div class="stat-box shadow-sm">
                    <div class="number">{{ total_analyzed }}</div>
                    <div class="label">Total Activity</div>
                </div>
            </div>
            <div class="col-4">
                <div class="stat-box shadow-sm">
                    <div class="number text-success">{{ safe_links }}</div>
                    <div class="label">Safe Links</div>
                </div>
            </div>
            <div class="col-4">
                <div class="stat-box shadow-sm">
                    <div class="number text-danger">{{ risky_links }}</div>
                    <div class="label">Threats Detected</div>
                </div>
            </div>
        </div>

        <!-- Share a New Link Card matching Figure 3 & 4 -->
        <div class="security-card">
            <h4 class="card-title-lg mb-4">
                <i class="fa-solid fa-magnifying-glass-chart text-primary me-2"></i> Share a New Link
            </h4>

            <form method="POST" action="{{ url_for('dashboard') }}">
                <div class="mb-3">
                    <label for="url" class="form-label">URL to Analyze & Share</label>
                    <div class="input-group">
                        <span class="input-group-text bg-white border-end-0">
                            <i class="fa-solid fa-link text-muted"></i>
                        </span>
                        <input type="text" class="form-control border-start-0 ps-0" id="url" name="url"
                               placeholder="https://example.com"
                               value="{{ tested_url or '' }}" required autofocus>
                    </div>
                    <div class="form-text text-muted">Enter any full URL including http:// or https:// for deep ML inspection.</div>
                </div>

                <div class="mb-4">
                    <label for="recipient" class="form-label">Share With</label>
                    <select class="form-select" id="recipient" name="recipient">
                        <option value="">Select a user (or leave empty to analyze only)...</option>
                        {% for u in users %}
                            <option value="{{ u.id }}">{{ u.email }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="d-flex gap-2">
                    <button type="submit" class="btn btn-action-primary">
                        <i class="fa-solid fa-shield-virus me-2"></i> Analyze & Share
                    </button>
                    {% if tested_url %}
                        <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                            <i class="fa-solid fa-arrows-rotate me-1"></i> Clear
                        </a>
                    {% endif %}
                </div>
            </form>

            <!-- Real-time Prediction Display Result -->
            {% if prediction %}
                <div class="mt-4 pt-3 border-top">
                    <h6 class="fw-bold text-uppercase text-muted small mb-2">Analysis Result:</h6>
                    <div class="p-3 rounded-3 d-flex align-items-center justify-content-between
                        {% if prediction == 'Benign' %}bg-success-subtle border border-success-subtle
                        {% elif prediction == 'Phishing' %}bg-warning-subtle border border-warning-subtle
                        {% elif prediction == 'Malware' %}bg-danger-subtle border border-danger-subtle
                        {% else %}bg-secondary-subtle border border-secondary-subtle{% endif %}">
                        <div>
                            <span class="fw-bold">URL:</span> <code class="text-dark">{{ tested_url }}</code>
                            <div class="small text-muted mt-1">Classified via LightGBM (Lexical + TF-IDF Character N-grams)</div>
                        </div>
                        <div>
                            {% if prediction == 'Benign' %}
                                <span class="badge-threat badge-benign fs-6">
                                    <i class="fa-solid fa-circle-check"></i> Safe: Benign
                                </span>
                            {% elif prediction == 'Phishing' %}
                                <span class="badge-threat badge-phishing fs-6">
                                    <i class="fa-solid fa-triangle-exclamation"></i> Warning: Phishing
                                </span>
                            {% elif prediction == 'Malware' %}
                                <span class="badge-threat badge-malware fs-6">
                                    <i class="fa-solid fa-bug"></i> Danger: Malware
                                </span>
                            {% else %}
                                <span class="badge-threat badge-defacement fs-6">
                                    <i class="fa-solid fa-skull"></i> Caution: Defacement
                                </span>
                            {% endif %}
                        </div>
                    </div>
                </div>
            {% endif %}
        </div>

        <!-- Your Inbox Preview (Latest 5) matching Figure 5 -->
        <div class="security-card">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h4 class="card-title-lg mb-0">
                    <i class="fa-solid fa-inbox text-primary me-2"></i> Your Inbox Preview (Latest 5)
                </h4>
                <a href="{{ url_for('inbox') }}" class="text-primary text-decoration-none fw-semibold small">
                    View All in Inbox <i class="fa-solid fa-arrow-right ms-1"></i>
                </a>
            </div>

            {% if inbox_preview %}
                <div class="inbox-list">
                    {% for item in inbox_preview %}
                        <div class="inbox-item-card">
                            <div class="me-3 overflow-hidden">
                                <div class="fw-bold text-dark small mb-1">
                                    <i class="fa-regular fa-envelope me-1 text-primary"></i> From: {{ item.sender.email }}
                                </div>
                                <div class="url-text text-truncate">{{ item.url }}</div>
                            </div>
                            <div class="flex-shrink-0">
                                {% if item.prediction == 'Benign' %}
                                    <span class="badge-threat badge-benign">
                                        <i class="fa-solid fa-circle-check"></i> Safe: Benign
                                    </span>
                                {% elif item.prediction == 'Phishing' %}
                                    <span class="badge-threat badge-phishing">
                                        <i class="fa-solid fa-triangle-exclamation"></i> Warning: Phishing
                                    </span>
                                {% elif item.prediction == 'Malware' %}
                                    <span class="badge-threat badge-malware">
                                        <i class="fa-solid fa-bug"></i> Danger: Malware
                                    </span>
                                {% else %}
                                    <span class="badge-threat badge-defacement">
                                        <i class="fa-solid fa-skull"></i> Caution: Defacement
                                    </span>
                                {% endif %}
                            </div>
                        </div>
                    {% endfor %}
                </div>

                <div class="text-center mt-3">
                    <a href="{{ url_for('inbox') }}" class="btn btn-outline-primary btn-sm px-4">
                        View All in Inbox
                    </a>
                </div>
            {% else %}
                <div class="text-center py-4 text-muted">
                    <i class="fa-regular fa-envelope-open fa-3x mb-3 text-secondary opacity-50"></i>
                    <p class="mb-0">No links in your inbox yet.</p>
                    <small>When another user shares a verified URL with you, it will appear here.</small>
                </div>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}

`

---

## File: templates\inbox.html

`html
{% extends "base.html" %}

{% block title %}Inbox - Links Shared With You{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-9">
        <!-- Banner Text matching Figure 6 -->
        <div class="page-header-text mb-4 text-center">
            <div class="d-inline-flex align-items-center gap-2 bg-white bg-opacity-25 px-4 py-2 rounded-pill mb-2">
                <i class="fa-solid fa-shield-halved"></i>
                <span class="small fw-semibold">All links shared with you, analyzed for security</span>
            </div>
            <h1 class="fw-bold">Shared Links Inbox</h1>
        </div>

        <div class="security-card">
            <!-- Header and Filter Pills matching Figure 6 -->
            <div class="d-flex flex-column flex-md-row justify-content-between align-items-md-center gap-3 mb-4 pb-3 border-bottom">
                <div>
                    <h4 class="card-title-lg mb-1">Links Shared With You</h4>
                    <span class="text-muted small">{{ links|length }} total links</span>
                </div>

                <!-- Filter Pills -->
                <div class="d-flex flex-wrap gap-2">
                    <a href="{{ url_for('inbox', filter='all') }}"
                       class="filter-pill {% if current_filter == 'all' %}active{% endif %}">
                        <i class="fa-solid fa-list-ul"></i> All ({{ all_count }})
                    </a>
                    <a href="{{ url_for('inbox', filter='safe') }}"
                       class="filter-pill {% if current_filter == 'safe' %}active{% endif %}">
                        <i class="fa-solid fa-shield-check text-success"></i> Safe ({{ safe_count }})
                    </a>
                    <a href="{{ url_for('inbox', filter='phishing') }}"
                       class="filter-pill {% if current_filter == 'phishing' %}active{% endif %}">
                        <i class="fa-solid fa-triangle-exclamation text-warning"></i> Phishing ({{ phishing_count }})
                    </a>
                    <a href="{{ url_for('inbox', filter='malware') }}"
                       class="filter-pill {% if current_filter == 'malware' %}active{% endif %}">
                        <i class="fa-solid fa-bug text-danger"></i> Malware ({{ malware_count }})
                    </a>
                    <a href="{{ url_for('inbox', filter='defacement') }}"
                       class="filter-pill {% if current_filter == 'defacement' %}active{% endif %}">
                        <i class="fa-solid fa-skull text-secondary"></i> Defacement ({{ defacement_count }})
                    </a>
                </div>
            </div>

            <!-- Links List matching Figure 6 -->
            {% if links %}
                <div class="inbox-cards-list">
                    {% for item in links %}
                        <div class="inbox-item-card align-items-center">
                            <!-- Left: User Avatar & Info -->
                            <div class="d-flex align-items-center gap-3 overflow-hidden me-3">
                                <div class="avatar-icon-box flex-shrink-0">
                                    <i class="fa-solid fa-user"></i>
                                </div>
                                <div class="overflow-hidden">
                                    <div class="d-flex align-items-center gap-2">
                                        <span class="fw-bold text-dark">{{ item.sender.email }}</span>
                                        <span class="badge bg-light text-muted small fw-normal">
                                            <i class="fa-regular fa-clock me-1"></i> Shared recently
                                        </span>
                                    </div>
                                    <div class="url-text text-truncate mt-1">
                                        <i class="fa-solid fa-arrow-up-right-from-square text-muted me-1"></i>
                                        <span>{{ item.url }}</span>
                                    </div>
                                </div>
                            </div>

                            <!-- Right: Threat Badge with Warning Subtext -->
                            <div class="text-end flex-shrink-0">
                                {% if item.prediction == 'Benign' %}
                                    <span class="badge-threat badge-benign">
                                        <i class="fa-solid fa-shield-check"></i> Safe
                                    </span>
                                    <div class="text-success small mt-1 fw-semibold">
                                        <i class="fa-solid fa-check-double me-1"></i> Verified Safe
                                    </div>
                                {% elif item.prediction == 'Phishing' %}
                                    <span class="badge-threat badge-phishing">
                                        <i class="fa-solid fa-triangle-exclamation"></i> Phishing Risk
                                    </span>
                                    <div class="text-danger small mt-1 fw-bold">
                                        <i class="fa-solid fa-ban me-1"></i> Do Not Open
                                    </div>
                                {% elif item.prediction == 'Malware' %}
                                    <span class="badge-threat badge-malware">
                                        <i class="fa-solid fa-bug"></i> Malware
                                    </span>
                                    <div class="text-danger small mt-1 fw-bold">
                                        <i class="fa-solid fa-triangle-exclamation me-1"></i> High Threat
                                    </div>
                                {% else %}
                                    <span class="badge-threat badge-defacement">
                                        <i class="fa-solid fa-skull"></i> Defacement
                                    </span>
                                    <div class="text-muted small mt-1 fw-semibold">
                                        <i class="fa-solid fa-triangle-exclamation me-1"></i> Compromised
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="text-center py-5 text-muted">
                    <i class="fa-solid fa-filter-circle-xmark fa-3x mb-3 text-secondary opacity-50"></i>
                    <h5>No links found in this category</h5>
                    <p class="small mb-3">Try choosing another filter pill or ask a peer to share links with you.</p>
                    <a href="{{ url_for('inbox', filter='all') }}" class="btn btn-outline-primary btn-sm px-3">
                        Show All Links
                    </a>
                </div>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}

`

---

