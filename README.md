# Malware Prediction Using Machine Learning System

A full-stack, machine learning-powered web application that detects and classifies malicious URLs into four categories:
1. **Benign** (Safe websites)
2. **Phishing** (Deceptive logins & spoofed portals)
3. **Malware** (Drive-by downloads, exploit payloads, trojans)
4. **Defacement** (Compromised & defaced websites)

Built strictly following the mini-project documentation, architecture diagrams (DFD Levels 0, 1, 2), ER model, table designs, and sample code.

---

## 🌟 Key Features

- **Hybrid Feature Engineering**:
  - **Lexical Analysis (8 features)**: URL length, dot count, dash count, slash count, digit count, special character count, HTTPS indicator, HTTP indicator.
  - **Character N-Gram Text Modeling**: TF-IDF vectorizer extracting (2, 5) character n-grams to uncover concealed string patterns.
- **LightGBM Classifier**: Fast gradient boosting multi-class model achieving high predictive accuracy.
- **Collaborative Threat Sharing**: Registered users can analyze URLs and securely share them with colleagues along with instant threat level warnings.
- **Security Dashboard & Inbox**:
  - Real-time URL threat prediction with color-coded safety indicators.
  - Category-filtered inbox (`All`, `Safe`, `Phishing`, `Malware`, `Defacement`).
  - High-level activity statistics (Total Analyzed, Safe Links, Threats Detected).
- **Authentication & Security**:
  - Password hashing with Werkzeug `generate_password_hash` and `check_password_hash`.
  - Session-based access control protecting the dashboard and inbox.
  - Parameterized ORM queries preventing SQL injection.

---

## 🏗️ System Architecture & Database Schema

### Table 1: `admin_user`
| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | INT | Primary Key, Auto Increment | Unique identifier for each user |
| `email` | VARCHAR(120) | NOT NULL, UNIQUE | User login email address |
| `password_hash` | VARCHAR(255) | NOT NULL | Secure salted password hash |

### Table 2: `shared_links`
| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | INT | Primary Key, Auto Increment | Unique identifier for shared link |
| `url` | VARCHAR(2048) | NOT NULL | Web link submitted by user |
| `prediction` | VARCHAR(50) | NOT NULL | Threat class (Benign, Phishing, Malware, Defacement) |
| `sender_id` | INT | Foreign Key (`admin_user.id`) | Sender's user ID |
| `recipient_id` | INT | Foreign Key (`admin_user.id`) | Recipient's user ID |

---

## 🚀 Quick Start Guide

### 1. Requirements Installation
Ensure you have Python 3.10+ installed:
```powershell
python -m pip install -r requirements.txt
```

### 2. (Optional) Re-train the Machine Learning Model
The model artifacts (`lightgbm_model.pkl` and `tfidf_vectorizer.pkl`) are already trained and ready. If you want to re-train:
```powershell
python train.py
```

### 3. Run the Web Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 Demo Accounts & Pre-seeded Data

To immediately showcase the application during the project presentation without manual setup, the database is pre-seeded with two accounts:

| User | Email | Password | Role |
|---|---|---|---|
| User 1 | `a@gmail.com` | `password123` | Sender & Analyst |
| User 2 | `b@gmail.com` | `password123` | Recipient (has shared inbox links) |

*(You can also register brand new accounts anytime via the **Get Started** page).*

---

## 🗄️ Database Options: SQLite vs MySQL

- **Default (Zero-Configuration)**: SQLite (`sqlite:///url_security.db`). Runs immediately out-of-the-box without requiring MySQL server to be running.
- **MySQL (As in PDF Specification)**:
  1. Start your local MySQL service (e.g. `Start-Service MySQL80`).
  2. Create the database: `CREATE DATABASE url_security;`
  3. Set the environment variable before running `app.py`:
     ```powershell
     $env:DATABASE_URL="mysql+pymysql://root:password@localhost/url_security"
     python app.py
     ```

---

## 🧪 Running the Software Testing Suite

Matches **Chapter 4 (Software Testing)** from the project report:
```powershell
python test_app.py
```
Covers:
- **Unit Testing**: Lexical feature extractor accuracy, TF-IDF vectorizer output, LightGBM multi-class prediction.
- **Integration Testing**: User registration, password hashing verification, URL submission, ML prediction storage.
- **Functional Testing**: End-to-end link sharing from User A to User B and inbox categorization.

---

## 📁 Project Structure

```
e:\Malware\
├── app.py                   # Main Flask web application
├── models.py                # SQLAlchemy models (admin_user, shared_links)
├── ml_pipeline.py           # Feature engineering & LightGBM classifier
├── train.py                 # ML training & model serialization script
├── generate_dataset.py      # Multi-class URL dataset generator
├── seed_demo_data.py        # Database seeder matching PDF figures
├── test_app.py              # Unit & integration test suite (Chapter 4)
├── requirements.txt         # Project dependencies
├── malicious_urls.csv       # Training dataset (2,065 URLs across 4 classes)
├── lightgbm_model.pkl       # Serialized LightGBM model artifact
├── tfidf_vectorizer.pkl     # Serialized TF-IDF character n-gram vectorizer
├── url_security.db          # Ready-to-demo SQLite database
├── templates/
│   ├── base.html            # Layout with navbar and flash messages
│   ├── home.html            # Landing page with system overview
│   ├── login.html           # Login page (Figure 1)
│   ├── register.html        # Account creation page (Figure 2)
│   ├── dashboard.html       # Link sharing & inbox preview (Figures 3, 4, 5)
│   └── inbox.html           # Shared links inbox with category filters (Figure 6)
└── static/
    └── css/
        └── style.css        # Violet gradient UI theme matching project report
```
