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
