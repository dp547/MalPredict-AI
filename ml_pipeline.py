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
