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
