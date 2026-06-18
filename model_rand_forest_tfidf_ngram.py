from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config
from data_processor import get_processed_data


def evaluate_model(y_test, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_anomaly": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_anomaly": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_anomaly": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score)
    }


def train_rf_tfidf_ngram(
    ngram_range=(3, 3),
    max_features=5000
) -> dict:
    print("Wczytywanie i przetwarzanie danych...")

    df = get_processed_data()

    if "request_text" not in df.columns:
        raise ValueError("Brakuje kolumny request_text. Sprawdź funkcję add_request_text().")

    X = df["request_text"]
    y = df["classification"]

    print("Dzielenie danych na zbiór treningowy i testowy...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print("Budowanie pipeline: TF-IDF char n-gram + Random Forest...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    print("Trenowanie modelu...")
    model.fit(X_train, y_train)

    print("Predykcja...")
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    results = evaluate_model(y_test, y_pred, y_score)

    results["model"] = "Random Forest"
    results["representation"] = "TF-IDF char n-gram"
    results["ngram_range"] = str(ngram_range)
    results["max_features"] = max_features

    print("\nWyniki:")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    train_rf_tfidf_ngram(
        ngram_range=(3, 3),
        max_features=20000
    )