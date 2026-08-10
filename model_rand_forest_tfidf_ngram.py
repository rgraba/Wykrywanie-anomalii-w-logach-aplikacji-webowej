import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from podzial_danych import (
    calculate_group_overlap,
    get_split_indices,
)
from sklearn.pipeline import Pipeline

import config
from metryki import calculate_binary_metrics
from przetwarzanie_danych import get_processed_data


def train_rf_tfidf_ngram(
    ngram_range=(3, 3),
    max_features=5000,
    split_type: str = "group",
) -> dict:

    df = get_processed_data()

    if "request_text" not in df.columns:
        raise ValueError("Brakuje kolumny request_text. Sprawdź funkcję add_request_text().")

    X = df["request_text"]
    y = df["classification"]

    print(f"Podział danych: {split_type}...")

    train_indices, test_indices = get_split_indices(
        df,
        split_type,
    )

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    group_overlap = calculate_group_overlap(
        df,
        train_indices,
        test_indices,
    )

    print(f"Liczba próbek treningowych: {len(train_indices)}")
    print(f"Liczba próbek testowych: {len(test_indices)}")
    print(f"Liczba wspólnych grup: {group_overlap}")

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

    results = calculate_binary_metrics(
        y_test,
        y_pred,
        y_score,
    )

    results["model"] = "Random Forest"
    results["representation"] = "TF-IDF char n-gram"
    results["ngram_range"] = str(ngram_range)
    results["max_features"] = max_features
    results["split"] = split_type
    results["train_size"] = len(train_indices)
    results["test_size"] = len(test_indices)
    results["group_overlap"] = group_overlap

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
            config.REPORTS_DIR
            / f"random_forest_tfidf_ngram_{split_type}.csv"
    )

    pd.DataFrame([results]).to_csv(
        output_path,
        index=False,
    )

    print("\nWyniki:")
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\nWyniki zapisano w: {output_path}")

    return results


if __name__ == "__main__":
    train_rf_tfidf_ngram(
        ngram_range=(2, 4),
        max_features=5000,
        split_type="group",
    )