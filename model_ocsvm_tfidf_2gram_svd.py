import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
from metryki import (
    calculate_binary_metrics,
    convert_one_class_predictions,
)
from przetwarzanie_danych import get_processed_data


def train_ocsvm_tfidf_svd(
    ngram_range=(2, 2),
    max_features=10000,
    n_components=256,
    nu=0.3,
    gamma="scale"
) -> dict:
    df = get_processed_data()

    if "request_text" not in df.columns:
        raise ValueError("Brakuje kolumny request_text. Sprawdź funkcję add_request_text().")

    X = df["request_text"]
    y = df["classification"]

    print("Podział danych na zbiór treningowy i testowy...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    X_train_normal = X_train[y_train == 0]

    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} próbek.")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek.")
    print(f"Liczba normalnych próbek użytych do treningu: {X_train_normal.shape[0]}.")

    print("Budowanie pipeline: TF-IDF char 2-gram + SVD + One-Class SVM...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True
        )),
        ("svd", TruncatedSVD(
            n_components=n_components,
            random_state=config.RANDOM_STATE
        )),
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(
            kernel="rbf",
            nu=nu,
            gamma=gamma
        ))
    ])

    print("Trenowanie modelu na próbkach normalnych...")
    model.fit(X_train_normal)

    print("Klasyfikacja próbek testowych...")
    raw_predictions = model.predict(X_test)
    y_pred = convert_one_class_predictions(
        raw_predictions
    )

    y_score = -model.decision_function(X_test)

    results = calculate_binary_metrics(
        y_test,
        y_pred,
        y_score,
    )

    results["model"] = "One-Class SVM"
    results["representation"] = "TF-IDF char 2-gram + SVD"
    results["ngram_range"] = str(ngram_range)
    results["max_features"] = max_features
    results["n_components"] = n_components
    results["nu"] = nu
    results["gamma"] = gamma

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
            config.REPORTS_DIR
            / "one_class_svm_tfidf_2gram_svd.csv"
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
    train_ocsvm_tfidf_svd(
        ngram_range=(2, 2),
        max_features=10000,
        n_components=512,
        nu=0.3,
        gamma=0.01
    )