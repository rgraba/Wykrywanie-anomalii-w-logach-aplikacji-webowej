import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from protokol_eksperymentalny import (
    get_development_and_final_test_indices,
)
from przetwarzanie_danych import get_processed_data


def train_iforest_tfidf_svd(
    ngram_range: tuple[int, int] = (2, 2),
    max_features: int = 10000,
    n_components: int = 512,
    contamination: float = 0.3,
    n_estimators: int = 200,
) -> None:
    df = get_processed_data()

    if "request_text" not in df.columns:
        raise ValueError(
            "Brakuje kolumny request_text."
        )

    development_indices, _ = (
        get_development_and_final_test_indices(df)
    )

    X_development = df.loc[
        development_indices,
        "request_text",
    ]

    y_development = df.loc[
        development_indices,
        "classification",
    ]

    X_train_normal = X_development.loc[
        y_development == 0
    ]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    max_features=max_features,
                    lowercase=True,
                ),
            ),
            (
                "svd",
                TruncatedSVD(
                    n_components=n_components,
                    random_state=config.RANDOM_STATE,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(
        "Trenowanie Isolation Forest z reprezentacją "
        "TF-IDF i SVD na normalnych próbkach development..."
    )

    model.fit(X_train_normal)

    artifact = {
        "model": model,
        "model_name": "Isolation Forest",
        "representation": "TF-IDF char 2-gram + SVD",
        "parameters": {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "n_components": n_components,
            "contamination": contamination,
            "n_estimators": n_estimators,
            "random_state": config.RANDOM_STATE,
        },
        "training_partition": "development_normal_only",
        "development_samples": len(development_indices),
        "normal_training_samples": len(X_train_normal),
        "protocol_seed": config.PROTOCOL_RANDOM_STATE,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        config.MODELS_DIR
        / "iforest_tfidf_svd_development.pkl"
    )

    joblib.dump(artifact, model_path)

    print(f"Model zapisano w: {model_path}")
    print(f"Próbki development: {len(development_indices)}")
    print(f"Normalne próbki treningowe: {len(X_train_normal)}")
    print("Zbiór final_test nie został użyty.")


if __name__ == "__main__":
    train_iforest_tfidf_svd()