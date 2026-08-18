import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import config
from protokol_eksperymentalny import (
    get_development_and_final_test_indices,
)
from przetwarzanie_danych import get_processed_data


def train_rf_tfidf_ngram(
    ngram_range: tuple[int, int] = (2, 4),
    max_features: int = 5000,
) -> None:
    df = get_processed_data()

    if "request_text" not in df.columns:
        raise ValueError(
            "Brakuje kolumny request_text."
        )

    development_indices, _ = (
        get_development_and_final_test_indices(df)
    )

    X_train = df.loc[
        development_indices,
        "request_text",
    ]

    y_train = df.loc[
        development_indices,
        "classification",
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
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(
        "Trenowanie Random Forest z reprezentacją "
        "TF-IDF na części development..."
    )

    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "model_name": "Random Forest",
        "representation": "TF-IDF char n-gram",
        "parameters": {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": config.RANDOM_STATE,
        },
        "training_partition": "development",
        "training_samples": len(development_indices),
        "protocol_seed": config.PROTOCOL_RANDOM_STATE,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        config.MODELS_DIR
        / "rf_tfidf_ngram_development.pkl"
    )

    joblib.dump(artifact, model_path)

    print(f"Model zapisano w: {model_path}")
    print(f"Próbki treningowe: {len(development_indices)}")
    print("Zbiór final_test nie został użyty.")


if __name__ == "__main__":
    train_rf_tfidf_ngram()