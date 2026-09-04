import joblib
from sklearn.ensemble import RandomForestClassifier

import config
from protokol_eksperymentalny import (
    get_development_indices,
)
from przetwarzanie_danych import get_processed_data


def train_random_forest() -> None:
    df = get_processed_data()
    features = config.ML_FEATURES

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Brakuje cech w danych: {missing_features}"
        )

    development_indices = get_development_indices(df)

    X_train = df.loc[development_indices, features]
    y_train = df.loc[development_indices, "classification"]

    model = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print("Trenowanie Random Forest na części development...")
    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "model_name": "Random Forest",
        "features": list(features),
        "parameters": {
            **config.RANDOM_FOREST_PARAMS,
            "random_state": config.RANDOM_STATE,
        },
        "training_partition": "development",
        "training_samples": len(development_indices),
        "protocol_seed": config.PROTOCOL_RANDOM_STATE,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        config.MODELS_DIR
        / "rf_model_development.pkl"
    )

    joblib.dump(artifact, model_path)

    print(f"Model zapisano w: {model_path}")
    print(f"Liczba próbek treningowych: {len(development_indices)}")
    print("Zbiór final_test nie został użyty.")


if __name__ == "__main__":
    train_random_forest()