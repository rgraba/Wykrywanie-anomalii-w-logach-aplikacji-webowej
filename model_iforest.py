import joblib
from sklearn.ensemble import IsolationForest

import config
from protokol_eksperymentalny import (
    get_development_and_final_test_indices,
)
from przetwarzanie_danych import get_processed_data


def train_isolation_forest() -> None:
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

    development_indices, _ = (
        get_development_and_final_test_indices(df)
    )

    X_development = df.loc[
        development_indices,
        features,
    ]

    y_development = df.loc[
        development_indices,
        "classification",
    ]

    X_train_normal = X_development.loc[
        y_development == 0
    ]

    model = IsolationForest(
        **config.ISOLATION_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        "Trenowanie Isolation Forest na normalnych "
        "próbkach części development..."
    )

    model.fit(X_train_normal)

    artifact = {
        "model": model,
        "model_name": "Isolation Forest",
        "features": list(features),
        "parameters": {
            **config.ISOLATION_FOREST_PARAMS,
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
        / "iforest_model_development.pkl"
    )

    joblib.dump(artifact, model_path)

    print(f"Model zapisano w: {model_path}")
    print(f"Próbki development: {len(development_indices)}")
    print(f"Normalne próbki treningowe: {len(X_train_normal)}")
    print("Zbiór final_test nie został użyty.")


if __name__ == "__main__":
    train_isolation_forest()