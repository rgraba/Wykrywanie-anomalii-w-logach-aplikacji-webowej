import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config
from data_processor import get_processed_data


def train_random_forest() -> None:
    print("Wczytywanie danych...")

    try:
        df = get_processed_data()
    except Exception as e:
        print(f"Błąd podczas pobierania danych: {e}")
        return

    missing_features = [
        feature for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        print(f"Błąd: brakuje cech w danych: {missing_features}")
        return

    if "classification" not in df.columns:
        print("Błąd: brakuje kolumny 'classification' w danych.")
        return

    print("Wyodrębnianie cech i etykiet...")

    X = df[config.ML_FEATURES]
    y = df["classification"]

    print("Dzielenie danych na zbiór treningowy i testowy...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print(f"Rozmiar zbioru treningowego: {X_train.shape[0]} próbek.")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek.")

    print("Inicjalizacja modelu Random Forest...")

    model = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE
    )

    print("Trenowanie modelu...")
    model.fit(X_train, y_train)
    print("Trening modelu zakończony.")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "rf_model.pkl"
    test_data_path = config.MODELS_DIR / "test_data_rf.pkl"

    joblib.dump(model, model_path)
    joblib.dump((X_test, y_test), test_data_path)

    print(f"Model zapisany jako: {model_path}")
    print(f"Zbiór testowy zapisany jako: {test_data_path}")

    print("\nZnaczenie cech w modelu Random Forest:")

    feature_importance = sorted(
        zip(config.ML_FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )

    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")

    print("\nZakończono proces.")


if __name__ == "__main__":
    train_random_forest()
