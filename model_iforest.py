import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import config
from data_processor import get_processed_data


def train_isolation_forest() -> None:

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

    print("\nRozkład klas w całym zbiorze:")
    print(y.value_counts())
    print("\nRozkład klas procentowo:")
    print(y.value_counts(normalize=True).round(4))

    print("\nDzielenie danych na zbiór treningowy i testowy...")

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

    print("\nInicjalizacja modelu Isolation Forest...")

    model = IsolationForest(
        **config.ISOLATION_FOREST_PARAMS,
        random_state=config.RANDOM_STATE
    )

    print("Trenowanie modelu na próbkach normalnych...")
    model.fit(X_train_normal)
    print("Trening modelu zakończony.")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "iforest_model.pkl"
    test_data_path = config.MODELS_DIR / "test_data_if.pkl"

    joblib.dump(model, model_path)
    joblib.dump((X_test, y_test), test_data_path)

    print(f"Model zapisany jako: {model_path}")
    print(f"Zbiór testowy zapisany jako: {test_data_path}")

    print("\nZakończono proces.")


if __name__ == "__main__":
    train_isolation_forest()