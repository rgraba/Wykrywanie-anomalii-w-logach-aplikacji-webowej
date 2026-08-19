import config
from przetwarzanie_danych import get_processed_data
from protokol_eksperymentalny import get_development_indices

def validate_features() -> None:
    all_data = get_processed_data()
    development_indices = get_development_indices(all_data)
    df = all_data.loc[development_indices]

    print(
        "Walidacja cech wyłącznie na zbiorze development: "
        f"{len(df)} rekordów"
    )

    feature_sets = {
        "ALTHUBITI_9": (
            config.ML_FEATURES_ALTHUBITI_9
        ),
        "ALTHUBITI_5": (
            config.ML_FEATURES_ALTHUBITI_5
        ),
    }

    for set_name, features in feature_sets.items():
        print("\n" + "=" * 70)
        print(f"WALIDACJA ZESTAWU: {set_name}")
        print("=" * 70)

        missing_features = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Brakujące cechy: {missing_features}"
            )

        missing_values = (
            df[features]
            .isna()
            .sum()
            .sum()
        )

        negative_values = (
            df[features] < 0
        ).sum().sum()

        print(f"Liczba cech: {len(features)}")
        print(
            "Liczba brakujących wartości: "
            f"{missing_values}"
        )
        print(
            "Liczba wartości ujemnych: "
            f"{negative_values}"
        )

        print("\nStatystyki cech:")
        print(
            df[features]
            .describe()
            .transpose()
            .round(2)
            .to_string()
        )

        print("\nPrzykładowe rekordy:")
        print(
            df[features]
            .head(5)
            .to_string(index=False)
        )

    top_five_is_subset = set(
        config.ML_FEATURES_ALTHUBITI_5
    ).issubset(
        set(config.ML_FEATURES_ALTHUBITI_9)
    )

    print("\n" + "=" * 70)
    print(
        "Czy ALTHUBITI_5 jest podzbiorem "
        f"ALTHUBITI_9: {top_five_is_subset}"
    )

    if not top_five_is_subset:
        raise ValueError(
            "ALTHUBITI_5 nie jest podzbiorem "
            "ALTHUBITI_9."
        )

    print("\nWalidacja zakończona poprawnie.")


if __name__ == "__main__":
    validate_features()