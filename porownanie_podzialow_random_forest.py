import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
from metryki import calculate_binary_metrics
from podzial_danych import (
    add_request_groups,
    create_random_split,
    load_group_split,
)
from przetwarzanie_danych import get_processed_data


def run_rf_experiment(
    df: pd.DataFrame,
    train_indices: pd.Index,
    test_indices: pd.Index,
    split_name: str,
) -> dict:
    features = config.ML_FEATURES_BASIC

    X = df[features]
    y = df["classification"]

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    model = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    results = calculate_binary_metrics(
        y_test,
        y_pred,
        y_score,
    )

    grouped_df = add_request_groups(df)

    train_groups = set(
        grouped_df.loc[
            train_indices,
            "request_group",
        ]
    )

    test_groups = set(
        grouped_df.loc[
            test_indices,
            "request_group",
        ]
    )

    results["split"] = split_name
    results["train_size"] = len(train_indices)
    results["test_size"] = len(test_indices)
    results["group_overlap"] = len(
        train_groups.intersection(test_groups)
    )

    return results


def compare_splits() -> None:
    df = get_processed_data()

    random_train, random_test = create_random_split(df)
    group_train, group_test = load_group_split(df)

    results = []

    results.append(
        run_rf_experiment(
            df,
            random_train,
            random_test,
            "random",
        )
    )

    results.append(
        run_rf_experiment(
            df,
            group_train,
            group_test,
            "group",
        )
    )

    results_df = pd.DataFrame(results)

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        config.REPORTS_DIR
        / "rf_split_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nPORÓWNANIE PODZIAŁÓW")
    print(results_df.round(4).to_string(index=False))

    print(f"\nWyniki zapisano w: {output_path}")


if __name__ == "__main__":
    compare_splits()