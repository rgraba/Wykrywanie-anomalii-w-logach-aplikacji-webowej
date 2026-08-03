import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import config

from data_processor import get_processed_data
from split_manager import (
    add_request_groups,
    create_random_split,
    load_group_split,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_score,
) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_anomaly": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_anomaly": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_anomaly": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


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

    results = calculate_metrics(
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