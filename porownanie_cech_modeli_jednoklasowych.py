from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold
from podzial_danych import add_request_groups
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
from metryki import (
    calculate_binary_metrics,
    convert_one_class_predictions,
)
from przetwarzanie_danych import get_processed_data
from selekcja_cech import (
    select_with_information_gain,
    select_with_random_forest,
)


def evaluate_one_class_model(
    df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
    features: list[str],
    train_indices,
    test_indices,
) -> dict:

    X_train = df.loc[
        train_indices,
        features,
    ]

    y_train = df.loc[
        train_indices,
        "classification",
    ]

    X_test = df.loc[
        test_indices,
        features,
    ]

    y_test = df.loc[
        test_indices,
        "classification",
    ]


    X_train_normal = X_train.loc[
        y_train == 0
    ]

    scaler = StandardScaler()

    X_train_normal_scaled = scaler.fit_transform(
        X_train_normal
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    if model_name == "ISOLATION_FOREST":
        model = IsolationForest(
            **config.ISOLATION_FOREST_PARAMS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "ONE_CLASS_SVM":
        model = OneClassSVM(
            **config.OCSVM_PARAMS,
        )

    else:
        raise ValueError(
            f"Nieznany model: {model_name}"
        )

    training_start = perf_counter()

    model.fit(
        X_train_normal_scaled
    )

    training_time = (
        perf_counter() - training_start
    )

    prediction_start = perf_counter()

    raw_predictions = model.predict(
        X_test_scaled
    )


    y_pred = convert_one_class_predictions(raw_predictions)


    y_score = -model.decision_function(
        X_test_scaled
    )

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    metrics["model"] = model_name
    metrics["feature_set"] = feature_set_name
    metrics["number_of_features"] = len(features)
    metrics["selected_features"] = ", ".join(
        features
    )
    metrics["normal_training_samples"] = len(
        X_train_normal
    )
    metrics["test_samples"] = len(X_test)
    metrics["training_time_seconds"] = (
        training_time
    )
    metrics["prediction_time_seconds"] = (
        prediction_time
    )

    return metrics


def compare_one_class_feature_sets() -> None:
    df = get_processed_data()

    grouped_df = add_request_groups(df)

    y = grouped_df["classification"]
    groups = grouped_df["request_group"]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    folds = list(
        splitter.split(
            df,
            y,
            groups=groups,
        )
    )

    test_positions = np.concatenate(
        [
            folds[0][1],
            folds[1][1],
        ]
    )

    train_positions = np.concatenate(
        [
            folds[2][1],
            folds[3][1],
            folds[4][1],
        ]
    )

    train_indices = df.index[
        train_positions
    ]

    test_indices = df.index[
        test_positions
    ]

    train_groups = set(
        groups.loc[train_indices]
    )

    test_groups = set(
        groups.loc[test_indices]
    )

    group_overlap = len(
        train_groups.intersection(
            test_groups
        )
    )

    if group_overlap != 0:
        raise RuntimeError(
            f"Wykryto {group_overlap} wspólnych grup "
            "pomiędzy treningiem i testem."
        )

    X_selection_train = df.loc[
        train_indices,
        config.ML_FEATURES_ALTHUBITI_9,
    ]

    y_selection_train = df.loc[
        train_indices,
        "classification",
    ]

    print("\nSelekcja cech wyłącznie na treningu...")

    ig_features, _ = (
        select_with_information_gain(
            X_selection_train,
            y_selection_train,
        )
    )

    rf_features, _ = (
        select_with_random_forest(
            X_selection_train,
            y_selection_train,
        )
    )

    print(f"Cechy IG: {ig_features}")
    print(f"Cechy RF: {rf_features}")

    feature_sets = {
        "BASIC": (
            config.ML_FEATURES_BASIC
        ),
        "ALTHUBITI_9": (
            config.ML_FEATURES_ALTHUBITI_9
        ),
        "INFORMATION_GAIN": (
            ig_features
        ),
        "RF_IMPORTANCE": (
            rf_features
        ),
    }

    model_names = [
        "ISOLATION_FOREST",
        "ONE_CLASS_SVM",
    ]

    normal_training_samples = int(
        (
            df.loc[
                train_indices,
                "classification",
            ]
            == 0
        ).sum()
    )

    print("\n" + "=" * 75)
    print("PROTOKÓŁ EKSPERYMENTALNY")
    print("=" * 75)
    print(
        "Podział grupowy: "
        "60% trening / 40% test"
    )
    print(
        f"Wspólne grupy: {group_overlap}"
    )
    print(f"Seed: {config.RANDOM_STATE}")
    print(
        "Normalne próbki treningowe: "
        f"{normal_training_samples}"
    )
    print(
        f"Liczba próbek testowych: "
        f"{len(test_indices)}"
    )

    metrics_rows = []

    for model_name in model_names:
        for feature_set_name, features in (
            feature_sets.items()
        ):
            print("\n" + "=" * 75)
            print(
                f"{model_name} — "
                f"{feature_set_name}"
            )
            print("=" * 75)
            print(f"Cechy: {features}")

            metrics = evaluate_one_class_model(
                df=df,
                model_name=model_name,
                feature_set_name=feature_set_name,
                features=features,
                train_indices=train_indices,
                test_indices=test_indices,
            )

            metrics["split"] = "group"
            metrics["group_overlap"] = group_overlap
            metrics["train_samples"] = len(
                train_indices
            )

            metrics_rows.append(metrics)

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )
            print(
                f"Balanced Accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(
                f"Precision anomaly: "
                f"{metrics['precision_anomaly']:.4f}"
            )
            print(
                f"Recall anomaly: "
                f"{metrics['recall_anomaly']:.4f}"
            )
            print(
                f"F1 anomaly: "
                f"{metrics['f1_anomaly']:.4f}"
            )
            print(
                f"ROC AUC: "
                f"{metrics['roc_auc']:.4f}"
            )
            print(
                f"PR AUC: "
                f"{metrics['pr_auc']:.4f}"
            )
            print(
                f"False-positive rate: "
                f"{metrics['false_positive_rate']:.4f}"
            )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        config.REPORTS_DIR
        / "one_class_group_literature_features_60_40.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    columns_to_display = [
        "model",
        "feature_set",
        "number_of_features",
        "accuracy",
        "balanced_accuracy",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "training_time_seconds",
    ]

    print("\n" + "=" * 75)
    print("PORÓWNANIE MODELI JEDNOKLASOWYCH")
    print("=" * 75)

    print(
        metrics_df[
            columns_to_display
        ].round(4).to_string(
            index=False
        )
    )

    print(
        f"\nWyniki zapisano w: "
        f"{output_path}"
    )


if __name__ == "__main__":
    compare_one_class_feature_sets()