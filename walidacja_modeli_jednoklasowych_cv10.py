import json
from time import perf_counter

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from podzial_danych import add_request_groups

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


TUNED_ISOLATION_FOREST = {
    "BASIC": {
        "contamination": 0.3,
        "n_estimators": 100,
    },
    "ALTHUBITI_9": {
        "contamination": 0.4,
        "n_estimators": 300,
    },
    "INFORMATION_GAIN": {
        "contamination": 0.2,
        "n_estimators": 300,
    },
    "RF_IMPORTANCE": {
        "contamination": 0.2,
        "n_estimators": 300,
    },
}


TUNED_ONE_CLASS_SVM = {
    "BASIC": {
        "gamma": 1.0,
        "nu": 0.4,
    },
    "ALTHUBITI_9": {
        "gamma": "scale",
        "nu": 0.4,
    },
    "INFORMATION_GAIN": {
        "gamma": 1.0,
        "nu": 0.4,
    },
    "RF_IMPORTANCE": {
        "gamma": 1.0,
        "nu": 0.4,
    },
}


def create_model(
    model_name: str,
    parameters: dict,
):
    if model_name == "ISOLATION_FOREST":
        return IsolationForest(
            **parameters,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "ONE_CLASS_SVM":
        return OneClassSVM(
            **parameters,
        )

    raise ValueError(
        f"Nieznany model: {model_name}"
    )


def prepare_fold_data(
    df: pd.DataFrame,
    features: list[str],
    train_indices,
    test_indices,
):
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

    # Uczenie wyłącznie na normalnych danych.
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

    return (
        X_train_normal_scaled,
        X_test_scaled,
        y_test,
        len(X_train_normal),
    )


def evaluate_configuration(
    model_name: str,
    feature_set_name: str,
    configuration_name: str,
    features: list[str],
    parameters: dict,
    X_train_normal,
    X_test,
    y_test,
    normal_training_samples: int,
    fold_number: int,
) -> dict:
    model = create_model(
        model_name,
        parameters,
    )

    training_start = perf_counter()

    model.fit(
        X_train_normal
    )

    training_time = (
        perf_counter() - training_start
    )

    prediction_start = perf_counter()

    raw_predictions = model.predict(
        X_test
    )

    y_pred = convert_one_class_predictions(raw_predictions)

    y_score = -model.decision_function(
        X_test
    )

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    metrics.update(
        {
            "fold": fold_number,
            "model": model_name,
            "feature_set": feature_set_name,
            "configuration": configuration_name,
            "number_of_features": len(features),
            "selected_features": ", ".join(
                features
            ),
            "parameters": json.dumps(
                parameters,
                sort_keys=True,
            ),
            "normal_training_samples": (
                normal_training_samples
            ),
            "test_samples": len(y_test),
            "training_time_seconds": (
                training_time
            ),
            "prediction_time_seconds": (
                prediction_time
            ),
        }
    )

    return metrics


def create_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    grouping_columns = [
        "model",
        "feature_set",
        "configuration",
    ]

    summary_df = (
        metrics_df
        .groupby(
            grouping_columns,
            as_index=False,
        )
        .agg(
            number_of_features_mean=(
                "number_of_features",
                "mean",
            ),
            number_of_features_std=(
                "number_of_features",
                "std",
            ),
            parameters=(
                "parameters",
                "first",
            ),
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            precision_anomaly_mean=(
                "precision_anomaly",
                "mean",
            ),
            precision_anomaly_std=(
                "precision_anomaly",
                "std",
            ),
            recall_anomaly_mean=(
                "recall_anomaly",
                "mean",
            ),
            recall_anomaly_std=(
                "recall_anomaly",
                "std",
            ),
            f1_anomaly_mean=(
                "f1_anomaly",
                "mean",
            ),
            f1_anomaly_std=(
                "f1_anomaly",
                "std",
            ),
            specificity_mean=(
                "specificity",
                "mean",
            ),
            specificity_std=(
                "specificity",
                "std",
            ),
            false_positive_rate_mean=(
                "false_positive_rate",
                "mean",
            ),
            false_positive_rate_std=(
                "false_positive_rate",
                "std",
            ),
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),
            pr_auc_std=(
                "pr_auc",
                "std",
            ),
            training_time_mean=(
                "training_time_seconds",
                "mean",
            ),
            training_time_std=(
                "training_time_seconds",
                "std",
            ),
        )
    )

    return summary_df


def validate_configs_cv10() -> None:
    df = get_processed_data()

    grouped_df = add_request_groups(df)

    y = grouped_df["classification"]
    groups = grouped_df["request_group"]

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    all_metrics = []
    selection_rows = []

    for fold_number, (
            train_positions,
            test_positions,
    ) in enumerate(
        splitter.split(
            df,
            y,
            groups=groups,
        ),
        start=1,
    ):
        print("\n" + "#" * 75)
        print(f"FOLD {fold_number}/10")
        print("#" * 75)

        train_groups = set(
            groups.iloc[train_positions]
        )

        test_groups = set(
            groups.iloc[test_positions]
        )

        group_overlap = len(
            train_groups.intersection(
                test_groups
            )
        )

        if group_overlap != 0:
            raise RuntimeError(
                f"Wykryto {group_overlap} wspólnych grup "
                f"w foldzie {fold_number}."
            )

        print(
            f"Wspólne grupy: {group_overlap}"
        )

        train_indices = df.index[
            train_positions
        ]

        test_indices = df.index[
            test_positions
        ]

        # Selekcja cech wyłącznie
        # na części treningowej foldu.
        X_selection_train = df.loc[
            train_indices,
            config.ML_FEATURES_ALTHUBITI_9,
        ]

        y_selection_train = df.loc[
            train_indices,
            "classification",
        ]

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

        print(f"IG: {ig_features}")
        print(f"RF: {rf_features}")

        selection_rows.append(
            {
                "fold": fold_number,
                "information_gain_features": (
                    ", ".join(ig_features)
                ),
                "information_gain_feature_count": (
                    len(ig_features)
                ),
                "rf_importance_features": (
                    ", ".join(rf_features)
                ),
                "rf_importance_feature_count": (
                    len(rf_features)
                ),
                "split": "group",
                "group_overlap": group_overlap,
            }
        )

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

        for feature_set_name, features in (
            feature_sets.items()
        ):
            (
                X_train_normal,
                X_test,
                y_test,
                normal_training_samples,
            ) = prepare_fold_data(
                df=df,
                features=features,
                train_indices=train_indices,
                test_indices=test_indices,
            )

            experiments = []

            default_if_parameters = {
                **config.ISOLATION_FOREST_PARAMS
            }

            tuned_if_parameters = {
                **config.ISOLATION_FOREST_PARAMS,
                **TUNED_ISOLATION_FOREST[
                    feature_set_name
                ],
            }

            experiments.append(
                (
                    "ISOLATION_FOREST",
                    "DEFAULT",
                    default_if_parameters,
                )
            )

            experiments.append(
                (
                    "ISOLATION_FOREST",
                    "TUNED",
                    tuned_if_parameters,
                )
            )

            default_ocsvm_parameters = {
                **config.OCSVM_PARAMS
            }

            experiments.append(
                (
                    "ONE_CLASS_SVM",
                    "DEFAULT",
                    default_ocsvm_parameters,
                )
            )

            if (
                feature_set_name
                in TUNED_ONE_CLASS_SVM
            ):
                tuned_ocsvm_parameters = {
                    **config.OCSVM_PARAMS,
                    **TUNED_ONE_CLASS_SVM[
                        feature_set_name
                    ],
                }

                experiments.append(
                    (
                        "ONE_CLASS_SVM",
                        "TUNED",
                        tuned_ocsvm_parameters,
                    )
                )

            for (
                model_name,
                configuration_name,
                parameters,
            ) in experiments:
                print(
                    f"{model_name} | "
                    f"{feature_set_name} | "
                    f"{configuration_name}"
                )

                metrics = evaluate_configuration(
                    model_name=model_name,
                    feature_set_name=(
                        feature_set_name
                    ),
                    configuration_name=(
                        configuration_name
                    ),
                    features=features,
                    parameters=parameters,
                    X_train_normal=(
                        X_train_normal
                    ),
                    X_test=X_test,
                    y_test=y_test,
                    normal_training_samples=(
                        normal_training_samples
                    ),
                    fold_number=fold_number,
                )

                metrics["split"] = "group"
                metrics["group_overlap"] = group_overlap

                all_metrics.append(metrics)

                print(
                    f"  BalAcc="
                    f"{metrics['balanced_accuracy']:.4f}, "
                    f"F1="
                    f"{metrics['f1_anomaly']:.4f}, "
                    f"AUC="
                    f"{metrics['roc_auc']:.4f}, "
                    f"FPR="
                    f"{metrics['false_positive_rate']:.4f}"
                )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    selection_df = pd.DataFrame(
        selection_rows
    )

    summary_df = create_summary(
        metrics_df
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_path = (
        config.REPORTS_DIR
        / "one_class_group_retuned_cv10_configs_folds.csv"
    )

    summary_path = (
        config.REPORTS_DIR
        / "one_class_group_retuned_cv10_configs_summary.csv"
    )

    selection_path = (
        config.REPORTS_DIR
        / "one_class_group_retuned_cv10_selection.csv"
    )

    metrics_df.to_csv(
        folds_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    selection_df.to_csv(
        selection_path,
        index=False,
    )

    columns_to_display = [
        "model",
        "feature_set",
        "configuration",
        "number_of_features_mean",
        "number_of_features_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_anomaly_mean",
        "f1_anomaly_std",
        "false_positive_rate_mean",
        "false_positive_rate_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "training_time_mean",
    ]

    print("\n" + "=" * 75)
    print("PODSUMOWANIE 10-KROTNEJ WALIDACJI")
    print("=" * 75)

    print(
        summary_df[
            columns_to_display
        ].round(4).to_string(
            index=False
        )
    )

    print("\nZapisane pliki:")
    print(f"- {folds_path}")
    print(f"- {summary_path}")
    print(f"- {selection_path}")


if __name__ == "__main__":
    validate_configs_cv10()