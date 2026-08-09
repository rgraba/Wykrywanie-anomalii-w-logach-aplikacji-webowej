import json
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import (
    ParameterGrid,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config

from data_processor import get_processed_data

from compare_rf_selekcja_cech import (
    calculate_metrics,
    select_with_information_gain,
    select_with_random_forest,
)


PARAMETER_GRIDS = {
    "ISOLATION_FOREST": {
        "n_estimators": [100, 200, 300],
        "contamination": [
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
        ],
    },
    "ONE_CLASS_SVM": {
        "nu": [
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        "gamma": [
            "scale",
            0.001,
            0.01,
            0.1,
            1.0,
        ],
    },
}


def create_model(
    model_name: str,
    parameters: dict,
):
    if model_name == "ISOLATION_FOREST":
        model_parameters = {
            **config.ISOLATION_FOREST_PARAMS,
            **parameters,
        }

        return IsolationForest(
            **model_parameters,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "ONE_CLASS_SVM":
        model_parameters = {
            **config.OCSVM_PARAMS,
            **parameters,
        }

        return OneClassSVM(
            **model_parameters,
        )

    raise ValueError(
        f"Nieznany model: {model_name}"
    )


def prepare_scaled_data(
    df: pd.DataFrame,
    features: list[str],
    train_indices,
    evaluation_indices,
):
    X_train = df.loc[
        train_indices,
        features,
    ]

    y_train = df.loc[
        train_indices,
        "classification",
    ]

    X_evaluation = df.loc[
        evaluation_indices,
        features,
    ]

    y_evaluation = df.loc[
        evaluation_indices,
        "classification",
    ]

    X_train_normal = X_train.loc[
        y_train == 0
    ]

    scaler = StandardScaler()

    X_train_normal_scaled = scaler.fit_transform(
        X_train_normal
    )

    X_evaluation_scaled = scaler.transform(
        X_evaluation
    )

    return (
        X_train_normal_scaled,
        X_evaluation_scaled,
        y_evaluation,
        len(X_train_normal),
    )


def predict_anomalies(
    model,
    X_evaluation,
):
    raw_predictions = model.predict(
        X_evaluation
    )

    y_pred = np.where(
        raw_predictions == -1,
        1,
        0,
    )

    y_score = -model.decision_function(
        X_evaluation
    )

    return y_pred, y_score


def tune_model(
    df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
    features: list[str],
    inner_train_indices,
    validation_indices,
):
    (
        X_train_normal,
        X_validation,
        y_validation,
        normal_training_samples,
    ) = prepare_scaled_data(
        df=df,
        features=features,
        train_indices=inner_train_indices,
        evaluation_indices=validation_indices,
    )

    parameter_candidates = list(
        ParameterGrid(
            PARAMETER_GRIDS[model_name]
        )
    )

    search_rows = []

    best_parameters = None
    best_metrics = None
    best_key = None

    for candidate_number, parameters in enumerate(
        parameter_candidates,
        start=1,
    ):
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

        y_pred, y_score = predict_anomalies(
            model,
            X_validation,
        )

        metrics = calculate_metrics(
            y_true=y_validation,
            y_pred=y_pred,
            y_score=y_score,
        )

        candidate_key = (
            metrics["balanced_accuracy"],
            metrics["f1_anomaly"],
            -metrics["false_positive_rate"],
        )

        if (
            best_key is None
            or candidate_key > best_key
        ):
            best_key = candidate_key
            best_parameters = dict(parameters)
            best_metrics = dict(metrics)

        row = {
            "model": model_name,
            "feature_set": feature_set_name,
            "candidate_number": candidate_number,
            "number_of_features": len(features),
            "selected_features": ", ".join(
                features
            ),
            "normal_training_samples": (
                normal_training_samples
            ),
            "training_time_seconds": training_time,
            **parameters,
        }

        for metric_name, value in metrics.items():
            row[
                f"validation_{metric_name}"
            ] = value

        search_rows.append(row)

        print(
            f"  {candidate_number}/"
            f"{len(parameter_candidates)} "
            f"{parameters} -> "
            f"BalAcc="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"F1={metrics['f1_anomaly']:.4f}, "
            f"FPR="
            f"{metrics['false_positive_rate']:.4f}"
        )

    return (
        best_parameters,
        best_metrics,
        search_rows,
    )


def evaluate_final_model(
    df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
    features: list[str],
    parameters: dict,
    outer_train_indices,
    test_indices,
    validation_metrics: dict,
):
    (
        X_train_normal,
        X_test,
        y_test,
        normal_training_samples,
    ) = prepare_scaled_data(
        df=df,
        features=features,
        train_indices=outer_train_indices,
        evaluation_indices=test_indices,
    )

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

    y_pred, y_score = predict_anomalies(
        model,
        X_test,
    )

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    metrics.update(
        {
            "model": model_name,
            "feature_set": feature_set_name,
            "number_of_features": len(features),
            "selected_features": ", ".join(
                features
            ),
            "best_parameters": json.dumps(
                parameters,
                sort_keys=True,
            ),
            "normal_training_samples": (
                normal_training_samples
            ),
            "test_samples": len(test_indices),
            "best_validation_balanced_accuracy": (
                validation_metrics[
                    "balanced_accuracy"
                ]
            ),
            "best_validation_f1_anomaly": (
                validation_metrics[
                    "f1_anomaly"
                ]
            ),
            "best_validation_fpr": (
                validation_metrics[
                    "false_positive_rate"
                ]
            ),
            "training_time_seconds": training_time,
            "prediction_time_seconds": (
                prediction_time
            ),
        }
    )

    return metrics


def optimize_models() -> None:
    print("Wczytywanie danych...")

    df = get_processed_data()

    y = df["classification"]

    outer_train_indices, test_indices = (
        train_test_split(
            df.index,
            test_size=0.4,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )
    )

    outer_train_y = df.loc[
        outer_train_indices,
        "classification",
    ]

    inner_train_indices, validation_indices = (
        train_test_split(
            outer_train_indices,
            test_size=0.2,
            random_state=config.RANDOM_STATE,
            stratify=outer_train_y,
        )
    )

    X_selection_train = df.loc[
        inner_train_indices,
        config.ML_FEATURES_ALTHUBITI_9,
    ]

    y_selection_train = df.loc[
        inner_train_indices,
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

    print(f"\nCechy IG: {ig_features}")
    print(f"Cechy RF: {rf_features}")

    feature_sets = {
        "BASIC": config.ML_FEATURES_BASIC,
        "ALTHUBITI_9": (
            config.ML_FEATURES_ALTHUBITI_9
        ),
    }

    if ig_features == rf_features:
        feature_sets["IG_RF_SELECTED"] = (
            ig_features
        )
    else:
        feature_sets["INFORMATION_GAIN"] = (
            ig_features
        )
        feature_sets["RF_IMPORTANCE"] = (
            rf_features
        )

    print("\n" + "=" * 75)
    print("PODZIAŁ DANYCH")
    print("=" * 75)
    print(
        f"Inner train: "
        f"{len(inner_train_indices)}"
    )
    print(
        f"Walidacja: "
        f"{len(validation_indices)}"
    )
    print(
        f"Outer train łącznie: "
        f"{len(outer_train_indices)}"
    )
    print(
        f"Test: {len(test_indices)}"
    )

    search_rows = []
    final_rows = []

    for model_name in PARAMETER_GRIDS:
        for feature_set_name, features in (
            feature_sets.items()
        ):
            print("\n" + "=" * 75)
            print(
                f"STROJENIE: {model_name} — "
                f"{feature_set_name}"
            )
            print("=" * 75)

            (
                best_parameters,
                best_validation_metrics,
                model_search_rows,
            ) = tune_model(
                df=df,
                model_name=model_name,
                feature_set_name=feature_set_name,
                features=features,
                inner_train_indices=(
                    inner_train_indices
                ),
                validation_indices=(
                    validation_indices
                ),
            )

            search_rows.extend(
                model_search_rows
            )

            print(
                f"\nNajlepsze parametry: "
                f"{best_parameters}"
            )
            print(
                "Walidacja: "
                f"BalAcc="
                f"{best_validation_metrics['balanced_accuracy']:.4f}, "
                f"F1="
                f"{best_validation_metrics['f1_anomaly']:.4f}"
            )

            final_metrics = evaluate_final_model(
                df=df,
                model_name=model_name,
                feature_set_name=feature_set_name,
                features=features,
                parameters=best_parameters,
                outer_train_indices=(
                    outer_train_indices
                ),
                test_indices=test_indices,
                validation_metrics=(
                    best_validation_metrics
                ),
            )

            final_rows.append(
                final_metrics
            )

            print(
                "TEST: "
                f"BalAcc="
                f"{final_metrics['balanced_accuracy']:.4f}, "
                f"F1="
                f"{final_metrics['f1_anomaly']:.4f}, "
                f"AUC="
                f"{final_metrics['roc_auc']:.4f}, "
                f"FPR="
                f"{final_metrics['false_positive_rate']:.4f}"
            )

    search_df = pd.DataFrame(
        search_rows
    )

    final_df = pd.DataFrame(
        final_rows
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_path = (
        config.REPORTS_DIR
        / "one_class_parameter_search_validation.csv"
    )

    final_path = (
        config.REPORTS_DIR
        / "one_class_tuned_test_60_40.csv"
    )

    search_df.to_csv(
        search_path,
        index=False,
    )

    final_df.to_csv(
        final_path,
        index=False,
    )

    columns_to_display = [
        "model",
        "feature_set",
        "number_of_features",
        "best_parameters",
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
    print("KOŃCOWE WYNIKI NA ZBIORZE TESTOWYM")
    print("=" * 75)

    print(
        final_df[
            columns_to_display
        ].round(4).to_string(
            index=False
        )
    )

    print("\nZapisane pliki:")
    print(f"- {search_path}")
    print(f"- {final_path}")


if __name__ == "__main__":
    optimize_models()