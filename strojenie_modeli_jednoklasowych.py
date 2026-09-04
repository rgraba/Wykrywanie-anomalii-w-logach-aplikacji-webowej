import json
from time import perf_counter

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
from metryki import calculate_binary_metrics, convert_one_class_predictions
from podzial_danych import ensure_no_group_overlap
from protokol_eksperymentalny import get_development_folds
from przetwarzanie_danych import get_processed_data
from selekcja_cech import (
    select_with_information_gain,
    select_with_random_forest,
)


PARAMETER_GRIDS = {
    "ISOLATION_FOREST": {
        "n_estimators": [100, 200, 300],
        "contamination": [0.05, 0.1, 0.2, 0.3, 0.4],
    },
    "ONE_CLASS_SVM": {
        "nu": [0.05, 0.1, 0.2, 0.3, 0.4],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
    },
}

FEATURE_SET_NAMES = [
    "BASIC",
    "ALTHUBITI_9",
    "INFORMATION_GAIN",
    "RF_IMPORTANCE",
]


def create_model(model_name: str, parameters: dict):
    if model_name == "ISOLATION_FOREST":
        model_parameters = {
            **config.ISOLATION_FOREST_PARAMS,
            **parameters,
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1,
        }
        return IsolationForest(**model_parameters)

    if model_name == "ONE_CLASS_SVM":
        model_parameters = {
            **config.OCSVM_PARAMS,
            **parameters,
        }
        return OneClassSVM(**model_parameters)

    raise ValueError(f"Nieznany model: {model_name}")


def select_features_for_fold(
    df: pd.DataFrame,
    feature_set_name: str,
    train_indices: pd.Index,
) -> list[str]:
    if feature_set_name == "BASIC":
        return list(config.ML_FEATURES_BASIC)

    if feature_set_name == "ALTHUBITI_9":
        return list(config.ML_FEATURES_ALTHUBITI_9)

    X_train = df.loc[train_indices, config.ML_FEATURES_ALTHUBITI_9]
    y_train = df.loc[train_indices, "classification"]

    if feature_set_name == "INFORMATION_GAIN":
        selected_features, _ = select_with_information_gain(X_train, y_train)
        return selected_features

    if feature_set_name == "RF_IMPORTANCE":
        selected_features, _ = select_with_random_forest(X_train, y_train)
        return selected_features

    raise ValueError(f"Nieznany zestaw cech: {feature_set_name}")


def prepare_fold_data(
    df: pd.DataFrame,
    features: list[str],
    train_indices: pd.Index,
    validation_indices: pd.Index,
):
    X_train = df.loc[train_indices, features]
    y_train = df.loc[train_indices, "classification"]

    X_validation = df.loc[validation_indices, features]
    y_validation = df.loc[validation_indices, "classification"]

    X_train_normal = X_train.loc[y_train == 0]

    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_validation_scaled = scaler.transform(X_validation)

    return (
        X_train_normal_scaled,
        X_validation_scaled,
        y_validation,
        len(X_train_normal),
    )


def evaluate_candidate(
    model,
    X_train_normal,
    X_validation,
    y_validation,
) -> tuple[dict, float, float]:
    training_start = perf_counter()
    model.fit(X_train_normal)
    training_time = perf_counter() - training_start

    prediction_start = perf_counter()

    raw_predictions = model.predict(X_validation)
    y_pred = convert_one_class_predictions(raw_predictions)
    y_score = -model.decision_function(X_validation)

    prediction_time = perf_counter() - prediction_start

    metrics = calculate_binary_metrics(
        y_true=y_validation,
        y_pred=y_pred,
        y_score=y_score,
    )

    return metrics, training_time, prediction_time


def create_parameter_summary(search_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "model",
        "feature_set",
        "candidate_number",
        "parameters",
    ]

    return (
        search_df.groupby(group_columns, as_index=False)
        .agg(
            folds_evaluated=("fold", "nunique"),
            balanced_accuracy_mean=(
                "validation_balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "validation_balanced_accuracy",
                "std",
            ),
            f1_anomaly_mean=("validation_f1_anomaly", "mean"),
            f1_anomaly_std=("validation_f1_anomaly", "std"),
            recall_anomaly_mean=("validation_recall_anomaly", "mean"),
            recall_anomaly_std=("validation_recall_anomaly", "std"),
            false_positive_rate_mean=(
                "validation_false_positive_rate",
                "mean",
            ),
            false_positive_rate_std=(
                "validation_false_positive_rate",
                "std",
            ),
            roc_auc_mean=("validation_roc_auc", "mean"),
            roc_auc_std=("validation_roc_auc", "std"),
            pr_auc_mean=("validation_pr_auc", "mean"),
            pr_auc_std=("validation_pr_auc", "std"),
            training_time_mean=("training_time_seconds", "mean"),
            prediction_time_mean=("prediction_time_seconds", "mean"),
        )
    )


def select_best_parameters(summary_df: pd.DataFrame) -> pd.DataFrame:
    ordered_summary = summary_df.sort_values(
        [
            "model",
            "feature_set",
            "balanced_accuracy_mean",
            "f1_anomaly_mean",
            "false_positive_rate_mean",
        ],
        ascending=[True, True, False, False, True],
    )

    return (
        ordered_summary.groupby(
            ["model", "feature_set"],
            sort=False,
        )
        .head(1)
        .reset_index(drop=True)
    )


def optimize_models() -> None:
    df = get_processed_data()
    folds = get_development_folds(df)

    search_rows = []

    print("\nStrojenie modeli wyłącznie na zbiorze development.")
    print("Zbiór final_test nie będzie używany.")
    print(f"Liczba wspólnych foldów: {len(folds)}")

    for fold_number, train_indices, validation_indices in folds:
        overlap = ensure_no_group_overlap(
            df,
            train_indices,
            validation_indices,
            first_name=f"fold_{fold_number}_train",
            second_name=f"fold_{fold_number}_validation",
        )

        print("\n" + "=" * 70)
        print(f"FOLD {fold_number}")
        print(f"Trening: {len(train_indices)}")
        print(f"Walidacja: {len(validation_indices)}")
        print(f"Group overlap: {overlap}")
        print("=" * 70)

        for feature_set_name in FEATURE_SET_NAMES:
            features = select_features_for_fold(
                df=df,
                feature_set_name=feature_set_name,
                train_indices=train_indices,
            )

            (
                X_train_normal,
                X_validation,
                y_validation,
                normal_training_samples,
            ) = prepare_fold_data(
                df=df,
                features=features,
                train_indices=train_indices,
                validation_indices=validation_indices,
            )

            print(
                f"\n{feature_set_name}: "
                f"{len(features)} cech, "
                f"{normal_training_samples} normalnych próbek treningowych"
            )

            for model_name, parameter_grid in PARAMETER_GRIDS.items():
                candidates = list(ParameterGrid(parameter_grid))

                for candidate_number, parameters in enumerate(
                    candidates,
                    start=1,
                ):
                    model = create_model(model_name, parameters)

                    metrics, training_time, prediction_time = (
                        evaluate_candidate(
                            model=model,
                            X_train_normal=X_train_normal,
                            X_validation=X_validation,
                            y_validation=y_validation,
                        )
                    )

                    row = {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "fold": fold_number,
                        "candidate_number": candidate_number,
                        "parameters": json.dumps(
                            parameters,
                            sort_keys=True,
                        ),
                        "number_of_features": len(features),
                        "selected_features": ", ".join(features),
                        "normal_training_samples": normal_training_samples,
                        "validation_samples": len(validation_indices),
                        "training_time_seconds": training_time,
                        "prediction_time_seconds": prediction_time,
                        "partition": "development",
                        "split": "development_group_cv10",
                        "scenario": "supervised_calibrated_one_class",
                        "group_overlap": overlap,
                        "protocol_seed": config.PROTOCOL_RANDOM_STATE,
                        "final_test_used": False,
                        **parameters,
                    }

                    for metric_name, value in metrics.items():
                        row[f"validation_{metric_name}"] = value

                    search_rows.append(row)

                    print(
                        f"  {model_name} {candidate_number}/{len(candidates)} "
                        f"{parameters}: "
                        f"BalAcc={metrics['balanced_accuracy']:.4f}, "
                        f"F1={metrics['f1_anomaly']:.4f}, "
                        f"FPR={metrics['false_positive_rate']:.4f}"
                    )

    search_df = pd.DataFrame(search_rows)
    summary_df = create_parameter_summary(search_df)
    best_parameters_df = select_best_parameters(summary_df)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    search_path = (
        config.REPORTS_DIR
        / "one_class_group_cv10_parameter_search.csv"
    )
    summary_path = (
        config.REPORTS_DIR
        / "one_class_group_cv10_parameter_summary.csv"
    )
    best_parameters_path = (
        config.REPORTS_DIR
        / "one_class_group_cv10_best_parameters.csv"
    )

    search_df.to_csv(search_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    best_parameters_df.to_csv(best_parameters_path, index=False)

    columns_to_display = [
        "model",
        "feature_set",
        "parameters",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_anomaly_mean",
        "false_positive_rate_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]

    print("\n" + "=" * 70)
    print("NAJLEPSZE PARAMETRY NA ZBIORZE DEVELOPMENT")
    print("=" * 70)
    print(
        best_parameters_df[columns_to_display]
        .round(4)
        .to_string(index=False)
    )

    print("\nZapisane pliki:")
    print(f"- {search_path}")
    print(f"- {summary_path}")
    print(f"- {best_parameters_path}")
    print("\nZbiór final_test nie został użyty.")


if __name__ == "__main__":
    optimize_models()