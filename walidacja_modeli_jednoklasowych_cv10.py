import json
from time import perf_counter

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from podzial_danych import (
    add_request_groups,
    ensure_no_group_overlap,
)
from protokol_eksperymentalny import get_development_folds

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
from sklearn.model_selection import GroupShuffleSplit


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

ONE_CLASS_SCENARIO = "supervised_calibrated_one_class"
STRICT_ONE_CLASS_SCENARIO = "strict_one_class"
STRICT_CALIBRATION_SIZE = 0.2
STRICT_TARGET_FPRS = (0.01, 0.05, 0.10)
STRICT_PRIMARY_FPR = 0.05


def create_strict_normal_split(
    df: pd.DataFrame,
    train_indices,
    fold_number: int,
) -> tuple[pd.Index, pd.Index]:
    normal_training_data = df.loc[train_indices]
    normal_training_data = normal_training_data.loc[
        normal_training_data["classification"] == 0
    ]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=STRICT_CALIBRATION_SIZE,
        random_state=(
            config.PROTOCOL_RANDOM_STATE
            + fold_number
        ),
    )

    fit_positions, calibration_positions = next(
        splitter.split(
            normal_training_data,
            groups=normal_training_data["request_group"],
        )
    )

    fit_indices = pd.Index(
        normal_training_data.index[fit_positions]
    )

    calibration_indices = pd.Index(
        normal_training_data.index[
            calibration_positions
        ]
    )

    calibration_overlap = ensure_no_group_overlap(
        df=df,
        first_indices=fit_indices,
        second_indices=calibration_indices,
        first_name=f"fold_{fold_number}_strict_fit",
        second_name=f"fold_{fold_number}_strict_calibration",
    )

    if calibration_overlap != 0:
        raise RuntimeError(
            "Wewnętrzny podział strict one-class "
            "zawiera wspólne grupy."
        )

    if len(fit_indices) + len(calibration_indices) != len(
        normal_training_data
    ):
        raise RuntimeError(
            "Wewnętrzny podział strict one-class "
            "nie obejmuje wszystkich normalnych rekordów."
        )

    return fit_indices, calibration_indices


def calculate_strict_threshold(
    calibration_scores,
    target_fpr: float,
) -> tuple[float, float]:
    scores = np.asarray(
        calibration_scores,
        dtype=float,
    )

    if scores.size == 0:
        raise ValueError(
            "Zbiór kalibracyjny nie może być pusty."
        )

    if not 0.0 < target_fpr < 1.0:
        raise ValueError(
            "Docelowy FPR musi należeć do przedziału (0, 1)."
        )

    allowed_false_positives = int(
        np.floor(target_fpr * len(scores))
    )

    ordered_scores = np.sort(scores)[::-1]
    threshold = ordered_scores[
        allowed_false_positives
    ]

    calibration_predictions = (
        scores > threshold
    )

    calibration_fpr = float(
        np.mean(calibration_predictions)
    )

    if calibration_fpr > target_fpr:
        raise RuntimeError(
            "Wyznaczony próg przekracza docelowy FPR "
            "na zbiorze kalibracyjnym."
        )

    return float(threshold), calibration_fpr


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
    validation_indices,
):
    X_train = df.loc[
        train_indices,
        features,
    ]

    y_train = df.loc[
        train_indices,
        "classification",
    ]

    X_validation = df.loc[
        validation_indices,
        features,
    ]

    y_validation = df.loc[
        validation_indices,
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

    X_validation_scaled = scaler.transform(
        X_validation
    )

    return (
        X_train_normal_scaled,
        X_validation_scaled,
        y_validation,
        len(X_train_normal),
    )


def evaluate_configuration(
    model_name: str,
    feature_set_name: str,
    configuration_name: str,
    features: list[str],
    parameters: dict,
    X_train_normal,
    X_validation,
    y_validation,
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
        X_validation
    )

    y_pred = convert_one_class_predictions(raw_predictions)

    y_score = -model.decision_function(
        X_validation
    )

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_binary_metrics(
        y_true=y_validation,
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
            "validation_samples": len(y_validation),
            "training_time_seconds": (
                training_time
            ),
            "prediction_time_seconds": (
                prediction_time
            ),
            "scenario": ONE_CLASS_SCENARIO,
        }
    )

    return metrics, y_pred, y_score


def evaluate_strict_configuration(
    df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
    features: list[str],
    parameters: dict,
    fit_indices,
    calibration_indices,
    validation_indices,
    fold_number: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    X_fit = df.loc[fit_indices, features]
    X_calibration = df.loc[
        calibration_indices,
        features,
    ]
    X_validation = df.loc[
        validation_indices,
        features,
    ]

    y_validation = df.loc[
        validation_indices,
        "classification",
    ]

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_calibration_scaled = scaler.transform(
        X_calibration
    )
    X_validation_scaled = scaler.transform(
        X_validation
    )

    model = create_model(
        model_name,
        parameters,
    )

    training_start = perf_counter()
    model.fit(X_fit_scaled)
    training_time = perf_counter() - training_start

    prediction_start = perf_counter()

    calibration_scores = -model.decision_function(
        X_calibration_scaled
    )

    validation_scores = -model.decision_function(
        X_validation_scaled
    )

    normal_mask = (
        y_validation.to_numpy() == 0
    )

    anomaly_mask = (
        y_validation.to_numpy() == 1
    )

    if not normal_mask.any() or not anomaly_mask.any():
        raise RuntimeError(
            "Fold walidacyjny musi zawierać "
            "rekordy normalne i anomalne."
        )

    operating_point_metrics = {}
    primary_predictions = None

    for target_fpr in STRICT_TARGET_FPRS:
        threshold, calibration_fpr = (
            calculate_strict_threshold(
                calibration_scores,
                target_fpr,
            )
        )

        validation_predictions = (
            validation_scores > threshold
        ).astype(int)

        validation_fpr = float(
            np.mean(
                validation_predictions[normal_mask]
            )
        )

        validation_tpr = float(
            np.mean(
                validation_predictions[anomaly_mask]
            )
        )

        fpr_label = (
            f"{int(target_fpr * 100)}_percent"
        )

        operating_point_metrics[
            f"strict_threshold_{fpr_label}"
        ] = threshold

        operating_point_metrics[
            f"strict_calibration_fpr_{fpr_label}"
        ] = calibration_fpr

        operating_point_metrics[
            f"strict_validation_fpr_{fpr_label}"
        ] = validation_fpr

        operating_point_metrics[
            f"strict_validation_tpr_{fpr_label}"
        ] = validation_tpr

        if np.isclose(
            target_fpr,
            STRICT_PRIMARY_FPR,
        ):
            primary_predictions = (
                validation_predictions
            )

    if primary_predictions is None:
        raise RuntimeError(
            "Nie wyznaczono predykcji dla głównego "
            "punktu pracy strict one-class."
        )

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_binary_metrics(
        y_true=y_validation,
        y_pred=primary_predictions,
        y_score=validation_scores,
    )

    metrics.update(operating_point_metrics)

    metrics.update({
        "fold": fold_number,
        "scenario": STRICT_ONE_CLASS_SCENARIO,
        "model": model_name,
        "feature_set": feature_set_name,
        "configuration": "FIXED_STRICT",
        "number_of_features": len(features),
        "selected_features": ", ".join(features),
        "parameters": json.dumps(
            parameters,
            sort_keys=True,
        ),
        "normal_training_samples": len(fit_indices),
        "normal_calibration_samples": len(
            calibration_indices
        ),
        "validation_samples": len(y_validation),
        "decision_target_fpr": STRICT_PRIMARY_FPR,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
    })

    expected_primary_fpr = (
        operating_point_metrics[
            "strict_validation_fpr_5_percent"
        ]
    )

    if not np.isclose(
        metrics["false_positive_rate"],
        expected_primary_fpr,
    ):
        raise RuntimeError(
            "FPR głównej predykcji nie zgadza się "
            "z punktem pracy 5%."
        )

    return (
        metrics,
        primary_predictions,
        validation_scores,
    )


def create_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    grouping_columns = [
        "scenario",
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
            tpr_at_fpr_1_percent_mean=(
                "tpr_at_fpr_1_percent",
                "mean",
            ),
            tpr_at_fpr_1_percent_std=(
                "tpr_at_fpr_1_percent",
                "std",
            ),
            tpr_at_fpr_5_percent_mean=(
                "tpr_at_fpr_5_percent",
                "mean",
            ),
            tpr_at_fpr_5_percent_std=(
                "tpr_at_fpr_5_percent",
                "std",
            ),
            tpr_at_fpr_10_percent_mean=(
                "tpr_at_fpr_10_percent",
                "mean",
            ),
            tpr_at_fpr_10_percent_std=(
                "tpr_at_fpr_10_percent",
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
            normal_training_samples_mean=(
                "normal_training_samples",
                "mean",
            ),
            normal_calibration_samples_mean=(
                "normal_calibration_samples",
                "mean",
            ),
            decision_target_fpr=(
                "decision_target_fpr",
                "first",
            ),
            strict_calibration_fpr_1_percent_mean=(
                "strict_calibration_fpr_1_percent",
                "mean",
            ),
            strict_validation_fpr_1_percent_mean=(
                "strict_validation_fpr_1_percent",
                "mean",
            ),
            strict_validation_fpr_1_percent_std=(
                "strict_validation_fpr_1_percent",
                "std",
            ),
            strict_validation_tpr_1_percent_mean=(
                "strict_validation_tpr_1_percent",
                "mean",
            ),
            strict_validation_tpr_1_percent_std=(
                "strict_validation_tpr_1_percent",
                "std",
            ),
            strict_calibration_fpr_5_percent_mean=(
                "strict_calibration_fpr_5_percent",
                "mean",
            ),
            strict_validation_fpr_5_percent_mean=(
                "strict_validation_fpr_5_percent",
                "mean",
            ),
            strict_validation_fpr_5_percent_std=(
                "strict_validation_fpr_5_percent",
                "std",
            ),
            strict_validation_tpr_5_percent_mean=(
                "strict_validation_tpr_5_percent",
                "mean",
            ),
            strict_validation_tpr_5_percent_std=(
                "strict_validation_tpr_5_percent",
                "std",
            ),
            strict_calibration_fpr_10_percent_mean=(
                "strict_calibration_fpr_10_percent",
                "mean",
            ),
            strict_validation_fpr_10_percent_mean=(
                "strict_validation_fpr_10_percent",
                "mean",
            ),
            strict_validation_fpr_10_percent_std=(
                "strict_validation_fpr_10_percent",
                "std",
            ),
            strict_validation_tpr_10_percent_mean=(
                "strict_validation_tpr_10_percent",
                "mean",
            ),
            strict_validation_tpr_10_percent_std=(
                "strict_validation_tpr_10_percent",
                "std",
            ),
        )
    )

    return summary_df


def validate_configs_cv10() -> None:
    df = add_request_groups(get_processed_data())
    folds = get_development_folds(df)

    all_metrics = []
    all_predictions = []
    selection_rows = []

    for fold_number, train_indices, validation_indices in folds:
        print("\n" + "#" * 75)
        print(
            f"FOLD {fold_number}/"
            f"{config.DEVELOPMENT_CV_N_SPLITS}"
        )
        print("#" * 75)

        group_overlap = ensure_no_group_overlap(
            df=df,
            first_indices=train_indices,
            second_indices=validation_indices,
            first_name=f"fold_{fold_number}_train",
            second_name=f"fold_{fold_number}_validation",
        )

        print(f"Trening: {len(train_indices)} rekordów")
        print(f"Walidacja: {len(validation_indices)} rekordów")
        print(f"Wspólne grupy: {group_overlap}")

        strict_fit_indices, strict_calibration_indices = (
            create_strict_normal_split(
                df=df,
                train_indices=train_indices,
                fold_number=fold_number,
            )
        )

        print(
            "Strict one-class — uczenie: "
            f"{len(strict_fit_indices)} normalnych rekordów"
        )
        print(
            "Strict one-class — kalibracja: "
            f"{len(strict_calibration_indices)} normalnych rekordów"
        )

        strict_feature_sets = {
            "BASIC": config.ML_FEATURES_BASIC,
            "ALTHUBITI_9": config.ML_FEATURES_ALTHUBITI_9,
        }

        strict_experiments = [
            (
                "ISOLATION_FOREST",
                {
                    **config.ISOLATION_FOREST_PARAMS
                },
            ),
            (
                "ONE_CLASS_SVM",
                {
                    **config.OCSVM_PARAMS
                },
            ),
        ]

        for (
            strict_feature_set_name,
            strict_features,
        ) in strict_feature_sets.items():
            for (
                strict_model_name,
                strict_parameters,
            ) in strict_experiments:
                print(
                    f"{strict_model_name} | "
                    f"{strict_feature_set_name} | "
                    "FIXED_STRICT"
                )

                (
                    strict_metrics,
                    strict_y_pred,
                    strict_y_score,
                ) = evaluate_strict_configuration(
                    df=df,
                    model_name=strict_model_name,
                    feature_set_name=(
                        strict_feature_set_name
                    ),
                    features=strict_features,
                    parameters=strict_parameters,
                    fit_indices=strict_fit_indices,
                    calibration_indices=(
                        strict_calibration_indices
                    ),
                    validation_indices=validation_indices,
                    fold_number=fold_number,
                )

                strict_metrics.update({
                    "split": "development_group_cv10",
                    "partition": "development",
                    "group_overlap": group_overlap,
                    "protocol_seed": (
                        config.PROTOCOL_RANDOM_STATE
                    ),
                    "final_test_used": False,
                })

                all_metrics.append(strict_metrics)

                strict_fold_predictions = pd.DataFrame({
                    "row_id": pd.Index(
                        validation_indices
                    ).to_numpy(),
                    "request_group": df.loc[
                        validation_indices,
                        "request_group",
                    ].to_numpy(),
                    "classification": df.loc[
                        validation_indices,
                        "classification",
                    ].to_numpy(),
                    "y_pred": strict_y_pred,
                    "y_score": strict_y_score,
                    "model": strict_model_name,
                    "feature_set": (
                        strict_feature_set_name
                    ),
                    "configuration": "FIXED_STRICT",
                    "scenario": (
                        STRICT_ONE_CLASS_SCENARIO
                    ),
                    "fold": fold_number,
                    "split": "development_group_cv10",
                    "group_overlap": group_overlap,
                    "protocol_seed": (
                        config.PROTOCOL_RANDOM_STATE
                    ),
                    "final_test_used": False,
                })

                all_predictions.append(
                    strict_fold_predictions
                )

                print(
                    "  Strict 5% FPR: "
                    f"TPR="
                    f"{strict_metrics['strict_validation_tpr_5_percent']:.4f}, "
                    f"rzeczywisty FPR="
                    f"{strict_metrics['strict_validation_fpr_5_percent']:.4f}"
                )

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
                "split": "development_group_cv10",
                "partition": "development",
                "group_overlap": group_overlap,
                "protocol_seed": config.PROTOCOL_RANDOM_STATE,
                "final_test_used": False,
                "scenario": ONE_CLASS_SCENARIO,
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
                X_validation,
                y_validation,
                normal_training_samples,
            ) = prepare_fold_data(
                df=df,
                features=features,
                train_indices=train_indices,
                validation_indices=validation_indices,
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

                metrics, y_pred, y_score = evaluate_configuration(
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
                    X_validation=X_validation,
                    y_validation=y_validation,
                    normal_training_samples=(
                        normal_training_samples
                    ),
                    fold_number=fold_number,
                )

                metrics["split"] = "development_group_cv10"
                metrics["partition"] = "development"
                metrics["group_overlap"] = group_overlap
                metrics["protocol_seed"] = config.PROTOCOL_RANDOM_STATE
                metrics["final_test_used"] = False

                all_metrics.append(metrics)
                fold_predictions = pd.DataFrame({
                    "row_id": pd.Index(validation_indices).to_numpy(),
                    "request_group": df.loc[
                        validation_indices,
                        "request_group",
                    ].to_numpy(),
                    "classification": y_validation.to_numpy(),
                    "y_pred": y_pred,
                    "y_score": y_score,
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "configuration": configuration_name,
                    "scenario": ONE_CLASS_SCENARIO,
                    "fold": fold_number,
                    "split": "development_group_cv10",
                    "group_overlap": group_overlap,
                    "protocol_seed": config.PROTOCOL_RANDOM_STATE,
                    "final_test_used": False,
                })

                all_predictions.append(fold_predictions)

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
                print(
                    f"  TPR@FPR 1%="
                    f"{metrics['tpr_at_fpr_1_percent']:.4f}, "
                    f"5%="
                    f"{metrics['tpr_at_fpr_5_percent']:.4f}, "
                    f"10%="
                    f"{metrics['tpr_at_fpr_10_percent']:.4f}"
                )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    selection_df = pd.DataFrame(
        selection_rows
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    expected_records = sum(
        len(validation_indices)
        for _, _, validation_indices in folds
    )

    configuration_counts = (
        predictions_df
        .groupby([
            "scenario",
            "model",
            "feature_set",
            "configuration",
        ])["row_id"]
        .nunique()
    )

    if not (configuration_counts == expected_records).all():
        raise RuntimeError(
            "Nie każda konfiguracja zawiera dokładnie jedną "
            "predykcję OOF dla każdego rekordu development."
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
        / "one_class_group_cv10_configs_folds.csv"
    )

    summary_path = (
        config.REPORTS_DIR
        / "one_class_group_cv10_configs_summary.csv"
    )

    selection_path = (
        config.REPORTS_DIR
        / "one_class_group_cv10_selection.csv"
    )

    predictions_path = (
            config.REPORTS_DIR
            / "one_class_group_cv10_oof_predictions.csv.gz"
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

    predictions_df.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    columns_to_display = [
        "scenario",
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
        "tpr_at_fpr_1_percent_mean",
        "tpr_at_fpr_1_percent_std",
        "tpr_at_fpr_5_percent_mean",
        "tpr_at_fpr_5_percent_std",
        "tpr_at_fpr_10_percent_mean",
        "tpr_at_fpr_10_percent_std",
        "training_time_mean",
    ]

    strict_columns_to_display = [
        "scenario",
        "model",
        "feature_set",
        "normal_training_samples_mean",
        "normal_calibration_samples_mean",
        "strict_calibration_fpr_1_percent_mean",
        "strict_validation_fpr_1_percent_mean",
        "strict_validation_tpr_1_percent_mean",
        "strict_calibration_fpr_5_percent_mean",
        "strict_validation_fpr_5_percent_mean",
        "strict_validation_tpr_5_percent_mean",
        "strict_calibration_fpr_10_percent_mean",
        "strict_validation_fpr_10_percent_mean",
        "strict_validation_tpr_10_percent_mean",
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

    strict_summary_df = summary_df.loc[
        summary_df["scenario"]
        == STRICT_ONE_CLASS_SCENARIO
        ]

    if not strict_summary_df.empty:
        print("\n" + "=" * 75)
        print("PUNKTY PRACY STRICT ONE-CLASS")
        print("=" * 75)

        print(
            strict_summary_df[
                strict_columns_to_display
            ].round(4).to_string(index=False)
        )

    print("\nZapisane pliki:")
    print(f"- {folds_path}")
    print(f"- {summary_path}")
    print(f"- {selection_path}")
    print(f"- {predictions_path}")


if __name__ == "__main__":
    validate_configs_cv10()