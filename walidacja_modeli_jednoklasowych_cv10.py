import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config
from metryki import calculate_binary_metrics, convert_one_class_predictions
from podzial_danych import add_request_groups, ensure_no_group_overlap
from protokol_eksperymentalny import get_development_folds
from przetwarzanie_danych import get_processed_data
from selekcja_cech import select_with_information_gain, select_with_random_forest


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

SPLIT_NAME = "development_group_cv10"


def create_strict_normal_split(
    df: pd.DataFrame,
    train_indices,
    fold_number: int,
) -> tuple[pd.Index, pd.Index]:
    normal_data = df.loc[train_indices]
    normal_data = normal_data.loc[normal_data["classification"] == 0]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=STRICT_CALIBRATION_SIZE,
        random_state=config.PROTOCOL_RANDOM_STATE + fold_number,
    )
    fit_positions, calibration_positions = next(
        splitter.split(normal_data, groups=normal_data["request_group"])
    )

    fit_indices = pd.Index(normal_data.index[fit_positions])
    calibration_indices = pd.Index(normal_data.index[calibration_positions])

    overlap = ensure_no_group_overlap(
        df=df,
        first_indices=fit_indices,
        second_indices=calibration_indices,
        first_name=f"fold_{fold_number}_strict_fit",
        second_name=f"fold_{fold_number}_strict_calibration",
    )
    if overlap != 0:
        raise RuntimeError("Wewnętrzny podział strict one-class zawiera wspólne grupy.")

    if len(fit_indices) + len(calibration_indices) != len(normal_data):
        raise RuntimeError(
            "Wewnętrzny podział strict one-class nie obejmuje wszystkich "
            "normalnych rekordów."
        )

    return fit_indices, calibration_indices


def calculate_strict_threshold(
    calibration_scores,
    target_fpr: float,
) -> tuple[float, float]:
    scores = np.asarray(calibration_scores, dtype=float)

    if scores.size == 0:
        raise ValueError("Zbiór kalibracyjny nie może być pusty.")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("Docelowy FPR musi należeć do przedziału (0, 1).")

    allowed_false_positives = int(np.floor(target_fpr * len(scores)))
    threshold = np.sort(scores)[::-1][allowed_false_positives]
    calibration_fpr = float(np.mean(scores > threshold))

    if calibration_fpr > target_fpr:
        raise RuntimeError(
            "Wyznaczony próg przekracza docelowy FPR na zbiorze kalibracyjnym."
        )

    return float(threshold), calibration_fpr


def create_model(model_name: str, parameters: dict):
    if model_name == "ISOLATION_FOREST":
        return IsolationForest(
            **parameters,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "ONE_CLASS_SVM":
        return OneClassSVM(**parameters)
    raise ValueError(f"Nieznany model: {model_name}")


def prepare_fold_data(
    df: pd.DataFrame,
    features: list[str],
    train_indices,
    validation_indices,
):
    X_train = df.loc[train_indices, features]
    y_train = df.loc[train_indices, "classification"]
    X_validation = df.loc[validation_indices, features]
    y_validation = df.loc[validation_indices, "classification"]

    X_train_normal = X_train.loc[y_train == 0]
    scaler = StandardScaler()

    return (
        scaler.fit_transform(X_train_normal),
        scaler.transform(X_validation),
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
) -> tuple[dict, np.ndarray, np.ndarray]:
    model = create_model(model_name, parameters)

    start = perf_counter()
    model.fit(X_train_normal)
    training_time = perf_counter() - start

    start = perf_counter()
    y_pred = convert_one_class_predictions(model.predict(X_validation))
    y_score = -model.decision_function(X_validation)
    prediction_time = perf_counter() - start

    metrics = calculate_binary_metrics(y_validation, y_pred, y_score)
    metrics.update(
        {
            "fold": fold_number,
            "model": model_name,
            "feature_set": feature_set_name,
            "configuration": configuration_name,
            "number_of_features": len(features),
            "selected_features": ", ".join(features),
            "parameters": json.dumps(parameters, sort_keys=True),
            "normal_training_samples": normal_training_samples,
            "validation_samples": len(y_validation),
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
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
    X_calibration = df.loc[calibration_indices, features]
    X_validation = df.loc[validation_indices, features]
    y_validation = df.loc[validation_indices, "classification"]

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_calibration = scaler.transform(X_calibration)
    X_validation = scaler.transform(X_validation)

    model = create_model(model_name, parameters)

    start = perf_counter()
    model.fit(X_fit)
    training_time = perf_counter() - start

    start = perf_counter()
    calibration_scores = -model.decision_function(X_calibration)
    validation_scores = -model.decision_function(X_validation)

    y_array = y_validation.to_numpy()
    normal_mask = y_array == 0
    anomaly_mask = y_array == 1
    if not normal_mask.any() or not anomaly_mask.any():
        raise RuntimeError("Fold walidacyjny musi zawierać obie klasy.")

    operating_points = {}
    primary_predictions = None

    for target_fpr in STRICT_TARGET_FPRS:
        threshold, calibration_fpr = calculate_strict_threshold(
            calibration_scores,
            target_fpr,
        )
        predictions = (validation_scores > threshold).astype(int)
        label = f"{int(target_fpr * 100)}_percent"

        operating_points.update(
            {
                f"strict_threshold_{label}": threshold,
                f"strict_calibration_fpr_{label}": calibration_fpr,
                f"strict_validation_fpr_{label}": float(
                    np.mean(predictions[normal_mask])
                ),
                f"strict_validation_tpr_{label}": float(
                    np.mean(predictions[anomaly_mask])
                ),
            }
        )

        if np.isclose(target_fpr, STRICT_PRIMARY_FPR):
            primary_predictions = predictions

    if primary_predictions is None:
        raise RuntimeError("Nie wyznaczono głównego punktu pracy strict one-class.")

    prediction_time = perf_counter() - start
    metrics = calculate_binary_metrics(
        y_validation,
        primary_predictions,
        validation_scores,
    )
    metrics.update(operating_points)
    metrics.update(
        {
            "fold": fold_number,
            "scenario": STRICT_ONE_CLASS_SCENARIO,
            "model": model_name,
            "feature_set": feature_set_name,
            "configuration": "FIXED_STRICT",
            "number_of_features": len(features),
            "selected_features": ", ".join(features),
            "parameters": json.dumps(parameters, sort_keys=True),
            "normal_training_samples": len(fit_indices),
            "normal_calibration_samples": len(calibration_indices),
            "validation_samples": len(y_validation),
            "decision_target_fpr": STRICT_PRIMARY_FPR,
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
        }
    )

    if not np.isclose(
        metrics["false_positive_rate"],
        operating_points["strict_validation_fpr_5_percent"],
    ):
        raise RuntimeError("FPR głównej predykcji nie zgadza się z punktem pracy 5%.")

    return metrics, primary_predictions, validation_scores


def add_protocol_metadata(metrics: dict, group_overlap: int) -> None:
    metrics.update(
        {
            "split": SPLIT_NAME,
            "partition": "development",
            "group_overlap": group_overlap,
            "protocol_seed": config.PROTOCOL_RANDOM_STATE,
            "final_test_used": False,
        }
    )


def create_prediction_frame(
    df: pd.DataFrame,
    validation_indices,
    y_pred,
    y_score,
    model_name: str,
    feature_set_name: str,
    configuration_name: str,
    scenario: str,
    fold_number: int,
    group_overlap: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": pd.Index(validation_indices).to_numpy(),
            "request_group": df.loc[
                validation_indices, "request_group"
            ].to_numpy(),
            "classification": df.loc[
                validation_indices, "classification"
            ].to_numpy(),
            "y_pred": y_pred,
            "y_score": y_score,
            "model": model_name,
            "feature_set": feature_set_name,
            "configuration": configuration_name,
            "scenario": scenario,
            "fold": fold_number,
            "split": SPLIT_NAME,
            "group_overlap": group_overlap,
            "protocol_seed": config.PROTOCOL_RANDOM_STATE,
            "final_test_used": False,
        }
    )


def create_calibrated_experiments(feature_set_name: str) -> list[tuple]:
    return [
        (
            "ISOLATION_FOREST",
            "DEFAULT",
            {**config.ISOLATION_FOREST_PARAMS},
        ),
        (
            "ISOLATION_FOREST",
            "TUNED",
            {
                **config.ISOLATION_FOREST_PARAMS,
                **TUNED_ISOLATION_FOREST[feature_set_name],
            },
        ),
        (
            "ONE_CLASS_SVM",
            "DEFAULT",
            {**config.OCSVM_PARAMS},
        ),
        (
            "ONE_CLASS_SVM",
            "TUNED",
            {
                **config.OCSVM_PARAMS,
                **TUNED_ONE_CLASS_SVM[feature_set_name],
            },
        ),
    ]


def create_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "number_of_features_mean": ("number_of_features", "mean"),
        "number_of_features_std": ("number_of_features", "std"),
        "parameters": ("parameters", "first"),
    }

    regular_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "specificity",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "tpr_at_fpr_1_percent",
        "tpr_at_fpr_5_percent",
        "tpr_at_fpr_10_percent",
    ]
    for metric in regular_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    aggregations.update(
        {
            "training_time_mean": ("training_time_seconds", "mean"),
            "training_time_std": ("training_time_seconds", "std"),
            "normal_training_samples_mean": ("normal_training_samples", "mean"),
            "normal_calibration_samples_mean": (
                "normal_calibration_samples",
                "mean",
            ),
            "decision_target_fpr": ("decision_target_fpr", "first"),
        }
    )

    for percent in (1, 5, 10):
        label = f"{percent}_percent"
        aggregations[f"strict_calibration_fpr_{label}_mean"] = (
            f"strict_calibration_fpr_{label}",
            "mean",
        )
        for metric in ("fpr", "tpr"):
            source = f"strict_validation_{metric}_{label}"
            aggregations[f"{source}_mean"] = (source, "mean")
            aggregations[f"{source}_std"] = (source, "std")

    return metrics_df.groupby(
        ["scenario", "model", "feature_set", "configuration"],
        as_index=False,
    ).agg(**aggregations)


def select_fold_features(
    df: pd.DataFrame,
    train_indices,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    X_train = df.loc[train_indices, config.ML_FEATURES_ALTHUBITI_9]
    y_train = df.loc[train_indices, "classification"]

    ig_features, _ = select_with_information_gain(X_train, y_train)
    rf_features, _ = select_with_random_forest(X_train, y_train)

    feature_sets = {
        "BASIC": config.ML_FEATURES_BASIC,
        "ALTHUBITI_9": config.ML_FEATURES_ALTHUBITI_9,
        "INFORMATION_GAIN": ig_features,
        "RF_IMPORTANCE": rf_features,
    }
    return feature_sets, ig_features, rf_features


def run_strict_experiments(
    df: pd.DataFrame,
    fold_number: int,
    fit_indices,
    calibration_indices,
    validation_indices,
    group_overlap: int,
) -> tuple[list[dict], list[pd.DataFrame]]:
    metrics_rows = []
    prediction_frames = []
    feature_sets = {
        "BASIC": config.ML_FEATURES_BASIC,
        "ALTHUBITI_9": config.ML_FEATURES_ALTHUBITI_9,
    }
    experiments = [
        ("ISOLATION_FOREST", {**config.ISOLATION_FOREST_PARAMS}),
        ("ONE_CLASS_SVM", {**config.OCSVM_PARAMS}),
    ]

    for feature_set_name, features in feature_sets.items():
        for model_name, parameters in experiments:
            print(f"{model_name} | {feature_set_name} | FIXED_STRICT")
            metrics, y_pred, y_score = evaluate_strict_configuration(
                df=df,
                model_name=model_name,
                feature_set_name=feature_set_name,
                features=features,
                parameters=parameters,
                fit_indices=fit_indices,
                calibration_indices=calibration_indices,
                validation_indices=validation_indices,
                fold_number=fold_number,
            )
            add_protocol_metadata(metrics, group_overlap)
            metrics_rows.append(metrics)
            prediction_frames.append(
                create_prediction_frame(
                    df=df,
                    validation_indices=validation_indices,
                    y_pred=y_pred,
                    y_score=y_score,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    configuration_name="FIXED_STRICT",
                    scenario=STRICT_ONE_CLASS_SCENARIO,
                    fold_number=fold_number,
                    group_overlap=group_overlap,
                )
            )
            print(
                "  Strict 5% FPR: "
                f"TPR={metrics['strict_validation_tpr_5_percent']:.4f}, "
                f"rzeczywisty FPR="
                f"{metrics['strict_validation_fpr_5_percent']:.4f}"
            )

    return metrics_rows, prediction_frames


def run_calibrated_experiments(
    df: pd.DataFrame,
    fold_number: int,
    train_indices,
    validation_indices,
    group_overlap: int,
    feature_sets: dict[str, list[str]],
) -> tuple[list[dict], list[pd.DataFrame]]:
    metrics_rows = []
    prediction_frames = []

    for feature_set_name, features in feature_sets.items():
        X_train, X_validation, y_validation, normal_samples = prepare_fold_data(
            df,
            features,
            train_indices,
            validation_indices,
        )

        for model_name, configuration_name, parameters in (
            create_calibrated_experiments(feature_set_name)
        ):
            print(f"{model_name} | {feature_set_name} | {configuration_name}")
            metrics, y_pred, y_score = evaluate_configuration(
                model_name=model_name,
                feature_set_name=feature_set_name,
                configuration_name=configuration_name,
                features=features,
                parameters=parameters,
                X_train_normal=X_train,
                X_validation=X_validation,
                y_validation=y_validation,
                normal_training_samples=normal_samples,
                fold_number=fold_number,
            )
            add_protocol_metadata(metrics, group_overlap)
            metrics_rows.append(metrics)
            prediction_frames.append(
                create_prediction_frame(
                    df=df,
                    validation_indices=validation_indices,
                    y_pred=y_pred,
                    y_score=y_score,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    configuration_name=configuration_name,
                    scenario=ONE_CLASS_SCENARIO,
                    fold_number=fold_number,
                    group_overlap=group_overlap,
                )
            )
            print(
                f"  BalAcc={metrics['balanced_accuracy']:.4f}, "
                f"F1={metrics['f1_anomaly']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}, "
                f"FPR={metrics['false_positive_rate']:.4f}"
            )

    return metrics_rows, prediction_frames


def validate_oof_predictions(
    predictions_df: pd.DataFrame,
    folds: list[tuple],
) -> None:
    expected_records = sum(len(validation) for _, _, validation in folds)
    counts = predictions_df.groupby(
        ["scenario", "model", "feature_set", "configuration"]
    )["row_id"].nunique()

    if not (counts == expected_records).all():
        raise RuntimeError(
            "Nie każda konfiguracja zawiera dokładnie jedną predykcję OOF "
            "dla każdego rekordu development."
        )


def save_results(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> list:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        config.REPORTS_DIR / "one_class_group_cv10_configs_folds.csv",
        config.REPORTS_DIR / "one_class_group_cv10_configs_summary.csv",
        config.REPORTS_DIR / "one_class_group_cv10_selection.csv",
        config.REPORTS_DIR / "one_class_group_cv10_oof_predictions.csv.gz",
    ]

    metrics_df.to_csv(paths[0], index=False)
    summary_df.to_csv(paths[1], index=False)
    selection_df.to_csv(paths[2], index=False)
    predictions_df.to_csv(paths[3], index=False, compression="gzip")
    return paths


def print_summary(summary_df: pd.DataFrame) -> None:
    columns = [
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
    strict_columns = [
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
    print(summary_df[columns].round(4).to_string(index=False))

    strict_summary = summary_df.loc[
        summary_df["scenario"] == STRICT_ONE_CLASS_SCENARIO
    ]
    if not strict_summary.empty:
        print("\n" + "=" * 75)
        print("PUNKTY PRACY STRICT ONE-CLASS")
        print("=" * 75)
        print(strict_summary[strict_columns].round(4).to_string(index=False))


def validate_configs_cv10() -> None:
    df = add_request_groups(get_processed_data())
    folds = get_development_folds(df)

    all_metrics = []
    all_predictions = []
    selection_rows = []

    for fold_number, train_indices, validation_indices in folds:
        print("\n" + "#" * 75)
        print(f"FOLD {fold_number}/{config.DEVELOPMENT_CV_N_SPLITS}")
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

        fit_indices, calibration_indices = create_strict_normal_split(
            df,
            train_indices,
            fold_number,
        )
        print(f"Strict one-class — uczenie: {len(fit_indices)} rekordów")
        print(f"Strict one-class — kalibracja: {len(calibration_indices)} rekordów")

        metrics, predictions = run_strict_experiments(
            df,
            fold_number,
            fit_indices,
            calibration_indices,
            validation_indices,
            group_overlap,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

        feature_sets, ig_features, rf_features = select_fold_features(
            df,
            train_indices,
        )
        print(f"IG: {ig_features}")
        print(f"RF: {rf_features}")
        selection_rows.append(
            {
                "fold": fold_number,
                "information_gain_features": ", ".join(ig_features),
                "information_gain_feature_count": len(ig_features),
                "rf_importance_features": ", ".join(rf_features),
                "rf_importance_feature_count": len(rf_features),
                "split": SPLIT_NAME,
                "partition": "development",
                "group_overlap": group_overlap,
                "protocol_seed": config.PROTOCOL_RANDOM_STATE,
                "final_test_used": False,
                "scenario": ONE_CLASS_SCENARIO,
            }
        )

        metrics, predictions = run_calibrated_experiments(
            df,
            fold_number,
            train_indices,
            validation_indices,
            group_overlap,
            feature_sets,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    selection_df = pd.DataFrame(selection_rows)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    validate_oof_predictions(predictions_df, folds)
    summary_df = create_summary(metrics_df)

    paths = save_results(metrics_df, summary_df, selection_df, predictions_df)
    print_summary(summary_df)

    print("\nZapisane pliki:")
    for path in paths:
        print(f"- {path}")
    print("Zbiór final_test nie został użyty.")


if __name__ == "__main__":
    validate_configs_cv10()