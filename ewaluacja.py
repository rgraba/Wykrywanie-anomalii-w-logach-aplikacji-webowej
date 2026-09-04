import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config
from metryki import calculate_binary_metrics, convert_one_class_predictions
from podzial_danych import add_request_groups, ensure_no_group_overlap
from protokol_eksperymentalny import (
    get_development_and_final_test_indices,
    get_development_folds,
)
from przetwarzanie_danych import get_processed_data
from walidacja_modeli_jednoklasowych_cv10 import (
    calculate_strict_threshold,
    create_model,
    create_strict_normal_split,
)


FINAL_METRICS_FILE = config.REPORTS_DIR / "final_test_metrics_seed_2026.csv"
FINAL_PREDICTIONS_FILE = (
    config.REPORTS_DIR / "final_test_predictions_seed_2026.csv.gz"
)

MODEL_CONFIGURATIONS = [
    ("random_forest", "RANDOM_FOREST", "supervised",
     config.FINAL_RANDOM_FOREST_FEATURE_SET,
     config.FINAL_RANDOM_FOREST_FEATURES, False),
    ("isolation_forest_calibrated", "ISOLATION_FOREST",
     "supervised_calibrated_one_class",
     config.FINAL_IFOREST_CALIBRATED_FEATURE_SET,
     config.FINAL_IFOREST_CALIBRATED_FEATURES, False),
    ("one_class_svm_calibrated", "ONE_CLASS_SVM",
     "supervised_calibrated_one_class",
     config.FINAL_OCSVM_CALIBRATED_FEATURE_SET,
     config.FINAL_OCSVM_CALIBRATED_FEATURES, False),
    ("isolation_forest_strict", "ISOLATION_FOREST", "strict_one_class",
     config.FINAL_IFOREST_STRICT_FEATURE_SET,
     config.FINAL_IFOREST_STRICT_FEATURES, True),
    ("one_class_svm_strict", "ONE_CLASS_SVM", "strict_one_class",
     config.FINAL_OCSVM_STRICT_FEATURE_SET,
     config.FINAL_OCSVM_STRICT_FEATURES, True),
]


def run_models(df, train_indices, evaluation_indices, split_number, final_test):
    overlap = ensure_no_group_overlap(
        df, train_indices, evaluation_indices, "training", "evaluation"
    )
    strict_fit, strict_calibration = create_strict_normal_split(
        df=df, train_indices=train_indices, fold_number=split_number
    )

    y_train = df.loc[train_indices, "classification"].to_numpy()
    y_true = df.loc[evaluation_indices, "classification"].to_numpy()
    normal_train = train_indices[y_train == 0]
    metrics_rows = []
    prediction_frames = []

    for key, model_name, scenario, feature_set, features, strict in (
        MODEL_CONFIGURATIONS
    ):
        print(f"Uczenie: {key}")
        threshold = None
        calibration_fpr = None

        if model_name == "RANDOM_FOREST":
            model = RandomForestClassifier(
                **config.RANDOM_FOREST_PARAMS,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(df.loc[train_indices, features], y_train)
            X_evaluation = df.loc[evaluation_indices, features]
            y_pred = model.predict(X_evaluation)
            y_score = model.predict_proba(X_evaluation)[:, 1]
            prediction_rule = "model_default"
        else:
            fit_indices = strict_fit if strict else normal_train
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(df.loc[fit_indices, features])
            parameters = (
                config.ISOLATION_FOREST_PARAMS
                if model_name == "ISOLATION_FOREST"
                else config.OCSVM_PARAMS
            )
            model = create_model(model_name, {**parameters})
            model.fit(X_fit)

            if strict:
                X_calibration = scaler.transform(
                    df.loc[strict_calibration, features]
                )
                calibration_scores = -model.decision_function(X_calibration)
                threshold, calibration_fpr = calculate_strict_threshold(
                    calibration_scores,
                    config.STRICT_ONE_CLASS_TARGET_FPR,
                )

            X_evaluation = scaler.transform(
                df.loc[evaluation_indices, features]
            )
            y_score = -model.decision_function(X_evaluation)
            if strict:
                y_pred = (y_score > threshold).astype(int)
                prediction_rule = (
                    "normal_only_calibration_target_fpr_5_percent"
                )
            else:
                y_pred = convert_one_class_predictions(
                    model.predict(X_evaluation)
                )
                prediction_rule = "model_default"

        metrics = calculate_binary_metrics(y_true, y_pred, y_score)
        metrics.update({
            "model_key": key,
            "model": model_name,
            "scenario": scenario,
            "feature_set": feature_set,
            "selected_features": ", ".join(features),
            "prediction_rule": prediction_rule,
            "decision_threshold": threshold,
            "calibration_fpr": calibration_fpr,
            "group_overlap": overlap,
            "protocol_seed": config.PROTOCOL_RANDOM_STATE,
            "final_test_used": final_test,
        })
        metrics_rows.append(metrics)

        prediction_frames.append(pd.DataFrame({
            "row_id": np.asarray(evaluation_indices),
            "request_group": df.loc[
                evaluation_indices, "request_group"
            ].to_numpy(),
            "classification": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "model_key": key,
            "scenario": scenario,
            "feature_set": feature_set,
            "prediction_rule": prediction_rule,
            "decision_threshold": threshold,
            "group_overlap": overlap,
            "protocol_seed": config.PROTOCOL_RANDOM_STATE,
            "final_test_used": final_test,
        }))

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(evaluation_indices) * len(MODEL_CONFIGURATIONS):
        raise RuntimeError("Nieprawidłowa liczba predykcji.")
    return metrics, predictions


def print_metrics(metrics):
    columns = [
        "model_key", "balanced_accuracy", "f1_anomaly",
        "false_positive_rate", "roc_auc", "pr_auc",
        "tpr_at_fpr_1_percent", "tpr_at_fpr_5_percent",
        "tpr_at_fpr_10_percent",
    ]
    print(metrics[columns].round(4).to_string(index=False))


def run_smoke_test():
    df = add_request_groups(get_processed_data())
    fold, train_indices, validation_indices = get_development_folds(df)[0]
    metrics, _ = run_models(
        df, train_indices, validation_indices, fold, final_test=False
    )
    print("\nTEST NA DEVELOPMENT")
    print_metrics(metrics)
    print("\nZbiór final_test nie został użyty.")


def run_final_test():
    existing = [
        path for path in (FINAL_METRICS_FILE, FINAL_PREDICTIONS_FILE)
        if path.exists()
    ]
    if existing:
        raise RuntimeError(f"Final_test został już oceniony: {existing}.")

    df = add_request_groups(get_processed_data())
    development, final_test = get_development_and_final_test_indices(df)
    metrics, predictions = run_models(
        df, development, final_test, split_number=0, final_test=True
    )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(FINAL_METRICS_FILE, index=False)
    predictions.to_csv(
        FINAL_PREDICTIONS_FILE, index=False, compression="gzip"
    )

    print("\nKOŃCOWA EWALUACJA FINAL_TEST")
    print_metrics(metrics)
    print(f"\nZapisano: {FINAL_METRICS_FILE}")
    print(f"Zapisano: {FINAL_PREDICTIONS_FILE}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true")
    group.add_argument("--run-final-test", action="store_true")
    arguments = parser.parse_args()

    if arguments.smoke_test:
        run_smoke_test()
    else:
        run_final_test()


if __name__ == "__main__":
    main()