import numpy as np
import pandas as pd

from podzial_danych import (
    add_request_groups,
    ensure_no_group_overlap,
)
from protokol_eksperymentalny import get_development_folds

import config
from metryki import calculate_binary_metrics
from przetwarzanie_danych import get_processed_data
from time import perf_counter
from sklearn.ensemble import RandomForestClassifier
from selekcja_cech import (
    select_with_information_gain,
    select_with_l1,
    select_with_random_forest,
)


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    train_indices: pd.Index,
    validation_indices: pd.Index,
) -> dict:
    X_train = df.loc[train_indices, features]
    y_train = df.loc[train_indices, "classification"]

    X_validation = df.loc[validation_indices, features]
    y_validation = df.loc[validation_indices, "classification"]

    model = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    training_start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - training_start

    prediction_start = perf_counter()
    y_pred = model.predict(X_validation)
    y_score = model.predict_proba(X_validation)[:, 1]
    prediction_time = perf_counter() - prediction_start

    metrics = calculate_binary_metrics(
        y_true=y_validation,
        y_pred=y_pred,
        y_score=y_score,
    )

    metrics.update(
        {
            "feature_set": feature_set_name,
            "number_of_features": len(features),
            "selected_features": ", ".join(features),
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
        }
    )

    return metrics


def create_metrics_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_anomaly",
        "recall_anomaly",
        "f1_anomaly",
        "specificity",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "training_time_seconds",
        "prediction_time_seconds",
    ]

    summary_df = (
        metrics_df
        .groupby("feature_set")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "feature_set"
        if column[0] == "feature_set"
        else f"{column[0]}_{column[1]}"
        for column in summary_df.columns
    ]

    return summary_df


def create_selection_stability(
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    stability_df = (
        scores_df
        .groupby(
            [
                "selection_method",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            selection_count=(
                "selected",
                "sum",
            ),
            selection_rate=(
                "selected",
                "mean",
            ),
            mean_score=(
                "score",
                "mean",
            ),
            std_score=(
                "score",
                "std",
            ),
        )
    )

    return stability_df


def validate_with_cv10() -> None:
    df = get_processed_data()
    grouped_df = add_request_groups(df)
    folds = get_development_folds(df)

    all_metrics = []
    all_scores = []
    l1_parameters = []

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

        X_train_all = df.loc[
            train_indices,
            config.ML_FEATURES_ALTHUBITI_9,
        ]

        y_train = df.loc[
            train_indices,
            "classification",
        ]

        X_train_all = df.loc[
            train_indices,
            config.ML_FEATURES_ALTHUBITI_9,
        ]

        y_train = df.loc[
            train_indices,
            "classification",
        ]

        train_groups = grouped_df.loc[
            train_indices,
            "request_group",
        ]

        print("Selekcja Information Gain...")

        ig_features, ig_scores = (
            select_with_information_gain(
                X_train_all,
                y_train,
            )
        )

        print(
            f"IG wybrało: {ig_features}"
        )

        print("Selekcja L1/LASSO...")

        (
            l1_features,
            l1_scores,
            best_l1_c,
        ) = select_with_l1(
            X_train=X_train_all,
            y_train=y_train,
            groups=train_groups,
        )

        print(
            f"L1 wybrało: {l1_features}"
        )

        print(
            f"Najlepsze C: {best_l1_c}"
        )

        print("Selekcja RF importance...")

        rf_features, rf_scores = (
            select_with_random_forest(
                X_train_all,
                y_train,
            )
        )

        print(
            f"RF wybrało: {rf_features}"
        )

        for score_row in (
            ig_scores
            + l1_scores
            + rf_scores
        ):
            score_row["fold"] = fold_number
            all_scores.append(score_row)

        l1_parameters.append(
            {
                "fold": fold_number,
                "best_c": best_l1_c,
                "number_of_selected_features": (
                    len(l1_features)
                ),
                "selected_features": ", ".join(
                    l1_features
                ),
            }
        )

        feature_sets = {
            "BASIC": (
                config.ML_FEATURES_BASIC
            ),
            "ALTHUBITI_9_NO_SELECTION": (
                config.ML_FEATURES_ALTHUBITI_9
            ),
            "ALTHUBITI_5_LITERATURE": (
                config.ML_FEATURES_ALTHUBITI_5
            ),
            "INFORMATION_GAIN": ig_features,
            "L1_LASSO": l1_features,
            "RF_IMPORTANCE": rf_features,
        }

        for feature_set_name, features in (
            feature_sets.items()
        ):
            metrics = evaluate_feature_set(
                df=df,
                feature_set_name=feature_set_name,
                features=features,
                train_indices=train_indices,
                validation_indices=validation_indices,
            )

            metrics["fold"] = fold_number
            metrics["split"] = "development_group_cv10"
            metrics["partition"] = "development"
            metrics["group_overlap"] = group_overlap
            metrics["protocol_seed"] = config.PROTOCOL_RANDOM_STATE
            metrics["final_test_used"] = False

            if feature_set_name == "L1_LASSO":
                metrics["l1_best_c"] = (
                    best_l1_c
                )
            else:
                metrics["l1_best_c"] = np.nan

            all_metrics.append(metrics)

            print(
                f"{feature_set_name}: "
                f"F1={metrics['f1_anomaly']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}"
            )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    scores_df = pd.DataFrame(
        all_scores
    )

    l1_parameters_df = pd.DataFrame(
        l1_parameters
    )

    summary_df = create_metrics_summary(
        metrics_df
    )

    stability_df = create_selection_stability(
        scores_df
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_group_cv10_folds.csv"
    )

    summary_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_group_cv10_summary.csv"
    )

    scores_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_group_cv10_scores.csv"
    )

    stability_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_group_cv10_stability.csv"
    )

    l1_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_group_cv10_l1_parameters.csv"
    )

    metrics_df.to_csv(
        folds_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    scores_df.to_csv(
        scores_path,
        index=False,
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    l1_parameters_df.to_csv(
        l1_path,
        index=False,
    )

    columns_to_display = [
        "feature_set",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "precision_anomaly_mean",
        "recall_anomaly_mean",
        "f1_anomaly_mean",
        "f1_anomaly_std",
        "false_positive_rate_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
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

    print("\n" + "=" * 75)
    print("STABILNOŚĆ SELEKCJI CECH")
    print("=" * 75)

    print(
        stability_df.round(4).to_string(
            index=False
        )
    )

    print("\nZapisane pliki:")
    print(f"- {folds_path}")
    print(f"- {summary_path}")
    print(f"- {scores_path}")
    print(f"- {stability_path}")
    print(f"- {l1_path}")


if __name__ == "__main__":
    validate_with_cv10()