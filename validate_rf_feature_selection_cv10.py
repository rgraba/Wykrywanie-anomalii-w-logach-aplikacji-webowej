import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import config

from data_processor import get_processed_data

from compare_rf_selekcja_cech import (
    evaluate_feature_set,
    select_with_information_gain,
    select_with_l1,
    select_with_random_forest,
)


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
    print("Wczytywanie danych...")

    df = get_processed_data()

    y = df["classification"]

    splitter = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    all_metrics = []
    all_scores = []
    l1_parameters = []

    for fold_number, (
        train_positions,
        validation_positions,
    ) in enumerate(
        splitter.split(df, y),
        start=1,
    ):
        print("\n" + "#" * 75)
        print(f"FOLD {fold_number}/10")
        print("#" * 75)

        train_indices = df.index[
            train_positions
        ]

        validation_indices = df.index[
            validation_positions
        ]

        X_train_all = df.loc[
            train_indices,
            config.ML_FEATURES_ALTHUBITI_9,
        ]

        y_train = df.loc[
            train_indices,
            "classification",
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
            X_train_all,
            y_train,
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
                test_indices=validation_indices,
            )

            metrics["fold"] = fold_number

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
        / "rf_feature_selection_cv10_folds.csv"
    )

    summary_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_cv10_summary.csv"
    )

    scores_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_cv10_scores.csv"
    )

    stability_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_cv10_stability.csv"
    )

    l1_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_cv10_l1_parameters.csv"
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