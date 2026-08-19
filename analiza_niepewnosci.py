import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import config


BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95

METRIC_NAMES = [
    "balanced_accuracy",
    "f1_anomaly",
    "false_positive_rate",
    "roc_auc",
    "average_precision",
]


def calculate_metrics(
    y_true,
    y_pred,
    y_score,
    sample_weight=None,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        sample_weight=sample_weight,
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    return {
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        ),
        "f1_anomaly": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
            sample_weight=sample_weight,
        ),
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(
            y_true,
            y_score,
            sample_weight=sample_weight,
        ),
        "average_precision": average_precision_score(
            y_true,
            y_score,
            sample_weight=sample_weight,
        ),
    }


def calculate_mean_fold_metrics(
    paired_predictions: pd.DataFrame,
    split_name: str,
    sample_weight=None,
) -> dict:
    split_columns = {
        "random_record": {
            "fold": "random_fold",
            "y_pred": "random_y_pred",
            "y_score": "random_y_score",
        },
        "group": {
            "fold": "group_fold",
            "y_pred": "group_y_pred",
            "y_score": "group_y_score",
        },
    }

    if split_name not in split_columns:
        raise ValueError(
            f"Nieznany wariant podziału: {split_name}."
        )

    columns = split_columns[split_name]

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

        if len(sample_weight) != len(paired_predictions):
            raise ValueError(
                "Liczba wag nie odpowiada liczbie predykcji."
            )

    fold_metrics = []

    for fold_number in range(1, 11):
        fold_mask = (
            paired_predictions[columns["fold"]].to_numpy()
            == fold_number
        )

        if not fold_mask.any():
            raise ValueError(
                f"Brak danych dla foldu {fold_number} "
                f"w podziale {split_name}."
            )

        fold_sample_weight = (
            sample_weight[fold_mask]
            if sample_weight is not None
            else None
        )

        if (
            fold_sample_weight is not None
            and fold_sample_weight.sum() == 0
        ):
            raise ValueError(
                f"Fold {fold_number} nie zawiera żadnej "
                "wylosowanej grupy."
            )

        metrics = calculate_metrics(
            y_true=paired_predictions.loc[
                fold_mask,
                "classification",
            ].to_numpy(),
            y_pred=paired_predictions.loc[
                fold_mask,
                columns["y_pred"],
            ].to_numpy(),
            y_score=paired_predictions.loc[
                fold_mask,
                columns["y_score"],
            ].to_numpy(),
            sample_weight=fold_sample_weight,
        )

        fold_metrics.append(metrics)

    return {
        metric_name: float(
            np.mean(
                [
                    metrics[metric_name]
                    for metrics in fold_metrics
                ]
            )
        )
        for metric_name in METRIC_NAMES
    }


def prepare_paired_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "row_id",
        "request_group",
        "classification",
        "y_pred",
        "y_score",
        "split",
        "fold",
        "protocol_seed",
        "final_test_used",
    }

    missing_columns = required_columns.difference(
        predictions_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Plik predykcji nie zawiera kolumn: "
            f"{sorted(missing_columns)}."
        )

    actual_splits = set(predictions_df["split"])

    if actual_splits != {"random_record", "group"}:
        raise ValueError(
            f"Nieprawidłowe warianty podziału: {actual_splits}."
        )

    final_test_used = (
        predictions_df["final_test_used"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
        .any()
    )

    if final_test_used:
        raise ValueError(
            "Predykcje nie mogą pochodzić ze zbioru final_test."
        )

    protocol_seeds = set(
        pd.to_numeric(
            predictions_df["protocol_seed"],
            errors="raise",
        ).astype(int)
    )

    if protocol_seeds != {config.PROTOCOL_RANDOM_STATE}:
        raise ValueError(
            "Nieprawidłowy seed protokołu: "
            f"{protocol_seeds}."
        )

    expected_folds = set(range(1, 11))

    for split_name in ["random_record", "group"]:
        split_predictions = predictions_df.loc[
            predictions_df["split"] == split_name
        ]

        actual_folds = set(
            pd.to_numeric(
                split_predictions["fold"],
                errors="raise",
            ).astype(int)
        )

        if actual_folds != expected_folds:
            raise ValueError(
                f"Podział {split_name} zawiera foldy: "
                f"{sorted(actual_folds)}."
            )

    random_predictions = predictions_df.loc[
        predictions_df["split"] == "random_record",
        [
            "row_id",
            "request_group",
            "classification",
            "fold",
            "y_pred",
            "y_score",
        ],
    ].rename(
        columns={
            "fold": "random_fold",
            "y_pred": "random_y_pred",
            "y_score": "random_y_score",
        }
    )

    group_predictions = predictions_df.loc[
        predictions_df["split"] == "group",
        [
            "row_id",
            "request_group",
            "classification",
            "fold",
            "y_pred",
            "y_score",
        ],
    ].rename(
        columns={
            "fold": "group_fold",
            "y_pred": "group_y_pred",
            "y_score": "group_y_score",
        }
    )

    if not random_predictions["row_id"].is_unique:
        raise ValueError(
            "Predykcje random_record zawierają powtórzone row_id."
        )

    if not group_predictions["row_id"].is_unique:
        raise ValueError(
            "Predykcje group zawierają powtórzone row_id."
        )

    paired_predictions = random_predictions.merge(
        group_predictions,
        on=[
            "row_id",
            "request_group",
            "classification",
        ],
        how="inner",
        validate="one_to_one",
    )

    if (
        len(paired_predictions) != len(random_predictions)
        or len(paired_predictions) != len(group_predictions)
    ):
        raise ValueError(
            "Nie udało się sparować wszystkich predykcji "
            "random_record i group."
        )

    return (
        paired_predictions
        .sort_values("row_id")
        .reset_index(drop=True)
    )


def run_group_bootstrap(
    paired_predictions: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    group_codes, unique_groups = pd.factorize(
        paired_predictions["request_group"],
        sort=False,
    )

    if (group_codes < 0).any():
        raise ValueError(
            "Co najmniej jeden rekord nie ma identyfikatora grupy."
        )

    number_of_groups = len(unique_groups)

    random_generator = np.random.default_rng(
        config.PROTOCOL_RANDOM_STATE
    )

    bootstrap_rows = []

    for iteration in range(1, iterations + 1):
        sampled_group_codes = random_generator.integers(
            low=0,
            high=number_of_groups,
            size=number_of_groups,
        )

        group_counts = np.bincount(
            sampled_group_codes,
            minlength=number_of_groups,
        )

        sample_weight = group_counts[group_codes]

        random_metrics = calculate_mean_fold_metrics(
            paired_predictions=paired_predictions,
            split_name="random_record",
            sample_weight=sample_weight,
        )

        group_metrics = calculate_mean_fold_metrics(
            paired_predictions=paired_predictions,
            split_name="group",
            sample_weight=sample_weight,
        )

        row = {
            "iteration": iteration,
            "sampled_groups": number_of_groups,
            "bootstrap_seed": config.PROTOCOL_RANDOM_STATE,
        }

        for metric_name in METRIC_NAMES:
            random_value = random_metrics[metric_name]
            group_value = group_metrics[metric_name]

            row[f"random_record_{metric_name}"] = random_value
            row[f"group_{metric_name}"] = group_value
            row[
                f"difference_random_minus_group_{metric_name}"
            ] = random_value - group_value

        bootstrap_rows.append(row)

        if iteration % 200 == 0:
            print(
                f"Bootstrap: {iteration}/{iterations}"
            )

    return pd.DataFrame(bootstrap_rows)


def create_bootstrap_summary(
    paired_predictions: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    random_metrics = calculate_mean_fold_metrics(
        paired_predictions=paired_predictions,
        split_name="random_record",
    )

    group_metrics = calculate_mean_fold_metrics(
        paired_predictions=paired_predictions,
        split_name="group",
    )

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    summary_rows = []

    for metric_name in METRIC_NAMES:
        random_values = bootstrap_df[
            f"random_record_{metric_name}"
        ]

        group_values = bootstrap_df[
            f"group_{metric_name}"
        ]

        difference_values = bootstrap_df[
            f"difference_random_minus_group_{metric_name}"
        ]

        difference_lower = float(
            np.percentile(
                difference_values,
                lower_percentile,
            )
        )

        difference_upper = float(
            np.percentile(
                difference_values,
                upper_percentile,
            )
        )

        summary_rows.append(
            {
                "metric": metric_name,
                "random_record_value": (
                    random_metrics[metric_name]
                ),
                "random_record_ci_lower": float(
                    np.percentile(
                        random_values,
                        lower_percentile,
                    )
                ),
                "random_record_ci_upper": float(
                    np.percentile(
                        random_values,
                        upper_percentile,
                    )
                ),
                "group_value": group_metrics[metric_name],
                "group_ci_lower": float(
                    np.percentile(
                        group_values,
                        lower_percentile,
                    )
                ),
                "group_ci_upper": float(
                    np.percentile(
                        group_values,
                        upper_percentile,
                    )
                ),
                "difference_random_minus_group": (
                    random_metrics[metric_name]
                    - group_metrics[metric_name]
                ),
                "difference_ci_lower": difference_lower,
                "difference_ci_upper": difference_upper,
                "difference_ci_excludes_zero": (
                    difference_lower > 0
                    or difference_upper < 0
                ),
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_iterations": len(bootstrap_df),
                "bootstrap_unit": "request_group",
                "aggregation": "mean_of_10_fold_metrics",
                "number_of_groups": (
                    paired_predictions[
                        "request_group"
                    ].nunique()
                ),
                "final_test_used": False,
            }
        )

    return pd.DataFrame(summary_rows)


def analyze_random_vs_group_uncertainty() -> None:
    predictions_path = (
        config.REPORTS_DIR
        / "rf_random_vs_group_cv10_oof_predictions.csv"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {predictions_path}"
        )

    predictions_df = pd.read_csv(predictions_path)
    paired_predictions = prepare_paired_predictions(
        predictions_df
    )

    print(
        "Bootstrap grupowy na "
        f"{paired_predictions['request_group'].nunique()} grupach."
    )
    print(
        f"Liczba iteracji: {BOOTSTRAP_ITERATIONS}"
    )
    print("Zbiór final_test nie jest używany.")

    bootstrap_df = run_group_bootstrap(
        paired_predictions
    )

    summary_df = create_bootstrap_summary(
        paired_predictions=paired_predictions,
        bootstrap_df=bootstrap_df,
    )

    samples_path = (
        config.REPORTS_DIR
        / "rf_random_vs_group_group_bootstrap_samples.csv"
    )
    summary_path = (
        config.REPORTS_DIR
        / "rf_random_vs_group_group_bootstrap_summary.csv"
    )

    bootstrap_df.to_csv(
        samples_path,
        index=False,
    )
    summary_df.to_csv(
        summary_path,
        index=False,
    )

    columns_to_display = [
        "metric",
        "random_record_value",
        "random_record_ci_lower",
        "random_record_ci_upper",
        "group_value",
        "group_ci_lower",
        "group_ci_upper",
        "difference_random_minus_group",
        "difference_ci_lower",
        "difference_ci_upper",
        "difference_ci_excludes_zero",
    ]

    print("\n95% PRZEDZIAŁY UFNOŚCI")
    print(
        summary_df[columns_to_display]
        .round(4)
        .to_string(index=False)
    )

    print("\nZapisane pliki:")
    print(f"- {samples_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    analyze_random_vs_group_uncertainty()