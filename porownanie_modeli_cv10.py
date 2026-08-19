import pandas as pd
import numpy as np
import config
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_curve,
    auc,
)

RF_PREDICTIONS_FILE = (
    config.REPORTS_DIR
    / "rf_random_vs_group_cv10_oof_predictions.csv"
)

ONE_CLASS_PREDICTIONS_FILE = (
    config.REPORTS_DIR
    / "one_class_group_cv10_oof_predictions.csv.gz"
)

KEY_COLUMNS = [
    "row_id",
    "request_group",
    "classification",
    "fold",
]

COMPARISON_SCENARIOS = {
    "supervised_calibrated": {
        "one_class_scenario": (
            "supervised_calibrated_one_class"
        ),
        "configuration": "DEFAULT",
    },
    "strict": {
        "one_class_scenario": "strict_one_class",
        "configuration": "FIXED_STRICT",
    },
}

MODEL_COLUMNS = {
    "RANDOM_FOREST": {
        "prediction": "random_forest_y_pred",
        "score": "random_forest_y_score",
    },
    "ISOLATION_FOREST": {
        "prediction": "isolation_forest_y_pred",
        "score": "isolation_forest_y_score",
    },
    "ONE_CLASS_SVM": {
        "prediction": "one_class_svm_y_pred",
        "score": "one_class_svm_y_score",
    },
}

TARGET_FPRS = (0.01, 0.05, 0.10)

METRIC_NAMES = [
    "balanced_accuracy",
    "f1_anomaly",
    "false_positive_rate",
    "roc_auc",
    "average_precision",
    "tpr_at_fpr_1_percent",
    "tpr_at_fpr_5_percent",
    "tpr_at_fpr_10_percent",
]

BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95

MODEL_OUTPUT_NAMES = {
    "RANDOM_FOREST": "random_forest",
    "ISOLATION_FOREST": "isolation_forest",
    "ONE_CLASS_SVM": "one_class_svm",
}

MODEL_PAIRS = [
    (
        "RANDOM_FOREST",
        "ISOLATION_FOREST",
    ),
    (
        "RANDOM_FOREST",
        "ONE_CLASS_SVM",
    ),
    (
        "ISOLATION_FOREST",
        "ONE_CLASS_SVM",
    ),
]


def load_predictions() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rf_predictions = pd.read_csv(
        RF_PREDICTIONS_FILE
    )

    one_class_predictions = pd.read_csv(
        ONE_CLASS_PREDICTIONS_FILE
    )

    return rf_predictions, one_class_predictions


def prepare_random_forest_predictions(
    rf_predictions: pd.DataFrame,
) -> pd.DataFrame:
    group_predictions = rf_predictions.loc[
        rf_predictions["split"] == "group"
    ].copy()

    if group_predictions["row_id"].duplicated().any():
        raise RuntimeError(
            "Random Forest zawiera powtórzone "
            "predykcje dla tych samych rekordów."
        )

    if group_predictions[
        "group_seen_in_training"
    ].any():
        raise RuntimeError(
            "Random Forest zawiera grupy widziane "
            "podczas treningu."
        )

    if group_predictions[
        "final_test_used"
    ].any():
        raise RuntimeError(
            "Predykcje Random Forest używają final_test."
        )

    if not (
        group_predictions["protocol_seed"]
        == config.PROTOCOL_RANDOM_STATE
    ).all():
        raise RuntimeError(
            "Random Forest używa innego seedu protokołu."
        )

    return group_predictions[
        KEY_COLUMNS + ["y_pred", "y_score"]
    ].rename(columns={
        "y_pred": "random_forest_y_pred",
        "y_score": "random_forest_y_score",
    })


def select_one_class_predictions(
    one_class_predictions: pd.DataFrame,
    scenario: str,
    configuration: str,
    model_name: str,
    output_name: str,
) -> pd.DataFrame:
    selected = one_class_predictions.loc[
        (
            one_class_predictions["scenario"]
            == scenario
        )
        & (
            one_class_predictions["model"]
            == model_name
        )
        & (
            one_class_predictions["feature_set"]
            == "BASIC"
        )
        & (
            one_class_predictions["configuration"]
            == configuration
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"Nie znaleziono predykcji: "
            f"{scenario}, {model_name}, BASIC, "
            f"{configuration}."
        )

    if selected["row_id"].duplicated().any():
        raise RuntimeError(
            f"{model_name} zawiera powtórzone "
            "predykcje OOF."
        )

    if not (
        selected["group_overlap"] == 0
    ).all():
        raise RuntimeError(
            f"{model_name} zawiera wspólne grupy."
        )

    if selected["final_test_used"].any():
        raise RuntimeError(
            f"{model_name} używa final_test."
        )

    if not (
        selected["protocol_seed"]
        == config.PROTOCOL_RANDOM_STATE
    ).all():
        raise RuntimeError(
            f"{model_name} używa innego seedu."
        )

    return selected[
        KEY_COLUMNS + ["y_pred", "y_score"]
    ].rename(columns={
        "y_pred": f"{output_name}_y_pred",
        "y_score": f"{output_name}_y_score",
    })


def prepare_scenario_predictions(
    rf_predictions: pd.DataFrame,
    one_class_predictions: pd.DataFrame,
    scenario_name: str,
    scenario_config: dict,
) -> pd.DataFrame:
    combined = prepare_random_forest_predictions(
        rf_predictions
    )

    isolation_forest = select_one_class_predictions(
        one_class_predictions=one_class_predictions,
        scenario=(
            scenario_config["one_class_scenario"]
        ),
        configuration=(
            scenario_config["configuration"]
        ),
        model_name="ISOLATION_FOREST",
        output_name="isolation_forest",
    )

    one_class_svm = select_one_class_predictions(
        one_class_predictions=one_class_predictions,
        scenario=(
            scenario_config["one_class_scenario"]
        ),
        configuration=(
            scenario_config["configuration"]
        ),
        model_name="ONE_CLASS_SVM",
        output_name="one_class_svm",
    )

    combined = combined.merge(
        isolation_forest,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    combined = combined.merge(
        one_class_svm,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    prediction_columns = [
        "random_forest_y_pred",
        "random_forest_y_score",
        "isolation_forest_y_pred",
        "isolation_forest_y_score",
        "one_class_svm_y_pred",
        "one_class_svm_y_score",
    ]

    if combined[prediction_columns].isna().any().any():
        raise RuntimeError(
            f"Nie udało się sparować wszystkich "
            f"predykcji dla scenariusza {scenario_name}."
        )

    if combined["request_group"].nunique() != 20690:
        raise RuntimeError(
            f"Scenariusz {scenario_name} ma "
            "nieprawidłową liczbę grup."
        )

    groups_in_multiple_folds = (
        combined
        .groupby("request_group")["fold"]
        .nunique()
        .gt(1)
        .sum()
    )

    if groups_in_multiple_folds > 0:
        raise RuntimeError(
            f"Scenariusz {scenario_name} zawiera "
            "grupy przypisane do kilku foldów."
        )

    return combined


def calculate_tpr_at_fpr(
    fpr_values: np.ndarray,
    tpr_values: np.ndarray,
    target_fpr: float,
) -> float:
    allowed_points = (
        fpr_values <= target_fpr
    )

    if not np.any(allowed_points):
        return 0.0

    return float(
        np.max(tpr_values[allowed_points])
    )


def calculate_model_metrics(
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

    true_positive_rate = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    true_negative_rate = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    f1_anomaly = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    fpr_values, tpr_values, _ = roc_curve(
        y_true,
        y_score,
        sample_weight=sample_weight,
    )

    metrics = {
        "balanced_accuracy": (
            true_positive_rate
            + true_negative_rate
        ) / 2,
        "f1_anomaly": f1_anomaly,
        "false_positive_rate": (
            false_positive_rate
        ),
        "roc_auc": auc(
            fpr_values,
            tpr_values,
        ),
        "average_precision": (
            average_precision_score(
                y_true,
                y_score,
                sample_weight=sample_weight,
            )
        ),
    }

    for target_fpr in TARGET_FPRS:
        label = (
            f"{int(target_fpr * 100)}_percent"
        )

        metrics[f"tpr_at_fpr_{label}"] = (
            calculate_tpr_at_fpr(
                fpr_values,
                tpr_values,
                target_fpr,
            )
        )

    return metrics


def calculate_point_metrics(
    predictions: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []

    for model_name, columns in MODEL_COLUMNS.items():
        fold_rows = []

        for fold_number, fold_predictions in (
            predictions.groupby("fold", sort=True)
        ):
            metrics = calculate_model_metrics(
                y_true=fold_predictions[
                    "classification"
                ],
                y_pred=fold_predictions[
                    columns["prediction"]
                ],
                y_score=fold_predictions[
                    columns["score"]
                ],
            )

            fold_rows.append({
                "fold": int(fold_number),
                **metrics,
            })

        fold_metrics_df = pd.DataFrame(
            fold_rows
        )

        if (
            scenario_name == "strict"
            and model_name != "RANDOM_FOREST"
        ):
            prediction_rule = (
                "normal_only_calibration_"
                "target_fpr_5_percent"
            )
        else:
            prediction_rule = "model_default"

        summary_row = {
            "scenario": scenario_name,
            "model": model_name,
            "feature_set": "BASIC",
            "prediction_rule": prediction_rule,
            "aggregation": "mean_of_10_folds",
            "records": len(predictions),
            "groups": (
                predictions[
                    "request_group"
                ].nunique()
            ),
            "final_test_used": False,
        }

        for metric_name in METRIC_NAMES:
            summary_row[
                f"{metric_name}_mean"
            ] = fold_metrics_df[
                metric_name
            ].mean()

            summary_row[
                f"{metric_name}_std"
            ] = fold_metrics_df[
                metric_name
            ].std(ddof=1)

        rows.append(summary_row)

    return pd.DataFrame(rows)


def run_group_bootstrap(
    predictions: pd.DataFrame,
    scenario_name: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    group_codes, unique_groups = pd.factorize(
        predictions["request_group"],
        sort=False,
    )

    number_of_groups = len(unique_groups)

    y_true = predictions[
        "classification"
    ].to_numpy()

    fold_numbers = predictions[
        "fold"
    ].to_numpy()

    fold_masks = {
        fold_number: (
            fold_numbers == fold_number
        )
        for fold_number in sorted(
            predictions["fold"].unique()
        )
    }

    model_arrays = {}

    for model_name, columns in MODEL_COLUMNS.items():
        model_arrays[model_name] = {
            "prediction": predictions[
                columns["prediction"]
            ].to_numpy(),
            "score": predictions[
                columns["score"]
            ].to_numpy(),
        }

    random_generator = np.random.default_rng(
        config.PROTOCOL_RANDOM_STATE
    )

    bootstrap_rows = []
    progress_interval = max(1, iterations // 10)

    print(
        f"\nBootstrap: {scenario_name}, "
        f"{number_of_groups} grup, "
        f"{iterations} iteracji."
    )

    for iteration in range(1, iterations + 1):
        sampled_group_codes = (
            random_generator.integers(
                low=0,
                high=number_of_groups,
                size=number_of_groups,
            )
        )

        sampled_group_counts = np.bincount(
            sampled_group_codes,
            minlength=number_of_groups,
        )

        sample_weight = sampled_group_counts[
            group_codes
        ]

        fold_metrics = {
            model_name: {
                metric_name: []
                for metric_name in METRIC_NAMES
            }
            for model_name in MODEL_COLUMNS
        }

        for fold_mask in fold_masks.values():
            fold_weight = sample_weight[
                fold_mask
            ]

            for model_name, arrays in (
                model_arrays.items()
            ):
                metrics = calculate_model_metrics(
                    y_true=y_true[fold_mask],
                    y_pred=arrays[
                        "prediction"
                    ][fold_mask],
                    y_score=arrays[
                        "score"
                    ][fold_mask],
                    sample_weight=fold_weight,
                )

                for metric_name in METRIC_NAMES:
                    fold_metrics[
                        model_name
                    ][metric_name].append(
                        metrics[metric_name]
                    )

        mean_metrics = {
            model_name: {
                metric_name: float(
                    np.mean(metric_values)
                )
                for metric_name, metric_values
                in model_metrics.items()
            }
            for model_name, model_metrics
            in fold_metrics.items()
        }

        result_row = {
            "scenario": scenario_name,
            "iteration": iteration,
            "sampled_groups": number_of_groups,
            "bootstrap_seed": (
                config.PROTOCOL_RANDOM_STATE
            ),
        }

        for model_name, model_metrics in (
            mean_metrics.items()
        ):
            model_output_name = (
                MODEL_OUTPUT_NAMES[model_name]
            )

            for metric_name, metric_value in (
                model_metrics.items()
            ):
                result_row[
                    f"{model_output_name}_"
                    f"{metric_name}"
                ] = metric_value

        for first_model, second_model in MODEL_PAIRS:
            first_output_name = (
                MODEL_OUTPUT_NAMES[first_model]
            )

            second_output_name = (
                MODEL_OUTPUT_NAMES[second_model]
            )

            for metric_name in METRIC_NAMES:
                difference = (
                    mean_metrics[
                        first_model
                    ][metric_name]
                    - mean_metrics[
                        second_model
                    ][metric_name]
                )

                result_row[
                    f"difference_"
                    f"{first_output_name}_minus_"
                    f"{second_output_name}_"
                    f"{metric_name}"
                ] = difference

        bootstrap_rows.append(result_row)

        if iteration % progress_interval == 0:
            print(
                f"Bootstrap {scenario_name}: "
                f"{iteration}/{iterations}"
            )

    return pd.DataFrame(bootstrap_rows)


def calculate_confidence_interval(
    values,
) -> tuple[float, float]:
    alpha = (
        1.0 - CONFIDENCE_LEVEL
    ) / 2.0

    lower = np.quantile(
        values,
        alpha,
    )

    upper = np.quantile(
        values,
        1.0 - alpha,
    )

    return float(lower), float(upper)


def create_model_bootstrap_summary(
    point_metrics_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for scenario_name in (
        point_metrics_df["scenario"].unique()
    ):
        scenario_bootstrap = bootstrap_df.loc[
            bootstrap_df["scenario"]
            == scenario_name
        ]

        for model_name in MODEL_COLUMNS:
            point_row = point_metrics_df.loc[
                (
                    point_metrics_df["scenario"]
                    == scenario_name
                )
                & (
                    point_metrics_df["model"]
                    == model_name
                )
            ].iloc[0]

            model_output_name = (
                MODEL_OUTPUT_NAMES[model_name]
            )

            for metric_name in METRIC_NAMES:
                bootstrap_column = (
                    f"{model_output_name}_"
                    f"{metric_name}"
                )

                lower, upper = (
                    calculate_confidence_interval(
                        scenario_bootstrap[
                            bootstrap_column
                        ]
                    )
                )

                summary_rows.append({
                    "scenario": scenario_name,
                    "model": model_name,
                    "feature_set": "BASIC",
                    "prediction_rule": (
                        point_row[
                            "prediction_rule"
                        ]
                    ),
                    "metric": metric_name,
                    "value": point_row[
                        f"{metric_name}_mean"
                    ],
                    "fold_std": point_row[
                        f"{metric_name}_std"
                    ],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": (
                        CONFIDENCE_LEVEL
                    ),
                    "bootstrap_iterations": len(
                        scenario_bootstrap
                    ),
                    "bootstrap_unit": "request_group",
                    "groups": int(
                        point_row["groups"]
                    ),
                    "aggregation": (
                        "mean_of_10_fold_metrics"
                    ),
                    "final_test_used": False,
                })

    return pd.DataFrame(summary_rows)


def determine_favored_model(
    first_model: str,
    second_model: str,
    metric_name: str,
    difference: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    ci_excludes_zero = (
        ci_lower > 0.0
        or ci_upper < 0.0
    )

    if not ci_excludes_zero:
        return "NO_CLEAR_DIFFERENCE"

    if metric_name == "false_positive_rate":
        if difference < 0.0:
            return first_model

        return second_model

    if difference > 0.0:
        return first_model

    return second_model


def create_pairwise_bootstrap_summary(
    point_metrics_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for scenario_name in (
        point_metrics_df["scenario"].unique()
    ):
        scenario_bootstrap = bootstrap_df.loc[
            bootstrap_df["scenario"]
            == scenario_name
        ]

        for first_model, second_model in MODEL_PAIRS:
            first_point = point_metrics_df.loc[
                (
                    point_metrics_df["scenario"]
                    == scenario_name
                )
                & (
                    point_metrics_df["model"]
                    == first_model
                )
            ].iloc[0]

            second_point = point_metrics_df.loc[
                (
                    point_metrics_df["scenario"]
                    == scenario_name
                )
                & (
                    point_metrics_df["model"]
                    == second_model
                )
            ].iloc[0]

            first_output_name = (
                MODEL_OUTPUT_NAMES[first_model]
            )

            second_output_name = (
                MODEL_OUTPUT_NAMES[second_model]
            )

            for metric_name in METRIC_NAMES:
                difference_column = (
                    f"difference_"
                    f"{first_output_name}_minus_"
                    f"{second_output_name}_"
                    f"{metric_name}"
                )

                lower, upper = (
                    calculate_confidence_interval(
                        scenario_bootstrap[
                            difference_column
                        ]
                    )
                )

                difference = (
                    first_point[
                        f"{metric_name}_mean"
                    ]
                    - second_point[
                        f"{metric_name}_mean"
                    ]
                )

                ci_excludes_zero = (
                    lower > 0.0
                    or upper < 0.0
                )

                if metric_name in {
                    "roc_auc",
                    "average_precision",
                    "tpr_at_fpr_1_percent",
                    "tpr_at_fpr_5_percent",
                    "tpr_at_fpr_10_percent",
                }:
                    comparison_scope = (
                        "score_based_same_target"
                    )
                else:
                    comparison_scope = (
                        "decision_rule_specific"
                    )

                favored_model = determine_favored_model(
                    first_model=first_model,
                    second_model=second_model,
                    metric_name=metric_name,
                    difference=difference,
                    ci_lower=lower,
                    ci_upper=upper,
                )

                summary_rows.append({
                    "scenario": scenario_name,
                    "comparison": (
                        f"{first_model}_minus_"
                        f"{second_model}"
                    ),
                    "first_model": first_model,
                    "second_model": second_model,
                    "first_prediction_rule": (
                        first_point[
                            "prediction_rule"
                        ]
                    ),
                    "second_prediction_rule": (
                        second_point[
                            "prediction_rule"
                        ]
                    ),
                    "comparison_scope": (
                        comparison_scope
                    ),
                    "metric": metric_name,
                    "difference": difference,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "difference_ci_excludes_zero": (
                        ci_excludes_zero
                    ),
                    "favored_model": favored_model,
                    "confidence_level": (
                        CONFIDENCE_LEVEL
                    ),
                    "bootstrap_iterations": len(
                        scenario_bootstrap
                    ),
                    "bootstrap_unit": "request_group",
                    "aggregation": (
                        "mean_of_10_fold_metrics"
                    ),
                    "final_test_used": False,
                })

    return pd.DataFrame(summary_rows)


def main() -> None:
    rf_predictions, one_class_predictions = (
        load_predictions()
    )

    point_metrics = []
    bootstrap_results = []

    print("SPRAWDZENIE DANYCH DO PORÓWNANIA MODELI")
    print("Zbiór final_test nie jest używany.")

    for scenario_name, scenario_config in (
        COMPARISON_SCENARIOS.items()
    ):
        predictions = prepare_scenario_predictions(
            rf_predictions=rf_predictions,
            one_class_predictions=(
                one_class_predictions
            ),
            scenario_name=scenario_name,
            scenario_config=scenario_config,
        )

        print("\n" + "=" * 70)
        print(f"Scenariusz: {scenario_name}")
        print(f"Rekordy: {len(predictions)}")
        print(
            "Unikalne rekordy: "
            f"{predictions['row_id'].nunique()}"
        )
        print(
            "Unikalne grupy: "
            f"{predictions['request_group'].nunique()}"
        )
        print(
            "Foldy: "
            f"{sorted(
                int(fold)
                for fold in predictions['fold'].unique()
            )}"
        )

        scenario_metrics = calculate_point_metrics(
            predictions=predictions,
            scenario_name=scenario_name,
        )

        point_metrics.append(
            scenario_metrics
        )

        scenario_bootstrap = run_group_bootstrap(
            predictions=predictions,
            scenario_name=scenario_name,
            iterations=BOOTSTRAP_ITERATIONS,
        )

        bootstrap_results.append(
            scenario_bootstrap
        )

    point_metrics_df = pd.concat(
        point_metrics,
        ignore_index=True,
    )

    bootstrap_df = pd.concat(
        bootstrap_results,
        ignore_index=True,
    )

    model_summary_df = (
        create_model_bootstrap_summary(
            point_metrics_df=point_metrics_df,
            bootstrap_df=bootstrap_df,
        )
    )

    pairwise_summary_df = (
        create_pairwise_bootstrap_summary(
            point_metrics_df=point_metrics_df,
            bootstrap_df=bootstrap_df,
        )
    )

    columns_to_display = [
        "scenario",
        "model",
        "prediction_rule",
    ]

    for metric_name in METRIC_NAMES:
        columns_to_display.extend([
            f"{metric_name}_mean",
            f"{metric_name}_std",
        ])

    print("\n" + "=" * 70)
    print(
        "METRYKI — ŚREDNIA I ODCHYLENIE "
        "Z 10 FOLDÓW"
    )
    print("=" * 70)

    print(
        point_metrics_df[
            columns_to_display
        ].round(4).to_string(index=False)
    )

    pairwise_columns = [
        "scenario",
        "comparison",
        "comparison_scope",
        "metric",
        "difference",
        "ci_lower",
        "ci_upper",
        "difference_ci_excludes_zero",
        "favored_model",
    ]

    print("\n" + "=" * 70)
    print("SPAROWANE RÓŻNICE — 95% CI")
    print("=" * 70)

    print(
        pairwise_summary_df[
            pairwise_columns
        ].round(4).to_string(index=False)
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap_path = (
        config.REPORTS_DIR
        / (
            "model_comparison_group_cv10_"
            "bootstrap_samples.csv.gz"
        )
    )

    model_summary_path = (
        config.REPORTS_DIR
        / (
            "model_comparison_group_cv10_"
            "bootstrap_model_summary.csv"
        )
    )

    pairwise_summary_path = (
        config.REPORTS_DIR
        / (
            "model_comparison_group_cv10_"
            "bootstrap_pairwise_summary.csv"
        )
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
        compression="gzip",
    )

    model_summary_df.to_csv(
        model_summary_path,
        index=False,
    )

    pairwise_summary_df.to_csv(
        pairwise_summary_path,
        index=False,
    )

    print("\nZapisane pliki:")
    print(f"- {bootstrap_path}")
    print(f"- {model_summary_path}")
    print(f"- {pairwise_summary_path}")

    print(
        "\nZbiór final_test nie został użyty."
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()