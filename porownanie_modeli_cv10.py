import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, confusion_matrix, roc_curve

import config


RF_PREDICTIONS_FILE = (
    config.REPORTS_DIR / "rf_random_vs_group_cv10_oof_predictions.csv"
)
ONE_CLASS_PREDICTIONS_FILE = (
    config.REPORTS_DIR / "one_class_group_cv10_oof_predictions.csv.gz"
)

KEY_COLUMNS = ["row_id", "request_group", "classification", "fold"]

COMPARISON_SCENARIOS = {
    "supervised_calibrated": {
        "one_class_scenario": "supervised_calibrated_one_class",
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

MODEL_OUTPUT_NAMES = {
    "RANDOM_FOREST": "random_forest",
    "ISOLATION_FOREST": "isolation_forest",
    "ONE_CLASS_SVM": "one_class_svm",
}

MODEL_PAIRS = [
    ("RANDOM_FOREST", "ISOLATION_FOREST"),
    ("RANDOM_FOREST", "ONE_CLASS_SVM"),
    ("ISOLATION_FOREST", "ONE_CLASS_SVM"),
]

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


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(RF_PREDICTIONS_FILE),
        pd.read_csv(ONE_CLASS_PREDICTIONS_FILE),
    )


def prepare_random_forest_predictions(
    rf_predictions: pd.DataFrame,
) -> pd.DataFrame:
    selected = rf_predictions.loc[rf_predictions["split"] == "group"].copy()

    if selected["row_id"].duplicated().any():
        raise RuntimeError("Random Forest zawiera powtórzone predykcje OOF.")
    if selected["group_seen_in_training"].any():
        raise RuntimeError("Random Forest zawiera grupy widziane podczas treningu.")
    if selected["final_test_used"].any():
        raise RuntimeError("Predykcje Random Forest używają final_test.")
    if not (selected["protocol_seed"] == config.PROTOCOL_RANDOM_STATE).all():
        raise RuntimeError("Random Forest używa innego seedu protokołu.")

    return selected[KEY_COLUMNS + ["y_pred", "y_score"]].rename(
        columns={
            "y_pred": "random_forest_y_pred",
            "y_score": "random_forest_y_score",
        }
    )


def select_one_class_predictions(
    one_class_predictions: pd.DataFrame,
    scenario: str,
    configuration: str,
    model_name: str,
    output_name: str,
) -> pd.DataFrame:
    selected = one_class_predictions.loc[
        (one_class_predictions["scenario"] == scenario)
        & (one_class_predictions["model"] == model_name)
        & (one_class_predictions["feature_set"] == "BASIC")
        & (one_class_predictions["configuration"] == configuration)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"Nie znaleziono predykcji: {scenario}, {model_name}, BASIC, "
            f"{configuration}."
        )
    if selected["row_id"].duplicated().any():
        raise RuntimeError(f"{model_name} zawiera powtórzone predykcje OOF.")
    if not (selected["group_overlap"] == 0).all():
        raise RuntimeError(f"{model_name} zawiera wspólne grupy.")
    if selected["final_test_used"].any():
        raise RuntimeError(f"{model_name} używa final_test.")
    if not (selected["protocol_seed"] == config.PROTOCOL_RANDOM_STATE).all():
        raise RuntimeError(f"{model_name} używa innego seedu protokołu.")

    return selected[KEY_COLUMNS + ["y_pred", "y_score"]].rename(
        columns={
            "y_pred": f"{output_name}_y_pred",
            "y_score": f"{output_name}_y_score",
        }
    )


def prepare_scenario_predictions(
    rf_predictions: pd.DataFrame,
    one_class_predictions: pd.DataFrame,
    scenario_name: str,
    scenario_config: dict,
) -> pd.DataFrame:
    combined = prepare_random_forest_predictions(rf_predictions)

    for model_name, output_name in (
        ("ISOLATION_FOREST", "isolation_forest"),
        ("ONE_CLASS_SVM", "one_class_svm"),
    ):
        selected = select_one_class_predictions(
            one_class_predictions=one_class_predictions,
            scenario=scenario_config["one_class_scenario"],
            configuration=scenario_config["configuration"],
            model_name=model_name,
            output_name=output_name,
        )
        combined = combined.merge(
            selected,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    prediction_columns = [
        column
        for model_columns in MODEL_COLUMNS.values()
        for column in model_columns.values()
    ]
    if combined[prediction_columns].isna().any().any():
        raise RuntimeError(
            f"Nie udało się sparować wszystkich predykcji dla {scenario_name}."
        )
    if combined["request_group"].nunique() != 20690:
        raise RuntimeError(f"Scenariusz {scenario_name} ma złą liczbę grup.")

    groups_in_multiple_folds = (
        combined.groupby("request_group")["fold"].nunique().gt(1).sum()
    )
    if groups_in_multiple_folds:
        raise RuntimeError(
            f"Scenariusz {scenario_name} zawiera grupy w kilku foldach."
        )

    return combined


def calculate_tpr_at_fpr(
    fpr_values: np.ndarray,
    tpr_values: np.ndarray,
    target_fpr: float,
) -> float:
    allowed_points = fpr_values <= target_fpr
    if not np.any(allowed_points):
        return 0.0
    return float(np.max(tpr_values[allowed_points]))


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

    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    tnr = tn / (tn + fp) if tn + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

    fpr_values, tpr_values, _ = roc_curve(
        y_true,
        y_score,
        sample_weight=sample_weight,
    )
    metrics = {
        "balanced_accuracy": (tpr + tnr) / 2,
        "f1_anomaly": f1,
        "false_positive_rate": fpr,
        "roc_auc": auc(fpr_values, tpr_values),
        "average_precision": average_precision_score(
            y_true,
            y_score,
            sample_weight=sample_weight,
        ),
    }

    for target_fpr in TARGET_FPRS:
        label = f"{int(target_fpr * 100)}_percent"
        metrics[f"tpr_at_fpr_{label}"] = calculate_tpr_at_fpr(
            fpr_values,
            tpr_values,
            target_fpr,
        )

    return metrics


def calculate_fold_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    sample_weight=None,
) -> list[dict]:
    columns = MODEL_COLUMNS[model_name]
    weights = None if sample_weight is None else np.asarray(sample_weight)
    fold_values = predictions["fold"].to_numpy()
    rows = []

    for fold_number in sorted(predictions["fold"].unique()):
        fold_mask = fold_values == fold_number
        fold_predictions = predictions.loc[fold_mask]
        fold_weights = None if weights is None else weights[fold_mask]

        rows.append(
            {
                "fold": int(fold_number),
                **calculate_model_metrics(
                    y_true=fold_predictions["classification"],
                    y_pred=fold_predictions[columns["prediction"]],
                    y_score=fold_predictions[columns["score"]],
                    sample_weight=fold_weights,
                ),
            }
        )

    return rows


def prediction_rule(scenario_name: str, model_name: str) -> str:
    if scenario_name == "strict" and model_name != "RANDOM_FOREST":
        return "normal_only_calibration_target_fpr_5_percent"
    return "model_default"


def calculate_point_metrics(
    predictions: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []

    for model_name in MODEL_COLUMNS:
        fold_metrics = pd.DataFrame(
            calculate_fold_metrics(predictions, model_name)
        )
        row = {
            "scenario": scenario_name,
            "model": model_name,
            "feature_set": "BASIC",
            "prediction_rule": prediction_rule(scenario_name, model_name),
            "aggregation": "mean_of_10_folds",
            "records": len(predictions),
            "groups": predictions["request_group"].nunique(),
            "final_test_used": False,
        }

        for metric_name in METRIC_NAMES:
            row[f"{metric_name}_mean"] = fold_metrics[metric_name].mean()
            row[f"{metric_name}_std"] = fold_metrics[metric_name].std(ddof=1)

        rows.append(row)

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
    random_generator = np.random.default_rng(config.PROTOCOL_RANDOM_STATE)
    progress_interval = max(1, iterations // 10)
    rows = []

    print(
        f"\nBootstrap: {scenario_name}, {number_of_groups} grup, "
        f"{iterations} iteracji."
    )

    for iteration in range(1, iterations + 1):
        sampled_codes = random_generator.integers(
            0,
            number_of_groups,
            size=number_of_groups,
        )
        group_counts = np.bincount(sampled_codes, minlength=number_of_groups)
        sample_weight = group_counts[group_codes]

        mean_metrics = {}
        for model_name in MODEL_COLUMNS:
            fold_metrics = pd.DataFrame(
                calculate_fold_metrics(predictions, model_name, sample_weight)
            )
            mean_metrics[model_name] = {
                metric: float(fold_metrics[metric].mean())
                for metric in METRIC_NAMES
            }

        row = {
            "scenario": scenario_name,
            "iteration": iteration,
            "sampled_groups": number_of_groups,
            "bootstrap_seed": config.PROTOCOL_RANDOM_STATE,
        }

        for model_name, metrics in mean_metrics.items():
            output_name = MODEL_OUTPUT_NAMES[model_name]
            for metric_name, value in metrics.items():
                row[f"{output_name}_{metric_name}"] = value

        for first_model, second_model in MODEL_PAIRS:
            first_name = MODEL_OUTPUT_NAMES[first_model]
            second_name = MODEL_OUTPUT_NAMES[second_model]
            for metric_name in METRIC_NAMES:
                column = (
                    f"difference_{first_name}_minus_{second_name}_{metric_name}"
                )
                row[column] = (
                    mean_metrics[first_model][metric_name]
                    - mean_metrics[second_model][metric_name]
                )

        rows.append(row)
        if iteration % progress_interval == 0:
            print(f"Bootstrap {scenario_name}: {iteration}/{iterations}")

    return pd.DataFrame(rows)


def calculate_confidence_interval(values) -> tuple[float, float]:
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def select_point_row(
    point_metrics_df: pd.DataFrame,
    scenario_name: str,
    model_name: str,
) -> pd.Series:
    return point_metrics_df.loc[
        (point_metrics_df["scenario"] == scenario_name)
        & (point_metrics_df["model"] == model_name)
    ].iloc[0]


def create_model_bootstrap_summary(
    point_metrics_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario_name in point_metrics_df["scenario"].unique():
        scenario_bootstrap = bootstrap_df.loc[
            bootstrap_df["scenario"] == scenario_name
        ]

        for model_name in MODEL_COLUMNS:
            point = select_point_row(point_metrics_df, scenario_name, model_name)
            output_name = MODEL_OUTPUT_NAMES[model_name]

            for metric_name in METRIC_NAMES:
                lower, upper = calculate_confidence_interval(
                    scenario_bootstrap[f"{output_name}_{metric_name}"]
                )
                rows.append(
                    {
                        "scenario": scenario_name,
                        "model": model_name,
                        "feature_set": "BASIC",
                        "prediction_rule": point["prediction_rule"],
                        "metric": metric_name,
                        "value": point[f"{metric_name}_mean"],
                        "fold_std": point[f"{metric_name}_std"],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "bootstrap_iterations": len(scenario_bootstrap),
                        "bootstrap_unit": "request_group",
                        "groups": int(point["groups"]),
                        "aggregation": "mean_of_10_fold_metrics",
                        "final_test_used": False,
                    }
                )

    return pd.DataFrame(rows)


def determine_favored_model(
    first_model: str,
    second_model: str,
    metric_name: str,
    difference: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    if not (ci_lower > 0.0 or ci_upper < 0.0):
        return "NO_CLEAR_DIFFERENCE"

    if metric_name == "false_positive_rate":
        return first_model if difference < 0.0 else second_model
    return first_model if difference > 0.0 else second_model


def comparison_scope(metric_name: str) -> str:
    score_based_metrics = {
        "roc_auc",
        "average_precision",
        "tpr_at_fpr_1_percent",
        "tpr_at_fpr_5_percent",
        "tpr_at_fpr_10_percent",
    }
    if metric_name in score_based_metrics:
        return "score_based_same_target"
    return "decision_rule_specific"


def create_pairwise_bootstrap_summary(
    point_metrics_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario_name in point_metrics_df["scenario"].unique():
        scenario_bootstrap = bootstrap_df.loc[
            bootstrap_df["scenario"] == scenario_name
        ]

        for first_model, second_model in MODEL_PAIRS:
            first_point = select_point_row(
                point_metrics_df,
                scenario_name,
                first_model,
            )
            second_point = select_point_row(
                point_metrics_df,
                scenario_name,
                second_model,
            )
            first_name = MODEL_OUTPUT_NAMES[first_model]
            second_name = MODEL_OUTPUT_NAMES[second_model]

            for metric_name in METRIC_NAMES:
                difference_column = (
                    f"difference_{first_name}_minus_{second_name}_{metric_name}"
                )
                lower, upper = calculate_confidence_interval(
                    scenario_bootstrap[difference_column]
                )
                difference = (
                    first_point[f"{metric_name}_mean"]
                    - second_point[f"{metric_name}_mean"]
                )

                rows.append(
                    {
                        "scenario": scenario_name,
                        "comparison": f"{first_model}_minus_{second_model}",
                        "first_model": first_model,
                        "second_model": second_model,
                        "first_prediction_rule": first_point["prediction_rule"],
                        "second_prediction_rule": second_point["prediction_rule"],
                        "comparison_scope": comparison_scope(metric_name),
                        "metric": metric_name,
                        "difference": difference,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "difference_ci_excludes_zero": (
                            lower > 0.0 or upper < 0.0
                        ),
                        "favored_model": determine_favored_model(
                            first_model,
                            second_model,
                            metric_name,
                            difference,
                            lower,
                            upper,
                        ),
                        "confidence_level": CONFIDENCE_LEVEL,
                        "bootstrap_iterations": len(scenario_bootstrap),
                        "bootstrap_unit": "request_group",
                        "aggregation": "mean_of_10_fold_metrics",
                        "final_test_used": False,
                    }
                )

    return pd.DataFrame(rows)


def save_results(
    bootstrap_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    pairwise_summary_df: pd.DataFrame,
) -> list:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        config.REPORTS_DIR
        / "model_comparison_group_cv10_bootstrap_samples.csv.gz",
        config.REPORTS_DIR
        / "model_comparison_group_cv10_bootstrap_model_summary.csv",
        config.REPORTS_DIR
        / "model_comparison_group_cv10_bootstrap_pairwise_summary.csv",
    ]

    bootstrap_df.to_csv(paths[0], index=False, compression="gzip")
    model_summary_df.to_csv(paths[1], index=False)
    pairwise_summary_df.to_csv(paths[2], index=False)
    return paths


def main() -> None:
    rf_predictions, one_class_predictions = load_predictions()
    point_results = []
    bootstrap_results = []

    print("SPRAWDZENIE DANYCH DO PORÓWNANIA MODELI")
    print("Zbiór final_test nie jest używany.")

    for scenario_name, scenario_config in COMPARISON_SCENARIOS.items():
        predictions = prepare_scenario_predictions(
            rf_predictions,
            one_class_predictions,
            scenario_name,
            scenario_config,
        )
        folds = sorted(int(fold) for fold in predictions["fold"].unique())

        print("\n" + "=" * 70)
        print(f"Scenariusz: {scenario_name}")
        print(f"Rekordy: {len(predictions)}")
        print(f"Unikalne rekordy: {predictions['row_id'].nunique()}")
        print(f"Unikalne grupy: {predictions['request_group'].nunique()}")
        print(f"Foldy: {folds}")

        point_results.append(calculate_point_metrics(predictions, scenario_name))
        bootstrap_results.append(
            run_group_bootstrap(
                predictions,
                scenario_name,
                iterations=BOOTSTRAP_ITERATIONS,
            )
        )

    point_metrics_df = pd.concat(point_results, ignore_index=True)
    bootstrap_df = pd.concat(bootstrap_results, ignore_index=True)
    model_summary_df = create_model_bootstrap_summary(
        point_metrics_df,
        bootstrap_df,
    )
    pairwise_summary_df = create_pairwise_bootstrap_summary(
        point_metrics_df,
        bootstrap_df,
    )

    display_columns = ["scenario", "model", "prediction_rule"]
    for metric_name in METRIC_NAMES:
        display_columns.extend(
            [f"{metric_name}_mean", f"{metric_name}_std"]
        )

    print("\n" + "=" * 70)
    print("METRYKI — ŚREDNIA I ODCHYLENIE Z 10 FOLDÓW")
    print("=" * 70)
    print(point_metrics_df[display_columns].round(4).to_string(index=False))

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
    print(pairwise_summary_df[pairwise_columns].round(4).to_string(index=False))

    paths = save_results(bootstrap_df, model_summary_df, pairwise_summary_df)
    print("\nZapisane pliki:")
    for path in paths:
        print(f"- {path}")
    print("\nZbiór final_test nie został użyty.")


if __name__ == "__main__":
    main()