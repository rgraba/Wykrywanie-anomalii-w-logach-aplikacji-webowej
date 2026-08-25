from itertools import combinations

import numpy as np
import pandas as pd

import config
from analiza_niepewnosci import calculate_metrics, METRIC_NAMES


BOOTSTRAP_ITERATIONS = 2000

PREDICTIONS_FILE = (
    config.REPORTS_DIR
    / "final_test_predictions_seed_2026.csv.gz"
)

SCENARIOS = {
    "supervised_calibrated": [
        "random_forest",
        "isolation_forest_calibrated",
        "one_class_svm_calibrated",
    ],
    "strict": [
        "random_forest",
        "isolation_forest_strict",
        "one_class_svm_strict",
    ],
}


def load_predictions():
    predictions = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "row_id",
        "request_group",
        "classification",
        "y_pred",
        "y_score",
        "model_key",
        "final_test_used",
    }

    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Brak kolumn: {sorted(missing)}")

    final_flags = (
        predictions["final_test_used"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    )

    if not final_flags.all():
        raise ValueError("Plik zawiera predykcje spoza final_test.")

    model_frames = {}
    reference = None

    for model_key, frame in predictions.groupby("model_key"):
        frame = (
            frame.sort_values("row_id")
            .reset_index(drop=True)
        )

        if frame["row_id"].duplicated().any():
            raise ValueError(
                f"Powtórzone row_id dla modelu {model_key}."
            )

        identifiers = frame[
            ["row_id", "request_group", "classification"]
        ]

        if reference is None:
            reference = identifiers.copy()
        elif not reference.equals(identifiers):
            raise ValueError(
                f"Predykcje modelu {model_key} "
                "nie są poprawnie sparowane."
            )

        model_frames[model_key] = frame

    return reference, model_frames


def model_metrics(frame, sample_weight=None):
    return calculate_metrics(
        y_true=frame["classification"].to_numpy(),
        y_pred=frame["y_pred"].to_numpy(),
        y_score=frame["y_score"].to_numpy(),
        sample_weight=sample_weight,
    )


def main():
    reference, models = load_predictions()

    group_codes, unique_groups = pd.factorize(
        reference["request_group"],
        sort=False,
    )

    number_of_groups = len(unique_groups)
    rng = np.random.default_rng(
        config.PROTOCOL_RANDOM_STATE
    )

    observed = {
        model_key: model_metrics(frame)
        for model_key, frame in models.items()
    }

    bootstrap_rows = []

    while len(bootstrap_rows) < BOOTSTRAP_ITERATIONS:
        sampled_codes = rng.integers(
            0,
            number_of_groups,
            size=number_of_groups,
        )

        group_counts = np.bincount(
            sampled_codes,
            minlength=number_of_groups,
        )

        # Wszystkie rekordy danej grupy otrzymują tę samą
        # wielokrotność w próbie bootstrapowej.
        sample_weight = group_counts[group_codes]

        selected_classes = reference.loc[
            sample_weight > 0,
            "classification",
        ].unique()

        if len(selected_classes) != 2:
            continue

        row = {
            "iteration": len(bootstrap_rows) + 1,
        }

        for model_key, frame in models.items():
            metrics = model_metrics(
                frame,
                sample_weight=sample_weight,
            )

            for metric_name in METRIC_NAMES:
                row[
                    f"{model_key}__{metric_name}"
                ] = metrics[metric_name]

        bootstrap_rows.append(row)

    bootstrap = pd.DataFrame(bootstrap_rows)

    model_summary_rows = []

    for model_key in models:
        for metric_name in METRIC_NAMES:
            values = bootstrap[
                f"{model_key}__{metric_name}"
            ]

            lower, upper = np.quantile(
                values,
                [0.025, 0.975],
            )

            model_summary_rows.append({
                "model_key": model_key,
                "metric": metric_name,
                "value": observed[model_key][metric_name],
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 0.95,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "number_of_groups": number_of_groups,
            })

    pairwise_rows = []

    for scenario, model_keys in SCENARIOS.items():
        for first_model, second_model in combinations(
            model_keys,
            2,
        ):
            for metric_name in METRIC_NAMES:
                differences = (
                    bootstrap[
                        f"{first_model}__{metric_name}"
                    ]
                    - bootstrap[
                        f"{second_model}__{metric_name}"
                    ]
                )

                lower, upper = np.quantile(
                    differences,
                    [0.025, 0.975],
                )

                pairwise_rows.append({
                    "scenario": scenario,
                    "comparison": (
                        f"{first_model} - {second_model}"
                    ),
                    "metric": metric_name,
                    "difference": (
                        observed[first_model][metric_name]
                        - observed[second_model][metric_name]
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_excludes_zero": (
                        lower > 0 or upper < 0
                    ),
                    "confidence_level": 0.95,
                    "bootstrap_iterations": (
                        BOOTSTRAP_ITERATIONS
                    ),
                    "number_of_groups": number_of_groups,
                })

    model_summary = pd.DataFrame(model_summary_rows)
    pairwise_summary = pd.DataFrame(pairwise_rows)

    samples_path = (
        config.REPORTS_DIR
        / "final_test_group_bootstrap_samples.csv.gz"
    )
    model_path = (
        config.REPORTS_DIR
        / "final_test_group_bootstrap_model_summary.csv"
    )
    pairwise_path = (
        config.REPORTS_DIR
        / "final_test_group_bootstrap_pairwise_summary.csv"
    )

    bootstrap.to_csv(
        samples_path,
        index=False,
        compression="gzip",
    )
    model_summary.to_csv(model_path, index=False)
    pairwise_summary.to_csv(pairwise_path, index=False)

    print(
        f"Bootstrap wykonany na {number_of_groups} grupach."
    )
    print("\nPrzedziały wyników modeli:")
    print(model_summary.round(4).to_string(index=False))
    print("\nPrzedziały różnic:")
    print(pairwise_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()