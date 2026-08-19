from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import config
from metryki import calculate_binary_metrics
from podzial_danych import (
    calculate_group_overlap,
    ensure_no_group_overlap,
    add_request_groups,
)
from protokol_eksperymentalny import (
    get_development_folds,
    get_development_indices,
)
from przetwarzanie_danych import get_processed_data


def create_random_record_folds(
    df: pd.DataFrame,
    development_indices: pd.Index,
) -> list[tuple[int, pd.Index, pd.Index]]:
    development_df = df.loc[development_indices]

    splitter = StratifiedKFold(
        n_splits=config.DEVELOPMENT_CV_N_SPLITS,
        shuffle=True,
        random_state=config.PROTOCOL_RANDOM_STATE,
    )

    folds = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        splitter.split(
            development_df,
            development_df["classification"],
        ),
        start=1,
    ):
        train_indices = development_df.index[train_positions]
        validation_indices = development_df.index[validation_positions]

        folds.append(
            (
                fold_number,
                train_indices,
                validation_indices,
            )
        )

    return folds


def run_rf_experiment(
    df: pd.DataFrame,
    train_indices: pd.Index,
    validation_indices: pd.Index,
    split_name: str,
    fold_number: int,
) -> tuple[dict, pd.DataFrame]:
    features = list(config.ML_FEATURES_BASIC)

    X_train = df.loc[train_indices, features]
    y_train = df.loc[train_indices, "classification"]

    X_validation = df.loc[validation_indices, features]
    y_validation = df.loc[validation_indices, "classification"]

    if split_name == "group":
        group_overlap = ensure_no_group_overlap(
            df,
            train_indices,
            validation_indices,
            first_name=f"group_fold_{fold_number}_train",
            second_name=f"group_fold_{fold_number}_validation",
        )
    else:
        group_overlap = calculate_group_overlap(
            df,
            train_indices,
            validation_indices,
        )

    train_groups = set(
        df.loc[
            train_indices,
            "request_group",
        ]
    )

    validation_groups = df.loc[
        validation_indices,
        "request_group",
    ]

    group_seen_in_training = validation_groups.isin(
        train_groups
    )

    validation_records_with_train_group = int(
        group_seen_in_training.sum()
    )

    validation_records_with_train_group_rate = (
            validation_records_with_train_group
            / len(validation_indices)
    )

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

    fold_predictions = pd.DataFrame(
        {
            "row_id": validation_indices.to_numpy(),
            "request_group": validation_groups.to_numpy(),
            "classification": y_validation.to_numpy(),
            "y_pred": y_pred,
            "y_score": y_score,
            "group_seen_in_training": (
                group_seen_in_training.to_numpy()
            ),
            "split": split_name,
            "fold": fold_number,
            "protocol_seed": config.PROTOCOL_RANDOM_STATE,
            "final_test_used": False,
        }
    )

    fold_metrics = {
        "model": "RANDOM_FOREST",
        "feature_set": "BASIC",
        "split": split_name,
        "fold": fold_number,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "group_overlap": group_overlap,
        "validation_records_with_train_group": (
            validation_records_with_train_group
        ),
        "validation_records_with_train_group_rate": (
            validation_records_with_train_group_rate
        ),
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "partition": "development",
        "protocol_seed": config.PROTOCOL_RANDOM_STATE,
        "model_seed": config.RANDOM_STATE,
        "final_test_used": False,
        **metrics,
    }

    return fold_metrics, fold_predictions


def create_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("split", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            group_overlap_mean=("group_overlap", "mean"),
            group_overlap_min=("group_overlap", "min"),
            group_overlap_max=("group_overlap", "max"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_anomaly_mean=("precision_anomaly", "mean"),
            precision_anomaly_std=("precision_anomaly", "std"),
            recall_anomaly_mean=("recall_anomaly", "mean"),
            recall_anomaly_std=("recall_anomaly", "std"),
            f1_anomaly_mean=("f1_anomaly", "mean"),
            f1_anomaly_std=("f1_anomaly", "std"),
            false_positive_rate_mean=("false_positive_rate", "mean"),
            false_positive_rate_std=("false_positive_rate", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            training_time_mean=("training_time_seconds", "mean"),
            prediction_time_mean=("prediction_time_seconds", "mean"),
            validation_records_with_train_group_mean=("validation_records_with_train_group", "mean"),
            validation_records_with_train_group_rate_mean=("validation_records_with_train_group_rate", "mean"),
            validation_records_with_train_group_rate_std=("validation_records_with_train_group_rate", "std"),
        )
    )


def compare_splits() -> None:
    df = add_request_groups(
        get_processed_data()
    )

    development_indices = get_development_indices(df)

    random_folds = create_random_record_folds(
        df=df,
        development_indices=development_indices,
    )
    group_folds = get_development_folds(df)

    results = []
    prediction_frames = []

    print("\nPORÓWNANIE PODZIAŁÓW NA ZBIORZE DEVELOPMENT")
    print(f"Development: {len(development_indices)} rekordów")
    print("Final test pozostaje zamrożony i nie jest używany.")

    for split_name, folds in [
        ("random_record", random_folds),
        ("group", group_folds),
    ]:
        print("\n" + "=" * 70)
        print(f"PODZIAŁ: {split_name}")
        print("=" * 70)

        for fold_number, train_indices, validation_indices in folds:
            fold_results, fold_predictions = run_rf_experiment(
                df=df,
                train_indices=train_indices,
                validation_indices=validation_indices,
                split_name=split_name,
                fold_number=fold_number,
            )

            results.append(fold_results)
            prediction_frames.append(fold_predictions)

            print(
                f"Fold {fold_number}: "
                f"overlap={fold_results['group_overlap']}, "
                f"record_overlap="
                f"{fold_results['validation_records_with_train_group_rate']:.2%}, "
                f"BalAcc={fold_results['balanced_accuracy']:.4f}, "
                f"F1={fold_results['f1_anomaly']:.4f}, "
                f"ROC AUC={fold_results['roc_auc']:.4f}"
            )

    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    development_rows = set(development_indices)

    for split_name in ["random_record", "group"]:
        split_predictions = predictions_df.loc[
            predictions_df["split"] == split_name
            ]

        if len(split_predictions) != len(development_indices):
            raise RuntimeError(
                f"Nieprawidłowa liczba predykcji OOF dla {split_name}."
            )

        if not split_predictions["row_id"].is_unique:
            raise RuntimeError(
                f"Powtórzone predykcje OOF dla {split_name}."
            )

        if set(split_predictions["row_id"]) != development_rows:
            raise RuntimeError(
                f"Predykcje OOF dla {split_name} nie obejmują "
                "dokładnie zbioru development."
            )

    random_results = results_df[
        results_df["split"] == "random_record"
    ]
    group_results = results_df[
        results_df["split"] == "group"
    ]

    if (random_results["group_overlap"] == 0).any():
        raise RuntimeError(
            "Co najmniej jeden losowy fold nie zawiera wspólnych grup. "
            "Sprawdź sposób tworzenia losowego podziału."
        )

    if (group_results["group_overlap"] != 0).any():
        raise RuntimeError(
            "W podziale grupowym wykryto group_overlap różny od zera."
        )

    summary_df = create_summary(results_df)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    folds_path = (
        config.REPORTS_DIR
        / "rf_random_vs_group_cv10_folds.csv"
    )
    summary_path = (
        config.REPORTS_DIR
        / "rf_random_vs_group_cv10_summary.csv"
    )
    predictions_path = (
            config.REPORTS_DIR
            / "rf_random_vs_group_cv10_oof_predictions.csv"
    )

    results_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False,)

    columns_to_display = [
        "split",
        "folds",
        "group_overlap_mean",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_anomaly_mean",
        "f1_anomaly_std",
        "false_positive_rate_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "validation_records_with_train_group_rate_mean",
    ]

    print("\n" + "=" * 70)
    print("PODSUMOWANIE PORÓWNANIA")
    print("=" * 70)
    print(
        summary_df[columns_to_display]
        .round(4)
        .to_string(index=False)
    )

    print("\nZapisane pliki:")
    print(f"- {folds_path}")
    print(f"- {summary_path}")
    print(f"- {predictions_path}")
    print("\nZbiór final_test nie został użyty.")



if __name__ == "__main__":
    compare_splits()