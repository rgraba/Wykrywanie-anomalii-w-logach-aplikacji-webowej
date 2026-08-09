from time import perf_counter

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config

from data_processor import get_processed_data


def calculate_metrics(
    y_true,
    y_pred,
    y_score,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_anomaly": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_anomaly": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_anomaly": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(
            y_true,
            y_score,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_score,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_experiment(
    df: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    train_indices,
    test_indices,
) -> tuple[dict, list[dict]]:
    print("\n" + "=" * 70)
    print(f"ZESTAW CECH: {feature_set_name}")
    print("=" * 70)

    X_train = df.loc[
        train_indices,
        features,
    ]

    X_test = df.loc[
        test_indices,
        features,
    ]

    y_train = df.loc[
        train_indices,
        "classification",
    ]

    y_test = df.loc[
        test_indices,
        "classification",
    ]

    model = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    training_start = perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        perf_counter() - training_start
    )

    prediction_start = perf_counter()

    y_pred = model.predict(X_test)

    y_score = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = (
        perf_counter() - prediction_start
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_score,
    )

    metrics["feature_set"] = feature_set_name
    metrics["number_of_features"] = len(features)
    metrics["train_size"] = len(train_indices)
    metrics["test_size"] = len(test_indices)
    metrics["training_time_seconds"] = (
        training_time
    )
    metrics["prediction_time_seconds"] = (
        prediction_time
    )

    importances = [
        {
            "feature_set": feature_set_name,
            "feature": feature,
            "importance": importance,
        }
        for feature, importance in sorted(
            zip(
                features,
                model.feature_importances_,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision anomaly: "
        f"{metrics['precision_anomaly']:.4f}"
    )

    print(
        f"Recall anomaly: "
        f"{metrics['recall_anomaly']:.4f}"
    )

    print(
        f"F1 anomaly: "
        f"{metrics['f1_anomaly']:.4f}"
    )

    print(
        f"ROC AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print(
        f"Czas treningu: "
        f"{training_time:.4f} s"
    )

    print("\nZnaczenie cech:")

    for importance_row in importances:
        print(
            f"{importance_row['feature']}: "
            f"{importance_row['importance']:.4f}"
        )

    return metrics, importances


def compare_feature_sets() -> None:
    print("Wczytywanie danych...")

    df = get_processed_data()

    y = df["classification"]

    # Podział 60/40 zgodny z publikacją
    # Althubiti et al.
    train_indices, test_indices = (
        train_test_split(
            df.index,
            test_size=0.4,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )
    )

    print("\nPROTOKÓŁ EKSPERYMENTALNY")
    print("Podział: 60% trening / 40% test")
    print(f"Seed: {config.RANDOM_STATE}")
    print(
        f"Liczba próbek treningowych: "
        f"{len(train_indices)}"
    )
    print(
        f"Liczba próbek testowych: "
        f"{len(test_indices)}"
    )

    print("\nRozkład klas w treningu:")
    print(
        y.loc[train_indices].value_counts()
    )

    print("\nRozkład klas w teście:")
    print(
        y.loc[test_indices].value_counts()
    )

    feature_sets = {
        "BASIC": config.ML_FEATURES_BASIC,
        "ALTHUBITI_9": (
            config.ML_FEATURES_ALTHUBITI_9
        ),
        "ALTHUBITI_5": (
            config.ML_FEATURES_ALTHUBITI_5
        ),
    }

    all_metrics = []
    all_importances = []

    for feature_set_name, features in (
        feature_sets.items()
    ):
        metrics, importances = run_experiment(
            df=df,
            feature_set_name=feature_set_name,
            features=features,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        all_metrics.append(metrics)
        all_importances.extend(importances)

    metrics_df = pd.DataFrame(all_metrics)

    importances_df = pd.DataFrame(
        all_importances
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        config.REPORTS_DIR
        / "rf_literature_features_60_40.csv"
    )

    importances_path = (
        config.REPORTS_DIR
        / "rf_literature_feature_importances.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    importances_df.to_csv(
        importances_path,
        index=False,
    )

    columns_to_display = [
        "feature_set",
        "number_of_features",
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
    ]

    print("\n" + "=" * 70)
    print("PORÓWNANIE ZESTAWÓW CECH")
    print("=" * 70)

    print(
        metrics_df[
            columns_to_display
        ].round(4).to_string(
            index=False
        )
    )

    print(f"\nMetryki zapisano w: {metrics_path}")

    print(
        "Znaczenie cech zapisano w: "
        f"{importances_path}"
    )


if __name__ == "__main__":
    compare_feature_sets()