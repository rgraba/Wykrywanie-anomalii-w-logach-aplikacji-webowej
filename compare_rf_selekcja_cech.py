from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
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
from sklearn.preprocessing import StandardScaler

import config

from data_processor import get_processed_data


def ensure_features_selected(
    features: list[str],
    scores: np.ndarray,
    selected_features: list[str],
) -> list[str]:
    if selected_features:
        return selected_features

    best_feature_index = int(np.argmax(scores))

    return [features[best_feature_index]]


def create_score_rows(
    method_name: str,
    features: list[str],
    scores: np.ndarray,
    threshold: float,
    selected_features: list[str],
) -> list[dict]:
    return [
        {
            "selection_method": method_name,
            "feature": feature,
            "score": float(score),
            "threshold": float(threshold),
            "selected": (
                feature in selected_features
            ),
        }
        for feature, score in zip(
            features,
            scores,
        )
    ]


def select_with_information_gain(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[str], list[dict]]:
    features = list(X_train.columns)

    scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=config.RANDOM_STATE,
    )

    threshold = float(np.mean(scores))

    selected_features = [
        feature
        for feature, score in zip(
            features,
            scores,
        )
        if score > threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    score_rows = create_score_rows(
        method_name="INFORMATION_GAIN",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )

    return selected_features, score_rows


def select_with_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[str], list[dict], float]:
    features = list(X_train.columns)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    selector = LogisticRegressionCV(
        Cs=[
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        refit=True,
    )

    selector.fit(
        X_train_scaled,
        y_train,
    )

    scores = np.abs(
        selector.coef_[0]
    )

    # Próg zgodny z podejściem opisanym
    # w publikacji.
    threshold = 1e-4

    selected_features = [
        feature
        for feature, score in zip(
            features,
            scores,
        )
        if score >= threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    score_rows = create_score_rows(
        method_name="L1_LASSO",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )

    best_c = float(selector.C_[0])

    return (
        selected_features,
        score_rows,
        best_c,
    )


def select_with_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[str], list[dict]]:
    features = list(X_train.columns)

    selector = RandomForestClassifier(
        **config.RANDOM_FOREST_PARAMS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    selector.fit(
        X_train,
        y_train,
    )

    scores = selector.feature_importances_

    threshold = float(np.mean(scores))

    selected_features = [
        feature
        for feature, score in zip(
            features,
            scores,
        )
        if score > threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    score_rows = create_score_rows(
        method_name="RF_IMPORTANCE",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )

    return selected_features, score_rows


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
        "false_positive_rate": (
            false_positive_rate
        ),
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


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
    train_indices,
    test_indices,
) -> dict:
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
    metrics["selected_features"] = ", ".join(
        features
    )
    metrics["training_time_seconds"] = (
        training_time
    )
    metrics["prediction_time_seconds"] = (
        prediction_time
    )

    return metrics


def compare_feature_selection() -> None:
    print("Wczytywanie danych...")

    df = get_processed_data()

    y = df["classification"]

    train_indices, test_indices = (
        train_test_split(
            df.index,
            test_size=0.4,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )
    )

    all_features = (
        config.ML_FEATURES_ALTHUBITI_9
    )

    X_train_all = df.loc[
        train_indices,
        all_features,
    ]

    y_train = df.loc[
        train_indices,
        "classification",
    ]

    print("\nSelekcja Information Gain...")

    ig_features, ig_scores = (
        select_with_information_gain(
            X_train_all,
            y_train,
        )
    )

    print(
        "Wybrane cechy IG: "
        f"{ig_features}"
    )

    print("\nSelekcja L1/LASSO...")

    (
        l1_features,
        l1_scores,
        best_l1_c,
    ) = select_with_l1(
        X_train_all,
        y_train,
    )

    print(
        "Wybrane cechy L1: "
        f"{l1_features}"
    )

    print(
        f"Najlepsze C dla L1: {best_l1_c}"
    )

    print("\nSelekcja RF importance...")

    rf_features, rf_scores = (
        select_with_random_forest(
            X_train_all,
            y_train,
        )
    )

    print(
        "Wybrane cechy RF: "
        f"{rf_features}"
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

    metrics_rows = []

    for feature_set_name, features in (
        feature_sets.items()
    ):
        print("\n" + "=" * 70)
        print(f"EWALUACJA: {feature_set_name}")
        print(f"Wybrane cechy: {features}")
        print("=" * 70)

        metrics = evaluate_feature_set(
            df=df,
            feature_set_name=feature_set_name,
            features=features,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        metrics_rows.append(metrics)

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision: "
            f"{metrics['precision_anomaly']:.4f}"
        )
        print(
            f"Recall: "
            f"{metrics['recall_anomaly']:.4f}"
        )
        print(
            f"F1: "
            f"{metrics['f1_anomaly']:.4f}"
        )
        print(
            f"ROC AUC: "
            f"{metrics['roc_auc']:.4f}"
        )
        print(
            f"False-positive rate: "
            f"{metrics['false_positive_rate']:.4f}"
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    scores_df = pd.DataFrame(
        ig_scores
        + l1_scores
        + rf_scores
    )

    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_60_40.csv"
    )

    scores_path = (
        config.REPORTS_DIR
        / "rf_feature_selection_scores.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    scores_df.to_csv(
        scores_path,
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
    print("PODSUMOWANIE SELEKCJI CECH")
    print("=" * 70)

    print(
        metrics_df[
            columns_to_display
        ].round(4).to_string(
            index=False
        )
    )

    print(
        f"\nWyniki zapisano w: "
        f"{metrics_path}"
    )

    print(
        f"Oceny cech zapisano w: "
        f"{scores_path}"
    )


if __name__ == "__main__":
    compare_feature_selection()