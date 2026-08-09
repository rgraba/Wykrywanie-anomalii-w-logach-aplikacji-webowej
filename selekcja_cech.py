import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import config


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
            "selected": feature in selected_features,
        }
        for feature, score in zip(features, scores)
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
        for feature, score in zip(features, scores)
        if score > threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    return selected_features, create_score_rows(
        method_name="INFORMATION_GAIN",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )


def select_with_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[list[str], list[dict], float]:
    features = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selector = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5,
        l1_ratios=(1.0,),
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=5000,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        refit=True,
        use_legacy_attributes=False,
    )

    selector.fit(X_train_scaled, y_train)

    scores = np.abs(selector.coef_[0])
    threshold = 1e-4

    selected_features = [
        feature
        for feature, score in zip(features, scores)
        if score >= threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    return (
        selected_features,
        create_score_rows(
            method_name="L1_LASSO",
            features=features,
            scores=scores,
            threshold=threshold,
            selected_features=selected_features,
        ),
        float(selector.C_),
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

    selector.fit(X_train, y_train)

    scores = selector.feature_importances_
    threshold = float(np.mean(scores))

    selected_features = [
        feature
        for feature, score in zip(features, scores)
        if score > threshold
    ]

    selected_features = ensure_features_selected(
        features,
        scores,
        selected_features,
    )

    return selected_features, create_score_rows(
        method_name="RF_IMPORTANCE",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )
