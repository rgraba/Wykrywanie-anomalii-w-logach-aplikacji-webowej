import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
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
    groups: pd.Series,
) -> tuple[list[str], list[dict], float]:
    features = list(X_train.columns)
    groups = groups.reindex(X_train.index)

    if groups.isna().any():
        raise ValueError(
            "Nie udało się przypisać grup do wszystkich rekordów "
            "używanych przez selekcję L1."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.PROTOCOL_RANDOM_STATE,
    )

    cv_splits = []

    for fold_number, (
        inner_train_positions,
        inner_validation_positions,
    ) in enumerate(
        splitter.split(
            X_train,
            y_train,
            groups=groups,
        ),
        start=1,
    ):
        inner_train_groups = set(
            groups.iloc[inner_train_positions]
        )
        inner_validation_groups = set(
            groups.iloc[inner_validation_positions]
        )

        group_overlap = len(
            inner_train_groups.intersection(
                inner_validation_groups
            )
        )

        if group_overlap != 0:
            raise RuntimeError(
                "Wewnętrzna walidacja L1 zawiera wspólne grupy "
                f"w foldzie {fold_number}: {group_overlap}."
            )

        cv_splits.append(
            (
                inner_train_positions,
                inner_validation_positions,
            )
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    l1_ratio=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__C": [
                0.001,
                0.01,
                0.1,
                1.0,
                10.0,
                100.0,
            ]
        },
        scoring="roc_auc",
        cv=cv_splits,
        refit=True,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    fitted_model = search.best_estimator_.named_steps["model"]
    scores = np.abs(fitted_model.coef_[0])
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

    score_rows = create_score_rows(
        method_name="L1_LASSO",
        features=features,
        scores=scores,
        threshold=threshold,
        selected_features=selected_features,
    )

    for row in score_rows:
        row.update(
            {
                "inner_cv": "StratifiedGroupKFold",
                "inner_cv_folds": len(cv_splits),
                "inner_group_overlap": 0,
                "inner_cv_best_roc_auc": float(search.best_score_),
            }
        )

    best_c = float(search.best_params_["model__C"])

    return selected_features, score_rows, best_c


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
