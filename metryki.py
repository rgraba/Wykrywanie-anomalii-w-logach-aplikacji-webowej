import numpy as np

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


def calculate_binary_metrics(
    y_true,
    y_pred,
    y_score,
) -> dict:
    # Oblicza wspólny zestaw metryk klasyfikacji binarnej.
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


def convert_one_class_predictions(
    raw_predictions,
) -> np.ndarray:
    #Zamienia format 1/-1 modeli jednoklasowych na etykiety 0/1.
    return np.where(
        np.asarray(raw_predictions) == -1,
        1,
        0,
    )