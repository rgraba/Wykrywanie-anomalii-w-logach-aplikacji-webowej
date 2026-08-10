import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import config
from metryki import (
    calculate_binary_metrics,
    convert_one_class_predictions,
)

MODELS_TO_EVALUATE = {
    "Isolation Forest": {
        "model_file": "iforest_model.pkl",
        "test_file": "test_data_if.pkl",
        "model_type": "anomaly"
    },
    "Random Forest": {
        "model_file": "rf_model.pkl",
        "test_file": "test_data_rf.pkl",
        "model_type": "classifier"
    },
    "One-Class SVM": {
        "model_file": "ocsvm_model.pkl",
        "test_file": "test_data_oc.pkl",
        "model_type": "anomaly"
    }
}

SPLIT_TYPE = "random"
SPLIT_LABEL = "Podział losowy 80/20"

def load_model_and_test_data(model_file: str, test_file: str):
    model_path = config.MODELS_DIR / model_file
    test_data_path = config.MODELS_DIR / test_file

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    return model, X_test, y_test


def predict_labels(model, X_test, model_type: str):

    if model_type == "classifier":
        return model.predict(X_test)

    if model_type == "anomaly":
        return convert_one_class_predictions(
            model.predict(X_test)
        )

    raise ValueError(f"Nieznany typ modelu: {model_type}")


def get_model_scores(model, X_test, model_type: str):
    if model_type == "classifier":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]

        if hasattr(model, "decision_function"):
            return model.decision_function(X_test)

        raise ValueError("Model klasyfikacyjny nie obsługuje predict_proba ani decision_function.")

    if model_type == "anomaly":
        if hasattr(model, "decision_function"):
            return -model.decision_function(X_test)

        if hasattr(model, "score_samples"):
            return -model.score_samples(X_test)

        raise ValueError("Model anomalii nie obsługuje decision_function ani score_samples.")

    raise ValueError(f"Nieznany typ modelu: {model_type}")


def calculate_metrics(
    model_name: str,
    y_true,
    y_pred,
    y_score,
) -> dict:
    raw_metrics = calculate_binary_metrics(
        y_true,
        y_pred,
        y_score,
    )

    return {
        "Model": model_name,
        "Accuracy": raw_metrics["accuracy"],
        "Balanced Accuracy": raw_metrics["balanced_accuracy"],
        "Precision (Anomaly)": raw_metrics["precision_anomaly"],
        "Recall (Anomaly)": raw_metrics["recall_anomaly"],
        "F1-Score (Anomaly)": raw_metrics["f1_anomaly"],
        "Specificity": raw_metrics["specificity"],
        "False Positive Rate": raw_metrics["false_positive_rate"],
        "ROC AUC": raw_metrics["roc_auc"],
        "PR AUC": raw_metrics["pr_auc"],
        "Split": SPLIT_TYPE,
        "Protokół": SPLIT_LABEL,
        "Seed": config.RANDOM_STATE,
    }


def plot_confusion_matrices(results: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(6 * len(results), 5)
    )

    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, result) in zip(axes, results.items()):
        cm = confusion_matrix(result["y_true"], result["y_pred"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Normalny", "Anomalia"],
            yticklabels=["Normalny", "Anomalia"]
        )

        ax.set_title(
            f"Macierz pomyłek: {model_name}\n"
            f"{SPLIT_LABEL}"
        )
        ax.set_xlabel("Przewidywana klasa")
        ax.set_ylabel("Rzeczywista klasa")

    plt.tight_layout()

    output_path = config.REPORTS_DIR / "confusion_matrices_random_80_20.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Macierze pomyłek zapisano jako: {output_path}")


def plot_roc_curves(results: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for model_name, result in results.items():
        y_true = result["y_true"]
        y_score = result.get("y_score")

        if y_score is None:
            print(f"Pominięto ROC dla modelu {model_name}: brak y_score.")
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = roc_auc_score(y_true, y_score)

            plt.plot(
                fpr,
                tpr,
                label=f"{model_name} (AUC = {auc_value:.4f})"
            )

        except ValueError as e:
            print(f"Nie udało się wygenerować ROC dla modelu {model_name}: {e}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Losowy klasyfikator")

    plt.xlabel("Odsetek fałszywie dodatnich (FPR)")
    plt.ylabel("Odsetek prawdziwie dodatnich (TPR)")
    plt.title(
        "Krzywe ROC dla modeli\n"
        f"{SPLIT_LABEL}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = config.REPORTS_DIR / "roc_curves_random_80_20.png"
    plt.savefig(output_path, dpi=300, bbox_inches = "tight")
    plt.show()
    plt.close()

    print(f"Krzywe ROC zapisano jako: {output_path}")


def evaluate_models() -> None:
    print("Rozpoczęto ewaluację modeli.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    metrics_rows = []

    for model_name, model_config in MODELS_TO_EVALUATE.items():
        print("\n" + "=" * 60)
        print(f"Ewaluacja modelu: {model_name}")
        print("=" * 60)

        try:
            model, X_test, y_test = load_model_and_test_data(
                model_config["model_file"],
                model_config["test_file"]
            )

            y_pred = predict_labels(
                model,
                X_test,
                model_config["model_type"]
            )

            y_score = get_model_scores(
                model,
                X_test,
                model_config["model_type"]
            )

        except Exception as e:
            print(f"Błąd podczas ewaluacji modelu {model_name}: {e}")
            continue

        print("\nRaport klasyfikacji:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["0 (Normal)", "1 (Anomaly)"],
                zero_division=0
            )
        )

        metrics = calculate_metrics(model_name, y_test, y_pred, y_score)
        metrics_rows.append(metrics)

        results[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_score": y_score
        }

    if not metrics_rows:
        print("Nie udało się obliczyć metryk dla żadnego modelu.")
        return

    summary_df = pd.DataFrame(metrics_rows)

    print("\nPODSUMOWANIE METRYK")
    print(summary_df.round(4).to_string(index=False))

    metrics_path = config.REPORTS_DIR / "metrics_summary_random_80_20.csv"
    summary_df.to_csv(metrics_path, index=False)

    print(f"\nTabela metryk zapisana jako: {metrics_path}")

    plot_confusion_matrices(results)
    plot_roc_curves(results)

    print("\nEwaluacja zakończona.")


if __name__ == "__main__":
    evaluate_models()
