import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

import config

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


def load_model_and_test_data(model_file: str, test_file: str):
    model_path = config.MODELS_DIR / model_file
    test_data_path = config.MODELS_DIR / test_file

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    return model, X_test, y_test


def predict_labels(model, X_test, model_type: str):
    # 0: normalny ruch, 1: anomalia / atak.

    if model_type == "classifier":
        return model.predict(X_test)

    if model_type == "anomaly":
        raw_predictions = model.predict(X_test)
        return [1 if pred == -1 else 0 for pred in raw_predictions]

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


def calculate_metrics(model_name: str, y_true, y_pred, y_score=None) -> dict:
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (Anomaly)": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "Recall (Anomaly)": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),
        "F1-Score (Anomaly)": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    }

    if y_score is not None:
        try:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["ROC AUC"] = None

    return metrics


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
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"]
        )

        ax.set_title(f"Macierz pomyłek: {model_name}")
        ax.set_xlabel("Przewidywana klasa")
        ax.set_ylabel("Rzeczywista klasa")

    plt.tight_layout()

    output_path = config.REPORTS_DIR / "confusion_matrices.png"
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

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywe ROC dla modeli")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = config.REPORTS_DIR / "roc_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Krzywe ROC zapisano jako: {output_path}")


def evaluate_models() -> None:
    print("Rozpoczynanie ewaluacji modeli...")

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

    metrics_path = config.REPORTS_DIR / "metrics_summary.csv"
    summary_df.to_csv(metrics_path, index=False)

    print(f"\nTabela metryk zapisana jako: {metrics_path}")

    plot_confusion_matrices(results)
    plot_roc_curves(results)

    print("\nEwaluacja zakończona.")


if __name__ == "__main__":
    evaluate_models()
