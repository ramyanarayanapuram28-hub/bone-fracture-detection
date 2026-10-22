
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["not fractured", "fractured"]


def load_features(split):
    path = RESULTS_DIR / f"{split}_hog_features.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = joblib.load(path)
    return data["X"], data["y"]


def show_metrics(model_name, model, X, y):
    predictions = model.predict(X)

    print(f"\n{model_name} — Validation Results")
    print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y, predictions, zero_division=0):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    return f1_score(y, predictions, zero_division=0)


def main():
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")

    print("Training feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)

    models = {
        "svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, random_state=42, max_iter=5000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name.upper()}...")
        start = time.time()

        model.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"Training time: {elapsed:.2f} seconds")

        val_f1 = show_metrics(name.upper(), model, X_val, y_val)
        results[name] = {
            "model": model,
            "validation_f1": val_f1,
            "training_seconds": elapsed,
        }

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        print(f"Saved model: {model_path}")

    print("\n" + "=" * 50)
    print("VALIDATION COMPARISON")

    for name, result in results.items():
        print(
            f"{name.upper():15s} "
            f"F1={result['validation_f1']:.4f} "
            f"Training time={result['training_seconds']:.2f}s"
        )

    best_name = max(results, key=lambda name: results[name]["validation_f1"])
    joblib.dump(
        {"model_name": best_name, "validation_f1": results[best_name]["validation_f1"]},
        RESULTS_DIR / "model_selection.joblib",
    )

    print(f"\nBest model by validation F1-score: {best_name.upper()}")


if __name__ == "__main__":
    main()
