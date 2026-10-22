
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["not fractured", "fractured"]


def main():
    test_data = joblib.load(RESULTS_DIR / "test_hog_features.joblib")
    X_test = test_data["X"]
    y_test = test_data["y"]

    print("Test feature shape:", X_test.shape)
    print("Test image count:", len(y_test))

    for name in ["svm", "random_forest"]:
        model_path = MODELS_DIR / f"{name}.joblib"
        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        print("\n" + "=" * 55)
        print(f"{name.upper()} — TEST RESULTS")
        print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
        print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        print("Confusion matrix:")
        print(matrix)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=CLASS_NAMES,
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"{name.upper()} — Test Confusion Matrix")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
