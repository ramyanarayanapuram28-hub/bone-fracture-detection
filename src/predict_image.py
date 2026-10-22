
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

CLASSES = ["not fractured", "fractured"]
IMAGE_SIZE = (224, 224)


def extract_features(image):
    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1).astype(np.float32), resized


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: .\\venv\\Scripts\\python.exe '
            '.\\src\\predict_image.py "path\\to\\xray.jpg"'
        )
        return

    image_path = Path(sys.argv[1])

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not read image. Check the image path and file.")
        return

    original_features, resized = extract_features(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(resized)

    enhanced_features, _ = extract_features(enhanced)

    print(f"\nImage: {image_path.name}")
    print(f"Original size: {image.shape[1]} x {image.shape[0]}")
    print(f"Model input size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")

    for name in ["svm", "random_forest"]:
        model_path = MODELS_DIR / f"{name}.joblib"

        if not model_path.exists():
            print(f"Missing model: {model_path}")
            continue

        model = joblib.load(model_path)

        original_prediction = CLASSES[int(model.predict(original_features)[0])]
        enhanced_prediction = CLASSES[int(model.predict(enhanced_features)[0])]

        print(f"\n{name.upper()}:")
        print(f"  Original image prediction: {original_prediction}")
        print(f"  Enhanced image prediction: {enhanced_prediction}")
        print(
            "  Prediction changed after enhancement:",
            original_prediction != enhanced_prediction
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Uploaded image")

    axes[1].imshow(resized, cmap="gray")
    axes[1].set_title("Resized image")

    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("CLAHE enhanced")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
