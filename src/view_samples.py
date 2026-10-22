
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Bone_Fracture_Binary_Classification"
    / "Bone_Fracture_Binary_Classification"
)

random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for row, class_name in enumerate(["fractured", "not fractured"]):
    class_dir = DATASET_DIR / "train" / class_name

    images = [
        path for path in class_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    selected_images = random.sample(images, min(3, len(images)))

    for col, image_path in enumerate(selected_images):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            axes[row, col].set_title("Could not load image")
            axes[row, col].axis("off")
            continue

        print(
            f"{class_name}: {image_path.name} | "
            f"Dimensions: {image.shape[1]} x {image.shape[0]}"
        )

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
