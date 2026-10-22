
from pathlib import Path

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

image_path = next(
    path for path in (DATASET_DIR / "train" / "fractured").iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if original is None:
    raise ValueError(f"Could not load image: {image_path}")

# Resize while preserving the original aspect ratio.
resized = cv2.resize(original, (224, 224), interpolation=cv2.INTER_AREA)

# Improve local contrast using CLAHE.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(resized)

print(f"Image: {image_path.name}")
print(f"Original dimensions: {original.shape[1]} x {original.shape[0]}")
print(f"Resized dimensions: {resized.shape[1]} x {resized.shape[0]}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(resized, cmap="gray")
axes[1].set_title("Resized (224 x 224)")

axes[2].imshow(enhanced, cmap="gray")
axes[2].set_title("CLAHE Enhanced")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

