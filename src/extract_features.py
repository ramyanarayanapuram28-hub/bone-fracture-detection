
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from skimage.feature import hog

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Bone_Fracture_Binary_Classification"
    / "Bone_Fracture_Binary_Classification"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
CLASSES = ["not fractured", "fractured"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = (224, 224)


def extract_hog_features(image):
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)


def main():
    start_time = time.time()

    for split in SPLITS:
        features_list = []
        labels_list = []
        filenames_list = []

        print(f"\nProcessing {split} dataset...")

        for label, class_name in enumerate(CLASSES):
            class_dir = DATASET_DIR / split / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            image_paths = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for index, image_path in enumerate(image_paths, start=1):
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    print(f"Skipping unreadable image: {image_path}")
                    continue

                features = extract_hog_features(image)

                features_list.append(features)
                labels_list.append(label)
                filenames_list.append(str(image_path.relative_to(DATASET_DIR)))

                if index % 500 == 0:
                    print(
                        f"  {class_name}: "
                        f"{index}/{len(image_paths)} images processed"
                    )

        if not features_list:
            raise RuntimeError(f"No readable images found in {split}")

        data = {
            "X": np.asarray(features_list, dtype=np.float32),
            "y": np.asarray(labels_list, dtype=np.int64),
            "filenames": filenames_list,
            "classes": CLASSES,
        }

        output_path = OUTPUT_DIR / f"{split}_hog_features.joblib"
        joblib.dump(data, output_path, compress=3)

        print(f"{split} feature shape: {data['X'].shape}")
        print(f"{split} label shape: {data['y'].shape}")
        print(f"Saved to: {output_path}")

    print(f"\nFeature extraction completed in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
