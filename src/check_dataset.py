
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Bone_Fracture_Binary_Classification"
    / "Bone_Fracture_Binary_Classification"
)

CLASSES = ["fractured", "not fractured"]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print(f"Dataset location: {DATASET_DIR}")
print(f"Dataset folder exists: {DATASET_DIR.exists()}")

total_images = 0

for split in SPLITS:
    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        print(f"\nMissing split: {split}")
        continue

    print(f"\n{split.upper()}:")

    for class_name in CLASSES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            print(f"  Missing class folder: {class_name}")
            continue

        count = sum(
            1
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += count
        print(f"  {class_name}: {count} images")

print(f"\nTotal images: {total_images}")
