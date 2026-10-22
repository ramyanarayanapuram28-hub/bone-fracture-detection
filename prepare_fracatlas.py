
from pathlib import Path
import random
import shutil

ROOT = Path("data/FracAtlas")
IMAGE_ROOT = ROOT / "images"
LABEL_ROOT = ROOT / "Annotations" / "YOLO"
OUTPUT = Path("data/fracatlas_yolo")

random.seed(42)
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

# Find images and annotation files.
images_by_stem = {}
for image in IMAGE_ROOT.rglob("*"):
    if image.is_file() and image.suffix.lower() in extensions:
        images_by_stem.setdefault(image.stem, []).append(image)

labels_by_stem = {
    p.stem: p for p in LABEL_ROOT.glob("*.txt") if p.is_file()
}

records = []
skipped = 0
box_count = 0

for stem, label_path in labels_by_stem.items():
    candidates = images_by_stem.get(stem, [])
    if len(candidates) != 1:
        skipped += 1
        continue

    image_path = candidates[0]
    valid_lines = []

    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            skipped += 1
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            skipped += 1
            continue

        if cls != 0 or not all(0 <= v <= 1 for v in (x, y, w, h)):
            skipped += 1
            continue
        if w <= 0 or h <= 0:
            skipped += 1
            continue

        valid_lines.append(
            f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    records.append((image_path, valid_lines))
    box_count += len(valid_lines)

if len(records) < 10:
    raise RuntimeError(
        f"Only {len(records)} matched images found. Check the folder paths."
    )

random.shuffle(records)
n = len(records)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": records[:n_train],
    "val": records[n_train:n_train + n_val],
    "test": records[n_train + n_val:],
}

# Re-create only the generated output folder.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split, items in splits.items():
    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, lines in items:
        shutil.copy2(image_path, image_dir / image_path.name)
        label_file = label_dir / f"{image_path.stem}.txt"
        label_file.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

yaml_text = f"""path: {OUTPUT.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fracture
"""
(OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

print("\nDataset preparation complete!")
print("Output:", OUTPUT.resolve())
print("Matched images:", n)
print("Valid bounding boxes:", box_count)
print("Invalid or unmatched items skipped:", skipped)
for split, items in splits.items():
    positives = sum(bool(lines) for _, lines in items)
    print(
        f"{split}: {len(items)} images, "
        f"{positives} with boxes, {len(items)-positives} without boxes"
    )
print("Config:", OUTPUT / "data.yaml")
