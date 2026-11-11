import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "clean_classification_dataset"
OUTPUT_FILE = DATASET_DIR / "labels.csv"
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    train_dir = DATASET_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError("Run prepare_dataset.py first.")
    classes = sorted(
        folder.name for folder in train_dir.iterdir() if folder.is_dir()
    )
    if not classes:
        raise ValueError("No class folders found in the training split.")

    records = []
    for label_id, class_name in enumerate(classes):
        for split in SPLITS:
            folder = DATASET_DIR / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    records.append(
                        {
                            "split": split,
                            "image_path": path.relative_to(ROOT).as_posix(),
                            "class_name": class_name,
                            "label_id": label_id,
                        }
                    )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} image labels to clean_classification_dataset/labels.csv")


if __name__ == "__main__":
    main()
