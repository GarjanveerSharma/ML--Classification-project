from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "clean_classification_dataset"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

    classes = sorted(
        (path for path in SOURCE_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if len(classes) < 2:
        raise ValueError(f"No class folders found in {SOURCE_DIR}")

    split_groups = {name: [] for name in ("train", "valid", "test")}
    unique_count = 0
    for folder in classes:
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 3:
            raise ValueError(f"{folder.name} needs at least 3 images.")
        groups = {}
        for path in images:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path)
        if len(groups) < 3:
            raise ValueError(
                f"{folder.name} needs at least 3 distinct images for splitting."
            )
        unique_count += len(groups)

        group_ids = sorted(groups)
        train_ids, held_ids = train_test_split(
            group_ids, test_size=0.30, random_state=SEED
        )
        valid_ids, test_ids = train_test_split(
            held_ids, test_size=0.50, random_state=SEED
        )
        for split, ids in (
            ("train", train_ids),
            ("valid", valid_ids),
            ("test", test_ids),
        ):
            for group_id in ids:
                split_groups[split].extend(groups[group_id])

    stage = ROOT / "clean_classification_dataset_build"
    if stage.exists():
        raise FileExistsError(f"Temporary folder already exists: {stage}")
    stage.mkdir()
    try:
        for split, paths in split_groups.items():
            for path in paths:
                destination = stage / split / path.parent.name / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

        (stage / ".dataset_source.json").write_text(
            json.dumps(
                {"source": "images", "classes": [folder.name for folder in classes]},
                indent=2,
            ),
            encoding="utf-8",
        )

        if OUTPUT_DIR.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = ROOT / f"clean_classification_dataset_backup_{timestamp}"
            OUTPUT_DIR.rename(backup)
            print(f"Previous dataset preserved at: {backup.name}")
        stage.rename(OUTPUT_DIR)
    except OSError:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    print("Dataset prepared from images/:")
    for split in SPLITS:
        count = sum(
            1
            for folder in (OUTPUT_DIR / split).iterdir()
            for path in folder.iterdir()
            if path.is_file()
        )
        print(f"  {split}: {count} images")
    print(f"Classes: {', '.join(folder.name for folder in classes)}")
    print(f"Distinct images kept together across splits: {unique_count}")


if __name__ == "__main__":
    main()
