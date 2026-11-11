import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "clean_classification_dataset"
FEATURE_DIR = DATASET_DIR / "features"
IMAGE_SIZE = (64, 64)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32).flatten() / 255.0


def main():
    train_dir = DATASET_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError("Run prepare_dataset.py first.")
    classes = sorted(
        folder.name for folder in train_dir.iterdir() if folder.is_dir()
    )
    if not classes:
        raise ValueError("No class folders found in the training split.")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        features, labels = [], []
        for label, class_name in enumerate(classes):
            folder = DATASET_DIR / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    features.append(extract_features(path))
                    labels.append(label)
                except ValueError as error:
                    print(f"WARNING: {error}")
        if not features:
            raise ValueError(f"No readable images in {split}.")
        np.savez_compressed(
            FEATURE_DIR / f"{split}_features.npz",
            X=np.stack(features),
            y=np.asarray(labels, dtype=np.int64),
            class_names=np.asarray(classes),
        )
        print(f"{split}: {len(features)} images, {len(features[0])} features each")

    (FEATURE_DIR / "class_mapping.json").write_text(
        json.dumps({name: i for i, name in enumerate(classes)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
