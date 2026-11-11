from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent / "clean_classification_dataset"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError("Run prepare_dataset.py first.")

    renamed = 0
    for split in ("train", "valid", "test"):
        for folder in sorted((ROOT / split).iterdir()):
            if not folder.is_dir():
                continue
            images = sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )
            temporary = []
            for path in images:
                temp_path = path.with_name(f".rename_{uuid4().hex}{path.suffix.lower()}")
                path.rename(temp_path)
                temporary.append((temp_path, path.suffix.lower()))
            for index, (path, extension) in enumerate(temporary, start=1):
                path.rename(folder / f"{folder.name}_{index:03d}{extension}")
                renamed += 1

    print(f"Renamed {renamed} images in clean_classification_dataset")


if __name__ == "__main__":
    main()
