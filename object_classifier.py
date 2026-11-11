from pathlib import Path
import argparse
import csv
import json
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "images"
MODEL_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = (64, 64)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def extract_features(image_path: Path) -> np.ndarray:
    """Resize an image and use its normalized RGB pixels as features."""
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable image")
    return extract_frame_features(image)


def extract_frame_features(image: np.ndarray) -> np.ndarray:
    """Create the same feature vector from an OpenCV BGR image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32).flatten() / 255.0


def load_dataset(dataset_dir: Path):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    class_dirs = sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if len(class_dirs) < 2:
        raise ValueError("Dataset must have at least two class folders.")

    features, labels = [], []
    class_names = [path.name for path in class_dirs]
    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_features = []
        for image_path in image_paths:
            try:
                class_features.append(extract_features(image_path))
            except ValueError:
                print(f"WARNING: Skipping unreadable image in {class_dir.name}")

        if len(class_features) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' needs at least 3 readable images."
            )
        features.extend(class_features)
        labels.extend([label] * len(class_features))

    return np.stack(features), np.asarray(labels), class_names


def make_models():
    """The six classifiers from the original project."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=SEED),
        "KNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=SEED)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def train(dataset_dir: Path):
    X, y, class_names = load_dataset(dataset_dir)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=SEED,
    )

    results = []
    fitted_models = {}
    for name, model in make_models().items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        row = {
            "Model": name,
            "Validation Accuracy (%)": round(
                accuracy_score(y_valid, predictions) * 100, 2
            ),
            "Validation F1-score (%)": round(
                f1_score(y_valid, predictions, average="macro") * 100, 2
            ),
            "Training Time (seconds)": round(time.perf_counter() - started, 3),
        }
        results.append(row)
        fitted_models[name] = model

    results.sort(
        key=lambda row: (
            row["Validation Accuracy (%)"],
            row["Validation F1-score (%)"],
        ),
        reverse=True,
    )
    best_name = results[0]["Model"]
    best_model = fitted_models[best_name]
    predictions = best_model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "class_names": class_names,
                "image_size": list(IMAGE_SIZE),
                "test_accuracy": test_accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (MODEL_DIR / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Images: {len(X)} | Classes: {', '.join(class_names)}")
    print(
        f"Train: {len(X_train)} | Validation: {len(X_valid)} | "
        f"Test: {len(X_test)}"
    )
    print("\nValidation results:")
    for row in results:
        print(
            f"{row['Model']}: accuracy "
            f"{row['Validation Accuracy (%)']:.2f}%"
        )
    print(f"\nSelected model: {best_name}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Saved model: models/best_model.joblib")


def predict(image_path: Path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {image_path.suffix}")

    model_path = MODEL_DIR / "best_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run 'python object_classifier.py train' first.")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label = int(model.predict(extract_features(image_path).reshape(1, -1))[0])
    print(f"Predicted object: {metadata['class_names'][label]}")


def webcam(camera_index: int):
    model_path = MODEL_DIR / "best_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run 'python object_classifier.py train' first.")

    model = joblib.load(model_path)
    class_names = json.loads(metadata_path.read_text(encoding="utf-8"))[
        "class_names"
    ]
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check webcam permissions."
        )

    print("Webcam classifier running. Show an object to the camera; press q to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            label = int(
                model.predict(extract_frame_features(frame).reshape(1, -1))[0]
            )
            cv2.putText(
                frame,
                f"Prediction: {class_names[label]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Object Classifier - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify objects in images.")
    parser.add_argument("command", choices=["train", "predict", "webcam"])
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    args = parser.parse_args()
    if args.command == "train":
        train(args.dataset)
    elif args.command == "webcam":
        webcam(args.camera)
    elif args.image is None:
        parser.error("predict requires an image path.")
    else:
        predict(args.image)
