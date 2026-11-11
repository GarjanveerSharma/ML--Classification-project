import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "clean_classification_dataset" / "features"
MODEL_DIR = FEATURE_DIR / "models"
IMAGE_SIZE = (64, 64)


def load_split(name):
    path = FEATURE_DIR / f"{name}_features.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Run feature_extraction.py first: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["X"], data["y"], data["class_names"].tolist()


def models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "KNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def features_from_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32).flatten().reshape(1, -1) / 255.0


def predict_image(path):
    metadata_path = MODEL_DIR / "model_metadata.json"
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Train a model first: python train_models.py")
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label = int(model.predict(features_from_frame(frame))[0])
    print(f"Predicted object: {metadata['class_names'][label]}")


def webcam(camera_index):
    metadata_path = MODEL_DIR / "model_metadata.json"
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Train a model first: python train_models.py")
    model = joblib.load(model_path)
    classes = json.loads(metadata_path.read_text(encoding="utf-8"))["class_names"]
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam {camera_index}.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from webcam.")
            label = int(model.predict(features_from_frame(frame))[0])
            cv2.putText(
                frame,
                f"Prediction: {classes[label]}",
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


def train():
    X_train, y_train, classes = load_split("train")
    X_valid, y_valid, valid_classes = load_split("valid")
    X_test, y_test, test_classes = load_split("test")
    if classes != valid_classes or classes != test_classes:
        raise ValueError("Class order differs across dataset splits.")

    results, fitted = [], {}
    for name, model in models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        predicted = model.predict(X_valid)
        results.append(
            {
                "Model": name,
                "Validation Accuracy (%)": round(
                    accuracy_score(y_valid, predicted) * 100, 2
                ),
                "Validation F1-score (%)": round(
                    f1_score(y_valid, predicted, average="macro") * 100, 2
                ),
                "Training Time (seconds)": round(time.perf_counter() - start, 3),
            }
        )
        fitted[name] = model

    results.sort(
        key=lambda row: (
            row["Validation Accuracy (%)"],
            row["Validation F1-score (%)"],
        ),
        reverse=True,
    )
    best_name = results[0]["Model"]
    best_model = fitted[best_name]
    predicted = best_model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predicted))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with (MODEL_DIR / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with (MODEL_DIR / "final_test_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["Model", "Test Accuracy (%)", "Test images"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Model": best_name,
                "Test Accuracy (%)": round(accuracy * 100, 2),
                "Test images": len(y_test),
            }
        )
    with (MODEL_DIR / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
        report = classification_report(
            y_test,
            predicted,
            labels=range(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        for name in classes:
            row = report[name]
            writer.writerow(
                [name, row["precision"], row["recall"], row["f1-score"], row["support"]]
            )
    with (MODEL_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual class", *classes])
        for name, row in zip(classes, confusion_matrix(y_test, predicted, labels=range(len(classes)))):
            writer.writerow([name, *row])

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "class_names": classes,
                "image_size": list(IMAGE_SIZE),
                "test_accuracy": accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in results:
        print(f"{row['Model']}: validation accuracy {row['Validation Accuracy (%)']:.2f}%")
    print(f"\nSelected model: {best_name}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("Saved model: clean_classification_dataset/features/models/best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use the object classifier.")
    parser.add_argument("command", nargs="?", choices=("train", "predict", "webcam"), default="train")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "webcam":
        webcam(args.camera)
    elif args.image is None:
        parser.error("predict requires an image path.")
    else:
        predict_image(args.image)
