import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "clean_classification_dataset" / "features" / "models"


def main():
    metadata_path = MODEL_DIR / "model_metadata.json"
    comparison_path = MODEL_DIR / "model_comparison.csv"
    results_path = MODEL_DIR / "final_test_results.csv"
    for path in (metadata_path, comparison_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run train_models.py first: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Test accuracy: {metadata['test_accuracy']:.2%}")
    print(f"Classes: {', '.join(metadata['class_names'])}")
    print("\nValidation results:")
    with comparison_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            print(f"{row['Model']}: {row['Validation Accuracy (%)']}%")
    print("\nReports and model saved in clean_classification_dataset/features/models/")


if __name__ == "__main__":
    main()
