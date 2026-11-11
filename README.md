# Simple Object Image Classifier

The source dataset is `images/`: one object class per folder. `prepare_dataset.py`
creates matching `train`, `valid`, and `test` folders in
`clean_classification_dataset/`. The old unrelated split is preserved in a
timestamped backup. Exact duplicate images are kept in the same split so they
cannot inflate the test score through train/test leakage.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run the six project steps

```powershell
python prepare_dataset.py
python rename_images.py
python annotate_images.py
python feature_extraction.py
python train_models.py
python performance_evaluation.py
```

Training compares only the original six algorithms: Logistic Regression,
Decision Tree, KNN, Random Forest, SVM, and Gaussian Naive Bayes. The best
validation model and test results are saved under
`clean_classification_dataset/features/models/`.

## Classify

```powershell
python train_models.py predict "images\Bottles\your-image.jpeg"
python train_models.py webcam
```

Press `q` to close the webcam window. The classifier predicts among the classes
in this dataset; it does not detect or reject unknown objects.
