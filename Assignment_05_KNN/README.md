# Assignment 05 — KNN Student Performance Classification

## Aim
Classify Pass/Fail (`G3 >= 10`) with K-Nearest Neighbours, selecting K by cross-validation.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly.
- **Target:** `Pass = 1 if G3 >= 10 else 0`; inputs exclude `G1`/`G2`/`G3`.

## Algorithm
- `ColumnTransformer(StandardScaler + OneHotEncoder)` + `KNeighborsClassifier`.
- K scanned over odd values 1–29; best-K = max 5-fold stratified CV accuracy (`StratifiedKFold`, `random_state=42`).
- Final evaluation: train/test accuracy, classification report, confusion matrix, ROC-AUC.

## Procedure
1. Open `AIML_Assignment_05_KNN.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads data, scans K, plots accuracy-vs-K, fits best-K model, writes `outputs/`.

## Outputs
- `outputs/k_scan.csv` — CV/train/test accuracy per K.
- `outputs/accuracy_vs_k.png` — training/testing/CV accuracy graph.
- `outputs/confusion_roc.png` — confusion matrix + ROC curve for best K.

## Result
Best-K and all metrics computed at runtime; see executed notebook.
