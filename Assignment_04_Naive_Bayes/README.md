# Assignment 04 — Naive Bayes Classification of Student Outcomes

## Aim
Classify students as Pass (`G3 >= 10`) or Fail (`G3 < 10`) with Gaussian Naive Bayes
under an early-prediction setup (no `G1`/`G2`/`G3` in inputs).

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly in the notebook.
- **Target:** `Pass = 1 if G3 >= 10 else 0`.
- **Features:** all columns except `G1`, `G2`, `G3`; numerics standardized, categoricals one-hot encoded via `ColumnTransformer`.

## Algorithm
- `sklearn.pipeline.Pipeline(ColumnTransformer + GaussianNB)`.
- Stratified `train_test_split(test_size=0.2, random_state=42)`.
- Metrics: accuracy, precision, recall, F1, classification report, confusion matrix, ROC curve, ROC-AUC.

## Procedure
1. Open `AIML_Assignment_04_Naive_Bayes.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads UCI data, splits stratified, trains GaussianNB, evaluates and writes `outputs/`.

## Outputs
- `outputs/metrics.csv` — accuracy, precision, recall, F1, ROC-AUC.
- `outputs/confusion_roc.png` — confusion matrix heatmap + ROC curve.

## Result
All metric values are computed at runtime; see the executed notebook.
