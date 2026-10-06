# Assignment 07 — Support Vector Regression for Grade Prediction

## Aim
Predict continuous `G3` with tuned Support Vector Regression and compare against Linear Regression.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly.
- **Target:** `G3`; **inputs:** all columns except `G1`, `G2`, `G3`.

## Algorithm
- `Pipeline(ColumnTransformer(StandardScaler + OneHotEncoder) + SVR)`.
- `GridSearchCV(cv=5)` over `kernel ∈ {rbf, linear}`, `C ∈ {0.5, 1.0, 5.0}`, `epsilon ∈ {0.1, 0.5}`, `gamma ∈ {scale, auto}`; baseline `LinearRegression` in the same preprocessing.
- Metrics: MAE, RMSE, R²; plots: actual-vs-predicted, error comparison bars.

## Procedure
1. Open `AIML_Assignment_07_SVR.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads data, grid-searches SVR, evaluates both models, writes `outputs/`.

## Outputs
- `outputs/best_params.csv` — best SVR hyperparameters.
- `outputs/comparison.csv` — MAE/RMSE/R² for SVR vs Linear Regression.
- `outputs/actual_vs_pred.png`, `outputs/comparison.png` — diagnostic graphs.

## Result
All values computed at runtime; see executed notebook.
