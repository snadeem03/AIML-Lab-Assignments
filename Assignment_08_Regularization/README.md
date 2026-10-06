# Assignment 08 — Regularized Student Grade Prediction

## Aim
Compare Ordinary Linear Regression against Ridge, Lasso and Elastic Net to show how
regularization controls overfitting when predicting `G3`.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly.
- **Target:** `G3`; **inputs:** all columns except `G1`, `G2`, `G3` (standardized + one-hot encoded).

## Algorithm
- `LinearRegression` (OLS) vs `RidgeCV`, `LassoCV`, `ElasticNetCV` (alphas `logspace(-3,2,30)`, 5-fold CV, `random_state=42`).
- Metrics: train/test R², test MAE/RMSE; coefficient comparison; Lasso zero-count.

## Procedure
1. Open `AIML_Assignment_08_Regularization.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads data, tunes alphas by CV, evaluates and writes `outputs/`.

## Outputs
- `outputs/model_comparison.csv` — train/test R², MAE, RMSE per model.
- `outputs/coefficients.csv` — per-feature coefficients for all four models.
- `outputs/comparison.png` — error/R² comparison bars.
- `outputs/coefficients.png` — top-12 coefficient shrinkage chart.

## Result
Best alphas, scores and sparsity computed at runtime; see executed notebook for the overfitting/regularization explanation.
