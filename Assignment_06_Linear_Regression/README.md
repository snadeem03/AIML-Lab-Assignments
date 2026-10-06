# Assignment 06 — Linear Regression for Final Grade Prediction

## Aim
Predict continuous final grade `G3` (0–20) with ordinary Linear Regression under the
early-prediction setup (no `G1`/`G2` in inputs).

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly.
- **Target:** `G3`; **inputs:** all other columns except `G1`, `G2`, `G3`.

## Algorithm
- `ColumnTransformer(StandardScaler + OneHotEncoder)` + `sklearn.linear_model.LinearRegression`.
- 80/20 `train_test_split(random_state=42)`; metrics MAE/MSE/RMSE/R²; coefficient table; residual diagnostics.

## Procedure
1. Open `AIML_Assignment_06_Linear_Regression.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads data, trains, evaluates and writes `outputs/`.

## Outputs
- `outputs/coefficients.csv` — feature coefficients sorted by |effect|.
- `outputs/metrics.csv` — test MAE, MSE, RMSE, R².
- `outputs/diagnostics.png` — actual-vs-predicted, residual plot, residual histogram.

## Result
All values computed at runtime; see executed notebook.
