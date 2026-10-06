# Assignment 01 — Association Between Dependent and Independent Variables

## Aim
Identify the association between independent variables (`studytime`, `failures`, `absences`)
and the dependent variable `G3` (final grade, 0–20) in the UCI Student Performance dataset.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (Portuguese language course, 649 rows × 33 columns, separator `;`)
- **Loading:** reproducible download in the notebook via `urllib` + `zipfile` + `pandas.read_csv(sep=';')`. No dataset file is committed.
- **Target:** `G3` (final grade 0–20).
- **Inputs for this assignment:** `studytime` (weekly study time ordinal 1–4), `failures` (number of past class failures), `absences` (number of school absences).

## Methods / Algorithms
- Descriptive statistics (mean, std, min/max, quartiles, variance, skew, kurtosis).
- Pearson correlation `scipy.stats.pearsonr` + `DataFrame.corr(method='pearson')`.
- Spearman rank correlation `scipy.stats.spearmanr` + `DataFrame.corr(method='spearman')`.
- Seaborn correlation heatmaps (Pearson and Spearman).
- Scatter plots with OLS trend lines (`seaborn.regplot`).
- Single-feature OLS slopes via `sklearn.linear_model.LinearRegression`.

## Procedure
1. Open `AIML_Assignment_01_Association.ipynb` in Google Colab (`File → Upload notebook`) or Jupyter.
2. `Runtime → Run all`. The notebook downloads `student.zip` from the UCI archive and extracts `student-por.csv` in memory.
3. Cells run: imports → loading → preview → selection → descriptive stats → Pearson/Spearman → heatmap → scatter+trend → OLS slopes → computed interpretation.
4. All figures/tables are written to `outputs/` when the notebook runs (both locally and in Colab).

## Outputs
- Notebook: `AIML_Assignment_01_Association.ipynb` (executed with outputs).
- `outputs/descriptive_statistics.csv` — full descriptive table.
- `outputs/pearson_spearman_vs_G3.csv` — Pearson r/p and Spearman rho/p vs G3.
- `outputs/correlation_heatmap.png` — Pearson and Spearman heatmaps.
- `outputs/scatter_trends.png` — scatter plots with regression trend lines for each independent variable vs G3.

## Result
Correlations are computed at runtime and interpreted in the notebook (no invented values).
Expected pattern: `studytime` positively associated with `G3`; `failures` and `absences` negatively associated — exact magnitudes, p-values and slopes are printed by the notebook.
