# Assignment 02 — K-Means Clustering of Student Data

## Aim
Apply unsupervised K-Means clustering to demographic, academic, social and lifestyle
features of the UCI Student Performance dataset and profile the discovered groups.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 rows × 33 columns, separator `;`), loaded reproducibly in the notebook.
- **Features:** all columns except `G1`, `G2`, `G3` (excluded to avoid leakage; `G3` is used only as a held-out profiling variable).
- Numerical columns are scaled; categorical columns are one-hot encoded.

## Algorithm
- `sklearn.compose.ColumnTransformer` + `sklearn.preprocessing.StandardScaler` + `OneHotEncoder`.
- `sklearn.cluster.KMeans` with `n_init=10`, `random_state=42`, K = 2…10 scanned.
- Model selection: elbow method (inertia vs K) and `sklearn.metrics.silhouette_score`.
- Final model: K with maximum silhouette score. 2-D visualization via `sklearn.decomposition.PCA`.

## Procedure
1. Open `AIML_Assignment_02_Clustering.ipynb` in Colab/Jupyter and `Run all`.
2. Notebook downloads the UCI archive, builds the preprocessing pipeline, scans K, plots elbow/silhouette, fits the best-K model, and writes cluster profiles + PCA scatter.
3. Outputs are written to `outputs/`.

## Outputs
- `outputs/k_metrics.csv` — inertia and silhouette per K.
- `outputs/elbow_silhouette.png` — elbow curve + silhouette bar chart.
- `outputs/cluster_sizes.csv` — counts per cluster.
- `outputs/cluster_profiles_numeric.csv` — numeric feature means per cluster (plus held-out G3 mean).
- `outputs/clusters_pca2d.png` — 2-D PCA scatter coloured by cluster.

## Result
Best-K, silhouette scores, sizes and profiles are computed at runtime; see the executed notebook for values and interpretation.
