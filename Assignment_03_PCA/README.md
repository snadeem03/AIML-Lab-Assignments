# Assignment 03 — PCA Dimensionality Reduction on Student Features

## Aim
Apply Principal Component Analysis to encoded/standardized student features and
determine how many components retain 95% of variance, with scree/cumulative plots,
PC1-vs-PC2 visualization and loading interpretation.

## Dataset
- **Source (official UCI):** `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`
- **File used:** `student-por.csv` (649 × 33, `;`-separated), loaded reproducibly in the notebook.
- **Features:** all columns except `G1`, `G2`, `G3` (early-prediction setup; grades excluded from PCA inputs).
- Numerical features standardized (`StandardScaler`); categoricals one-hot encoded (`OneHotEncoder`) via `ColumnTransformer`.

## Algorithm
- `sklearn.decomposition.PCA(random_state=42)` on the processed matrix.
- Explained variance ratio, eigenvalues, cumulative variance; `n95` = smallest number of components with cumulative variance ≥ 0.95.
- PC1-vs-PC2 scatter coloured by held-out `Pass = (G3 >= 10)`; loadings = `pca.components_`.

## Procedure
1. Open `AIML_Assignment_03_PCA.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook downloads UCI data, preprocesses, fits PCA, writes variance tables and all plots to `outputs/`.

## Outputs
- `outputs/explained_variance.csv` — per-component variance ratio, cumulative variance, eigenvalue.
- `outputs/scree_cumulative.png` — scree plot + cumulative variance with 95% line and n95 marker.
- `outputs/pc1_pc2.png` — PC1 vs PC2 scatter (Pass/Fail colour).
- `outputs/pca_loadings_pc1_pc5.csv` — loadings for PC1–PC5.
- `outputs/pca_loadings.png` — top-10 absolute loadings for PC1 and PC2.

## Result
All variance numbers and `n95` are computed at runtime; see the executed notebook for values.
