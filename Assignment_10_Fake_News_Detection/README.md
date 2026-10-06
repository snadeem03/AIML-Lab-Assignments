# Assignment 10 — Multimodal Fake-News Detection (Text + Image)

## Aim
Design a multimodal fake-news classifier with text-only, image-only and fused
baselines, comparing accuracy/precision/recall/F1 with confusion matrices.

## Dataset
- **In-notebook synthetic set (default, Colab-ready):** 1000 items, balanced
  real/fake, `seed=42`. Text headlines from class-conditioned vocabularies;
  32×32 RGB images with class-correlated tint + noise. Generated at runtime —
  no download, no committed data.
- **Real-dataset format (to swap in):** CSV with `text` (string), `image_path`
  (or `image` array) and `label` (`0`=real, `1`=fake), e.g. Fakeddit
  (`text`, `image_url`, `label`) or FakeNewsNet. Resize images to 32×32, flatten,
  and reuse the same TF-IDF + scaling + fusion pipeline documented in the notebook.

## Method
- **Text preprocessing:** lowercase, strip non-letters, collapse whitespace.
- **Image preprocessing:** resize → 32×32×3, flatten to 3072, `StandardScaler`.
- **Text features:** `TfidfVectorizer(max_features=800, ngram(1,2), stop_words=english)`.
- **Image features:** scaled pixel vectors.
- **Models:** `LogisticRegression` × 3 — text-only, image-only, fusion
  (`hstack([TF-IDF, pixels])`). Stratified 80/20 split, `random_state=42`.

## Procedure
1. Open `AIML_Assignment_10_Fake_News_Detection.ipynb` in Colab/Jupyter, `Run all`.
2. Notebook generates data, trains all three baselines, writes `outputs/`.
3. To use real data, replace the generator cell with a loader producing the same
   `(texts, images, labels)` variables (format documented in notebook §2).

## Outputs
- `outputs/metrics.csv` — accuracy/precision/recall/F1 per model.
- `outputs/confusion_matrices.png` — all three confusion matrices.
- `outputs/comparison.png` — metric comparison bars.
- `outputs/sample_images.png` — synthetic image samples.

## Result & ethics
Fusion is expected to outperform single modalities on the synthetic set (exact
numbers computed at runtime). This is a lab demo only — see the notebook's
ethical-limitations section before any real-world use (no automated censorship,
human review required, bias/privacy caveats).
