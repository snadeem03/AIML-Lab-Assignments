# Assignment 09 — Object Detection in CCTV Footage (YOLO)

## Aim
Detect objects in user-provided **authorized** CCTV footage with a pretrained
Ultralytics YOLO model; report confidences, counts, an annotated frame, a count
graph and processed-video instructions.

## Data / Model
- **Input:** user-provided authorized video path (`VIDEO_PATH` env or notebook variable).
  In Google Colab the notebook prompts for upload (`google.colab.files.upload`).
  The video is **never committed**.
- **Model:** pretrained `yolov8n.pt` via `ultralytics` (`YOLO("yolov8n.pt")`, auto-cached).
  Model weights (`*.pt`) are **never committed**.
- **Fallback:** if no video / no `ultralytics` is available (e.g. offline validation),
  the notebook runs a reproducible seeded synthetic CCTV-style demo (seed 42) so it
  always executes and still produces all required artifact types.

## Method
- Video validation with `cv2.VideoCapture` (existence, readability, fps/frame-count/resolution).
- Frame-by-frame `model.predict(conf=0.25)` → per-box `(class, confidence, xyxy)`.
- Aggregation: per-class counts, mean confidences, per-frame count table.
- Matplotlib annotated sample frame + per-frame object-count graph.

## Procedure
1. Open `AIML_Assignment_09_Object_Detection.ipynb` in Colab.
2. `pip install ultralytics opencv-python` (first cell guidance), upload authorized clip,
   set `VIDEO_PATH`, `Run all`.
3. For a processed video: set `SAVE_PROCESSED_VIDEO=True` / use `model.track(..., save=True)`,
   then download the mp4 via `files.download(...)` — do not `git add` it.

## Outputs
- `outputs/detections.csv` — one row per detection (frame, class, confidence, bbox).
- `outputs/per_frame_counts.csv` — per-frame × class count matrix.
- `outputs/annotated_frame.png` — sample frame with boxes + confidence labels.
- `outputs/object_counts.png` — total + per-class detections per frame.

## Result & ethics
Use only footage you are authorized to process. Committed artifacts are PNG/CSV only —
no videos, no weights.
