# AfterMath — Design

## Overview

AfterMath is a hurricane building-damage classifier. Given a before/after
satellite image pair of a single building, it predicts damage severity
(`no-damage` / `minor-damage` / `major-damage` / `destroyed`) and shows
Grad-CAM heatmaps explaining the prediction. This is a single, tightly
scoped build (no multi-phase roadmap) delivered as notebooks + a results
README — no live web demo.

Built on **xBD** (xView2 challenge dataset, real disaster satellite
imagery), scoped to four hurricane events: Harvey, Florence, Matthew,
Michael.

## Data Pipeline

**Source**: xBD, downloaded from xview2.org (free registration required).
Only the four hurricane events are pulled — not the full multi-disaster
dataset.

**Building crops**: xBD provides, per image, a JSON of building polygons
with a damage label (`no-damage`, `minor-damage`, `major-damage`,
`destroyed`, `un-classified`). For each labeled building:
- Crop the polygon's bounding region from both the pre- and
  post-disaster image for that location.
- Drop any building labeled `un-classified`.
- Resize both crops to a fixed size (224×224).
- Store the pair (pre-crop, post-crop, label) as one training example.

**Class imbalance**: `no-damage` dominates most storms. Handled via
class-weighted loss (weights computed from training-set class
frequencies), not by discarding data.

**Split — by storm, not randomly**: train on Harvey + Florence + Matthew;
hold out Michael entirely as the test set. This avoids leaking
near-duplicate buildings across train/test (a risk with random crop-level
splitting) and evaluates true generalization to an unseen disaster event,
mirroring the time-based holdout approach already used in WorldCup.

**Scale target**: subsample to a manageable size — a rough target of a
few thousand crops per class, capped so training finishes in reasonable
time on a free Kaggle/Colab GPU.

## Model Architecture

**Type**: Siamese dual-branch network.

**Backbone**: A pretrained CNN (ResNet18 or EfficientNet-B0), fine-tuned
end-to-end from ImageNet weights — same transfer-learning pattern used in
IgnisEye. **One shared set of weights**, applied twice: once to the
pre-crop, once to the post-crop, producing two feature vectors.

**Merge**: Concatenate the pre- and post-feature vectors (not subtract).
Concatenation preserves both "what it looked like before" and "what it
looks like after" for the classifier, rather than collapsing them into a
single difference signal.

**Head**: A small fully-connected layer on the concatenated vector,
softmax over the 4 damage classes.

**Loss**: Weighted cross-entropy, using class weights from the training
set.

## Training & Evaluation

**Training**: Adam optimizer, full end-to-end fine-tuning (backbone not
frozen). Data augmentation: random horizontal/vertical flips and
90°-rotations (safe for satellite imagery, which has no canonical
orientation). Early stopping on validation performance rather than a
fixed epoch count.

**Evaluation**: Headline metric is **macro-F1** (not raw accuracy) —
given the `no-damage` class imbalance, raw accuracy could look
deceptively good from majority-class guessing, while macro-F1 weights all
4 classes equally. Also report a confusion matrix and per-class
precision/recall. All test-set numbers come from the held-out Hurricane
Michael storm specifically, reported separately from validation
performance (computed on the Harvey/Florence/Matthew split).

## Explainability (Grad-CAM)

For sample predictions, compute Grad-CAM on **both** branches:
- Post-disaster crop: shows what damage evidence drove the prediction.
- Pre-disaster crop: sanity-checks that the model isn't picking up
  spurious cues (lighting, vegetation, etc.) from the "before" image and
  miscalling it damage.

Both heatmaps are overlaid on their respective crops and shown side by
side with the predicted class and confidence. These are saved as static
figures for the README/notebooks, not wired into a live app.

## Repo Structure

```
AfterMath/
├── README.md              # results, setup, tech stack
├── LICENSE                # MIT
├── requirements.txt / requirements-dev.txt
├── config.yaml            # model/data config (backbone choice, paths, hyperparams)
├── notebooks/
│   ├── 01_eda.ipynb              # class distribution per storm, sample pairs
│   ├── 02_data_prep.ipynb        # parse xBD labels, crop, resize, save processed dataset
│   ├── 03_model.ipynb            # define + train the siamese model
│   ├── 04_evaluation.ipynb       # held-out storm test metrics, confusion matrix
│   └── 05_gradcam.ipynb          # explainability visualizations
├── utils/                  # shared code: dataset class, model class, training loop
├── data/                   # xBD subset (gitignored; download instructions in README)
├── models/                 # saved weights (gitignored; download link in README)
├── docs/superpowers/specs/
└── tests/
```

This mirrors the conventions already used in PitchPulse, IgnisEye, and
WorldCup. AfterMath is a separate, standalone project/repo — no existing
project's files are touched as part of this work.

## Testing

Tests cover deterministic logic, not model training itself:
- **Data pipeline**: polygon-to-crop cropping produces correctly-sized
  outputs; `un-classified` labels are filtered out; class-weight
  computation is correct given a set of label counts.
- **Model**: forward pass produces correctly-shaped output for a batch;
  the concatenation/merge step produces the expected feature dimension.

Training convergence and accuracy are judged via notebook metrics
(confusion matrix, macro-F1), not unit tests.
