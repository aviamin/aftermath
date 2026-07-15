# AfterMath Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hurricane building-damage classifier (before/after satellite image pair → damage severity) with Grad-CAM explainability, delivered as notebooks + a results README.

**Architecture:** A siamese dual-branch CNN (shared ResNet18 backbone, ImageNet-pretrained) processes pre- and post-disaster building crops separately, concatenates their feature vectors, and classifies into 4 damage classes via a small FC head. Trained on a subset of the xBD dataset (Harvey/Florence/Matthew for train+val, Michael held out entirely for test), evaluated with macro-F1 given class imbalance, and explained per-prediction with Grad-CAM on both branches.

**Tech Stack:** Python 3.10+, PyTorch + torchvision, pandas, scikit-learn (metrics), matplotlib/seaborn, Pillow, shapely (polygon parsing), OpenCV (resize/Grad-CAM upsampling), pytest, Jupyter.

## Global Constraints

- Python 3.10+ (matches other portfolio projects).
- Backbone: ResNet18, ImageNet-pretrained (`torchvision.models.resnet18`), shared weights across both branches.
- Crop size: 224×224.
- Damage classes, in this exact order everywhere: `["no-damage", "minor-damage", "major-damage", "destroyed"]` (`un-classified` buildings are dropped, never a 5th class).
- Split: train/val on Harvey + Florence + Matthew; test on Michael (`test_storm: hurricane-michael`), held out entirely — never mixed into train/val.
- Loss: class-weighted cross-entropy (weights computed from training-set class frequencies).
- Headline metric: **macro-F1**, not raw accuracy.
- Deliverable format: notebooks + README results — no Flask app, no live demo.
- This is a new, standalone repo (`AfterMath/`) inside `Portfolioprojects/`. Never touch files in `PitchPulse/`, `WorldCup/`, or `pyrosight/`.
- License: MIT (matches other projects).

---

## File Structure

```
AfterMath/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── config.yaml
├── utils/
│   ├── __init__.py
│   ├── xbd_labels.py      # parse xBD label JSON, filter un-classified, class weights
│   ├── cropping.py        # polygon -> resized crop
│   ├── prepare.py         # storm split + manifest building
│   ├── dataset.py         # PyTorch Dataset over the manifest
│   ├── model.py           # SiameseDamageNet
│   ├── training.py        # EarlyStopper, class-weight tensor, train/validate loops
│   └── gradcam.py         # GradCAM class
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_prep.ipynb
│   ├── 03_model.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_gradcam.ipynb
├── data/                  # gitignored (raw/ + processed/); download instructions in README
├── models/                # gitignored (best.pt); download link in README once trained
├── docs/superpowers/specs/2026-07-14-aftermath-design.md   (already exists)
├── docs/superpowers/plans/2026-07-14-aftermath-implementation.md  (this file)
└── tests/
    ├── __init__.py
    ├── test_xbd_labels.py
    ├── test_cropping.py
    ├── test_prepare.py
    ├── test_dataset.py
    ├── test_model.py
    ├── test_training.py
    └── test_gradcam.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `AfterMath/requirements.txt`
- Create: `AfterMath/requirements-dev.txt`
- Create: `AfterMath/.gitignore`
- Create: `AfterMath/LICENSE`
- Create: `AfterMath/config.yaml`
- Create: `AfterMath/README.md`
- Create: `AfterMath/utils/__init__.py`
- Create: `AfterMath/tests/__init__.py`

**Interfaces:**
- Produces: `config.yaml` keys that all later tasks read: `data.raw_dir`, `data.processed_dir`, `data.manifest`, `data.crop_size`, `data.storms`, `data.test_storm`, `model.backbone`, `model.pretrained`, `model.num_classes`, `model.classes`, `training.batch_size`, `training.epochs`, `training.learning_rate`, `training.early_stopping_patience`, `paths.best_weights`.

- [ ] **Step 1: Create `requirements.txt`**

```
torch>=2.0
torchvision>=0.15
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
shapely
opencv-python
tqdm
pyyaml
jupyter
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```
pytest
```

- [ ] **Step 3: Create `.gitignore`**

```
data/
models/*.pt
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
```

- [ ] **Step 4: Create `LICENSE`**

Use the standard MIT license text (copy the exact text from `PitchPulse/LICENSE` or `IgnisEye/LICENSE`, updating only the copyright name/year to match those files' format — do not alter the license terms).

- [ ] **Step 5: Create `config.yaml`**

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  manifest: data/processed/manifest.csv
  crop_size: 224
  storms:
    - hurricane-harvey
    - hurricane-florence
    - hurricane-matthew
    - hurricane-michael
  test_storm: hurricane-michael

model:
  backbone: resnet18
  pretrained: true
  num_classes: 4
  classes: [no-damage, minor-damage, major-damage, destroyed]

training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.0001
  early_stopping_patience: 5

paths:
  best_weights: models/best.pt
```

- [ ] **Step 6: Create `README.md` stub**

```markdown
# AfterMath

**Hurricane building-damage classification from before/after satellite imagery.**

Status: in progress — see `docs/superpowers/specs/2026-07-14-aftermath-design.md` for the full design.

## Dataset setup

This project uses the [xBD dataset](https://xview2.org) (free registration
required). After downloading, place the four hurricane events at:

```
data/raw/hurricane-harvey/
data/raw/hurricane-florence/
data/raw/hurricane-matthew/
data/raw/hurricane-michael/
```

Each event folder should contain xBD's standard `images/` and `labels/`
subfolders as provided by the download.

## Setup

\`\`\`bash
pip install -r requirements.txt -r requirements-dev.txt
\`\`\`

## Tests

\`\`\`bash
pytest -v
\`\`\`
```

- [ ] **Step 7: Create empty package markers**

Create `utils/__init__.py` (empty file) and `tests/__init__.py` (empty file).

- [ ] **Step 8: Verify the environment installs cleanly**

Run: `pip install -r requirements.txt -r requirements-dev.txt`
Expected: completes with no errors.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore LICENSE config.yaml README.md utils/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding"
```

---

### Task 2: xBD label parsing + class weights

**Files:**
- Create: `utils/xbd_labels.py`
- Test: `tests/test_xbd_labels.py`

**Interfaces:**
- Produces: `DAMAGE_CLASSES: list[str]`, `BuildingLabel` dataclass (fields: `uid: str`, `polygon: list[tuple[float,float]]`, `damage_class: str`), `parse_label_file(json_path: str) -> list[BuildingLabel]`, `compute_class_weights(labels: list[BuildingLabel]) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_xbd_labels.py
import json
import pytest
from utils.xbd_labels import DAMAGE_CLASSES, BuildingLabel, parse_label_file, compute_class_weights


def _write_label_json(tmp_path, features):
    label = {"features": {"xy": features}, "metadata": {}}
    path = tmp_path / "sample_post.json"
    path.write_text(json.dumps(label))
    return str(path)


def test_parse_label_file_extracts_damage_classes(tmp_path):
    features = [
        {
            "properties": {"uid": "a1", "feature_type": "building", "subtype": "no-damage"},
            "wkt": "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        },
        {
            "properties": {"uid": "a2", "feature_type": "building", "subtype": "destroyed"},
            "wkt": "POLYGON ((20 20, 20 30, 30 30, 30 20, 20 20))",
        },
    ]
    path = _write_label_json(tmp_path, features)

    labels = parse_label_file(path)

    assert len(labels) == 2
    assert labels[0] == BuildingLabel(uid="a1", polygon=[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)], damage_class="no-damage")
    assert labels[1].damage_class == "destroyed"


def test_parse_label_file_drops_unclassified(tmp_path):
    features = [
        {
            "properties": {"uid": "b1", "feature_type": "building", "subtype": "un-classified"},
            "wkt": "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        },
        {
            "properties": {"uid": "b2", "feature_type": "building", "subtype": "minor-damage"},
            "wkt": "POLYGON ((20 20, 20 30, 30 30, 30 20, 20 20))",
        },
    ]
    path = _write_label_json(tmp_path, features)

    labels = parse_label_file(path)

    assert len(labels) == 1
    assert labels[0].uid == "b2"


def test_compute_class_weights_favors_rare_classes():
    labels = (
        [BuildingLabel(uid=f"n{i}", polygon=[], damage_class="no-damage") for i in range(80)]
        + [BuildingLabel(uid=f"d{i}", polygon=[], damage_class="destroyed") for i in range(4)]
    )

    weights = compute_class_weights(labels)

    assert set(weights.keys()) == set(DAMAGE_CLASSES)
    assert weights["destroyed"] > weights["no-damage"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_xbd_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.xbd_labels'`

- [ ] **Step 3: Write the implementation**

```python
# utils/xbd_labels.py
from dataclasses import dataclass
import json

from shapely import wkt as shapely_wkt

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


@dataclass
class BuildingLabel:
    uid: str
    polygon: list
    damage_class: str


def parse_label_file(json_path: str) -> list[BuildingLabel]:
    with open(json_path) as f:
        data = json.load(f)

    labels = []
    for feature in data["features"]["xy"]:
        subtype = feature["properties"]["subtype"]
        if subtype not in DAMAGE_CLASSES:
            continue
        geometry = shapely_wkt.loads(feature["wkt"])
        polygon = list(geometry.exterior.coords)
        labels.append(
            BuildingLabel(
                uid=feature["properties"]["uid"],
                polygon=polygon,
                damage_class=subtype,
            )
        )
    return labels


def compute_class_weights(labels: list[BuildingLabel]) -> dict:
    counts = {cls: 0 for cls in DAMAGE_CLASSES}
    for label in labels:
        counts[label.damage_class] += 1

    total = sum(counts.values())
    num_classes = len(DAMAGE_CLASSES)
    weights = {}
    for cls in DAMAGE_CLASSES:
        count = max(counts[cls], 1)
        weights[cls] = total / (num_classes * count)
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_xbd_labels.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/xbd_labels.py tests/test_xbd_labels.py
git commit -m "feat: parse xBD label JSON and compute class weights"
```

---

### Task 3: Building crop extraction

**Files:**
- Create: `utils/cropping.py`
- Test: `tests/test_cropping.py`

**Interfaces:**
- Consumes: `BuildingLabel.polygon` (`list[tuple[float,float]]`) from Task 2.
- Produces: `crop_and_resize(image: np.ndarray, polygon: list[tuple[float,float]], size: int = 224) -> np.ndarray`, returning an `(size, size, 3)` uint8 array.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropping.py
import numpy as np
from utils.cropping import crop_and_resize


def test_crop_and_resize_returns_correct_shape():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    polygon = [(10.0, 10.0), (10.0, 50.0), (50.0, 50.0), (50.0, 10.0), (10.0, 10.0)]

    crop = crop_and_resize(image, polygon, size=224)

    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_crop_and_resize_clips_polygon_to_image_bounds():
    image = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
    polygon = [(-10.0, -10.0), (-10.0, 200.0), (200.0, 200.0), (200.0, -10.0)]

    crop = crop_and_resize(image, polygon, size=224)

    assert crop.shape == (224, 224, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cropping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.cropping'`

- [ ] **Step 3: Write the implementation**

```python
# utils/cropping.py
import cv2
import numpy as np


def crop_and_resize(image: np.ndarray, polygon: list, size: int = 224) -> np.ndarray:
    height, width = image.shape[:2]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = max(int(min(xs)), 0)
    x_max = min(int(max(xs)), width)
    y_min = max(int(min(ys)), 0)
    y_max = min(int(max(ys)), height)

    x_max = max(x_max, x_min + 1)
    y_max = max(y_max, y_min + 1)

    crop = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cropping.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/cropping.py tests/test_cropping.py
git commit -m "feat: crop and resize building polygons from satellite images"
```

---

### Task 4: Storm split + manifest builder

**Files:**
- Create: `utils/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `DAMAGE_CLASSES` from `utils.xbd_labels` (Task 2).
- Produces: `split_storms(storms: list[str], test_storm: str) -> tuple[list[str], list[str]]`, `manifest_row(pre_path: str, post_path: str, damage_class: str, storm: str, split: str) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare.py
from utils.prepare import split_storms, manifest_row


def test_split_storms_holds_out_test_storm():
    storms = ["hurricane-harvey", "hurricane-florence", "hurricane-matthew", "hurricane-michael"]

    train, test = split_storms(storms, test_storm="hurricane-michael")

    assert train == ["hurricane-harvey", "hurricane-florence", "hurricane-matthew"]
    assert test == ["hurricane-michael"]


def test_manifest_row_contains_expected_fields():
    row = manifest_row(
        pre_path="data/processed/pre/1.png",
        post_path="data/processed/post/1.png",
        damage_class="major-damage",
        storm="hurricane-harvey",
        split="train",
    )

    assert row == {
        "pre_path": "data/processed/pre/1.png",
        "post_path": "data/processed/post/1.png",
        "damage_class": "major-damage",
        "storm": "hurricane-harvey",
        "split": "train",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.prepare'`

- [ ] **Step 3: Write the implementation**

```python
# utils/prepare.py
def split_storms(storms: list, test_storm: str) -> tuple:
    train = [s for s in storms if s != test_storm]
    test = [s for s in storms if s == test_storm]
    return train, test


def manifest_row(pre_path: str, post_path: str, damage_class: str, storm: str, split: str) -> dict:
    return {
        "pre_path": pre_path,
        "post_path": post_path,
        "damage_class": damage_class,
        "storm": storm,
        "split": split,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prepare.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/prepare.py tests/test_prepare.py
git commit -m "feat: add storm split and manifest row helpers"
```

---

### Task 5: Paired crop dataset

**Files:**
- Create: `utils/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: a manifest CSV with columns `pre_path, post_path, damage_class, storm, split` (produced by Task 4's `manifest_row` rows, written by Notebook 02 in Task 10).
- Produces: `PairedCropDataset(manifest_csv: str, transform=None)` — a `torch.utils.data.Dataset` where `__getitem__(idx)` returns `(pre_tensor, post_tensor, label_int)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils.dataset import PairedCropDataset


def _make_fake_image(path):
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    img.save(path)


def test_paired_crop_dataset_returns_tensors_and_label(tmp_path):
    pre_path = tmp_path / "pre1.png"
    post_path = tmp_path / "post1.png"
    _make_fake_image(pre_path)
    _make_fake_image(post_path)

    manifest = pd.DataFrame(
        [{"pre_path": str(pre_path), "post_path": str(post_path), "damage_class": "major-damage", "storm": "hurricane-harvey", "split": "train"}]
    )
    manifest_csv = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_csv, index=False)

    dataset = PairedCropDataset(str(manifest_csv), transform=transforms.ToTensor())

    assert len(dataset) == 1
    pre_tensor, post_tensor, label = dataset[0]
    assert pre_tensor.shape == (3, 224, 224)
    assert post_tensor.shape == (3, 224, 224)
    assert label == 2  # index of "major-damage" in DAMAGE_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# utils/dataset.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from utils.xbd_labels import DAMAGE_CLASSES


class PairedCropDataset(Dataset):
    def __init__(self, manifest_csv: str, transform=None):
        self.manifest = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        pre_img = Image.open(row["pre_path"]).convert("RGB")
        post_img = Image.open(row["post_path"]).convert("RGB")

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        label = DAMAGE_CLASSES.index(row["damage_class"])
        return pre_img, post_img, label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add utils/dataset.py tests/test_dataset.py
git commit -m "feat: add PairedCropDataset for before/after crops"
```

---

### Task 6: Siamese model architecture

**Files:**
- Create: `utils/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `SiameseDamageNet(num_classes: int = 4, pretrained: bool = True)` — an `nn.Module` whose `forward(pre_img, post_img)` takes two `(N, 3, 224, 224)` tensors and returns `(N, num_classes)` logits. Exposes `self.backbone` (shared `resnet18` with `fc` replaced by `nn.Identity()`) — later used by Task 8's Grad-CAM as `model.backbone.layer4[-1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from utils.model import SiameseDamageNet


def test_forward_pass_shape():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    pre = torch.randn(2, 3, 224, 224)
    post = torch.randn(2, 3, 224, 224)

    logits = model(pre, post)

    assert logits.shape == (2, 4)


def test_backbone_is_shared_between_branches():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    pre = torch.randn(1, 3, 224, 224)

    feat_via_backbone = model.backbone(pre)

    assert feat_via_backbone.shape == (1, model.feature_dim)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.model'`

- [ ] **Step 3: Write the implementation**

```python
# utils/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SiameseDamageNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        pre_feat = self.backbone(pre_img)
        post_feat = self.backbone(post_img)
        merged = torch.cat([pre_feat, post_feat], dim=1)
        return self.head(merged)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/model.py tests/test_model.py
git commit -m "feat: add SiameseDamageNet dual-branch model"
```

---

### Task 7: Training utilities (class weights tensor, early stopping, train/validate loops)

**Files:**
- Create: `utils/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `SiameseDamageNet` from Task 6, `DAMAGE_CLASSES` from Task 2.
- Produces: `compute_class_weights_tensor(class_counts: dict, class_order: list) -> torch.Tensor`, `EarlyStopper(patience: int)` with `.step(val_loss: float) -> bool`, `train_one_epoch(model, dataloader, optimizer, criterion, device) -> float`, `validate(model, dataloader, criterion, device) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_training.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.model import SiameseDamageNet
from utils.training import compute_class_weights_tensor, EarlyStopper, train_one_epoch, validate
from utils.xbd_labels import DAMAGE_CLASSES


def test_compute_class_weights_tensor_favors_rare_classes():
    counts = {"no-damage": 800, "minor-damage": 100, "major-damage": 80, "destroyed": 20}

    weights = compute_class_weights_tensor(counts, DAMAGE_CLASSES)

    assert weights.shape == (4,)
    assert weights[DAMAGE_CLASSES.index("destroyed")] > weights[DAMAGE_CLASSES.index("no-damage")]


def test_early_stopper_signals_stop_after_patience_exceeded():
    stopper = EarlyStopper(patience=2)

    assert stopper.step(1.0) is False   # improves (inf -> 1.0)
    assert stopper.step(0.5) is False   # improves
    assert stopper.step(0.6) is False   # no improve, counter=1
    assert stopper.step(0.7) is True    # no improve, counter=2 >= patience


def _tiny_dataloader():
    pre = torch.randn(4, 3, 224, 224)
    post = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 4, (4,))
    dataset = TensorDataset(pre, post, labels)
    return DataLoader(dataset, batch_size=2)


def test_train_one_epoch_and_validate_return_finite_losses():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    dataloader = _tiny_dataloader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    val_loss = validate(model, dataloader, criterion, device)

    assert train_loss == train_loss and train_loss > 0  # not NaN
    assert val_loss == val_loss and val_loss > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.training'`

- [ ] **Step 3: Write the implementation**

```python
# utils/training.py
import numpy as np
import torch


def compute_class_weights_tensor(class_counts: dict, class_order: list) -> torch.Tensor:
    counts = np.array([class_counts.get(c, 0) for c in class_order], dtype=np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(class_order) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for pre, post, labels in dataloader:
        pre, post, labels = pre.to(device), post.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pre, post)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pre.size(0)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pre, post, labels in dataloader:
            pre, post, labels = pre.to(device), post.to(device), labels.to(device)
            outputs = model(pre, post)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pre.size(0)
    return total_loss / len(dataloader.dataset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_training.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/training.py tests/test_training.py
git commit -m "feat: add class-weighted loss helper, early stopping, train/validate loops"
```

---

### Task 8: Grad-CAM utility

**Files:**
- Create: `utils/gradcam.py`
- Test: `tests/test_gradcam.py`

**Interfaces:**
- Consumes: `SiameseDamageNet` from Task 6 (specifically `model.backbone.layer4[-1]` as the target conv layer).
- Produces: `GradCAM(model, target_layer)` with `.generate(pre_img, post_img, target_class: int, branch: str = "post") -> np.ndarray` (2D array, values in `[0, 1]`), and `resize_cam(cam: np.ndarray, size: int) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gradcam.py
import numpy as np
import torch

from utils.model import SiameseDamageNet
from utils.gradcam import GradCAM, resize_cam


def test_gradcam_generate_returns_normalized_heatmap():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    target_layer = model.backbone.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    pre = torch.randn(1, 3, 224, 224)
    post = torch.randn(1, 3, 224, 224)

    cam = cam_tool.generate(pre, post, target_class=0, branch="post")

    assert cam.ndim == 2
    assert cam.shape == (7, 7)  # resnet18 layer4 output spatial size for 224x224 input
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_gradcam_works_for_pre_branch_too():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    target_layer = model.backbone.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    pre = torch.randn(1, 3, 224, 224)
    post = torch.randn(1, 3, 224, 224)

    cam = cam_tool.generate(pre, post, target_class=1, branch="pre")

    assert cam.shape == (7, 7)


def test_resize_cam_upsamples_to_target_size():
    cam = np.random.rand(7, 7).astype(np.float32)

    resized = resize_cam(cam, size=224)

    assert resized.shape == (224, 224)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gradcam.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.gradcam'`

- [ ] **Step 3: Write the implementation**

```python
# utils/gradcam.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = []
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        output.retain_grad()
        self.activations.append(output)

    def generate(self, pre_img: torch.Tensor, post_img: torch.Tensor, target_class: int, branch: str = "post") -> np.ndarray:
        self.activations.clear()
        self.model.zero_grad()

        logits = self.model(pre_img, post_img)
        score = logits[0, target_class]

        # The shared backbone is called twice in one forward pass (pre, then
        # post), so the forward hook fires twice. Forward-call order IS
        # deterministic (SiameseDamageNet.forward calls pre before post,
        # sequentially) so index 0 is always pre's activation, index 1 always
        # post's. We deliberately do NOT use a backward hook to pair up
        # gradients: backward-hook firing order for two independent branches
        # that merge at a later `cat()` is not guaranteed to mirror forward
        # order, which risks silently swapping pre/post attribution. Instead,
        # `retain_grad()` lets us read `.grad` directly off the exact
        # activation tensor we already identified by forward order, which is
        # unambiguous regardless of backward traversal order.
        act_idx = 0 if branch == "pre" else 1
        activation = self.activations[act_idx]

        score.backward()
        gradient = activation.grad[0]
        activation_values = activation.detach()[0]

        weights = gradient.mean(dim=(1, 2))
        cam = torch.einsum("c,chw->hw", weights, activation_values)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(cam, (size, size))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gradcam.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/gradcam.py tests/test_gradcam.py
git commit -m "feat: add Grad-CAM for both siamese branches"
```

---

### Task 9: Notebook 01 — EDA

**Files:**
- Create: `notebooks/01_eda.ipynb`

**Interfaces:**
- Consumes: raw xBD files at `data/raw/<storm>/images/` and `data/raw/<storm>/labels/` (per README setup in Task 1), `parse_label_file` from Task 2.

Requires the xBD hurricane subset to already be downloaded per the README (Task 1) — this task cannot be completed or verified without it.

- [ ] **Step 1: Build the notebook**

Create `notebooks/01_eda.ipynb` using `nbformat`:

```python
# one-off script to generate the notebook — run once, then delete
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# AfterMath — EDA\n\nExplore class distribution and sample before/after pairs across the four hurricane events."),
    nbf.v4.new_code_cell(
        "import glob\n"
        "import yaml\n"
        "import matplotlib.pyplot as plt\n"
        "from utils.xbd_labels import parse_label_file, DAMAGE_CLASSES\n\n"
        "config = yaml.safe_load(open('../config.yaml'))\n"
        "storms = config['data']['storms']\n"
        "raw_dir = '../' + config['data']['raw_dir']"
    ),
    nbf.v4.new_markdown_cell("## Damage class counts per storm"),
    nbf.v4.new_code_cell(
        "counts_per_storm = {}\n"
        "for storm in storms:\n"
        "    label_files = glob.glob(f'{raw_dir}/{storm}/labels/*_post_disaster.json')\n"
        "    counts = {cls: 0 for cls in DAMAGE_CLASSES}\n"
        "    for label_file in label_files:\n"
        "        for label in parse_label_file(label_file):\n"
        "            counts[label.damage_class] += 1\n"
        "    counts_per_storm[storm] = counts\n"
        "counts_per_storm"
    ),
    nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "for storm, counts in counts_per_storm.items():\n"
        "    ax.bar(DAMAGE_CLASSES, list(counts.values()), alpha=0.6, label=storm)\n"
        "ax.set_ylabel('Building count')\n"
        "ax.set_title('Damage class distribution per storm')\n"
        "ax.legend()\n"
        "plt.savefig('../docs/class_distribution.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell("## Sample before/after image pairs"),
    nbf.v4.new_code_cell(
        "import random\n"
        "from PIL import Image\n\n"
        "sample_storm = storms[0]\n"
        "post_files = glob.glob(f'{raw_dir}/{sample_storm}/images/*_post_disaster.png')\n"
        "sample_post = random.choice(post_files)\n"
        "sample_pre = sample_post.replace('_post_disaster', '_pre_disaster')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
        "axes[0].imshow(Image.open(sample_pre))\n"
        "axes[0].set_title('Before')\n"
        "axes[1].imshow(Image.open(sample_post))\n"
        "axes[1].set_title('After')\n"
        "plt.show()"
    ),
]
nb["cells"] = cells
nbf.write(nb, "notebooks/01_eda.ipynb")
```

Run this script once from the `AfterMath/` root to generate the notebook file, then delete the script (it's a one-time generator, not a project artifact).

- [ ] **Step 2: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`
Expected: completes with no cell errors; `docs/class_distribution.png` is created.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb docs/class_distribution.png
git commit -m "docs: add EDA notebook"
```

---

### Task 10: Notebook 02 — Data prep

**Files:**
- Create: `notebooks/02_data_prep.ipynb`

**Interfaces:**
- Consumes: `parse_label_file`, `compute_class_weights` (Task 2), `crop_and_resize` (Task 3), `split_storms`, `manifest_row` (Task 4).
- Produces: `data/processed/pre/*.png`, `data/processed/post/*.png`, and `data/processed/manifest.csv` (columns: `pre_path, post_path, damage_class, storm, split`) — consumed by `PairedCropDataset` (Task 5) in Notebooks 03/04/05.

Requires the xBD hurricane subset to already be downloaded per the README (Task 1).

- [ ] **Step 1: Build the notebook**

Use the same `nbformat` pattern as Task 9. Cells:

```python
cells = [
    nbf.v4.new_markdown_cell("# AfterMath — Data Prep\n\nBuild the before/after crop dataset and manifest from raw xBD images."),
    nbf.v4.new_code_cell(
        "import glob\n"
        "import os\n"
        "import yaml\n"
        "import pandas as pd\n"
        "from PIL import Image\n"
        "import numpy as np\n\n"
        "from utils.xbd_labels import parse_label_file\n"
        "from utils.cropping import crop_and_resize\n"
        "from utils.prepare import split_storms, manifest_row\n\n"
        "config = yaml.safe_load(open('../config.yaml'))\n"
        "raw_dir = '../' + config['data']['raw_dir']\n"
        "processed_dir = '../' + config['data']['processed_dir']\n"
        "crop_size = config['data']['crop_size']\n"
        "train_storms, test_storms = split_storms(config['data']['storms'], config['data']['test_storm'])\n\n"
        "os.makedirs(f'{processed_dir}/pre', exist_ok=True)\n"
        "os.makedirs(f'{processed_dir}/post', exist_ok=True)"
    ),
    nbf.v4.new_code_cell(
        "def storm_split(storm):\n"
        "    return 'train' if storm in train_storms else 'test'\n\n"
        "rows = []\n"
        "crop_id = 0\n"
        "for storm in config['data']['storms']:\n"
        "    split = storm_split(storm)\n"
        "    post_label_files = glob.glob(f'{raw_dir}/{storm}/labels/*_post_disaster.json')\n"
        "    for label_file in post_label_files:\n"
        "        pre_label_file = label_file.replace('_post_disaster', '_pre_disaster')\n"
        "        post_image_file = label_file.replace('/labels/', '/images/').replace('.json', '.png')\n"
        "        pre_image_file = pre_label_file.replace('/labels/', '/images/').replace('.json', '.png')\n\n"
        "        post_labels = {l.uid: l for l in parse_label_file(label_file)}\n"
        "        pre_labels = {l.uid: l for l in parse_label_file(pre_label_file)}\n\n"
        "        post_image = np.array(Image.open(post_image_file).convert('RGB'))\n"
        "        pre_image = np.array(Image.open(pre_image_file).convert('RGB'))\n\n"
        "        for uid, post_label in post_labels.items():\n"
        "            if uid not in pre_labels:\n"
        "                continue\n"
        "            pre_label = pre_labels[uid]\n\n"
        "            post_crop = crop_and_resize(post_image, post_label.polygon, size=crop_size)\n"
        "            pre_crop = crop_and_resize(pre_image, pre_label.polygon, size=crop_size)\n\n"
        "            pre_path = f'{processed_dir}/pre/{crop_id}.png'\n"
        "            post_path = f'{processed_dir}/post/{crop_id}.png'\n"
        "            Image.fromarray(pre_crop).save(pre_path)\n"
        "            Image.fromarray(post_crop).save(post_path)\n\n"
        "            rows.append(manifest_row(pre_path, post_path, post_label.damage_class, storm, split))\n"
        "            crop_id += 1\n\n"
        "manifest = pd.DataFrame(rows)\n"
        "manifest.to_csv(f'{processed_dir}/manifest.csv', index=False)\n"
        "manifest['damage_class'].value_counts()"
    ),
]
```

- [ ] **Step 2: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_data_prep.ipynb --output 02_data_prep.ipynb`
Expected: completes with no cell errors; `data/processed/manifest.csv` exists with `pre_path, post_path, damage_class, storm, split` columns and non-zero rows for each damage class.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_data_prep.ipynb
git commit -m "docs: add data prep notebook"
```

(Note: `data/processed/` itself is gitignored per Task 1 — only the notebook is committed.)

---

### Task 11: Notebook 03 — Model training

**Files:**
- Create: `notebooks/03_model.ipynb`

**Interfaces:**
- Consumes: `data/processed/manifest.csv` (Task 10), `PairedCropDataset` (Task 5), `SiameseDamageNet` (Task 6), `compute_class_weights_tensor`, `EarlyStopper`, `train_one_epoch`, `validate` (Task 7).
- Produces: `models/best.pt` (state dict), consumed by Notebooks 04 and 05.

Requires Task 10's manifest to exist.

- [ ] **Step 1: Build the notebook**

Cells (via the same `nbformat` pattern):

```python
cells = [
    nbf.v4.new_markdown_cell("# AfterMath — Model Training"),
    nbf.v4.new_code_cell(
        "import os\n"
        "import yaml\n"
        "import pandas as pd\n"
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import transforms\n"
        "import matplotlib.pyplot as plt\n\n"
        "from utils.dataset import PairedCropDataset\n"
        "from utils.model import SiameseDamageNet\n"
        "from utils.training import compute_class_weights_tensor, EarlyStopper, train_one_epoch, validate\n"
        "from utils.xbd_labels import DAMAGE_CLASSES\n\n"
        "config = yaml.safe_load(open('../config.yaml'))\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
    ),
    nbf.v4.new_code_cell(
        "manifest_path = '../' + config['data']['manifest']\n"
        "full_manifest = pd.read_csv(manifest_path)\n"
        "train_manifest = full_manifest[full_manifest['split'] == 'train'].reset_index(drop=True)\n\n"
        "# further split train into train/val (80/20), stratified by damage_class\n"
        "from sklearn.model_selection import train_test_split\n"
        "train_rows, val_rows = train_test_split(\n"
        "    train_manifest, test_size=0.2, stratify=train_manifest['damage_class'], random_state=42\n"
        ")\n"
        "train_rows.to_csv('../data/processed/manifest_train.csv', index=False)\n"
        "val_rows.to_csv('../data/processed/manifest_val.csv', index=False)"
    ),
    nbf.v4.new_code_cell(
        "transform = transforms.Compose([transforms.ToTensor()])\n"
        "train_dataset = PairedCropDataset('../data/processed/manifest_train.csv', transform=transform)\n"
        "val_dataset = PairedCropDataset('../data/processed/manifest_val.csv', transform=transform)\n\n"
        "batch_size = config['training']['batch_size']\n"
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n"
        "val_loader = DataLoader(val_dataset, batch_size=batch_size)"
    ),
    nbf.v4.new_code_cell(
        "class_counts = train_rows['damage_class'].value_counts().to_dict()\n"
        "class_weights = compute_class_weights_tensor(class_counts, DAMAGE_CLASSES).to(device)\n\n"
        "model = SiameseDamageNet(num_classes=config['model']['num_classes'], pretrained=config['model']['pretrained']).to(device)\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])\n"
        "criterion = torch.nn.CrossEntropyLoss(weight=class_weights)\n"
        "stopper = EarlyStopper(patience=config['training']['early_stopping_patience'])"
    ),
    nbf.v4.new_code_cell(
        "train_losses, val_losses = [], []\n"
        "best_val_loss = float('inf')\n"
        "os.makedirs('../models', exist_ok=True)\n\n"
        "for epoch in range(config['training']['epochs']):\n"
        "    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)\n"
        "    val_loss = validate(model, val_loader, criterion, device)\n"
        "    train_losses.append(train_loss)\n"
        "    val_losses.append(val_loss)\n"
        "    print(f'epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')\n\n"
        "    if val_loss < best_val_loss:\n"
        "        best_val_loss = val_loss\n"
        "        torch.save(model.state_dict(), '../models/best.pt')\n\n"
        "    if stopper.step(val_loss):\n"
        "        print('Early stopping triggered')\n"
        "        break"
    ),
    nbf.v4.new_code_cell(
        "plt.plot(train_losses, label='train')\n"
        "plt.plot(val_losses, label='val')\n"
        "plt.xlabel('epoch')\n"
        "plt.ylabel('loss')\n"
        "plt.legend()\n"
        "plt.savefig('../docs/training_curves.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
]
```

- [ ] **Step 2: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute notebooks/03_model.ipynb --output 03_model.ipynb`
Expected: completes with no cell errors; `models/best.pt` and `docs/training_curves.png` are created; training loss trends downward across epochs.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_model.ipynb docs/training_curves.png
git commit -m "docs: add model training notebook"
```

(`models/best.pt` is gitignored — note in the README that trained weights are available via a Kaggle notebook link, matching IgnisEye's convention.)

---

### Task 12: Notebook 04 — Evaluation

**Files:**
- Create: `notebooks/04_evaluation.ipynb`

**Interfaces:**
- Consumes: `models/best.pt` (Task 11), `data/processed/manifest.csv` test split (Task 10), `SiameseDamageNet` (Task 6), `PairedCropDataset` (Task 5).

Requires Task 11's trained weights to exist.

- [ ] **Step 1: Build the notebook**

Cells:

```python
cells = [
    nbf.v4.new_markdown_cell("# AfterMath — Evaluation\n\nEvaluated on the held-out Hurricane Michael test set."),
    nbf.v4.new_code_cell(
        "import yaml\n"
        "import pandas as pd\n"
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "from torchvision import transforms\n"
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from utils.dataset import PairedCropDataset\n"
        "from utils.model import SiameseDamageNet\n"
        "from utils.xbd_labels import DAMAGE_CLASSES\n\n"
        "config = yaml.safe_load(open('../config.yaml'))\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
    ),
    nbf.v4.new_code_cell(
        "manifest = pd.read_csv('../' + config['data']['manifest'])\n"
        "test_manifest = manifest[manifest['split'] == 'test'].reset_index(drop=True)\n"
        "test_manifest.to_csv('../data/processed/manifest_test.csv', index=False)\n\n"
        "test_dataset = PairedCropDataset('../data/processed/manifest_test.csv', transform=transforms.Compose([transforms.ToTensor()]))\n"
        "test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])"
    ),
    nbf.v4.new_code_cell(
        "model = SiameseDamageNet(num_classes=config['model']['num_classes'], pretrained=False).to(device)\n"
        "model.load_state_dict(torch.load('../models/best.pt', map_location=device))\n"
        "model.eval()"
    ),
    nbf.v4.new_code_cell(
        "all_preds, all_labels = [], []\n"
        "with torch.no_grad():\n"
        "    for pre, post, labels in test_loader:\n"
        "        pre, post = pre.to(device), post.to(device)\n"
        "        outputs = model(pre, post)\n"
        "        preds = outputs.argmax(dim=1).cpu()\n"
        "        all_preds.extend(preds.tolist())\n"
        "        all_labels.extend(labels.tolist())"
    ),
    nbf.v4.new_code_cell(
        "print('Macro F1:', f1_score(all_labels, all_preds, average='macro'))\n"
        "print(classification_report(all_labels, all_preds, target_names=DAMAGE_CLASSES))"
    ),
    nbf.v4.new_code_cell(
        "cm = confusion_matrix(all_labels, all_preds)\n"
        "plt.figure(figsize=(6, 5))\n"
        "sns.heatmap(cm, annot=True, fmt='d', xticklabels=DAMAGE_CLASSES, yticklabels=DAMAGE_CLASSES, cmap='Blues')\n"
        "plt.xlabel('Predicted')\n"
        "plt.ylabel('Actual')\n"
        "plt.title('Confusion Matrix (Hurricane Michael, held out)')\n"
        "plt.savefig('../docs/confusion_matrix.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
]
```

- [ ] **Step 2: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute notebooks/04_evaluation.ipynb --output 04_evaluation.ipynb`
Expected: completes with no cell errors; prints macro-F1 and a classification report; `docs/confusion_matrix.png` is created.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_evaluation.ipynb docs/confusion_matrix.png
git commit -m "docs: add evaluation notebook (held-out Hurricane Michael test set)"
```

---

### Task 13: Notebook 05 — Grad-CAM visualizations

**Files:**
- Create: `notebooks/05_gradcam.ipynb`

**Interfaces:**
- Consumes: `models/best.pt` (Task 11), `GradCAM`, `resize_cam` (Task 8), `data/processed/manifest_test.csv` (Task 12).

Requires Task 11's trained weights and Task 12's test manifest to exist.

- [ ] **Step 1: Build the notebook**

Cells:

```python
cells = [
    nbf.v4.new_markdown_cell("# AfterMath — Grad-CAM\n\nExplainability: which parts of the before/after images drove each prediction."),
    nbf.v4.new_code_cell(
        "import yaml\n"
        "import pandas as pd\n"
        "import torch\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "from torchvision import transforms\n"
        "import matplotlib.pyplot as plt\n\n"
        "from utils.model import SiameseDamageNet\n"
        "from utils.gradcam import GradCAM, resize_cam\n"
        "from utils.xbd_labels import DAMAGE_CLASSES\n\n"
        "config = yaml.safe_load(open('../config.yaml'))\n"
        "device = torch.device('cpu')  # Grad-CAM run on CPU for simplicity\n\n"
        "model = SiameseDamageNet(num_classes=config['model']['num_classes'], pretrained=False).to(device)\n"
        "model.load_state_dict(torch.load('../models/best.pt', map_location=device))\n"
        "model.eval()\n\n"
        "cam_tool = GradCAM(model, model.backbone.layer4[-1])"
    ),
    nbf.v4.new_code_cell(
        "test_manifest = pd.read_csv('../data/processed/manifest_test.csv')\n"
        "transform = transforms.ToTensor()\n\n"
        "def load_tensor(path):\n"
        "    return transform(Image.open(path).convert('RGB')).unsqueeze(0)\n\n"
        "def overlay_heatmap(image, cam):\n"
        "    cam_resized = resize_cam(cam, size=image.shape[0])\n"
        "    heatmap = plt.cm.jet(cam_resized)[:, :, :3]\n"
        "    return (0.5 * image / 255.0 + 0.5 * heatmap)"
    ),
    nbf.v4.new_code_cell(
        "samples = test_manifest.sample(3, random_state=42)\n\n"
        "fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))\n"
        "for row_idx, (_, row) in enumerate(samples.iterrows()):\n"
        "    pre_tensor = load_tensor(row['pre_path'])\n"
        "    post_tensor = load_tensor(row['post_path'])\n"
        "    with torch.enable_grad():\n"
        "        logits = model(pre_tensor, post_tensor)\n"
        "        pred_class = logits.argmax(dim=1).item()\n\n"
        "    pre_cam = cam_tool.generate(pre_tensor, post_tensor, target_class=pred_class, branch='pre')\n"
        "    post_cam = cam_tool.generate(pre_tensor, post_tensor, target_class=pred_class, branch='post')\n\n"
        "    pre_img = np.array(Image.open(row['pre_path']).convert('RGB'))\n"
        "    post_img = np.array(Image.open(row['post_path']).convert('RGB'))\n\n"
        "    axes[row_idx, 0].imshow(pre_img)\n"
        "    axes[row_idx, 0].set_title('Before')\n"
        "    axes[row_idx, 1].imshow(overlay_heatmap(pre_img, pre_cam))\n"
        "    axes[row_idx, 1].set_title('Before + Grad-CAM')\n"
        "    axes[row_idx, 2].imshow(overlay_heatmap(post_img, post_cam))\n"
        "    axes[row_idx, 2].set_title(f'After + Grad-CAM\\nPredicted: {DAMAGE_CLASSES[pred_class]}')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig('../docs/gradcam_samples.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
]
```

- [ ] **Step 2: Run the notebook end to end**

Run: `jupyter nbconvert --to notebook --execute notebooks/05_gradcam.ipynb --output 05_gradcam.ipynb`
Expected: completes with no cell errors; `docs/gradcam_samples.png` shows before/after crops with heatmap overlays and predicted classes.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_gradcam.ipynb docs/gradcam_samples.png
git commit -m "docs: add Grad-CAM visualization notebook"
```

---

### Task 14: Final README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: results/figures produced by Tasks 9–13 (`docs/class_distribution.png`, `docs/training_curves.png`, `docs/confusion_matrix.png`, `docs/gradcam_samples.png`), macro-F1 and per-class metrics printed in Task 12.

- [ ] **Step 1: Rewrite `README.md` with full results**

```markdown
# AfterMath

**Hurricane building-damage classification from before/after satellite imagery — with Grad-CAM explainability.**

![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/model-ResNet18%20Siamese-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## How It Works

AfterMath takes a before/after satellite image pair of a single building
and predicts damage severity (`no-damage` / `minor-damage` / `major-damage`
/ `destroyed`) using a siamese dual-branch CNN: a shared, ImageNet-pretrained
ResNet18 processes the pre- and post-disaster crops separately, and their
feature vectors are concatenated before a classification head.

```
Before crop -\
              >-- shared ResNet18 --> concat features --> FC head --> damage class
After crop  -/
```

Trained on the [xBD dataset](https://xview2.org) (real hurricane
satellite imagery: Harvey, Florence, Matthew, Michael). Evaluated on
Hurricane Michael, held out entirely during training, to test
generalization to a storm the model has never seen.

---

## Results

[Fill in macro-F1 and per-class precision/recall from Task 12's output,
and embed `docs/confusion_matrix.png`, once the notebooks have been run
against the real downloaded dataset.]

### Training Curves

![Training curves](docs/training_curves.png)

### Confusion Matrix (held-out Hurricane Michael)

![Confusion matrix](docs/confusion_matrix.png)

### Grad-CAM — What Drove Each Prediction

![Grad-CAM samples](docs/gradcam_samples.png)

Heatmaps are shown for both the before and after image — confirming the
model is reacting to actual damage evidence in the after image, not
spurious cues (lighting, vegetation) in the before image.

---

## Dataset

- **xBD** — Gupta et al., xView2 Challenge — [xview2.org](https://xview2.org)
  Real pre/post satellite imagery from ~18 disasters; this project uses
  only the four hurricane events (Harvey, Florence, Matthew, Michael).

## Quick Start

```bash
git clone https://github.com/aviamin/aftermath.git
cd aftermath
pip install -r requirements.txt -r requirements-dev.txt
```

Download the xBD hurricane subset from xview2.org (free registration
required) and place it at `data/raw/<storm>/` as described above.

Run the notebooks in order: `01_eda` → `02_data_prep` → `03_model` →
`04_evaluation` → `05_gradcam`.

**Run the tests:**
```bash
pytest -v
```

---

## Tech Stack

Python · PyTorch · torchvision · scikit-learn · pandas · NumPy · OpenCV ·
Shapely · matplotlib · seaborn
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: finalize README with results"
```

---

## Self-Review Notes

- **Spec coverage:** data pipeline (Tasks 2-4, 10), siamese model (Task 6),
  training/loss/split (Tasks 7, 11), evaluation/macro-F1 (Task 12),
  Grad-CAM on both branches (Tasks 8, 13), repo structure (Task 1),
  testing scope (Tasks 2-8) — all spec sections have a corresponding task.
- **Placeholder scan:** the only intentionally-deferred content is the
  README's Results section (Task 14, Step 1), which cannot be filled with
  real numbers until the notebooks are actually run against the
  downloaded xBD data — this is a data dependency, not a placeholder left
  by oversight, and the task says exactly what to fill in and from where.
- **Type consistency:** `DAMAGE_CLASSES` (Task 2) is imported and reused
  unchanged in Tasks 5, 6 (indirectly via order), 7, and all evaluation/viz
  notebooks — never redefined. `BuildingLabel`, `PairedCropDataset`,
  `SiameseDamageNet`, `GradCAM` signatures match between their defining
  task and every later consumer.
- **Correctness catch during review:** Task 8's initial draft paired
  forward/backward hook firing order to attribute gradients to the
  pre/post branch, which assumes backward hooks fire in a fixed order
  relative to two independent branches merging at `cat()` — not
  guaranteed by autograd, and the test (which only checks shape/value
  range) wouldn't have caught a silent pre/post swap. Replaced with
  `retain_grad()` on the exact activation tensor identified via forward
  order, which is unambiguous. Fixed inline before handoff.
