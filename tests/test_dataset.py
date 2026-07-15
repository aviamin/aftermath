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
