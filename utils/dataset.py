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
