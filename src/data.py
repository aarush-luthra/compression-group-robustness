"""Waterbirds loader using the official split in metadata.csv.

Expected layout (the standard Sagawa et al. release):
    <root>/metadata.csv      columns: img_id,img_filename,y,split,place,place_filename
    <root>/<img_filename>    images
split: 0=train, 1=val, 2=test.
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .metrics import make_group

SPLITS = {"train": 0, "val": 1, "test": 2}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transform():
    """ONE preprocessing pipeline shared by every model and every split.
    No augmentation, so all compared models see identical inputs."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class Waterbirds(Dataset):
    def __init__(self, root, split, transform=None):
        meta = pd.read_csv(os.path.join(root, "metadata.csv"))
        meta = meta[meta["split"] == SPLITS[split]].reset_index(drop=True)
        self.root = root
        self.files = meta["img_filename"].tolist()
        self.y = meta["y"].to_numpy()
        self.place = meta["place"].to_numpy()
        self.group = make_group(self.y, self.place)
        self.transform = transform or get_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("RGB")
        return self.transform(img), int(self.y[i]), int(self.group[i])


def group_counts(ds):
    return {g: int((ds.group == g).sum()) for g in range(4)}


def get_loaders(root, batch_size=64, num_workers=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda s, shuffle: DataLoader(
        Waterbirds(root, s), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        generator=g if shuffle else None)
    return {"train": mk("train", True), "val": mk("val", False), "test": mk("test", False)}
