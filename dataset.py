"""Chest X-ray dataset loading utilities."""
import os

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset):
    """Loads all JPEGs under data/chest_xray/<split>/{NORMAL,PNEUMONIA}."""

    def __init__(self, root, splits=("train",), img_size=128, train_aug=False):
        self.files = []
        for split in splits:
            for label in ("NORMAL", "PNEUMONIA"):
                d = os.path.join(root, split, label)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.files.append(os.path.join(d, f))
        aug = [transforms.RandomHorizontalFlip()] if train_aug else []
        self.transform = transforms.Compose(
            aug
            + [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.ToTensor(),  # [0, 1]
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # A few JPEGs in the original dataset are truncated; retry next file.
        for off in range(len(self.files)):
            path = self.files[(idx + off) % len(self.files)]
            try:
                img = Image.open(path)
                img = img.convert("L")
                return self.transform(img), 0
            except Exception:
                continue
        raise RuntimeError("no readable image found")


def _make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def get_train_loader(root, img_size=128, batch_size=64, num_workers=4):
    ds = ConcatDataset(
        [
            ChestXrayDataset(root, ("train",), img_size, train_aug=True),
            ChestXrayDataset(root, ("val",), img_size, train_aug=True),
        ]
    )
    return _make_loader(ds, batch_size, True, num_workers, drop_last=True)


def get_eval_loader(root, splits=("test",), img_size=128, batch_size=64,
                    num_workers=4):
    ds = ChestXrayDataset(root, splits, img_size, train_aug=False)
    return _make_loader(ds, batch_size, False, num_workers)
