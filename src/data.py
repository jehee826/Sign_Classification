from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SignDataset(Dataset):
    def __init__(self, samples, indices, image_size, transform, cached_images=None):
        self.samples = samples
        self.indices = indices
        self.image_size = image_size
        self.transform = transform
        self.cached_images = cached_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        path, label = self.samples[source_index]

        if self.cached_images is None:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        else:
            image = Image.fromarray(self.cached_images[source_index])

        return self.transform(image), label


def build_loaders(data_dir, image_size=64, batch_size=32, seed=42,
                  augment=True, cache=True, workers=0):
    source = datasets.ImageFolder(str(Path(data_dir)))
    classes = source.classes
    expected = [f"c{i:02d}" for i in range(14)]
    if classes != expected:
        raise ValueError(f"c00~c13 클래스 폴더 14개가 필요합니다: {classes}")

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for class_id in range(14):
        indices = [i for i, label in enumerate(source.targets) if label == class_id]
        if len(indices) < 2:
            raise ValueError(f"{classes[class_id]} 클래스의 이미지가 2장 미만입니다.")
        rng.shuffle(indices)
        val_count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    cached_images = None
    if cache:
        print(f"이미지 {len(source.samples)}장 RAM 캐싱 중...", flush=True)
        cached_images = []
        for path, _ in source.samples:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                cached_images.append(np.array(image, dtype=np.uint8, copy=True))
        print("RAM 캐싱 완료", flush=True)

    train_steps = []
    if augment:
        train_steps.append(transforms.RandomRotation(7))
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = SignDataset(source.samples, train_indices, image_size,
                            train_transform, cached_images)
    val_set = SignDataset(source.samples, val_indices, image_size,
                          val_transform, cached_images)

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0
    }
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader, classes
