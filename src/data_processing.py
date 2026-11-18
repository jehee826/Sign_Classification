import csv
import hashlib
import math
import random
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
INPUT_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREPROCESSING = {
    "version": 1,
    "orientation": "exif_transpose",
    "color": "RGB",
    "resize": "fit_longest_side_bilinear",
    "padding": "center_black",
    "layout": "CHW",
    "normalization": "float32_div_255",
}
METADATA_FIELDS = ("file", "class_id", "signer", "background", "lighting", "angle", "scale")
SPLIT_FIELDS = ("file", "partition", "group_id")


def read_csv(path, fields):
    if not path.is_file():
        raise ValueError(f"필수 CSV가 없습니다: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"{path.name} 헤더는 {','.join(fields)} 순서여야 합니다.")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if None in row or any(not (value or "").strip() for value in row.values()):
                raise ValueError(f"{path.name}:{line_number}: 누락되거나 초과한 값이 있습니다.")
            rows.append({key: value.strip() for key, value in row.items()})
    if not rows:
        raise ValueError(f"비어 있는 CSV입니다: {path}")
    return rows


def image_path(data_dir, filename):
    relative = PurePosixPath(filename)
    if "\\" in filename or relative.is_absolute() or ".." in relative.parts or ":" in filename:
        raise ValueError(f"data 기준 상대 경로를 /로 작성하세요: {filename}")
    path = (data_dir / relative).resolve()
    if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
        raise ValueError(f"data 안에 있는 파일을 찾을 수 없습니다: {filename}")
    return path


def load_rgb(path):
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}: {error}") from error


def preprocess(image, augmentation_rng=None):
    width, height = image.size
    brightness = 1.0
    if augmentation_rng is None:
        ratio = INPUT_SIZE / max(width, height)
        new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        resized = image.resize(new_size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
        canvas.paste(resized, ((INPUT_SIZE - new_size[0]) // 2, (INPUT_SIZE - new_size[1]) // 2))
    else:
        angle = math.radians(augmentation_rng.uniform(-10.0, 10.0))
        zoom = augmentation_rng.uniform(0.90, 1.00)
        brightness = augmentation_rng.uniform(0.90, 1.10)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated_width = abs(cosine) * width + abs(sine) * height
        rotated_height = abs(sine) * width + abs(cosine) * height
        ratio = INPUT_SIZE / max(rotated_width, rotated_height) * zoom
        # 출력 좌표에서 원본 좌표로 역변환해 회전 후에도 전체 이미지가 들어오게 한다.
        a, b = cosine / ratio, sine / ratio
        d, e = -sine / ratio, cosine / ratio
        center = INPUT_SIZE / 2
        coefficients = (a, b, width / 2 - (a + b) * center,
                        d, e, height / 2 - (d + e) * center)
        canvas = image.transform(
            (INPUT_SIZE, INPUT_SIZE), Image.Transform.AFFINE, coefficients,
            resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
        )
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    pixels = np.clip(pixels * brightness, 0.0, 1.0)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def inspect_training_data(data_dir):
    metadata = read_csv(data_dir / "metadata.csv", METADATA_FIELDS)
    split = read_csv(data_dir / "split.csv", SPLIT_FIELDS)
    records = {}
    paths = set()
    hashes = []
    digest_files = {}
    for row in metadata:
        filename = row["file"]
        path = image_path(data_dir, filename)
        if filename != path.relative_to(data_dir.resolve()).as_posix():
            raise ValueError(f"경로 표기를 정규화하세요: {filename}")
        if filename in records or path in paths:
            raise ValueError(f"metadata.csv 중복 파일입니다: {filename}")
        if row["class_id"] not in {str(index) for index in range(len(LABELS))}:
            raise ValueError(f"class_id는 정수 0~13이어야 합니다: {filename}")
        class_id = int(row["class_id"])
        if len(Path(filename).parts) != 2 or Path(filename).parts[0] != f"c{class_id:02d}":
            raise ValueError(f"클래스 폴더와 라벨이 다릅니다: {filename}")
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"학습 원본은 JPG여야 합니다: {filename}")
        with Image.open(path) as image:
            if image.format != "JPEG" or max(image.size) < 640:
                raise ValueError(f"JPEG 형식과 긴 변 640px 이상을 확인하세요: {filename}")
        load_rgb(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in digest_files:
            raise ValueError(f"동일 원본이 중복되었습니다: {digest_files[digest]}, {filename}")
        digest_files[digest] = filename
        hashes.append({"file": filename, "sha256": digest})
        records[filename] = {**row, "class_id": class_id, "path": path}
        paths.add(path)

    actual_files = {
        path.resolve() for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if actual_files != paths:
        missing = sorted(str(path) for path in actual_files - paths)
        raise ValueError(f"메타데이터에 없는 이미지가 있습니다: {missing[:5]}")
    for signer in sorted({row["signer"] for row in metadata}):
        signed = [row for row in metadata if row["signer"] == signer]
        counts = Counter(int(row["class_id"]) for row in signed)
        if any(counts[index] < 30 for index in range(len(LABELS))):
            raise ValueError(f"{signer}: 모든 자음에 30장 이상 필요합니다. 현재 수량: {dict(counts)}")
        if len({row["background"] for row in signed}) < 3 or len({row["lighting"] for row in signed}) < 2:
            raise ValueError(f"{signer}: 배경 3종 이상, 조명 2종 이상이 필요합니다.")

    assigned = {}
    groups = {}
    for row in split:
        filename, partition, group_id = row["file"], row["partition"], row["group_id"]
        if filename not in records or filename in assigned:
            raise ValueError(f"split.csv에 알 수 없거나 중복된 파일이 있습니다: {filename}")
        if partition not in {"train", "val"}:
            raise ValueError(f"partition은 train 또는 val이어야 합니다: {filename}")
        if group_id in groups and groups[group_id] != partition:
            raise ValueError(f"학습·검증에 같은 촬영 그룹이 있습니다: {group_id}")
        groups[group_id] = partition
        assigned[filename] = partition
        records[filename].update(partition=partition, group_id=group_id)
    if set(assigned) != set(records):
        raise ValueError("metadata.csv와 split.csv의 파일 목록이 일치하지 않습니다.")
    for partition in ("train", "val"):
        present = {row["class_id"] for row in records.values() if row["partition"] == partition}
        if present != set(range(len(LABELS))):
            raise ValueError(f"{partition} 분할에 14클래스가 모두 있어야 합니다.")
    for signer in sorted({row["signer"] for row in metadata}):
        for class_id in range(len(LABELS)):
            partitions = {row["partition"] for row in records.values()
                          if row["signer"] == signer and row["class_id"] == class_id}
            if partitions != {"train", "val"}:
                raise ValueError(f"{signer}, {LABELS[class_id]}: 그룹을 보존하며 양쪽 분할을 확보하세요.")
    return [records[filename] for filename in sorted(records)], hashes


class SignDataset(Dataset):
    def __init__(self, records, partition, augmentation_seed=None):
        self.records = [row for row in records if row["partition"] == partition]
        self.augmentation_rng = random.Random(augmentation_seed) if augmentation_seed is not None else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = load_rgb(row["path"])
        return preprocess(image, self.augmentation_rng), row["class_id"]
