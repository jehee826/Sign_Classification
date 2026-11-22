import argparse
import csv
import random
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import SignClassifier


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "T06_data"
LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
CLASS_FOLDERS = [f"c{index:02d}" for index in range(len(LABELS))]
SPLIT_SEED = 42
SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
LOG_FIELDS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds", "lr")


# 원본의 긴 변과 모델에 넣는 정사각형 크기를 300px로 맞춘다.
INPUT_SIZE = 300


# 비율을 유지하며 이미지의 긴 변을 INPUT_SIZE에 맞춘다.
def resize_image(image):
    width, height = image.size
    longest_side = max(width, height)
    if longest_side == INPUT_SIZE:
        return image
    scale = INPUT_SIZE / longest_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


# 원본을 리사이즈하고 밝기의 양끝 2%를 기준으로 대비를 보정한다.
def preprocess_image(image):
    image = resize_image(image).convert("RGB")
    return ImageOps.autocontrast(image, cutoff=2, preserve_tone=True)


# 사진에 검은 여백을 넣고 0~1 범위의 텐서로 바꾼다.
def to_tensor(image):
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    left = (INPUT_SIZE - image.width) // 2
    top = (INPUT_SIZE - image.height) // 2
    canvas.paste(image, (left, top))
    tensor = v2.functional.to_image(canvas)
    return v2.functional.to_dtype(tensor, torch.float32, scale=True)


# 공통 분할표에 모든 사진이 한 번씩 나오고 촬영 그룹이 겹치지 않는지 확인한다.
def read_split(split_file, images_by_class):
    remaining = {path.as_posix(): path for images in images_by_class for path in images}
    partitions = {"train": [], "val": []}
    groups = {}
    with split_file.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            partition, group = row["partition"].strip(), row["group_id"].strip()
            if not group or groups.setdefault(group, partition) != partition:
                raise ValueError(f"split.csv의 촬영 그룹을 확인하세요: {group!r}")
            partitions[partition].append((remaining.pop(row["file"].strip()), group))
    if remaining:
        raise ValueError(f"split.csv에 없는 사진이 {len(remaining)}장 있습니다: {next(iter(remaining))}")
    return partitions


# 자음별 사진을 읽고 공통 CSV 또는 고정 시드로 학습·검증을 나눈다.
def split_images(source):
    images_by_class = []
    for label in CLASS_FOLDERS:
        images = sorted(path.relative_to(source) for path in (source / label).rglob("*")
                        if path.is_file() and path.suffix.lower() == ".jpg")
        images_by_class.append(images)

    split_file = source / "split.csv"
    if split_file.is_file():
        partitions = read_split(split_file, images_by_class)
    else:
        print("촬영 그룹 정보 없이 자음별 약 80:20으로 분할합니다.")
        rng = random.Random(SPLIT_SEED)
        partitions = {"train": [], "val": []}
        for images in images_by_class:
            rng.shuffle(images)
            val_count = max(1, round(len(images) * 0.2))
            partitions["train"].extend((path, "") for path in images[val_count:])
            partitions["val"].extend((path, "") for path in images[:val_count])

    for partition, images in partitions.items():
        if {path.parts[0] for path, _ in images} != set(CLASS_FOLDERS):
            raise ValueError(f"{partition} 분할에 c00~c13 사진이 모두 필요합니다.")
    return partitions


# 리사이즈한 사진을 train·val 폴더에 나눠 저장하고 분할표를 만든다.
def prepare_data(source, output):
    if not source.is_dir():
        raise ValueError(f"원본 폴더를 찾을 수 없습니다: {source}")

    partitions = split_images(source)
    if output.is_dir():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with (output / "split.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "partition", "group_id"))
        for partition, images in partitions.items():
            for path, group in sorted(images):
                target = output / partition / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source / path) as image:
                    preprocess_image(image).save(target)
                writer.writerow((path.as_posix(), partition, group))

    print(f"전처리 완료: 학습 {len(partitions['train'])}장, 검증 {len(partitions['val'])}장 → {output}")


# 저장된 가중치를 모델에 불러온다.
def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SignClassifier()
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


# 손실과 정확도를 계산하고, optimizer가 있을 때만 가중치를 갱신한다.
def run_epoch(model, loader, criterion, device, optimizer=None, matrix=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            correct += (predictions == labels).sum().item()
            if matrix is not None:
                for label, prediction in zip(labels.tolist(), predictions.tolist()):
                    matrix[label, prediction] += 1
    count = len(loader.dataset)
    return total_loss / count, correct / count


# 최고 가중치를 골라 제출용 결과와 가장 많이 혼동한 자음 쌍을 기록한다.
def result(run_dir, results, val_loader, device):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    best = max(results, key=lambda row: (row["val_acc"], -row["val_loss"]))
    winner = run_dir / best["augmentation"]
    model = load_model(winner / "best.pt", device)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    run_epoch(model, val_loader, nn.CrossEntropyLoss(), device, matrix=matrix)

    selected = run_dir / "selected"
    selected.mkdir()
    for filename in ("best.pt", "train_log.csv"):
        shutil.copyfile(winner / filename, selected / filename)
    if len(results) > 1:
        with (selected / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=best.keys())
            writer.writeheader()
            writer.writerows(results)

    # A→B와 B→A 오분류를 더해 가장 많이 혼동한 쌍을 찾는다.
    pairs = np.triu(matrix + matrix.T, k=1)
    first, second = np.unravel_index(pairs.argmax(), pairs.shape)
    count = pairs[first, second]

    fig, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    plot = axis.imshow(matrix, cmap="Blues")
    ticks = range(len(LABELS))
    title = (
        f"Confusion Matrix (Most confused: class {first}/{second}, {count})"
        if count
        else "Confusion Matrix (No misclassifications)"
    )
    axis.set(xticks=ticks, yticks=ticks, xlabel="Predicted class", ylabel="True class", title=title)
    threshold = matrix.max() / 2
    for (row, column), value in np.ndenumerate(matrix):
        color = "white" if value > threshold else "black"
        axis.text(column, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(plot, ax=axis)
    fig.savefig(selected / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    outputs = ["best.pt", "train_log.csv", "confusion_matrix.png"]
    if len(results) > 1:
        outputs.append("comparison.csv")
    for filename in outputs:
        shutil.copyfile(selected / filename, ROOT / filename)

    print(f"최다 혼동: {LABELS[first]}/{LABELS[second]} {count}건" if count else "검증 오분류가 없습니다.")
    print(f"선택: 증강 {best['augmentation']}, 검증 정확도 {best['val_acc']:.4f} → {ROOT} (기록: {selected})")


# 데이터를 읽고 epoch마다 학습과 검증을 실행한다.
def main():
    parser = argparse.ArgumentParser(description="한글 지문자 분류 모델 학습")
    parser.add_argument("--augmentation", choices=("off", "on", "both"), default="both")
    augmentation = parser.parse_args().augmentation
    conditions = ("off", "on") if augmentation == "both" else (augmentation,)

    with TemporaryDirectory(dir=ROOT) as temp_dir:
        data_dir = Path(temp_dir) / "data_prosessing"
        source_dir = ROOT / "data" if not SOURCE_DIR.is_dir() and (ROOT / "data").is_dir() else SOURCE_DIR
        prepare_data(source_dir, data_dir)

        # 증강을 켜면 학습 사진을 조금 회전시키고 밝기를 바꾼다.
        augmented_transform = v2.Compose([
            v2.RandomRotation(10, interpolation=v2.InterpolationMode.BILINEAR, expand=True),
            resize_image,
            v2.ColorJitter(brightness=(0.9, 1.1)),
            to_tensor,
        ])

        train_dataset = datasets.ImageFolder(data_dir / "train")
        val_dataset = datasets.ImageFolder(data_dir / "val", transform=to_tensor)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"학습 장치 {device}", flush=True)
        criterion = nn.CrossEntropyLoss()
        result_dir = ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result_dir.mkdir(parents=True)

        # 같은 초기 가중치와 배치 순서로 증강 켬과 끔을 비교한다.
        results = []
        for condition in conditions:
            train_dataset.transform = augmented_transform if condition == "on" else to_tensor
            shuffle_rng = torch.Generator().manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_rng)

            torch.manual_seed(SEED)
            model = SignClassifier().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            # 검증 손실이 10회 연속 개선되지 않으면 학습률을 낮춘다.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.3, patience=9, threshold=0.0, min_lr=1e-5
            )
            directory = result_dir / condition
            directory.mkdir()
            best_score = (-1, 0)

            with (directory / "train_log.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
                writer.writeheader()

                for epoch in range(1, EPOCHS + 1):
                    # 학습과 검증에 걸린 시간을 잰다.
                    started = perf_counter()
                    train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
                    val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
                    epoch_seconds = perf_counter() - started
                    metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                               "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": epoch_seconds,
                               "lr": optimizer.param_groups[0]["lr"]}
                    scheduler.step(val_loss)
                    writer.writerow(metrics)
                    stream.flush()
                    print(f"[{condition}] {epoch}/{EPOCHS} "
                          f"train loss={train_loss:.4f}, acc={train_acc:.4f} "
                          f"val loss={val_loss:.4f}, acc={val_acc:.4f}, "
                          f"lr={metrics['lr']:.2g}, {epoch_seconds:.2f}s", flush=True)

                    # 검증 정확도가 높거나 동률에서 손실이 작으면 최고 가중치를 갱신한다.
                    score = (val_acc, -val_loss)
                    if score > best_score:
                        best_score = score
                        best = {"augmentation": condition, "best_epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
                        torch.save({"model_state": model.state_dict()}, directory / "best.pt")

            results.append(best)

        result(result_dir, results, val_loader, device)
    return result_dir


if __name__ == "__main__":
    main()
