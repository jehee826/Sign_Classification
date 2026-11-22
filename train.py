import argparse
import csv
import random
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
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
INPUT_SIZE = 300
SPLIT_SEED = 42
SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
LOG_FIELDS = ("augmentation", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds", "lr")


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


# 원본을 자음별 80:20으로 나누고 전처리해 출력 폴더에 덮어쓴다.
def prepare_data(source, output):
    if not source.is_dir():
        raise ValueError(f"원본 폴더를 찾을 수 없습니다: {source}")
    if output.is_dir():
        shutil.rmtree(output)

    rng = random.Random(SPLIT_SEED)
    train_count, val_count = 0, 0
    for label in CLASS_FOLDERS:
        images = sorted(
            path.relative_to(source) for path in (source / label).rglob("*")
            if path.is_file() and path.suffix.lower() == ".jpg"
        )
        if len(images) < 2:
            raise ValueError(f"{label} 폴더에 JPG가 2장 이상 필요합니다.")
        rng.shuffle(images)
        cut = max(1, round(len(images) * 0.2))
        splits = (("val", images[:cut]), ("train", images[cut:]))
        for partition, group in splits:
            for rel_path in group:
                target = output / partition / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source / rel_path) as image:
                    preprocess_image(image).save(target)
        val_count += cut
        train_count += len(images) - cut

    print(f"전처리 완료: 학습 {train_count}장, 검증 {val_count}장 → {output}")


# 저장된 가중치를 모델에 불러온다.
def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SignClassifier()
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


# 손실과 정확도를 계산하고, 검증 시에는 14×14 혼동행렬도 함께 만든다.
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    matrix = None if training else np.zeros((len(LABELS), len(LABELS)), dtype=int)
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
    return total_loss / count, correct / count, matrix


# 14×14 혼동행렬을 저장하고 가장 많이 혼동한 자음 쌍을 출력한다.
def save_confusion_matrix(matrix, best, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pairs = np.triu(matrix + matrix.T, k=1)
    first, second = np.unravel_index(pairs.argmax(), pairs.shape)
    count = pairs[first, second]

    fig, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    plot = axis.imshow(matrix, cmap="Blues")
    ticks = range(len(LABELS))
    confused = f"Most confused: class {first}/{second} ({count})" if count else "No misclassifications"
    title = f"Confusion Matrix (Best: {best['augmentation']}, val_acc={best['val_acc']:.4f} | {confused})"
    axis.set(xticks=ticks, yticks=ticks, xlabel="Predicted class", ylabel="True class", title=title)
    threshold = matrix.max() / 2
    for (row, column), value in np.ndenumerate(matrix):
        color = "white" if value > threshold else "black"
        axis.text(column, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(plot, ax=axis)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"최다 혼동: {LABELS[first]}/{LABELS[second]} {count}건" if count else "검증 오분류가 없습니다.")


# 데이터를 전처리하고 증강 조건별로 학습·검증을 실행한 뒤 최종 결과를 루트에 저장한다.
def main():
    parser = argparse.ArgumentParser(description="한글 지문자 분류 모델 학습")
    parser.add_argument("--augmentation", choices=("off", "on", "both"), default="both")
    augmentation = parser.parse_args().augmentation
    conditions = ("off", "on") if augmentation == "both" else (augmentation,)

    source_dir = ROOT / "data" if not SOURCE_DIR.is_dir() and (ROOT / "data").is_dir() else SOURCE_DIR
    data_dir = ROOT / f"{source_dir.name}_processing"
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

    # 같은 초기 가중치와 배치 순서로 증강 끔과 켬을 비교한다.
    results = []
    best_overall_score = (-1, 0)
    best_overall = None
    best_matrix = None

    with (ROOT / "train_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for condition in conditions:
            train_dataset.transform = augmented_transform if condition == "on" else to_tensor
            shuffle_rng = torch.Generator().manual_seed(SEED)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_rng)

            torch.manual_seed(SEED)
            model = SignClassifier().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.3, patience=9, threshold=0.0, min_lr=1e-5
            )

            cond_best_score = (-1, 0)
            cond_best = None

            for epoch in range(1, EPOCHS + 1):
                started = perf_counter()
                train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
                val_loss, val_acc, val_matrix = run_epoch(model, val_loader, criterion, device)
                epoch_seconds = perf_counter() - started
                metrics = {
                    "augmentation": condition,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_seconds": epoch_seconds,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                scheduler.step(val_loss)
                writer.writerow(metrics)
                stream.flush()
                print(
                    f"[{condition}] {epoch}/{EPOCHS} "
                    f"train loss={train_loss:.4f}, acc={train_acc:.4f} "
                    f"val loss={val_loss:.4f}, acc={val_acc:.4f}, "
                    f"lr={metrics['lr']:.2g}, {epoch_seconds:.2f}s",
                    flush=True,
                )

                score = (val_acc, -val_loss)
                if score > cond_best_score:
                    cond_best_score = score
                    cond_best = {"augmentation": condition, "best_epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = cond_best
                    best_matrix = val_matrix
                    torch.save({"model_state": model.state_dict()}, ROOT / "best.pt")

            results.append(cond_best)

    if len(results) > 1:
        for row in results:
            row["selected"] = row["augmentation"] == best_overall["augmentation"]
        with (ROOT / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    save_confusion_matrix(best_matrix, best_overall, ROOT / "confusion_matrix.png")
    print(f"선택: 증강 {best_overall['augmentation']}, 검증 정확도 {best_overall['val_acc']:.4f} → {ROOT}")


if __name__ == "__main__":
    main()
