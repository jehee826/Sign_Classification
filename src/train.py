import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import build_loaders
from model import CNNSON


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        correct += (logits.argmax(1) == labels).sum().item()
        total += count

    return total_loss / total, 100 * correct / total


def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)

            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (preds == labels).sum().item()
            total += count

            if return_predictions:
                actual.extend(labels.cpu().tolist())
                predicted.extend(preds.cpu().tolist())

    if return_predictions:
        return actual, predicted
    return total_loss / total, 100 * correct / total


def save_confusion_matrix(actual, predicted, classes, output_file):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, guess in zip(actual, predicted):
        matrix[target, guess] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CNNSON Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def save_learning_curve(rows, output_file):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["epoch"] for r in rows], [r["train_accuracy"] for r in rows],
            label="Train accuracy")
    ax.plot([r["epoch"] for r in rows], [r["val_accuracy"] for r in rows],
            label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def run_experiment(args, augment, output_dir):
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, classes = build_loaders(
        args.data, image_size=args.image_size, batch_size=args.batch_size,
        seed=args.seed, augment=augment, cache=not args.no_cache,
        workers=args.workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSON(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"장치: {device} | 증강: {augment} | 학습: {len(train_loader.dataset)}장 | 검증: {len(val_loader.dataset)}장")
    print(f"파라미터: {params:,} | 배치: {len(train_loader)} | 에포크: {args.epochs}")

    log_file = output_dir / "train_log.csv"
    weights_file = output_dir / "best_cnnson.pth"
    rows, best_acc = [], -1.0
    columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"]

    with log_file.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            seconds = time.perf_counter() - start

            row = {
                "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                "val_loss": val_loss, "val_accuracy": val_acc, "seconds": seconds
            }
            rows.append(row)
            writer.writerow(row)
            file.flush()
            print(f"Epoch [{epoch}/{args.epochs}] | Train {train_acc:.2f}% | Val {val_acc:.2f}% | {seconds:.1f}초", flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "augmentation": augment,
                    "epoch": epoch,
                    "val_accuracy": val_acc
                }, weights_file)

    checkpoint = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    actual, predicted = evaluate(
        model, val_loader, criterion, device, return_predictions=True
    )
    save_confusion_matrix(actual, predicted, classes, output_dir / "confusion_matrix.png")
    save_learning_curve(rows, output_dir / "learning_curve.png")
    print(f"최고 검증 정확도: {best_acc:.2f}% | 저장: {weights_file}")
    return best_acc, weights_file


def parse_args():
    parser = argparse.ArgumentParser(description="CNNSON 14-class Korean fingerspelling training")
    parser.add_argument("--data", type=Path, default=Path("team_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/results"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--compare-augmentation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        raise ValueError("epochs, batch-size는 1 이상, image-size는 16 이상이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_augmentation:
        scores = []
        for augment, name in [(True, "aug_on"), (False, "aug_off")]:
            acc, weight_path = run_experiment(args, augment, args.output_dir / name)
            scores.append((name, augment, acc, weight_path))

        with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["experiment", "augmentation", "best_val_accuracy"])
            for name, augment, acc, _ in scores:
                writer.writerow([name, augment, acc])

        best = max(scores, key=lambda item: item[2])
        shutil.copy2(best[3], args.output_dir / "best_cnnson.pth")
        print(f"증강 비교 종료 | 최적 실험: {best[0]} | 정확도: {best[2]:.2f}%")
    else:
        run_experiment(args, not args.no_augmentation, args.output_dir)


if __name__ == "__main__":
    main()
