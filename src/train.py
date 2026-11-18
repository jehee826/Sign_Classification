import copy
import csv
import json
import math
import os
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# CUDA 행렬 연산도 같은 환경에서 재현할 수 있도록 초기화 전에 설정한다.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
import numpy as np
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_processing import INPUT_SIZE, LABELS, PREPROCESSING, SignDataset, inspect_training_data
from model import SignClassifier

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt


# 실행 폴더와 관계없이 데이터와 결과는 저장소 루트를 기준으로 찾는다.
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
LOG_FIELDS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset, shuffle, seed):
    # DataLoader의 난수가 Dropout과 증강 난수 진행에 영향을 주지 않게 한다.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, drop_last=False, generator=generator)


def evaluate(model, loader, criterion, device, with_confusion=False):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64) if with_confusion else None
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not math.isfinite(loss.item()):
                raise ValueError("평가 손실이 유한한 값이 아닙니다.")
            predictions = logits.argmax(dim=1)
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            if matrix is not None:
                for truth, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                    matrix[truth, prediction] += 1
    if count == 0:
        raise ValueError("평가할 이미지가 없습니다.")
    return total_loss / count, correct / count, matrix


def train_condition(records, initial_state, condition, run_dir, device):
    seed_everything()
    model = SignClassifier().to(device)
    model.load_state_dict(initial_state, strict=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = SignDataset(records, "train", SEED + 1 if condition == "on" else None)
    train_loader = make_loader(training, True, SEED)
    train_eval_loader = make_loader(SignDataset(records, "train"), False, SEED + 2)
    val_loader = make_loader(SignDataset(records, "val"), False, SEED + 3)
    condition_dir = run_dir / condition
    condition_dir.mkdir()
    best = None
    with (condition_dir / "train_log.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                if not math.isfinite(loss.item()):
                    raise ValueError(f"{condition}, epoch {epoch}: 학습 손실이 유한하지 않습니다.")
                loss.backward()
                optimizer.step()

            train_loss, train_acc, _ = evaluate(model, train_eval_loader, criterion, device)
            val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
            metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc}
            writer.writerow(metrics)
            stream.flush()
            print(f"[{condition}] {epoch:02d}/{EPOCHS}: "
                  f"train loss={train_loss:.4f}, acc={train_acc:.4f}, "
                  f"val loss={val_loss:.4f}, acc={val_acc:.4f}", flush=True)
            if best is None or (val_acc, -val_loss) > (best["val_acc"], -best["val_loss"]):
                best = {"augmentation": condition, "best_epoch": epoch,
                        "val_loss": val_loss, "val_acc": val_acc}
                checkpoint = {
                    "format_version": 1,
                    "model_name": "SignClassifier_v1",
                    "model_state": {name: tensor.detach().cpu().clone()
                                    for name, tensor in model.state_dict().items()},
                    "input_size": INPUT_SIZE,
                    "labels": LABELS,
                    "preprocessing": PREPROCESSING,
                    "epoch": epoch,
                    "augmentation": condition,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                torch.save(checkpoint, condition_dir / "best.pt")
    return best


def save_confusion_plot(matrix, path):
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ("Malgun Gothic", "AppleGothic", "NanumGothic")
                        if name in available_fonts), None)
    if korean_font:
        plt.rcParams["font.family"] = korean_font
    fig, axis = plt.subplots(figsize=(11, 10))
    plot = axis.imshow(matrix, cmap="Blues", vmin=0)
    ticks = [f"{index}: {label}" if korean_font else str(index)
             for index, label in enumerate(LABELS)]
    axis.set_xticks(range(len(LABELS)), labels=ticks, rotation=45, ha="right")
    axis.set_yticks(range(len(LABELS)), labels=ticks)
    axis.set_xlabel("예측" if korean_font else "Predicted class")
    axis.set_ylabel("정답" if korean_font else "True class")
    axis.set_title("검증 혼동행렬" if korean_font else "Validation confusion matrix")
    threshold = matrix.max() / 2
    for truth in range(len(LABELS)):
        for prediction in range(len(LABELS)):
            value = int(matrix[truth, prediction])
            axis.text(prediction, truth, str(value), ha="center", va="center",
                      color="white" if value > threshold else "black", fontsize=8)
    fig.colorbar(plot, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data_dir = ROOT / "data"
    records, hashes = inspect_training_data(data_dir)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_state = copy.deepcopy(SignClassifier().state_dict())
    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(data_dir / "metadata.csv", run_dir / "metadata.csv")
    shutil.copy2(data_dir / "split.csv", run_dir / "split.csv")
    write_csv(run_dir / "hashes.csv", ("file", "sha256"), hashes)
    counts = []
    for signer in sorted({row["signer"] for row in records}):
        for class_id in range(len(LABELS)):
            counts.append({
                "signer": signer, "class_id": class_id,
                **{partition: sum(row["signer"] == signer and row["class_id"] == class_id
                                  and row["partition"] == partition for row in records)
                   for partition in ("train", "val")},
            })
    write_csv(run_dir / "split_counts.csv", ("signer", "class_id", "train", "val"), counts)
    config = {
        "status": "running", "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "num_workers": 0,
        "input_size": INPUT_SIZE, "labels": LABELS, "preprocessing": PREPROCESSING,
        "model_name": "SignClassifier_v1",
        "augmentation": {"degrees": [-10, 10], "scale": [0.90, 1.00], "brightness": [0.90, 1.10]},
        "selection": "val_acc descending, val_loss ascending, earlier epoch, off first",
        "device": str(device), "platform": platform.platform(), "python": platform.python_version(),
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                     "Pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"실행 폴더: {run_dir}\n장치: {device}", flush=True)
    try:
        results = [train_condition(records, initial_state, condition, run_dir, device)
                   for condition in ("off", "on")]
        # max는 동점이면 앞의 off를 유지한다.
        selected = max(results, key=lambda row: (row["val_acc"], -row["val_loss"]))
        checkpoint = torch.load(run_dir / selected["augmentation"] / "best.pt",
                                map_location="cpu", weights_only=True)
        model = SignClassifier().to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        val_loader = make_loader(SignDataset(records, "val"), False, SEED + 3)
        val_loss, val_acc, matrix = evaluate(model, val_loader, nn.CrossEntropyLoss(), device, True)
        val_count = len(val_loader.dataset)
        if int(matrix.sum()) != val_count or not math.isclose(float(matrix.trace()) / val_count, val_acc):
            raise ValueError("혼동행렬 수량과 검증 정확도가 일치하지 않습니다.")
        if not math.isclose(val_acc, selected["val_acc"], abs_tol=1e-12) or not math.isclose(
                val_loss, selected["val_loss"], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("최고 가중치 재평가와 저장된 검증 지표가 일치하지 않습니다.")
        selected_dir = run_dir / "selected"
        selected_dir.mkdir()
        for name in ("best.pt", "train_log.csv"):
            shutil.copy2(run_dir / selected["augmentation"] / name, selected_dir / name)
        write_csv(selected_dir / "comparison.csv",
                  ("augmentation", "best_epoch", "val_loss", "val_acc"), results)
        save_confusion_plot(matrix, selected_dir / "confusion_matrix.png")
        pair = max(((int(matrix[a, b] + matrix[b, a]), a, b)
                    for a in range(len(LABELS)) for b in range(a + 1, len(LABELS))),
                   key=lambda item: item[0])
        config.update(status="complete", selected=selected,
                      confusion_pair={"class_ids": [pair[1], pair[2]], "count": pair[0]})
        print(f"선택 결과: 증강 {selected['augmentation']}, epoch {selected['best_epoch']}, "
              f"검증 정확도 {val_acc:.4f}")
        print(f"최다 혼동: {LABELS[pair[1]]}/{LABELS[pair[2]]}, {pair[0]}건"
              if pair[0] else "검증 오분류가 없습니다.")
        print(f"제출용 결과: {selected_dir}")
    except BaseException as error:
        config.update(status="failed", error=str(error))
        raise
    finally:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"학습 실패: {error}", file=sys.stderr)
        sys.exit(1)
