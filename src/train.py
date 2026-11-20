import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from data_processing import pad_image, prepare_data, resize_image
from model import CHECKPOINT_INFO, LABELS, SignClassifier

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_prosessing"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
LOG_FIELDS = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")


# 전체 학습 배치에서 가중치를 갱신하고 손실과 정확도를 계산한다.
def train_epoch(model, train_loader, criterion, optimizer, device):
    # 학습모드로 변경
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 기존 "기울기" 제거 => "가중치"와 "기울기"는 서로 다르다 "기울기"를 계산해서 "가중치"에 반영하는 방식이다
        # 만약 제거하지않는다면? 
        # 한번 계산 == 기울기 2.0 => 가중치반영 => 가중치 2.0 => 다음 epoch에 기울기 계산 => 여기서 기존기울기와 합쳐져버려 가중치에 한번에 들어가버림 => 기존기울기 2.0 + 그다음 기울기 1.0 == 3.0이 가중치에 합해져버린다 => 그 전 epoch에서 업데이트된 가중치에 1.0이 더해지는게 아니라 3.0이 더해져버려서 2.0(기존) + 3.0(초기화 안된가중치) == 5.0이 되버림 
        optimizer.zero_grad()
        # 순전파 => pytorch의 nn.module이 제공하는 __call__매직메서드로인해 forward()가 실행됨
        logits = model(images)
        # 손실 계산
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    count = len(train_loader.dataset)
    return total_loss / count, correct / count


# 가중치를 바꾸지 않고 전체 검증 배치의 손실과 정확도를 계산한다.
def evaluate_model(model, val_loader, criterion, device, matrix=None):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += (predictions == labels).sum().item()
            if matrix is not None:
                for label, prediction in zip(labels.tolist(), predictions.tolist()):
                    matrix[label, prediction] += 1
    count = len(val_loader.dataset)
    return total_loss / count, correct / count


# 최고 가중치를 골라 제출용 결과와 가장 많이 혼동한 자음 쌍을 기록한다.
def result(run_dir, results, val_loader, device):
    best = max(results, key=lambda row: (row["val_acc"], -row["val_loss"]))
    winner = run_dir / best["augmentation"]
    checkpoint = torch.load(winner / "best.pt", map_location="cpu", weights_only=True)
    model = SignClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    evaluate_model(model, val_loader, nn.CrossEntropyLoss(), device, matrix=matrix)

    selected = run_dir / "selected"
    selected.mkdir()
    for filename in ("best.pt", "train_log.csv"):
        shutil.copyfile(winner / filename, selected / filename)
    if len(results) > 1:
        with (selected / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("augmentation", "best_epoch", "val_loss", "val_acc"))
            writer.writeheader()
            writer.writerows(results)

    fig, axis = plt.subplots(figsize=(9, 8), layout="constrained")
    plot = axis.imshow(matrix, cmap="Blues")
    ticks = range(len(LABELS))
    axis.set(xticks=ticks, yticks=ticks, xlabel="Predicted class", ylabel="True class")
    threshold = matrix.max() / 2
    for row in ticks:
        for column in ticks:
            count = matrix[row, column]
            color = "white" if count > threshold else "black"
            axis.text(column, row, str(count), ha="center", va="center", color=color)
    fig.colorbar(plot, ax=axis)
    fig.savefig(selected / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # A→B와 B→A 오분류를 더해 가장 많이 혼동한 쌍을 찾는다.
    pairs = np.triu(matrix + matrix.T, k=1)
    first, second = np.unravel_index(pairs.argmax(), pairs.shape)
    count = pairs[first, second]
    print(f"최다 혼동: {LABELS[first]}/{LABELS[second]} {count}건" if count else "검증 오분류가 없습니다.")
    print(f"선택: 증강 {best['augmentation']}, 검증 정확도 {best['val_acc']:.4f} → {selected}")


# 데이터를 읽고 epoch마다 학습과 검증을 실행한다.
def main():
    parser = argparse.ArgumentParser(description="한글 지문자 분류 모델 학습")
    parser.add_argument("--augmentation", choices=("off", "on", "both"), default="both",
                        help="증강 끔(off), 켬(on), 또는 두 조건 비교(both, 기본값)")
    args = parser.parse_args()
    conditions = ("off", "on") if args.augmentation == "both" else (args.augmentation,)

    if not DATA_DIR.exists():
        prepare_data(ROOT / "data", DATA_DIR)

    # 이미지 => 텐서변환 파이프라인
    to_tensor = v2.Compose([
        pad_image,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ])

    # 이미지 증강 파이프라인
    augmented_transform = v2.Compose([
        v2.RandomRotation(10, interpolation=v2.InterpolationMode.BILINEAR, expand=True),
        resize_image,
        v2.ColorJitter(brightness=(0.9, 1.1)),
        to_tensor,
    ])

    # 먼저 데이터셋과 검증 데이터로더를 준비한다
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=to_tensor)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=to_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 학습에 사용할 디바이스 선택(CPU vs GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 손실함수정의
    criterion = nn.CrossEntropyLoss()
    result_dir = ROOT / "result" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir.mkdir(parents=True)

    # 증강여부(both, on, off)에 따라 학습 => 튜플의 요소를 하나씩 꺼내 진행한다
    results = []
    for condition in conditions:
        # 증강이 on => 튜플에서 on을 꺼냈을때 있다면 증강적용, 아니라면 바로 텐서변환
        train_dataset.transform = augmented_transform if condition == "on" else to_tensor
        shuffle_rng = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_rng)

        torch.manual_seed(SEED)
        model = SignClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        directory = result_dir / condition
        directory.mkdir()
        best = None

        with (directory / "train_log.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
            writer.writeheader()

            # 학습 시작
            for epoch in range(1, EPOCHS + 1):
                # metrics에 저장할 값 저장
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

                # metrics에 추가 => csv에 넣고 best.pt 판가름할때도 사용함
                metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                           "val_loss": val_loss, "val_acc": val_acc}
                # csv파일에 저장
                writer.writerow(metrics)
                stream.flush()
                print(f"[{condition}] {epoch}/{EPOCHS} "
                      f"train loss={train_loss:.4f}, acc={train_acc:.4f} "
                      f"val loss={val_loss:.4f}, acc={val_acc:.4f}", flush=True)

                # 최적 가중치 저장로직 => 검증정확도가 더 높은모델을 저장한다, 같을경우 손실이 더 낮은모델을 저장한다
                if best is None or (val_acc, -val_loss) > (best["val_acc"], -best["val_loss"]):
                    best = metrics # 기존 metrics를 best로 저장(다음 epoch비교)
                    checkpoint = CHECKPOINT_INFO.copy()
                    checkpoint.update(metrics) # 지정한 dict변수에 key가 없다면 추가, key가 있다면 value업데이트
                    checkpoint["model_state"] = model.state_dict() # 현재 모델의 가중치, 편향, BatchNorm 누적 통계
                    checkpoint["augmentation"] = condition # 증강 on/off 여부 기록
                    torch.save(checkpoint, directory / "best.pt") # 가중치 저장

        results.append({"augmentation": condition, "best_epoch": best["epoch"],
                        "val_loss": best["val_loss"], "val_acc": best["val_acc"]})

    result(result_dir, results, val_loader, device)


if __name__ == "__main__":
    main()
