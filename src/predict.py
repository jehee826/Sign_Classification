import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import CNNSON


JAMO = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(path):
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        return [path]
    if path.is_dir():
        images = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if images:
            return images
    raise FileNotFoundError(f"예측할 이미지가 없습니다: {path}")


def predict_images(weights, source, output, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    classes = checkpoint.get("classes", [f"c{i:02d}" for i in range(14)])
    image_size = checkpoint.get("image_size", 64)
    model = CNNSON(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    paths = collect_images(source)
    rows = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            group = paths[start:start + batch_size]
            tensors = []
            for path in group:
                with Image.open(path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    tensors.append(transform(image))
            logits = model(torch.stack(tensors).to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
            for path, idx, prob in zip(group, predicted.tolist(), confidence.tolist()):
                code = classes[idx]
                jamo = JAMO[idx] if classes == [f"c{i:02d}" for i in range(14)] else ""
                rows.append({
                    "filename": str(path),
                    "class_index": idx,
                    "class_code": code,
                    "jamo": jamo,
                    "confidence": round(float(prob), 6)
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["filename", "class_index", "class_code", "jamo", "confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[:20]:
        print(f"{row['filename']} → {row['class_code']} ({row['jamo']}) / {row['confidence']:.1%}")
    if len(rows) > 20:
        print(f"... 총 {len(rows)}개 예측")
    print(f"예측 결과 저장: {output}")


def main():
    parser = argparse.ArgumentParser(description="CNNSON image prediction")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("src/results/best_cnnson.pth"))
    parser.add_argument("--output", type=Path, default=Path("src/results/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    predict_images(args.weights, args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
