import argparse
import csv
import os
import pickle
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from data_processing import IMAGE_EXTENSIONS, pad_image, resize_image
from model import load_model


ROOT = Path(__file__).resolve().parents[1]


# 입력 폴더의 사진을 분류하고 파일명과 정수 라벨을 저장한다.
def main():
    parser = argparse.ArgumentParser(description="한글 지문자 이미지 폴더 예측")
    parser.add_argument("--input", required=True, type=Path, help="원본 이미지 폴더")
    parser.add_argument("--output", required=True, type=Path, help="새 예측 CSV 경로")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"이미 출력 파일이 있습니다: {args.output}")
    images = sorted(path for path in args.input.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"JPG/JPEG/PNG 이미지가 없습니다: {args.input}")
    if any(path.parent != args.input for path in images):
        raise ValueError("이미지는 입력 폴더 바로 아래에 배치하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ROOT / "best.pt", device)
    model.eval()
    to_tensor = v2.Compose([pad_image, v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    rows = []
    with torch.no_grad():
        for path in images:
            with Image.open(path) as original:
                image = resize_image(original).convert("RGB")
            tensor = to_tensor(image)
            batch = tensor.unsqueeze(0).to(device)
            scores = model(batch)
            if not torch.isfinite(scores).all().item():
                raise ValueError(f"예측 점수가 NaN 또는 무한대입니다: {path.name}")
            prediction = scores.argmax(dim=1).item()
            rows.append((path.name, prediction))

    # CSV가 완성되면 최종 경로에 옮긴다.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as folder:
        temporary = Path(folder) / args.output.name
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("filename", "pred"))
            writer.writerows(rows)
        if os.name == "nt":
            temporary.rename(args.output)
        else:
            os.link(temporary, args.output)
    print(f"예측 완료: {len(rows)}장 → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as error:
        sys.exit(f"예측 실패: {error}")
