import argparse
import csv
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image

from train import ROOT, load_model, preprocess_image, to_tensor


# 입력 폴더의 사진을 분류하고 파일명과 정수 라벨을 저장한다.
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="한글 지문자 이미지 폴더 예측")
    parser.add_argument("--input", required=True, type=Path, help="원본 이미지 폴더")
    parser.add_argument("--output", required=True, type=Path, help="새 예측 CSV 경로")
    args = parser.parse_args()
    images = sorted(path for path in args.input.iterdir()
                    if path.is_file() and path.suffix.lower() == ".jpg")
    if not images:
        raise ValueError(f"JPG 이미지가 없습니다: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ROOT / "best.pt", device).eval()
    rows = []
    for path in images:
        with Image.open(path) as original:
            image = preprocess_image(original)
        scores = model(to_tensor(image).unsqueeze(0).to(device))
        if not torch.isfinite(scores).all().item():
            raise ValueError(f"예측 점수가 NaN 또는 무한대입니다: {path.name}")
        rows.append((path.name, scores.argmax(dim=1).item()))

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
    main()
