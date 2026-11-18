import argparse
import csv
import os
import pickle
import sys
import tempfile
from pathlib import Path

import torch

from data_processing import IMAGE_EXTENSIONS, INPUT_SIZE, LABELS, PREPROCESSING, load_rgb, preprocess
from model import SignClassifier


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="한글 지문자 14종 이미지 폴더 예측")
    parser.add_argument("--input", required=True, type=Path, help="원본 이미지 폴더")
    parser.add_argument("--output", required=True, type=Path, help="새 예측 CSV 경로")
    args = parser.parse_args()
    if not args.input.is_dir():
        raise ValueError(f"입력 폴더가 없습니다: {args.input}")
    if args.output.exists():
        raise ValueError(f"기존 결과는 덮어쓰지 않습니다. 다른 출력 경로를 지정하세요: {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError(f"출력 폴더가 없습니다: {args.output.parent}")
    images = []
    for path in sorted(args.input.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.parent != args.input:
            raise ValueError(f"하위 폴더의 이미지를 입력 폴더 바로 아래로 배치하세요: {path}")
        images.append(path)
    if not images:
        raise ValueError(f"JPG/JPEG/PNG 이미지가 없습니다: {args.input}")
    weights_path = ROOT / "best.pt"
    if not weights_path.is_file():
        raise ValueError(f"선택한 best.pt를 저장소 루트에 배치하세요: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    expected = {"format_version": 1, "model_name": "SignClassifier_v1", "input_size": INPUT_SIZE,
                "labels": LABELS, "preprocessing": PREPROCESSING}
    if not isinstance(checkpoint, dict) or any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("가중치의 모델·입력 크기·라벨 순서·전처리 규칙이 현재 코드와 다릅니다.")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("가중치에 model_state가 없습니다.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for path in images:
            tensor = preprocess(load_rgb(path)).unsqueeze(0).to(device)
            logits = model(tensor)
            if logits.shape != (1, len(LABELS)) or not torch.isfinite(logits).all().item():
                raise ValueError(f"정상적인 14클래스 점수가 아닙니다: {path.name}")
            rows.append((path.name, int(logits.argmax(dim=1).item())))

    # 완성된 임시 CSV만 공개한다. Windows rename은 기존 파일을 덮어쓰지 않는다.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".tmp",
                                         dir=args.output.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("filename", "pred"))
            writer.writerows(rows)
        if os.name == "nt":
            os.rename(temporary_path, args.output)
        else:
            os.link(temporary_path, args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"예측 완료: {len(rows)}장 → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        print(f"예측 실패: {error}", file=sys.stderr)
        sys.exit(1)
