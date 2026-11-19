import argparse
import csv
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_SIZE = 128
SPLIT_SEED = 42
LABELS = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
CLASS_FOLDERS = [f"c{index:02d}" for index in range(len(LABELS))]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# 비율을 유지하며 이미지의 긴 변을 128px로 맞춘다.
def resize_image(image):
    width, height = image.size
    longest_side = max(width, height)
    if longest_side == INPUT_SIZE:
        return image
    scale = INPUT_SIZE / longest_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


# 텐서로 변환하기 전에 사진을 정사각형 검은 배경의 가운데에 놓는다.
def pad_image(image):
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    left = (INPUT_SIZE - image.width) // 2
    top = (INPUT_SIZE - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


# 팀 공통 분할을 읽고 사진과 촬영 그룹이 한쪽에만 속하는지 확인한다.
def read_split(source, images_by_class):
    remaining = {}
    for images in images_by_class.values():
        for path in images:
            remaining[path.relative_to(source).as_posix()] = path

    partitions = {"train": [], "val": []}
    groups = {}
    with (source / "split.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"file", "partition", "group_id"}.issubset(reader.fieldnames or ()):
            raise ValueError("split.csv에는 file, partition, group_id 열이 필요합니다.")
        for row in reader:
            filename = (row["file"] or "").strip()
            partition = (row["partition"] or "").strip()
            group = (row["group_id"] or "").strip()
            path = remaining.pop(filename, None)
            if path is None or partition not in partitions or not group:
                raise ValueError(f"split.csv {reader.line_num}행의 사진·분할·그룹을 확인하세요: {filename}")
            if group in groups and groups[group] != partition:
                raise ValueError(f"같은 촬영 그룹이 학습과 검증에 있습니다: {group}")
            groups[group] = partition
            partitions[partition].append((path, group))

    if remaining:
        raise ValueError(f"split.csv에 없는 사진이 {len(remaining)}장 있습니다: {next(iter(remaining))}")
    for partition, images in partitions.items():
        folders = {path.relative_to(source).parts[0] for path, _ in images}
        if folders != set(CLASS_FOLDERS):
            raise ValueError(f"{partition} 분할에 c00~c13 사진이 모두 필요합니다.")
    return partitions


# 자음별 사진을 읽고 공통 CSV 또는 고정 시드로 학습·검증을 나눈다.
def split_images(source):
    images_by_class = {}
    for label in CLASS_FOLDERS:
        images = sorted(path for path in (source / label).rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 2:
            raise ValueError(f"{label}/에 학습·검증용 사진이 최소 2장 필요합니다.")
        images_by_class[label] = images

    if (source / "split.csv").is_file():
        return read_split(source, images_by_class)

    print("촬영 그룹 정보 없이 자음별 약 80:20으로 분할합니다.")
    rng = random.Random(SPLIT_SEED)
    partitions = {"train": [], "val": []}
    for images in images_by_class.values():
        rng.shuffle(images)
        val_count = max(1, round(len(images) * 0.2))
        partitions["train"].extend((path, "") for path in images[val_count:])
        partitions["val"].extend((path, "") for path in images[:val_count])
    return partitions


# 리사이즈한 사진을 train·val 폴더에 나눠 저장하고 분할표를 만든다.
def prepare_data(source, output):
    if not source.is_dir():
        raise ValueError(f"원본 폴더를 찾을 수 없습니다: {source}")
    if output.exists():
        raise ValueError(f"결과 폴더가 이미 있습니다: {output}")

    partitions = split_images(source)
    with TemporaryDirectory(dir=output.parent) as folder:
        resized = Path(folder) / output.name
        resized.mkdir()
        with (resized / "split.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("file", "partition", "group_id"))
            for partition, images in partitions.items():
                for path, group in sorted(images):
                    relative = path.relative_to(source)
                    target = resized / partition / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(path) as image:
                        resize_image(image).save(target)
                    writer.writerow((relative.as_posix(), partition, group))
        resized.rename(output)

    print(f"전처리 완료: 학습 {len(partitions['train'])}장, 검증 {len(partitions['val'])}장 → {output}")


# 원본 폴더를 인자로 받아 전처리를 따로 실행한다.
def main():
    parser = argparse.ArgumentParser(description="사진을 비율 유지 리사이즈하고 학습·검증으로 분할")
    parser.add_argument("--input", type=Path, default=ROOT / "data", help="원본 사진 폴더")
    source = parser.parse_args().input.resolve()
    output = source.with_name(f"{source.name}_prosessing")
    prepare_data(source, output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"전처리 실패: {error}")
