# Sign Classification

## 주제

수화 이미지 분류 모델: 팀원들이 직접 촬영한 손 사진으로 한글 지문자 자음 14개(ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ)를 구별하는 다중분류 모델을 만든다. 쌍자음과 모음은 제외한다.

## 내용

- 팀원 전원이 자음별로 30장 이상의 사진을 촬영하고, 다양한 배경·조명·각도·크기를 포함한 데이터셋과 촬영 정보(`metadata.csv`)를 구성한다.
- 팀 데이터만으로 이미지 분류 모델을 처음부터 학습한다. 모든 가중치는 무작위로 초기화하며, 직접 설계한 CNN 또는 학습된 가중치가 없는 모델 구조를 사용할 수 있다.
- 학습·검증 손실과 정확도를 첫 epoch부터 마지막 epoch까지 기록하고, 데이터 증강 적용 여부에 따른 검증 정확도와 14×14 혼동행렬로 성능을 분석한다.
- 학습 데이터, 학습·예측 코드, 최종 가중치, 학습 로그, 혼동행렬 이미지, 2분 이내 실행 영상을 제출한다. 시험 당일에는 제공되는 테스트 이미지에 대한 예측 결과를 30분 안에 제출하고, 각자 팀 코드와 실행 결과를 바탕으로 지필고사를 치른다.

세부 촬영 규칙, 제출 파일 및 예측 입출력 형식은 [기말과제 공지](docs/기말과제공지.pdf)를 따른다. 구현 기준과 남은 확인은 [상세 계획](docs/plan.md)에 정리한다.

## 주의사항

- **모든 코드는 팀원 모두가 확실하게 설명할 수 있어야 한다.** 코드의 역할, 사용한 기술의 이유, 학습과 예측 흐름을 이해한 상태로 작성한다.
- **과설계를 지양한다.** 과제에 필요한 범위에서 단순하고 이해하기 쉬운 구조를 유지하며, 모든 딥러닝 기술을 사용할 필요는 없다.
- **모델의 복잡함보다 정확도를 우선한다.** 복잡한 구조나 기술의 개수보다 검증 결과를 근거로 개선한다.
- **과제의 데이터·모델 제한을 준수한다.** 인터넷 이미지, 공개 데이터셋, 다른 팀의 사진, 사전학습 가중치, 손 검출·키포인트 모델(MediaPipe 등)은 사용하지 않는다.
- **Test 시연시 코드안에서 전처리 수행후 예측한다.** 필요한 전처리는 학습 조건에 맞춰 `predict.py` 안에서 수행한다.

## 구현 파일

- `src/data_processing.py`: 원본 폴더와 같은 구조로 비율을 유지해 긴 변이 128px인 사진을 저장하는 명령.
- `src/model.py`: 직접 구성한 합성곱 3블록과 14클래스 분류부.
- `src/train.py`: 무작위 초기화 학습, 증강 off/on 비교, epoch별 평가, 가중치 선택과 혼동행렬.
- `src/predict.py`: 원본 이미지 폴더를 읽어 `filename,pred` CSV 생성.

`requirements.txt`는 저장소 루트에 둔다. 원본 사진은 기본 `data/`, 학습 입력은 기본 `data_prosessing/`, 결과는 `result/`, 추론 가중치는 `best.pt`에 둔다. 상대 경로 인자는 명령을 실행한 현재 폴더를 기준으로 해석한다.

사진은 저장된 픽셀 방향과 가로세로 비율을 유지하면서 긴 변을 128px로 맞춘다. 학습과 예측에서는 검은 여백을 채워 128×128 RGB 입력을 만들고, Torchvision v2의 `ToImage()`와 `ToDtype(torch.float32, scale=True)`로 0~1 범위의 CHW 텐서로 바꾼다. 증강을 켠 학습에는 회전 -10~10도와 밝기 0.90~1.10을 적용한다.

## 실행 환경

Windows의 `nail` 환경에서 PyTorch 2.13.0+cpu와 Torchvision 0.28.0+cpu를 사용했다. 의존성 버전은 `requirements.txt`에 기록했다.

아래는 Windows에서 프로젝트 폴더를 연 뒤 사용하는 명령이다. 새 환경에서는 먼저 의존성을 설치한다.

```cmd
python -m pip install -r requirements.txt
```

## 전처리 후 학습

과제 공지대로 `data/c00/`부터 `data/c13/`까지 클래스 폴더 바로 아래에 JPG 원본을 둔다. `c00`은 ㄱ, `c01`은 ㄴ, …, `c13`은 ㅎ이다. 학습·검증을 나누려면 자음당 최소 2장이 필요하다.

학습 명령을 실행하면 `data_prosessing/`이 없을 때 `data/`의 원본을 자동으로 전처리한다. 사진을 리사이즈해 `train/c00/`~`train/c13/`과 `val/c00/`~`val/c13/`으로 나누고, `split.csv`를 함께 저장한다. 이미 `data_prosessing/`이 있으면 학습에 재사용한다. 원본을 바꾼 뒤 다시 전처리하려면 기존 결과 폴더를 별도로 옮겨 둔다.

전처리를 따로 실행하려면 아래 명령을 사용한다. 다른 원본 폴더는 `--input 폴더명`으로 지정하며, 원본 옆에 `폴더명_prosessing/`을 만든다.

```cmd
python src/data_processing.py
```

사진만 넣으면 전처리에서 시드 42로 자음별 약 80:20 분할을 만든다. 이때 결과 CSV의 `group_id`는 비어 있다. 유사한 사진이 양쪽에 들어갈 수 있으므로 팀 비교에는 촬영 그룹을 확인해 만든 `data/split.csv`를 사용한다.

```csv
file,partition,group_id
c00/img_0001.jpg,train,session_a
```

`file`은 원본 데이터 폴더 기준 상대 경로다. 예시 행에 해당하는 결과 사진은 `train/c00/img_0001.jpg`에 저장된다. 원본에 팀 공통 CSV가 있으면 전처리에서 그 분할을 사용한다. 모든 사진을 한 번씩 기록하고, `train`과 `val` 양쪽에 14개 자음이 모두 있어야 한다. 같은 촬영 그룹은 한쪽 분할에만 둔다.

데이터를 추가하거나 제출할 때는 팀원당 자음별 30장 이상, 배경 3종·조명 2종 이상, JPG 긴 변 640px 이상, 중복과 흐림 여부를 확인한다. 학습은 폴더에서 라벨을 읽으므로 사진만으로도 실행할 수 있다.

```cmd
python src/train.py --augmentation off
python src/train.py --augmentation on
python src/train.py --augmentation both
```

저장소 루트에서 원하는 명령 하나를 실행한다. 학습 입력 경로는 `src/train.py` 상단의 `DATA_DIR`에서 정한다. `--augmentation`을 생략하면 `both`로 비교한다. 기본 설정은 Adam 0.001, 배치 32, 시드 42, 조건별 30 epoch다. 메모리 부족이면 `BATCH_SIZE`를 16 또는 8로 바꾸고 처음부터 실행한다.

새 결과는 `result/실행ID/`에 저장한다. 실행한 조건의 폴더에 학습 로그와 최고 가중치를, `selected/`에는 선택한 `best.pt`, `train_log.csv`, `confusion_matrix.png`를 둔다. 두 조건을 실행하면 `comparison.csv`도 저장한다. 최다 혼동 자음 쌍은 콘솔에 표시한다.

검증 정확도 우선, 동률이면 낮은 검증 손실, 둘 다 같으면 이른 epoch를 선택한다. 조건 간 완전 동률이면 off를 선택한다. 학습 지표는 실제 학습 배치에서 계산하고, 검증 지표는 증강 없이 전체 검증 배치에서 계산한다. 정확도 단위는 0~1이다.

## 제출용 촬영 정보

`data.zip`에는 팀 번호를 넣은 원본 데이터 폴더를 담는다. 공지의 03팀 예시는 `T03_data/`다. 이 폴더의 `c00/`~`c13/` 옆에 `metadata.csv`를 UTF-8로 저장한다. 사진 한 장마다 한 행을 만들고, 경로는 클래스 폴더부터 적는다.

```csv
file,class_id,signer,background,lighting,angle,scale
c00/img_0001.jpg,0,s01,wall,bright,front,medium
```

`file`과 `class_id`는 파일 목록과 폴더명에서 자동으로 채울 수 있다. 촬영자 ID와 촬영 조건은 실제 기록을 사용해, 조건이 같은 사진을 묶어 한꺼번에 입력한다. 예시 값은 팀에서 정한 일관된 표기로 바꿔도 된다.

## 학습된 가중치로 예측

새 전처리로 학습한 실행의 `selected/best.pt`를 저장소 루트의 `best.pt`로 복사한 뒤 실행한다. 예측은 시험 원본 사진을 읽어 같은 비율 유지 리사이즈와 여백 추가를 적용한다.

```cmd
python src/predict.py --input test_images/ --output predict_팀번호.csv
```

입력 폴더 바로 아래의 JPG/JPEG/PNG를 파일명 순으로 처리한다. 하위 폴더의 이미지, 빈 입력, 읽기 실패, 가중치 불일치는 중단 사유다. 모든 예측이 성공한 뒤 완성된 CSV를 최종 경로에 저장하며 기존 출력은 덮어쓰지 않는다.

라벨 0~13은 `ㄱ, ㄴ, ㄷ, ㄹ, ㅁ, ㅂ, ㅅ, ㅇ, ㅈ, ㅊ, ㅋ, ㅌ, ㅍ, ㅎ` 순서다.

## 과제 명령과 제출 폴더

학습과 예측을 실행할 때는 아래 폴더 구조를 사용한다. `data_processing.py`를 포함한 네 구현 파일은 같은 `src/`에 두고, 의존성 목록과 가중치는 한 단계 위에 둔다. 데이터 ZIP·로그·혼동행렬과 2분 이내 영상도 함께 준비한다.

```text
제출폴더/
  requirements.txt
  best.pt
  train_log.csv
  confusion_matrix.png
  data/                 c00~c13·metadata.csv, 팀 공통 split.csv가 있으면 함께 보관
  data_prosessing/      처음 학습할 때 생성하는 train/, val/, split.csv
  test_images/          시험 입력 폴더 예시
  src/
    data_processing.py
    model.py
    train.py
    predict.py
```

과제의 `python train.py`, `python predict.py --input ... --output ...` 명령은 제출폴더의 `src/`에서 실행한다. 아래 CMD 예시는 제출폴더에서 시작하며, 루트의 시험 입력을 읽고 루트에 결과 CSV를 만든다.

```cmd
cd src
python train.py
python predict.py --input ../test_images/ --output ../predict_팀번호.csv
```

이 경우에도 처음 학습할 때 필요하면 `../data_prosessing/`을 만들고, 학습 결과는 `../result/`에 저장한다. 추론은 `../best.pt`를 읽는다. 시험 이미지의 실제 위치에 맞춰 상대 경로를 지정한다.

## 현재 확인 범위

현재 작업용 데이터는 자음 14종의 JPG 420장이다. 조건별 30 epoch의 팀 전체 데이터 학습과 시험 입력 처리 시간 측정은 남아 있다.
