# CNNSON 제출용 코드

팀에서 촬영한 한글 지문자 자음 14종(c00~c13)을 분류하는 PyTorch CNN 코드입니다. 
사전학습 가중치와 외부 데이터셋을 사용하지 않습니다. 기존 main 브랜치의 파일은 수정하지 않았습니다.

## 파일

- model.py: 32 → 64 → 128 → 256 구성의 CNNSON, 14개 출력
- data.py: 클래스별 80:20 분할(seed=42), 이미지 캐싱, 전처리
- train.py: 학습/검증, 최고 체크포인트 저장, CSV/그래프/혼동행렬, 증강 비교
- predict.py: 예측 시 전처리 포함, 이미지/폴더 단위 CSV 생성
- requirements.txt: 실행 의존성
- run_train.bat: Windows 실행

## 실행 (저장소 루트에서)

Python 3.12 및 NVIDIA CUDA가 지원되는 PyTorch를 준비합니다.

~~~powershell
python -m pip install -r src/requirements.txt
python src/train.py --data team_data --epochs 10
~~~

클래스 폴더는 team_data/c00 ~ team_data/c13 형태여야 합니다. 
실행 결과는 기본적으로 src/results/에 저장됩니다. 
원본 이미지와 체크포인트는 GitHub 브랜치에 포함하지 않았습니다.

증강 여부 비교:

~~~powershell
python src/train.py --data team_data --epochs 30 --compare-augmentation
~~~

이미지 하나 예측:

~~~powershell
python src/predict.py --input "test.jpg" --weights "src/results/best_cnnson.pth"
~~~

여러 이미지 폴더 예측:

~~~powershell
python src/predict.py --input "test_images" --output "src/results/predictions.csv"
~~~

## 실험 설정

- 원본: 팀 촬영 데이터만 사용, 각 클래스의 정답은 폴더 이름
- 전처리: EXIF 방향 보정, RGB, 64×64 변환, 평균/표준편차 0.5 정규화
- 증강 실험: 학습 데이터에만 ±7° 회전
- 학습: Adam, lr=0.001, batch_size=32, CrossEntropyLoss
- 파라미터: 약 2,238,126개(14개 클래스 기준)
- 검증 정확도는 독립 시험 데이터 정확도가 아님

## 결과 파일

- src/results/best_cnnson.pth
- src/results/train_log.csv
- src/results/learning_curve.png
- src/results/confusion_matrix.png
- 비교 실험 시 src/results/aug_on/, src/results/aug_off/, src/results/comparison.csv

## 제출/시험 전 확인

GitHub README를 기준으로 작성했습니다. 공지 PDF에만 있는 예측 입출력 규칙,
제출 경로 및 파일명은 공지와 대조해 최종 확정해야 합니다.
예측 CSV는 현재 filename, class_index, class_code, jamo, confidence 열을 사용합니다.
실제 metadata.csv의 c00~c13 대응 순서도 시험 전에 확인하십시오.
2분 이내 시연 영상, 원본 학습 데이터, 최종 가중치 파일은 별도로 준비해야 합니다.
