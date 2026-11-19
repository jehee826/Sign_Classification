import torch
from torch import nn

from data_processing import INPUT_SIZE, LABELS


PREPROCESSING = {
    "version": 2,
    "orientation": "as_stored",
    "color": "RGB",
    "resize": "fit_longest_side_bilinear",
    "padding": "center_black",
    "layout": "CHW",
    "normalization": "float32_div_255",
}


CHECKPOINT_INFO = {
    "format_version": 1,
    "model_name": "SignClassifier_v1",
    "input_size": INPUT_SIZE,
    "labels": LABELS,
    "preprocessing": PREPROCESSING,
}


class SignClassifier(nn.Module):
    # 특징 추출용 합성곱 3블록과 자음 분류층을 만든다.
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(LABELS)),
        )

    # 이미지에서 특징을 추출하고 자음별 점수를 계산한다.
    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


# 가중치의 형식을 확인하고 모델에 불러온다.
def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("가중치 파일에 모델 상태가 없습니다.")
    for key, expected in CHECKPOINT_INFO.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"가중치 설정이 현재 코드와 다릅니다: {key}")

    model = SignClassifier()
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)
