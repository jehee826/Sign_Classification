from torch import nn


class SignClassifier(nn.Module):
    # 합성곱 다섯 층으로 손의 특징을 추출하고 자음 분류층을 만든다.
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

        # 특징 맵을 채널마다 4×4로 줄여 자음 14개의 점수를 계산한다.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 14),
        )
        # 최고 기록 실험과 같은 순서로 초기화하려고 분류층 뒤에 두 층을 만든다.
        self.features.extend([
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ])

    # 이미지에서 특징을 추출하고 자음별 점수를 계산한다.
    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)
