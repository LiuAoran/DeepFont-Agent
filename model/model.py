import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFontModel(nn.Module):

    def __init__(self, num_classes=10):
        super(DeepFontModel, self).__init__()

        # ==================================================
        # Conv Block 1
        # ==================================================
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ==================================================
        # Conv Block 2
        # ==================================================
        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.bn2 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ==================================================
        # Conv Block 3
        # ==================================================
        self.conv3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(256)

        # ==================================================
        # Conv Block 4
        # ==================================================
        self.conv4 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(256)

        self.pool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ==================================================
        # Adaptive Pool
        # ==================================================
        self.global_pool = nn.AdaptiveAvgPool2d((6, 6))

        # ==================================================
        # Fully Connected
        # ==================================================
        self.fc1 = nn.Linear(
            256 * 6 * 6,
            1024
        )

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            1024,
            1024
        )

        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(
            1024,
            num_classes
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(
                    m.weight,
                    0,
                    0.01
                )

                nn.init.constant_(
                    m.bias,
                    0
                )

    def forward(self, x):

        # Block 1
        x = self.pool1(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        # Block 2
        x = self.pool2(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # Block 3
        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        # Block 4
        x = self.pool4(
            F.relu(
                self.bn4(
                    self.conv4(x)
                )
            )
        )

        # Adaptive Pool
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            1
        )

        # FC1
        x = self.dropout1(
            F.relu(
                self.fc1(x)
            )
        )

        # FC2
        x = self.dropout2(
            F.relu(
                self.fc2(x)
            )
        )

        # Output
        x = self.fc3(x)

        return x


if __name__ == "__main__":

    model = DeepFontModel(
        num_classes=100
    )

    mock_input = torch.randn(
        4,
        1,
        105,
        300
    )

    output = model(
        mock_input
    )

    print("Model loaded successfully!")
    print(f"Input shape : {mock_input.shape}")
    print(f"Output shape: {output.shape}")