import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFontModel(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepFontModel, self).__init__()
        
        # Branch 1: Main Feature Extractor (Input: 1 x 105 x 150)
        # Conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Conv4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        # Input size calculation after convolutions and poolings:
        # 105x150 -> Conv1/Pool1: 26x37 -> Conv2/Pool2: 13x18 -> Conv3/4/Pool4: 6x9
        self.fc1 = nn.Linear(256 * 6 * 9, 1024)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Output Layer (Matches your font count, e.g., 10)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Layer 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Layer 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the feature maps for dense layers
        x = x.view(x.size(0), -1)
        
        # Dense 1
        x = self.dropout1(F.relu(self.fc1(x)))
        
        # Dense 2
        x = self.dropout2(F.relu(self.fc2(x)))
        
        # Final Logits output
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Quick dimensions sanity check
    model = DeepFontModel(num_classes=10)
    
    # Simulate a batch of 4 grayscale sample images from our pipeline (Batch_size, Channels, Height, Width)
    mock_input = torch.randn(4, 1, 105, 150)
    output = model(mock_input)
    
    print("Model loaded successfully!")
    print(f"Input shape:  {mock_input.shape}")
    print(f"Output shape: {output.shape} -> (Batch_size, Num_classes)")