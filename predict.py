import torch
from torchvision import transforms
from PIL import Image
import os

# 定义数据预处理的变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),         # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 定义基础模型结构（需和训练时一致）
class DeepFontModel(torch.nn.Module):
    def __init__(self):
        super(DeepFontModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 112 * 112, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)  # 假设有 10 个分类
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 加载模型权重
def load_model(weights_path):
    model = DeepFontModel()
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式
    return model

# 进行单张图片的预测
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)  # Softmax 转化为概率
        predicted_class = torch.argmax(probs).item()
    
    print("Prediction:")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {class_names[predicted_class]} (Confidence: {probs[predicted_class]:.2f})")

# 主函数
if __name__ == "__main__":
    # 定义模型权重路径和类别名称
    weights_path = "model_weights.pth"  # 模型权重文件（需要用户提供）
    class_names = ['Font1', 'Font2', 'Font3', 'Font4', 'Font5', 'Font6', 'Font7', 'Font8', 'Font9', 'Font10']
    
    # 加载模型
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
    else:
        model = load_model(weights_path)
        print("Model loaded successfully.")
    
        # 对图片进行预测
        image_path = "test_image.jpg"  # 测试图片路径（需要用户提供）
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
        else:
            predict_image(image_path, model, class_names)