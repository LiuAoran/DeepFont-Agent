import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import DeepFontModel  # 确保 model.py 在同一目录下

def predict_font(image_path, weights_path="deepfont_weights.pth", dataset_dir="./font_dataset"):
    # 1. 硬件检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # 2. 获取类别映射
    if not os.path.exists(dataset_dir):
        print(f"Error: Need '{dataset_dir}' to map class indices to font names.")
        return
    
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    num_classes = len(class_names)

    # 3. 图像预处理 ⬇️ 已经在这里帮你改好啦 ⬇️
    data_transforms = transforms.Compose([
        transforms.Resize((105, 150)),  # 👈 核心修改：强制缩放到高 105，宽 150
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 4. 加载图片并处理
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
        
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    model = DeepFontModel(num_classes=num_classes).to(device)
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]

    print("\n" + "="*30)
    print(f"Predicted Font: {predicted_class}")
    print(f"Confidence:     {confidence.item() * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    test_image_url = "font_dataset_for_eval/AlienBlock/AlienBlock_0.png"  
    predict_font(test_image_url)