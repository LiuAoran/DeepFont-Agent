import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from model import DeepFontModel  # 确保 model.py 在同一目录下

def evaluate_model(dataset_dir="./font_dataset_for_eval", weights_path="deepfont_weights.pth"):
    # 1. 硬件检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device for evaluation: {device}")

    # 2. 严格一致的图像预处理
    data_transforms = transforms.Compose([
        transforms.Resize((105, 150)),  # 必须与数据生成和推理完全一致
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 3. 加载评估数据集
    if not os.path.exists(dataset_dir):
        print(f"Error: '{dataset_dir}' folder not found.")
        return

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    eval_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    # 4. 初始化模型并加载已训练的权重
    model = DeepFontModel(num_classes=num_classes).to(device)
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # 💡 切换到评估模式

    # 5. 开始收集预测结果
    all_preds = []
    all_labels = []
    correct_top1 = 0
    correct_top3 = 0
    total_samples = 0

    print("\nEvaluating model performance...")
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # --- 计算 Top-1 和 Top-3 准确率 ---
            total_samples += labels.size(0)
            
            # Top-1
            _, max_preds = torch.max(outputs, 1)
            correct_top1 += (max_preds == labels).sum().item()
            
            # Top-3
            _, top3_preds = torch.topk(outputs, k=min(3, num_classes), dim=1)
            # labels.view(-1, 1) 变成二维以支持 broadcast 比较
            correct_top3 += (top3_preds == labels.view(-1, 1)).sum().item()

            # 收集整条流水线的数据用于统计报告
            all_preds.extend(max_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. 计算最终指标
    top1_acc = (correct_top1 / total_samples) * 100
    top3_acc = (correct_top3 / total_samples) * 100

    # 7. 打印评估报告
    print("\n" + "="*50)
    print("                EVALUATION REPORT                ")
    print("="*50)
    print(f"Total Samples Tested: {total_samples}")
    print(f"Top-1 Accuracy (首选准确率): {top1_acc:.2f}%")
    print(f"Top-3 Accuracy (前三准确率): {top3_acc:.2f}%")
    print("-"*50)
    
    # 打印详细的每个类别的 Precision, Recall, F1-score
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 打印简易混淆矩阵
    print("-"*50)
    print("Confusion Matrix (混淆矩阵 - 横轴预测，纵轴真实):")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("="*50)

if __name__ == "__main__":
    # 指向你的数据集路径
    evaluate_model(dataset_dir="./font_dataset_for_eval", weights_path="deepfont_weights.pth")