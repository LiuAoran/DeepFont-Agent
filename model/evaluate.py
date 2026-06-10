import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from model import DeepFontModel  # Ensure model.py is in the same directory

def evaluate_model(dataset_dir="./font_dataset_for_eval", weights_path="deepfont_weights.pth"):
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device for evaluation: {device}")

    # 2. Image Preprocessing (Must match data generation and inference exactly)
    data_transforms = transforms.Compose([
        transforms.Resize((105, 150)),  
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 3. Load Evaluation Dataset
    if not os.path.exists(dataset_dir):
        print(f"Error: '{dataset_dir}' folder not found.")
        return

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    eval_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    # 4. Initialize Model and Load Trained Weights
    model = DeepFontModel(num_classes=num_classes).to(device)
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Switch to evaluation mode

    # 5. Collect Predictions
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
            
            # --- Calculate Top-1 and Top-3 Accuracy ---
            total_samples += labels.size(0)
            
            # Top-1
            _, max_preds = torch.max(outputs, 1)
            correct_top1 += (max_preds == labels).sum().item()
            
            # Top-3
            _, top3_preds = torch.topk(outputs, k=min(3, num_classes), dim=1)
            # Reshape labels to 2D to support broadcast comparison
            correct_top3 += (top3_preds == labels.view(-1, 1)).sum().item()

            # Collect metrics data
            all_preds.extend(max_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Compute Final Metrics
    top1_acc = (correct_top1 / total_samples) * 100
    top3_acc = (correct_top3 / total_samples) * 100

    # 7. Print Evaluation Report
    print("\n" + "="*50)
    print("                EVALUATION REPORT                ")
    print("="*50)
    print(f"Total Samples Tested: {total_samples}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print("-"*50)
    
    # Detailed Precision, Recall, and F1-score per class
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    print("-"*50)
    print("Confusion Matrix (Rows: True, Columns: Predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("="*50)

if __name__ == "__main__":
    evaluate_model(dataset_dir="./font_dataset_for_eval", weights_path="deepfont_weights.pth")