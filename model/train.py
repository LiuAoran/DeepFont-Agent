import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import DeepFontModel

def train_model():
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Data Transforms
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # 3. Load Dataset
    dataset_dir = "./font_dataset"
    if not os.path.exists(dataset_dir):
        print(f"Error: '{dataset_dir}' folder not found.")
        return

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
    num_classes = len(full_dataset.classes)
    print(f"Detected Classes ({num_classes}): {full_dataset.classes}")

    # 4. Train/Val Split (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data Loaders (Increased batch_size to 64 for more stable gradients)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset Split: {train_size} training items, {val_size} validation items.")

    # 5. Initialize Network, Loss, and Optimizer
    model = DeepFontModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Lower initial learning rate for better stability
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    # Learning rate decays by half every 4 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # 6. Training Loop
    epochs = 20
    print("\nStarting Optimized Training Pipeline...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # ⬇️ CRITICAL FIX: Insert this line to clip gradients ⬇️
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        
        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f} "
              f"| Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
              
        # Step the learning rate scheduler
        scheduler.step()

    # 7. Save Weights
    model_save_path = "deepfont_weights.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining complete! Optimized weights saved to '{model_save_path}'.")

if __name__ == "__main__":
    train_model()