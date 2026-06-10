import os
import torch
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import DeepFontModel


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def evaluate():

    device = get_device()

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    dataset_dir = "./font_dataset_for_eval"

    dataset = datasets.ImageFolder(
        root=dataset_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )

    num_classes = len(dataset.classes)

    print(
        f"Classes: {num_classes}"
    )

    model = DeepFontModel(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        torch.load(
            "deepfont_weights.pth",
            map_location=device
        )
    )

    model.eval()

    total = 0
    top1_correct = 0
    top3_correct = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = F.softmax(
                outputs,
                dim=1
            )

            # Top-1
            _, pred1 = torch.max(
                probs,
                dim=1
            )

            top1_correct += (
                pred1 == labels
            ).sum().item()

            # Top-3
            _, pred3 = torch.topk(
                probs,
                k=min(3, num_classes),
                dim=1
            )

            for i in range(
                labels.size(0)
            ):

                if labels[i] in pred3[i]:
                    top3_correct += 1

            total += labels.size(0)

    top1_acc = (
        top1_correct /
        total * 100
    )

    top3_acc = (
        top3_correct /
        total * 100
    )

    print("\n========== RESULT ==========")

    print(
        f"Samples       : {total}"
    )

    print(
        f"Top-1 Accuracy: "
        f"{top1_acc:.2f}%"
    )

    print(
        f"Top-3 Accuracy: "
        f"{top3_acc:.2f}%"
    )

    print("============================")


if __name__ == "__main__":
    evaluate()