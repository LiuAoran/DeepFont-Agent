import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

from model import DeepFontModel


class ResizeWithPadding:
    """
    保持长宽比缩放，并填充到指定尺寸
    """

    def __init__(self, target_size, fill=255):
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img):

        # 保持比例缩放
        img.thumbnail(
            (self.target_w, self.target_h),
            Image.Resampling.LANCZOS
        )

        width, height = img.size

        pad_w = self.target_w - width
        pad_h = self.target_h - height

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )

        img = ImageOps.expand(
            img,
            border=padding,
            fill=self.fill
        )

        return img


def predict_font(
        image_path,
        weights_path="deepfont_weights.pth",
        dataset_dir="./font_dataset"
):

    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 2. Get Class Mapping
    if not os.path.exists(dataset_dir):
        print(
            f"Error: Need '{dataset_dir}' "
            f"to map class indices to font names."
        )
        return

    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    num_classes = len(class_names)

    # 3. Image Preprocessing
    data_transforms = transforms.Compose([
        ResizeWithPadding((105, 300)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 4. Load Image
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = (
        data_transforms(image)
        .unsqueeze(0)
        .to(device)
    )

    # 5. Load Model
    model = DeepFontModel(
        num_classes=num_classes
    ).to(device)

    if not os.path.exists(weights_path):
        print(
            f"Error: Weights file "
            f"'{weights_path}' not found."
        )
        return

    model.load_state_dict(
        torch.load(
            weights_path,
            map_location=device
        )
    )

    model.eval()

    # 6. Inference
    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = F.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=0
        )

        predicted_class = class_names[
            predicted_idx.item()
        ]

    # 7. Print Result
    print("\n" + "=" * 40)
    print(f"Predicted Font : {predicted_class}")
    print(
        f"Confidence     : "
        f"{confidence.item()*100:.2f}%"
    )
    print("=" * 40)

    # Top-5
    top5_prob, top5_idx = torch.topk(
        probabilities,
        k=min(5, len(class_names))
    )

    print("\nTop-5 Predictions:")

    for rank, (idx, prob) in enumerate(
            zip(top5_idx, top5_prob),
            start=1):

        print(
            f"{rank}. "
            f"{class_names[idx.item()]} "
            f"({prob.item()*100:.2f}%)"
        )


if __name__ == "__main__":

    test_image = (
        "test_data/"
        "image.png"
    )

    predict_font(test_image)