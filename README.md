# DeepFont-Easy

A lightweight font recognition system built with PyTorch, inspired by the [DeepFont](https://arxiv.org/abs/1507.03196) paper. It trains a CNN to classify fonts from text-image patches and includes a full pipeline: font downloading, synthetic dataset generation, training, evaluation, and inference.

---

## Project Structure

```
DeepFont-Easy/
├── model/
│   ├── model.py          # CNN architecture (DeepFontModel)
│   ├── train.py          # Training script
│   ├── evaluate.py       # Top-1 / Top-3 accuracy evaluation
│   └── predict.py        # Single-image font prediction
└── utils/
    ├── catch_fonts.py         # Download fonts from Google Fonts
    ├── fonts_list.txt         # List of fonts to download
    ├── generate_dataset.py    # Generate training dataset
    └── generate_dataset_for_eval.py  # Generate evaluation dataset
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- OpenCV (`cv2`)
- NumPy

Install dependencies:

```bash
pip install torch torchvision pillow opencv-python numpy
```

---

## Quick Start

### Step 1 — Download Fonts

Edit `utils/fonts_list.txt` to list the fonts you want, then run:

```bash
python utils/catch_fonts.py
```

This downloads `.ttf` files from Google Fonts into `./fonts/`.  
You can also manually place any `.ttf` or `.otf` files into `./fonts/`.

### Step 2 — Generate Training Dataset

```bash
python utils/generate_dataset.py
```

Generates synthetic text-image patches (default: 1000 images per font) in `./font_dataset/`.

### Step 3 — Train the Model

```bash
cd model
python train.py
```

Trains for 20 epochs with an 80/20 train/val split. Saves weights to `model/deepfont_weights.pth`.

### Step 4 — Evaluate

Generate an evaluation dataset (separate from training data):

```bash
python utils/generate_dataset_for_eval.py
```

Then run evaluation:

```bash
cd model
python evaluate.py
```

Reports **Top-1** and **Top-3** accuracy.

### Step 5 — Predict a Font

```bash
cd model
python predict.py
```

By default, it reads `test_data/image.png`. Edit `predict.py` to point to your own image. Outputs the predicted font name and a Top-5 ranking with confidence scores.

---

## Model Architecture

`DeepFontModel` is a CNN with the following layers:

| Layer | Output Channels | Kernel | Notes |
|-------|----------------|--------|-------|
| Conv1 + BN + MaxPool | 64 | 5×5, stride 2 | |
| Conv2 + BN + MaxPool | 128 | 5×5, stride 1 | |
| Conv3 + BN | 256 | 3×3 | |
| Conv4 + BN + MaxPool | 256 | 3×3 | |
| AdaptiveAvgPool | — | 6×6 output | |
| FC1 (Dropout 0.5) | 1024 | | |
| FC2 (Dropout 0.5) | 1024 | | |
| FC3 | num_classes | | |

Input: grayscale patches of size **1 × 105 × 300**.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
