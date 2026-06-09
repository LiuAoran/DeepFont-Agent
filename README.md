# DeepFont-Agent

### A Reproduction of the DeepFont Font Recognition System

This project aims to replicate the **DeepFont** font recognition model using PyTorch. The goal is to identify a font's type from an input image of text. The project pipeline includes data collection, model construction, training, validation, and deployment.

### Project Structure
```
DeepFont-Agent/
├── data/                   # Generated dataset from Google Fonts
├── datasets/               # Custom PyTorch data loaders
├── models/                 # CNN-based model architecture
├── utils/                  # Helper functions (e.g., data augmentation)
├── train.py                # Training script
├── validate.py             # Validation script
├── predict.py              # Script for prediction on new images
└── README.md               # Project documentation
```

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/LiuAoran/DeepFont-Agent.git
   cd DeepFont-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate dataset from Google Fonts:
   ```bash
   python utils/generate_dataset.py
   ```

4. Train the model:
   ```bash
   python train.py
   ```

5. Validate the model:
   ```bash
   python validate.py
   ```

6. Predict using a trained model:
   ```bash
   python predict.py --image_path ./sample.jpg
   ```

### Features
- **Google Fonts Integration**: Data generated dynamically from open-source fonts.
- **CNN Architecture**: Built using PyTorch for font classification.
- **End-to-End Pipeline**: Data preparation, training, validation, and prediction.