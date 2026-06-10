import os
import random
import string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_random_text():
    """Generate a random string with mixed lengths and characters"""
    length = random.randint(4, 8)
    # Mix uppercase, lowercase, and digits
    pool = string.ascii_letters + string.digits
    return ''.join(random.choice(pool) for _ in range(length))

def apply_shading(img):
    """Simulate uneven illumination (gradient shading)"""
    h, w = img.shape[:2]
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    alpha = random.uniform(0.3, 0.8)
    
    # Randomly choose linear gradient direction
    mask = alpha * u + (1 - alpha) * v if random.random() > 0.5 else alpha * v + (1 - alpha) * u
    mask = (mask * 255).astype(np.uint8)
    
    # Blend the shading mask into the image
    return cv2.addWeighted(img, 0.7, mask, 0.3, 0)

def apply_perspective(img):
    """Simulate camera tilt and perspective distortion"""
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    
    # Random pixel offsets for corners
    max_offset = int(h * 0.15)
    d1 = random.randint(0, max_offset)
    d2 = random.randint(0, max_offset)
    
    dst_pts = np.float32([
        [d1, d2], 
        [w - 1 - d1, d2], 
        [0, h - 1], 
        [w - 1, h - 1]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderValue=255)

def apply_noise_and_blur(img):
    """Apply random Gaussian blur and digital noise"""
    # 1. Gaussian Blur
    if random.random() > 0.5:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # 2. Gaussian Noise
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return img

def generate_font_dataset(font_dir, output_dir, images_per_font=100):
    """Loop through all fonts and generate synthetic patches with noise"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Target normalized height matching DeepFont requirements
    target_height = 105
    target_width = 150  # Fixed width for network feeding consistency
    
    # Find all font files
    font_files = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
    print(f"Found {len(font_files)} fonts in '{font_dir}'. starting synthesis...")

    for font_file in font_files:
        font_name = os.path.splitext(font_file)[0]
        font_path = os.path.join(font_dir, font_file)
        
        # Create a specific directory for this font class
        class_dir = os.path.join(output_dir, font_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generating patches for: {font_name}")
        
        for i in range(images_per_font):
            text = generate_random_text()
            
            # Base text rendering using Pillow
            font_size = random.randint(55, 70)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                continue
                
            # Render text on a generous canvas to avoid truncation
            canvas_w, canvas_h = 400, 150
            img_pil = Image.new('L', (canvas_w, canvas_h), color=255)
            draw = ImageDraw.Draw(img_pil)
            
            # Draw text near center
            draw.text((30, 30), text, fill=0, font=font)
            
            # Convert to OpenCV format for geometric/pixel transformations
            img_cv = np.array(img_pil)
            
            # Apply DeepFont pipeline expansions
            img_cv = apply_perspective(img_cv)
            img_cv = apply_shading(img_cv)
            img_cv = apply_noise_and_blur(img_cv)
            
            # Resize step: Normalize height to 105, crop/resize width to 150
            h, w = img_cv.shape
            scale = target_height / h
            new_w = int(w * scale)
            img_resized = cv2.resize(img_cv, (new_w, target_height))
            
            # Center crop or pad horizontally to match strict target_width
            if new_w >= target_width:
                start_x = (new_w - target_width) // 2
                final_patch = img_resized[:, start_x:start_x + target_width]
            else:
                # Pad with white pixels if text width is smaller than 150
                pad_width = target_width - new_w
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                final_patch = cv2.copyMakeBorder(img_resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
            
            # Save the final synthesized patch
            save_path = os.path.join(class_dir, f"{font_name}_{i}.png")
            cv2.imwrite(save_path, final_patch)

if __name__ == "__main__":
    # Configure your input directory (where your 10 fonts are)
    INPUT_FONT_DIR = "./fonts" 
    OUTPUT_DATASET_DIR = "./font_dataset_for_eval"
    
    # Generate 100 training images per font family
    generate_font_dataset(INPUT_FONT_DIR, OUTPUT_DATASET_DIR, images_per_font=100)
    print("\nDataset synthesis complete successfully!")