import os
from PIL import Image, ImageDraw, ImageFont

def generate_dataset(output_dir="data", image_size=(224, 224), fonts_dir="fonts", num_samples=100):
    """
    Generate font dataset by creating images with text rendered in different fonts.

    Parameters:
        output_dir (str): Directory where the dataset will be saved.
        image_size (tuple): Size of the generated images.
        fonts_dir (str): Directory containing .ttf or .otf font files.
        num_samples (int): Number of images per font to generate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fonts = [f for f in os.listdir(fonts_dir) if f.endswith(".ttf") or f.endswith(".otf")]
    
    if not fonts:
        raise ValueError(f"No font files found in directory: {fonts_dir}")

    for font_file in fonts:
        font_name = os.path.splitext(font_file)[0]
        font_path = os.path.join(fonts_dir, font_file)
        font_output_dir = os.path.join(output_dir, font_name)
        os.makedirs(font_output_dir, exist_ok=True)

        try:
            font = ImageFont.truetype(font_path, size=40)
        except Exception as e:
            print(f"Error loading font {font_file}: {e}")
            continue

        for i in range(num_samples):
            img = Image.new("RGB", image_size, color="white")
            draw = ImageDraw.Draw(img)
            text = f"Sample {i+1}"

            text_width, text_height = draw.textsize(text, font=font)
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

            draw.text(position, text, fill="black", font=font)

            output_path = os.path.join(font_output_dir, f"sample_{i+1}.jpg")
            img.save(output_path)

        print(f"Generated {num_samples} images for font: {font_name}")

if __name__ == "__main__":
    generate_dataset()