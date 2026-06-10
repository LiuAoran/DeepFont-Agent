import os
import random
import string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_random_text():
    """Generate random text"""

    length = random.randint(5, 15)

    pool = (
        string.ascii_letters +
        string.digits +
        " "
    )

    text = ''.join(
        random.choice(pool)
        for _ in range(length)
    )

    text = " ".join(text.split())

    if len(text) == 0:
        text = random.choice(string.ascii_letters)

    return text


def apply_shading(img):
    """Simulate uneven illumination"""

    h, w = img.shape[:2]

    u, v = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h)
    )

    alpha = random.uniform(0.3, 0.8)

    if random.random() > 0.5:
        mask = alpha * u + (1 - alpha) * v
    else:
        mask = alpha * v + (1 - alpha) * u

    mask = (mask * 255).astype(np.uint8)

    return cv2.addWeighted(
        img,
        0.7,
        mask,
        0.3,
        0
    )


def apply_perspective(img):
    """Simulate camera tilt and perspective distortion"""

    h, w = img.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    max_offset = int(h * 0.15)

    d1 = random.randint(0, max_offset)
    d2 = random.randint(0, max_offset)

    dst_pts = np.float32([
        [d1, d2],
        [w - 1 - d1, d2],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    matrix = cv2.getPerspectiveTransform(
        src_pts,
        dst_pts
    )

    return cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        borderValue=255
    )


def apply_noise_and_blur(img):
    """Apply Gaussian blur and noise"""

    if random.random() > 0.5:

        kernel_size = random.choice([3, 5])

        img = cv2.GaussianBlur(
            img,
            (kernel_size, kernel_size),
            0
        )

    if random.random() > 0.5:

        noise = np.random.normal(
            0,
            random.uniform(5, 15),
            img.shape
        ).astype(np.float32)

        img = np.clip(
            img.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    return img


def crop_text_region(img):
    """Crop text bounding box"""

    coords = cv2.findNonZero(255 - img)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    pad = 5

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)

    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)

    return img[y1:y2, x1:x2]


def resize_keep_ratio(
    img,
    target_width=300,
    target_height=105
):
    """Resize while preserving aspect ratio"""

    h, w = img.shape

    scale = min(
        target_width / w,
        target_height / h
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img_resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.ones(
        (target_height, target_width),
        dtype=np.uint8
    ) * 255

    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w
    ] = img_resized

    return canvas


def generate_font_dataset(
    font_dir,
    output_dir,
    images_per_font=100
):
    """Generate synthetic font dataset"""

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    target_height = 105
    target_width = 300

    font_files = [
        f for f in os.listdir(font_dir)
        if f.endswith(('.ttf', '.otf'))
    ]

    print(
        f"Found {len(font_files)} fonts in '{font_dir}'"
    )

    for font_file in font_files:

        font_name = os.path.splitext(
            font_file
        )[0]

        font_path = os.path.join(
            font_dir,
            font_file
        )

        class_dir = os.path.join(
            output_dir,
            font_name
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )

        print(
            f"Generating patches for: {font_name}"
        )

        for i in range(images_per_font):

            text = generate_random_text()

            font_size = random.randint(
                20,
                90
            )

            try:
                font = ImageFont.truetype(
                    font_path,
                    font_size
                )
            except IOError:
                continue

            canvas_w = 600
            canvas_h = 200

            img_pil = Image.new(
                'L',
                (canvas_w, canvas_h),
                color=255
            )

            draw = ImageDraw.Draw(
                img_pil
            )

            bbox = draw.textbbox(
                (0, 0),
                text,
                font=font
            )

            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            x = (canvas_w - text_w) // 2
            y = (canvas_h - text_h) // 2

            draw.text(
                (x, y),
                text,
                fill=0,
                font=font
            )

            img_cv = np.array(
                img_pil
            )

            img_cv = apply_perspective(
                img_cv
            )

            img_cv = apply_shading(
                img_cv
            )

            img_cv = apply_noise_and_blur(
                img_cv
            )

            img_cv = crop_text_region(
                img_cv
            )

            final_patch = resize_keep_ratio(
                img_cv,
                target_width,
                target_height
            )

            save_path = os.path.join(
                class_dir,
                f"{font_name}_{i}.png"
            )

            cv2.imwrite(
                save_path,
                final_patch
            )

        print(
            f"Finished {font_name}"
        )


if __name__ == "__main__":

    INPUT_FONT_DIR = "./fonts"

    OUTPUT_DATASET_DIR = "./font_dataset_for_eval"

    generate_font_dataset(
        INPUT_FONT_DIR,
        OUTPUT_DATASET_DIR,
        images_per_font=200
    )

    print("\nDataset synthesis complete successfully!")