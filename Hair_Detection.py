import mediapipe as mp
import numpy as np
import cv2
import os
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hair color to emoji mapping
hair_emoji_map = {
    'brown': 'emojis/brown_short.png',
    'blonde': 'emojis/blonde_short.png',
    'black': 'emojis/black_short.png',
    'red': 'emojis/red_short.png',
    'gray': 'emojis/gray_short.png',
    
}

# Custom color category mapping
color_category_map = {
    'darkolivegreen': 'brown',
    'darkslategray': 'brown',
    'darkkhaki': 'blonde',
    'saddlebrown': 'ginger',
    'black': 'black',
    'darkred': 'red',
    'gray': 'gray',
    
}

def mediapipeHairSegmentation(image):
    """Segment hair region using MediaPipe ImageSegmenter."""
    kernel = np.ones((5, 5), np.uint8)
    model_asset_path = os.path.abspath("hair_segmenter.tflite")
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        binary_mask = (category_mask > 0.1).astype(np.uint8)
        refined_mask = cv2.erode(binary_mask, kernel, iterations=3)
        return refined_mask

def overlay_emoji(image, emoji_path, mask):
    """Overlay an emoji image onto the detected hair region using the mask."""
    # Load emoji image with alpha channel
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

    # Find bounding box of the largest contour in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No hair region detected in the mask.")
        return image

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Resize the emoji to the bounding box size
    emoji_resized = cv2.resize(emoji, (w, h))
    emoji_rgb = emoji_resized[:, :, :3]
    emoji_alpha = emoji_resized[:, :, 3] / 255.0

    # Overlay emoji onto the original image at the bounding box location
    for c in range(3):
        image[y:y+h, x:x+w, c] = (1 - emoji_alpha) * image[y:y+h, x:x+w, c] + emoji_alpha * emoji_rgb[:, :, c]
    return image

def process_images_from_csv(csv_path, image_dir, output_dir):
    """Read hair colors from CSV and overlay emojis on corresponding images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load hair colors from the CSV file
    hair_data = pd.read_csv(csv_path)

    for _, row in hair_data.iterrows():
        image_name = row['Image']
        detected_color = row['Color']

        # Map detected color to emoji category
        hair_category = color_category_map.get(detected_color.lower(), None)
        if not hair_category:
            print(f"No emoji mapping found for color: {detected_color}")
            continue

        emoji_path = hair_emoji_map.get(hair_category, None)
        if not emoji_path:
            print(f"No emoji file found for hair category: {hair_category}")
            continue

        # Load the image
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        print(f"Processing image: {image_name} with hair color: {detected_color} ({hair_category})")

        # Generate hair mask and overlay emoji
        mask = mediapipeHairSegmentation(image)
        final_image = overlay_emoji(image, emoji_path, mask)

        # Save the final image
        output_path = os.path.join(output_dir, f"HairAddition_{os.path.basename(image_name)}")
        cv2.imwrite(output_path, final_image)
        print(f"Output saved: {output_path}")

# Paths
csv_path = "hair_colors.csv"       # Path to the CSV file
image_dir = "Data"                 # Directory with input images
output_dir = "Output"              # Directory to save output images

# Process images using the hair color CSV
process_images_from_csv(csv_path, image_dir, output_dir)
