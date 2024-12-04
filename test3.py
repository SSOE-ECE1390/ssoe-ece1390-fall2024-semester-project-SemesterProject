import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

# Image paths
image_path = 'Data/crying_stock_photo.png'

image = cv2.imread(image_path)

# Create the options for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

# Morphological kernel for erosion
kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size for more/less erosion

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    
    # Convert the image to MediaPipe format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Retrieve the segmentation result
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask.numpy_view()

    # Convert the BGR image to RGB for processing
    image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a binary mask from category_mask
    binary_mask = (category_mask > 0.1).astype(np.uint8)  # Convert to binary (0 or 1)

    # Erode the mask to remove extra areas
    refined_mask = cv2.erode(binary_mask, kernel, iterations=5)

    # Apply the refined mask to the original image
    refined_condition = np.stack((refined_mask,) * 3, axis=-1).astype(bool)
    refined_hair_region = np.where(refined_condition, image_data, 0)

    # Visualize the refined hair region
    cv2.imshow(f'Refined Hair Region', refined_hair_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate mean RGB values for the refined hair region
    hair_pixels = refined_hair_region[refined_condition].reshape(-1, 3)  # Extract RGB pixels
    if hair_pixels.size > 0:  # Ensure there are hair pixels
        mean_rgb = hair_pixels.mean(axis=0)  # Calculate mean for each channel
        print(f'Mean R: {mean_rgb[0]}, Mean G: {mean_rgb[1]}, Mean B: {mean_rgb[2]}')
    else:
        print("No hair detected in the refined region.")