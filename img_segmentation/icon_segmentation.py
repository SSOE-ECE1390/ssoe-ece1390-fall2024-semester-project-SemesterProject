import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import math
import matplotlib.pyplot as plt

def segment_icon(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

     # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours from the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(image)

    # Fill the contours in the mask
    if contours:
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    output_path = os.path.abspath("Output/SeparateIcon/test.png")
    cv2.imwrite(output_path, mask)
    return [output_path, mask]

# Source: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#scrollTo=Yl_Oiye4mUuo
def segment_iconv2(path, keypoint=(0.68, 0.68), output_path="test2", model_path=os.path.relpath("Input/Other/magic_touch.tflite")):
    image = cv2.imread(path)
    cv2.imwrite(path, image)
    BG_COLOR = (0, 0, 0) # black
    MASK_COLOR = (255, 255, 255) # white

    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

    # Create the options that will be used for InteractiveSegmenter
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)
    
    # Create the interactive segmenter
    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        # Create the MediaPipe image file that will be segmented
        image = mp.Image.create_from_file(path)

        # Retrieve the masks for the segmented image
        (cx, cy) = keypoint
        roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                            keypoint=NormalizedKeypoint(x=cx, y=cy))
        
        segmentation_result = segmenter.segment(image, roi)
        category_mask = segmentation_result.category_mask
        # Generate solid color images for showing the output segmentation mask.
        image_data = image.numpy_view()
        image_shape = list(image_data.shape)
        if image_shape[-1] > 3:
            image_shape[-1] = 3
        
        fg_image = np.zeros(image_shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        # fg_image = fg_image[:3]
        
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        output_image = np.where(condition, fg_image, bg_image)

        # # Draw a white dot with black border to denote the point of interest
        # radius, thickness = 6, -1
        # cv2.circle(output_image, (center_x, center_y), radius + 5, (0, 0, 0), thickness)
        # cv2.circle(output_image, (center_x, center_y), radius, (255, 255, 255), thickness)
        # plt.imshow(output_image)
        # plt.title("icon mask")
        # plt.show()
        output_path = os.path.abspath(f"Output/SeparateIcon/{output_path}.jpeg")
        cv2.imwrite(output_path, output_image)
        return output_image
    
# # example usage
# # NOTE: input must be jpeg
image_path = os.path.relpath("Input/Icon/spongebob.jpeg")
image = cv2.imread(image_path)
print(image.shape)
segment_iconv2(image_path, model_path="magic_touch.tflite")
