import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt

# Source: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md
def segment_person(image_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    BG_COLOR = (0, 0, 0) # black
    MASK_COLOR = (255, 255, 255) # white

    with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        output_path = os.path.abspath("Output/SeparateBackground/test.png")
        cv2.imwrite(output_path, output_image)
        # plt.imshow(results.segmentation_mask > 0.1, cmap='gray')
        # plt.show()

        binary_mask = (results.segmentation_mask > 0.1).astype(np.uint8) * 255
        return binary_mask

# # example usage
# image_path = os.path.abspath("Input/Face/1 (1).jpeg")
# segment_person(image_path)