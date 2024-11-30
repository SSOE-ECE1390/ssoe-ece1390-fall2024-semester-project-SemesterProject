import cv2
import os

def resize(image, target_image, output_path="test"):
    target_height, target_width = target_image.shape[:2]
    resized_image = cv2.resize(image, (target_width, target_height))
    output_path = os.path.abspath(f"Output/OtherFeatures/{output_path}.jpeg")
    cv2.imwrite(output_path, resized_image)
    return resized_image