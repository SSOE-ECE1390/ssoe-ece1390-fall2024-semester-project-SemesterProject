import cv2
import numpy as np
import os

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

# example usage
image_path = os.path.abspath("Input/Icon/Comiccon_Decals_Square_for_Shopify-42.webp")
segment_icon(image_path)