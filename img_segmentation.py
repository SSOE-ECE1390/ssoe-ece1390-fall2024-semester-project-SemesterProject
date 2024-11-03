import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def img_segmentation(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

     # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 100, 200)

    # Show the edges for debugging
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)  # Wait for a key press to proceed
    cv2.destroyAllWindows()

    # Find contours from the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(image)

    # Fill the contours in the mask
    if contours:
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    output_path = "Output/segmented_img.png"
    cv2.imwrite(output_path, mask)
    plt.imshow(mask, cmap='gray')
    plt.show()
    return [output_path, mask]