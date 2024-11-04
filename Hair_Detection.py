import mediapipe as mp
import numpy as np

# TODO: Polish hair and facial hair detection

def detect_hair(image, forehead_points, jawline_points):
    hair_present = False
    facial_hair_present = False

    # Check area above forehead points
    forehead_y = int(np.mean([point[1] for point in forehead_points])) - 30  # Move slightly above forehead
    forehead_x = int(np.mean([point[0] for point in forehead_points]))
    
    # Sample a small area above forehead for hair intensity
    hair_area = image[max(forehead_y - 20, 0):forehead_y, forehead_x - 20:forehead_x + 20]
    if np.mean(hair_area) < 100:  # Arbitrary threshold for dark areas (indicating hair)
        hair_present = True

    # Check jawline area for potential facial hair
    jawline_y = int(np.mean([point[1] for point in jawline_points])) + 10  # Slightly below jawline
    jawline_x = int(np.mean([point[0] for point in jawline_points]))
    
    facial_hair_area = image[jawline_y:jawline_y + 20, jawline_x - 20:jawline_x + 20]
    if np.std(facial_hair_area) > 20:  # Texture threshold to indicate possible facial hair
        facial_hair_present = True

    return hair_present, facial_hair_present
    