import cv2
import mediapipe as mp
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Paths to images and hair emoji
image_paths = ['Data/jim.jpg', 'Data/crying_stock_photo.png', 'Data/AngryMan.jpg']
hair_emoji_path = 'emojis/brown_short.png'

# Function to detect brown & short hair
def detect_brown_short_hair(image, face_landmarks, width, height):
    # Extract landmarks for the forehead region
    forehead_landmarks = [10, 338, 297, 332, 284]  # Example Mediapipe landmark indices around the forehead
    forehead_points = np.array(
        [[int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)] for i in forehead_landmarks]
    )

    # Define the region above the forehead
    x_min, y_min = np.min(forehead_points, axis=0)
    x_max, y_max = np.max(forehead_points, axis=0)
    return (x_min, y_min, x_max, y_max)

# Function to overlay the emoji
def overlay_hair_emoji(image, emoji_path, forehead_coords):
    x_min, y_min, x_max, y_max = forehead_coords
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Error loading emoji: {emoji_path}")
        return image

    # Calculate size and position for the emoji
    emoji_width = int((x_max - x_min) * 3.0)  # Keep emoji large
    emoji_height = int(emoji_width * (emoji.shape[0] / emoji.shape[1]))
    emoji_resized = cv2.resize(emoji, (emoji_width, emoji_height), interpolation=cv2.INTER_AREA)

    # Adjust the emoji placement
    y_start = max(0, y_min - int(0.6 * emoji_height))  # Vertical position
    y_end = y_start + emoji_height
    x_start = max(0, x_min - int(0.3 * emoji_width)) - 17 # Manual left shift
    x_end = x_start + emoji_width

    # Handle transparency (alpha channel)
    for c in range(3):  # Apply to each color channel
        alpha_mask = emoji_resized[:, :, 3] / 255.0
        if y_start < y_end and x_start < x_end and alpha_mask.shape[:2] == image[y_start:y_end, x_start:x_end].shape[:2]:
            image[y_start:y_end, x_start:x_end, c] = (
                alpha_mask * emoji_resized[:, :, c] +
                (1 - alpha_mask) * image[y_start:y_end, x_start:x_end, c]
            )

    return image

for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}. Please check the image path and file.")
        continue

    # Resize the image for processing
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Convert to RGB for Mediapipe
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Initialize Face Mesh model
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # Process the image to detect facial landmarks
        results = face_mesh.process(image_rgb)

        # Check if landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Detect brown & short hair
                forehead_coords = detect_brown_short_hair(image_resized, face_landmarks, width, height)
                # Overlay the hair emoji
                image_resized = overlay_hair_emoji(image_resized, hair_emoji_path, forehead_coords)

    # Resize the final image for output (increase size)
    output_scale = 2.0  # Scale up the image by 2x
    output_width = int(image_resized.shape[1] * output_scale)
    output_height = int(image_resized.shape[0] * output_scale)
    output_image = cv2.resize(image_resized, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Save and display the output image
    output_path = f'output_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, output_image)
    print(f"Saved processed image: {output_path}")
    cv2.imshow('Processed Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
