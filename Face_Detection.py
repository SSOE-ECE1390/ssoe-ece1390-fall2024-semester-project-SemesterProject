import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=True)

# Paths to images and emojis
image_paths = ['Data/Tiger_Woods.jpeg', 'Data/jim.jpg', 'Data/crying_stock_photo.png']
emoji_folder = 'emojis'

# Emotion to emoji mapping
emotion_emoji_dict = {
    'happy': 'smiling.png', 
    'sad': 'disappointed.png',
    'angry': 'angry.png',
    'surprise': 'astonished.png',
    'fear': 'fearful.png',
    'disgust': 'nauseated.png',
    'neutral': 'neutral.png',
    'contempt': 'unamused.png'
}

for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}. Please check the image path and file.")
        continue

    # Resize the image if needed
    scale_percent = 50  # Adjust this value
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Convert the resized image to RGB as Mediapipe requires
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Detect emotions using FER
    emotion_results = emotion_detector.detect_emotions(image_resized)

    emotion = 'neutral'  # Default emotion if none are detected
    if emotion_results:
        # Get the top emotion
        emotions = emotion_results[0]['emotions']
        emotion = max(emotions, key=emotions.get)
        print(f"Detected emotion: {emotion}")
    else:
        print("No emotions detected, defaulting to 'neutral'.")

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
                # Draw facial landmarks on the image (optional)
                mp_drawing.draw_landmarks(
                    image_resized,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

                # Get bounding box coordinates
                face_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
                face_coords = np.array(face_coords)
                x_min = int(np.min(face_coords[:, 0]) * width)
                x_max = int(np.max(face_coords[:, 0]) * width)
                y_min = int(np.min(face_coords[:, 1]) * height)
                y_max = int(np.max(face_coords[:, 1]) * height)

                # Scale the bounding box to cover more face area
                scale_factor = 1.2  # Scale factor for emoji size
                box_width = int((x_max - x_min) * scale_factor)
                box_height = int((y_max - y_min) * scale_factor)

                # Recalculate x_min, y_min, x_max, y_max for the new box dimensions
                x_min = max(0, x_min - (box_width - (x_max - x_min)) // 2)
                y_min = max(0, y_min - (box_height - (y_max - y_min)) // 2)
                x_max = min(width, x_min + box_width)
                y_max = min(height, y_min + box_height)

                # Calculate face orientation angle
                left_eye, right_eye = face_landmarks.landmark[33], face_landmarks.landmark[263]
                x1, y1 = int(left_eye.x * width), int(left_eye.y * height)
                x2, y2 = int(right_eye.x * width), int(right_eye.y * height)
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

                # Load the corresponding emoji
                emoji_filename = emotion_emoji_dict.get(emotion, 'neutral_face.png')
                emoji_path = os.path.join(emoji_folder, emoji_filename)
                emoji_image = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

                if emoji_image is None:
                    print(f"Error loading emoji image: {emoji_path}")
                    continue

                # Resize and rotate the emoji
                emoji_resized = cv2.resize(emoji_image, (box_width, box_height), interpolation=cv2.INTER_AREA)
                rotation_matrix = cv2.getRotationMatrix2D((box_width // 2, box_height // 2), angle, 1.0)
                emoji_rotated = cv2.warpAffine(emoji_resized, rotation_matrix, (box_width, box_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                # Overlay emoji onto face region
                if emoji_rotated.shape[2] == 4:
                    emoji_bgr, alpha_mask = emoji_rotated[:, :, :3], emoji_rotated[:, :, 3] / 255.0
                    roi = image_resized[y_min:y_max, x_min:x_max]

                    if roi.shape[:2] == emoji_bgr.shape[:2]:
                        for c in range(3):
                            roi[:, :, c] = (alpha_mask * emoji_bgr[:, :, c] + (1 - alpha_mask) * roi[:, :, c])
                        image_resized[y_min:y_max, x_min:x_max] = roi
                    else:
                        print("Size mismatch between ROI and emoji. Skipping this face.")
                else:
                    print("Emoji image does not have an alpha channel.")
        else:
            print("No facial landmarks detected.")

    # Resize the final image for a larger display
    display_scale = 1.5
    display_width = int(image_resized.shape[1] * display_scale)
    display_height = int(image_resized.shape[0] * display_scale)
    image_display = cv2.resize(image_resized, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    # Show the enlarged image
    cv2.imshow('Emoji Face Swap', image_display)
    cv2.moveWindow('Emoji Face Swap', 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
