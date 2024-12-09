import os
import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Mediapipe Face Mesh and FER
mp_face_mesh = mp.solutions.face_mesh
emotion_detector = FER(mtcnn=True)

# Paths to images and emojis
image_paths = [
    'output/HairAddition_jim.jpg',
    'output/HairAddition_crying_stock_photo.png',
    'output/HairAddition_AngryMan.jpg',
    'output/HairAddition_GingerMan.jpg'
]

emoji_folder = 'emojis'
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

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

# Enhanced Emotion Detection
def detect_emotion(image):
    """Detects facial emotion using the FER library."""
    emotion_results = emotion_detector.detect_emotions(image)
    if not emotion_results:
        return 'neutral'

    emotions = emotion_results[0]['emotions']
    if max(emotions.values()) < 0.2:  # Confidence threshold
        return 'neutral'
    return max(emotions, key=emotions.get)

# Function to overlay emoji based on face bounding box
def overlay_face_emoji(image, emoji_path, face_bbox):
    """Overlay an emoji image onto the detected face region."""
    x_min, y_min, x_max, y_max = face_bbox

    # Load emoji with alpha channel
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Error loading emoji: {emoji_path}")
        return image

    # Resize the emoji to fit the face bounding box
    emoji_width = int((x_max - x_min) * 1.3)
    emoji_height = int((y_max - y_min) * 1.3)
    emoji_resized = cv2.resize(emoji, (emoji_width, emoji_height))

    # Adjust position slightly upwards for better alignment
    y_min -= int(0.15 * emoji_height)
    y_max = y_min + emoji_height
    x_min -= int(0.15 * emoji_width)
    x_max = x_min + emoji_width

    # Ensure the overlay stays within the image bounds
    y_min, y_max = max(0, y_min), min(image.shape[0], y_max)
    x_min, x_max = max(0, x_min), min(image.shape[1], x_max)

    # Overlay the emoji using the alpha channel
    for c in range(3):
        alpha = emoji_resized[:, :, 3] / 255.0
        image[y_min:y_max, x_min:x_max, c] = (
            alpha * emoji_resized[:, :, c] +
            (1 - alpha) * image[y_min:y_max, x_min:x_max, c]
        )
    return image

# Main Processing Loop
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Resize image for faster processing
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Emotion detection
    emotion = detect_emotion(image_resized)
    print(f"Detected emotion: {emotion}")

    # Facial landmarks detection to get face bounding box
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * width)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * width)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * height)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * height)

                face_bbox = (x_min, y_min, x_max, y_max)

                # Select the emoji based on detected emotion
                emoji_filename = emotion_emoji_dict.get(emotion, 'neutral.png')
                emoji_path = os.path.join(emoji_folder, emoji_filename)

                # Overlay emoji on the detected face
                image_resized = overlay_face_emoji(image_resized, emoji_path, face_bbox)

    # Save the final processed image
    output_path = os.path.join(output_folder, f"FaceAddition_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image_resized)
    print(f"Saved output to {output_path}")
