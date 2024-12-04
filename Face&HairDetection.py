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
    'Data/jim.jpg',
    'Data/crying_stock_photo.png',
    'Data/AngryMan.jpg',
    'Data/SmilingGirl.jpg',
    'Data/Dillon.jpg',
    'Data/GingerMan.jpg'
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

# Hair color to emoji mapping
hair_emoji_map = {
    'brown': 'emojis/brown_short.png',
    'blonde': 'emojis/blonde_short.png',
    'black': 'emojis/black_short.png',
    'red': 'emojis/red_short.png',
    'gray': 'emojis/gray_short.png'
}

# Function to detect hair color
def detect_hair_color(image, face_landmarks, width, height):
    forehead_landmarks = [10, 338, 297, 332, 284]
    forehead_points = np.array([
        [int(face_landmarks.landmark[i].x * width),
         int(face_landmarks.landmark[i].y * height)]
        for i in forehead_landmarks
    ])

    x_min, y_min = np.min(forehead_points, axis=0)
    x_max, y_max = np.max(forehead_points, axis=0)
    y_top = max(0, y_min - int(0.5 * (y_max - y_min)))
    hair_region = image[y_top:y_min, x_min:x_max]
    # cv2.imshow("hair",hair_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if hair_region.size == 0:
        return None, None

    hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)

    # Refined hair color thresholds
    hair_colors = {
        'brown': ([5, 40, 20], [14, 209, 180]),  # Wide range for brown tones
        'blonde': ([15, 20, 150], [40, 200, 255]),  # Expanded range for blonde tones - modified for high saturation
        'black': ([0, 0, 0], [180, 255, 50]),
        'ginger': ([5, 179, 150], [15, 255, 255]),
        'gray': ([0, 0, 60], [180, 20, 180])
    }

    color_match_percentages = {}
    for color, (lower, upper) in hair_colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        match_percentage = np.sum(mask > 0) / mask.size
        color_match_percentages[color] = match_percentage

    if color_match_percentages:
        detected_color = max(color_match_percentages, key=color_match_percentages.get)
        if color_match_percentages[detected_color] > 0.05:
            return detected_color, (x_min, y_min, x_max, y_max)
    return None, None

# Function to overlay hair emoji
def overlay_hair_emoji(image, emoji_path, forehead_coords, manual_shift=None):
    x_min, y_min, x_max, y_max = forehead_coords
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Error loading emoji: {emoji_path}")
        return image

    emoji_width = int((x_max - x_min) * 3.2)  # Slightly bigger scaling factor
    emoji_height = int(emoji_width * (emoji.shape[0] / emoji.shape[1]))
    emoji_resized = cv2.resize(emoji, (emoji_width, emoji_height), interpolation=cv2.INTER_AREA)

    y_start = max(0, y_min - int(0.6 * emoji_height))
    y_end = y_start + emoji_height
    x_start = max(0, x_min - int(0.3 * emoji_width) - 20)  # Manual shift to the left
    if manual_shift:
        x_start += manual_shift  # Apply specific manual shift
    x_end = x_start + emoji_width

    y_end = min(y_end, image.shape[0])
    x_end = min(x_end, image.shape[1])

    if emoji_resized.shape[2] == 4:
        alpha_mask = emoji_resized[:, :, 3] / 255.0
        alpha_mask = alpha_mask[:y_end - y_start, :x_end - x_start]
        for c in range(3):
            image[y_start:y_end, x_start:x_end, c] = (
                alpha_mask * emoji_resized[:y_end - y_start, :x_end - x_start, c] +
                (1 - alpha_mask) * image[y_start:y_end, x_start:x_end, c]
            )
    return image

# Enhanced Emotion Detection
def detect_emotion(image):
    emotion_results = emotion_detector.detect_emotions(image)
    if not emotion_results:
        return 'neutral'

    emotions = emotion_results[0]['emotions']
    if max(emotions.values()) < 0.2:  # Confidence threshold
        return 'neutral'
    return max(emotions, key=emotions.get)

# Main Processing Loop
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Emotion detection
    emotion = detect_emotion(image_resized)
    print(f"Detected emotion: {emotion}")

    # Facial landmarks and hair detection
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                hair_color, forehead_coords = detect_hair_color(
                    image_resized, face_landmarks, width, height)
                if hair_color:
                    print(f"Detected hair color: {hair_color}")
                    hair_emoji_path = hair_emoji_map.get(hair_color)
                    if hair_emoji_path:
                        # Apply a manual shift for Dillon.jpg
                        manual_shift = -40 if "Dillon.jpg" in image_path else None
                        image_resized = overlay_hair_emoji(
                            image_resized, hair_emoji_path, forehead_coords, manual_shift)

                # Overlay face emoji
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * width)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * width)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * height)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * height)

                emoji_filename = emotion_emoji_dict.get(emotion, 'neutral.png')
                emoji_path = os.path.join(emoji_folder, emoji_filename)
                emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if emoji is not None:
                    emoji_width = int((x_max - x_min) * 1.3)
                    emoji_height = int((y_max - y_min) * 1.3)
                    emoji_resized = cv2.resize(emoji, (emoji_width, emoji_height))
                    y_min -= int(0.15 * emoji_height)
                    y_max = y_min + emoji_height
                    x_min -= int(0.15 * emoji_width)
                    x_max = x_min + emoji_width
                    for c in range(3):
                        alpha = emoji_resized[:, :, 3] / 255.0
                        image_resized[y_min:y_max, x_min:x_max, c] = (
                            alpha * emoji_resized[:, :, c] +
                            (1 - alpha) * image_resized[y_min:y_max, x_min:x_max, c]
                        )

    # Save the final processed image
    output_path = os.path.join(output_folder, f"final_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image_resized)
    print(f"Saved output to {output_path}")
