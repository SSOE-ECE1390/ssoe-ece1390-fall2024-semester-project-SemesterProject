import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import math
import os
from HairDetectionDillon import detect_brown_short_hair
from HairDetectionDillon import overlay_hair_emoji

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Mediapipe Face Mesh and FER
mp_face_mesh = mp.solutions.face_mesh
emotion_detector = FER(mtcnn=True)

# Paths to images and emojis
image_paths = ['Data/jim.jpg', 'Data/crying_stock_photo.png', 'Data/AngryMan.jpg', 'Data/SmilingGirl.jpg']
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

# Hair color to emoji mapping
hair_emoji_map = {
    'brown': 'emojis/brown_short.png',
    'blonde': 'emojis/blonde_short.png'
}

# Function to detect hair color
def detect_hair_color(image, face_landmarks, width, height):
    # Extract landmarks for the forehead region
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

    hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)

    hair_colors = {
        'brown': ([5, 50, 20], [20, 255, 180]),
        'blonde': ([15, 20, 180], [45, 150, 255])
    }

    color_match_percentages = {}
    for color, (lower, upper) in hair_colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        match_percentage = np.sum(mask > 0) / mask.size
        color_match_percentages[color] = match_percentage

    # Choose the color with the highest match percentage
    if color_match_percentages:
        detected_color = max(color_match_percentages, key=color_match_percentages.get)
        max_match_percentage = color_match_percentages[detected_color]
        if max_match_percentage > 0.05:  # Threshold of 5%
            return detected_color, (x_min, y_min, x_max, y_max)
    return None, None

# Function to overlay hair emoji
def overlay_hair_emoji(image, emoji_path, forehead_coords):
    x_min, y_min, x_max, y_max = forehead_coords
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Error loading emoji: {emoji_path}")
        return image

    emoji_width = int((x_max - x_min) * 3.0)
    emoji_height = int(emoji_width * (emoji.shape[0] / emoji.shape[1]))
    emoji_resized = cv2.resize(emoji, (emoji_width, emoji_height),
                               interpolation=cv2.INTER_AREA)

    y_start = max(0, y_min - int(0.6 * emoji_height))
    y_end = y_start + emoji_height
    x_start = max(0, x_min - int(0.3 * emoji_width)) - 17
    x_end = x_start + emoji_width

    y_end = min(y_end, image.shape[0])
    x_end = min(x_end, image.shape[1])

    if emoji_resized.shape[2] == 4:
        alpha_mask = emoji_resized[:, :, 3] / 255.0
        alpha_mask = alpha_mask[:y_end - y_start, :x_end - x_start]
        for c in range(3):
            image[y_start:y_end, x_start:x_end, c] = (
                alpha_mask * emoji_resized[:y_end - y_start,
                                           :x_end - x_start, c] +
                (1 - alpha_mask) * image[y_start:y_end,
                                         x_start:x_end, c]
            )
    else:
        print("Emoji image does not have an alpha channel.")

    return image

# Main processing loop
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image_resized = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_AREA)

    # Image enhancement
    image_yuv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    image_to_process = image_enhanced

    # Image filtering
    image_filtered = cv2.GaussianBlur(image_to_process, (5, 5), 0)

    # Edge detection
    image_gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)

    # Segmentation
    _, segmented_image = cv2.threshold(image_gray, 127, 255,
                                       cv2.THRESH_BINARY)

    image_rgb = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB)

    # Emotion detection
    emotion_results = emotion_detector.detect_emotions(image_filtered)
    emotion = 'neutral'
    if emotion_results:
        emotions = emotion_results[0]['emotions']
        emotion = max(emotions, key=emotions.get)
        print(f"Detected emotion: {emotion}")
    else:
        print("No emotions detected, defaulting to 'neutral'.")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_coords = [
                    (landmark.x, landmark.y)
                    for landmark in face_landmarks.landmark
                ]
                face_coords = np.array(face_coords)
                x_min = int(np.min(face_coords[:, 0]) * width)
                x_max = int(np.max(face_coords[:, 0]) * width)
                y_min = int(np.min(face_coords[:, 1]) * height)
                y_max = int(np.max(face_coords[:, 1]) * height)

                # Adjusted scale factor to make the emoji slightly smaller
                scale_factor = 1.3  # Reduced scale factor
                box_width = int((x_max - x_min) * scale_factor)
                box_height = int((y_max - y_min) * scale_factor)

                # Recalculate x_min, y_min for the new box dimensions
                x_min = max(0, x_min - (box_width - (x_max - x_min)) // 2)
                y_min = max(0, y_min - (box_height - (y_max - y_min)) // 2)
                x_max = min(width, x_min + box_width)
                y_max = min(height, y_min + box_height)

                # Adjust y_min to move the emoji upwards slightly
                y_offset = int(0.15 * box_height)  # Move emoji upwards
                y_min = max(0, y_min - y_offset)
                y_max = min(height, y_min + box_height)

                # Calculate face orientation angle
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                x1, y1 = int(left_eye.x * width), int(left_eye.y * height)
                x2, y2 = int(right_eye.x * width), int(right_eye.y * height)
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

                emoji_filename = emotion_emoji_dict.get(
                    emotion, 'neutral_face.png')
                emoji_path = os.path.join(emoji_folder, emoji_filename)
                emoji_image = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

                if emoji_image is None:
                    print(f"Error loading emoji image: {emoji_path}")
                    continue

                # Resize and rotate the emoji
                emoji_resized = cv2.resize(
                    emoji_image, (box_width, box_height),
                    interpolation=cv2.INTER_AREA)
                rotation_matrix = cv2.getRotationMatrix2D(
                    (box_width // 2, box_height // 2), angle, 1.0)
                emoji_rotated = cv2.warpAffine(
                    emoji_resized, rotation_matrix, (box_width, box_height),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))

                if emoji_rotated.shape[2] == 4:
                    emoji_bgr = emoji_rotated[:, :, :3]
                    alpha_mask = emoji_rotated[:, :, 3] / 255.0
                    roi = image_resized[y_min:y_max, x_min:x_max]
                    image_altered = image_resized.copy()
                    if roi.shape[:2] == emoji_bgr.shape[:2]:
                        for c in range(3):
                            roi[:, :, c] = (alpha_mask * emoji_bgr[:, :, c] +
                                            (1 - alpha_mask) * roi[:, :, c])
                        image_resized[y_min:y_max, x_min:x_max] = roi
                    else:
                        print("Size mismatch between ROI and emoji.")
                else:
                    print("Emoji image does not have an alpha channel.")
        else:
            print("No facial landmarks detected.")

    # Morphological Transformation
    kernel = np.ones((5, 5), np.uint8)
    image_morph = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

    # Save processed images
    cv2.imwrite(f"output/enhanced_{os.path.basename(image_path)}",
                image_enhanced)
    cv2.imwrite(f"output/filtered_{os.path.basename(image_path)}",
                image_filtered)
    cv2.imwrite(f"output/edges_{os.path.basename(image_path)}", edges)
    cv2.imwrite(f"output/segmented_{os.path.basename(image_path)}",
                segmented_image)
    cv2.imwrite(f"output/morph_{os.path.basename(image_path)}", image_morph)
    # Save the final image with emoji overlay
    cv2.imwrite(f"output/final_{os.path.basename(image_path)}", image_resized)

    # Display the final image
    display_scale = 1.5
    display_width = int(image_resized.shape[1] * display_scale)
    display_height = int(image_resized.shape[0] * display_scale)
    image_display = cv2.resize(image_resized, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow('Emoji Face Swap', image_display)
    cv2.moveWindow('Emoji Face Swap', 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
