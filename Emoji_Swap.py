from fer import FER
import os
import cv2

# Emotion to emoji mapping
emotion_emoji_dict = {
    'happy': 'smiling.png', 
    'sad': 'disappointed.png',
    'angry': 'angry.png',
    'surprise': 'astonished.png',
    'fear': 'fearful.png',
    'disgust': 'nauseated.png',
    'neutral': 'neutral.png',
    'contempt': 'unamused.png'  # Add more mappings as needed
}

# Path to emoji
emoji_folder = 'emojis'

def detect_emotion(image):
    # Initialize FER for emotion detection
    emotion_detector = FER(mtcnn=True)
    # Detect emotions using FER
    emotion_results = emotion_detector.detect_emotions(image)

    if emotion_results:
        # Get the top emotion
        emotions = emotion_results[0]['emotions']
        emotion = max(emotions, key=emotions.get)
        print(f"Detected emotion: {emotion}")
    else:
        emotion = 'neutral'
        print("No emotions detected, defaulting to 'neutral'.")
    
    return emotion

def swap_emoji(image_resized, emotion, angle, x_max, x_min, y_max, y_min):

    # Load the corresponding emoji
    emoji_filename = emotion_emoji_dict.get(emotion, 'neutral_face.png')
    emoji_path = os.path.join(emoji_folder, emoji_filename)
    emoji_image = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

    if emoji_image is None:
        print(f"Error loading emoji image: {emoji_path}")
        return
    
    # Get box width and height
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Resize the emoji to match the face bounding box
    emoji_resized = cv2.resize(emoji_image, (box_width, box_height), interpolation=cv2.INTER_AREA)

    # Rotate the emoji image
    center = (box_width // 2, box_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    emoji_rotated = cv2.warpAffine(emoji_resized, rotation_matrix, (box_width, box_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Overlay the emoji onto the face region
    if emoji_rotated.shape[2] == 4:
        # Split the emoji image into BGR and Alpha channels
        emoji_bgr = emoji_rotated[:, :, :3]
        alpha_mask = emoji_rotated[:, :, 3] / 255.0

        # Get the region of interest on the original image
        roi = image_resized[y_min:y_max, x_min:x_max]

        # Check if ROI size matches emoji size
        if roi.shape[0] != emoji_bgr.shape[0] or roi.shape[1] != emoji_bgr.shape[1]:
            print("Size mismatch between ROI and emoji. Skipping this face.")
            return

        # Blend the emoji and the ROI
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * emoji_bgr[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

        # Put the blended ROI back into the original image
        image_resized[y_min:y_max, x_min:x_max] = roi
        return image_resized
    else:
        print("Emoji image does not have an alpha channel.")