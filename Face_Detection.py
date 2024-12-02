
import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import math
import os
from HairDetectionDillon import detect_brown_short_hair
from HairDetectionDillon import overlay_hair_emoji

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=True)

# Paths to images and emojis
image_paths = ['Data/jim.jpg', 'Data/crying_stock_photo.png', 'Data/AngryMan.jpg', 'Data/SmilingGirl.jpg']
hair_emoji_path = 'emojis/brown_short.png'
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

    ### Requirement: Include an image enhancement method (Histogram Equalization)
    # Convert image to YUV color space
    image_yuv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2YUV)
    # Equalize the histogram of the Y channel
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    # Convert back to BGR color space
    image_enhanced = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    # Use the enhanced image for further processing
    image_to_process = image_enhanced

    ### Requirement: Include an image filtering method (Gaussian Blur)
    image_filtered = cv2.GaussianBlur(image_to_process, (5, 5), 0)

    ### Requirement: Include an edge detection method (Canny Edge Detection)
    # Convert to grayscale for edge detection
    image_gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)

    ### Requirement: Demonstrate segmentation (Thresholding)
    # Apply binary thresholding
    _, segmented_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # Convert the filtered image to RGB as Mediapipe requires
    image_rgb = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB)

    # Detect emotions using FER
    emotion_results = emotion_detector.detect_emotions(image_filtered)

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
                # Get bounding box coordinates
                face_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
                face_coords = np.array(face_coords)
                x_min = int(np.min(face_coords[:, 0]) * width)
                x_max = int(np.max(face_coords[:, 0]) * width)
                y_min = int(np.min(face_coords[:, 1]) * height)
                y_max = int(np.max(face_coords[:, 1]) * height)

                # Scale the bounding box to cover more face area
                scale_factor = 1.12  # Reduced scale factor for slightly smaller emoji
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
                emoji_rotated = cv2.warpAffine(
                    emoji_resized, rotation_matrix, (box_width, box_height),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                # Overlay emoji onto face region
                if emoji_rotated.shape[2] == 4:
                    emoji_bgr, alpha_mask = emoji_rotated[:, :, :3], emoji_rotated[:, :, 3] / 255.0
                    roi = image_resized[y_min:y_max, x_min:x_max]
                    image_altered = image_resized.copy()
                    if roi.shape[:2] == emoji_bgr.shape[:2]:
                        for c in range(3):
                            roi[:, :, c] = (alpha_mask * emoji_bgr[:, :, c] +
                                            (1 - alpha_mask) * roi[:, :, c])
                        image_altered[y_min:y_max, x_min:x_max] = roi
                    else:
                        print("Size mismatch between ROI and emoji. Skipping this face.")
                else:
                    print("Emoji image does not have an alpha channel.")
                forehead_coords = detect_brown_short_hair(image_resized, face_landmarks, width, height)
                # Overlay the hair emoji
                image_altered = overlay_hair_emoji(image_altered, hair_emoji_path, forehead_coords)
        else:
            print("No facial landmarks detected.")

    ### Requirement: Incorporate 1 additional method from class code (Morphological Transformation)
    # Apply morphological closing to the segmented image
    kernel = np.ones((5, 5), np.uint8)
    image_morph = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

    ### Save the processed images
    # Save the enhanced image
    cv2.imwrite(f"output/enhanced_{os.path.basename(image_path)}", image_enhanced)
    # Save the filtered image
    cv2.imwrite(f"output/filtered_{os.path.basename(image_path)}", image_filtered)
    # Save the edge-detected image
    cv2.imwrite(f"output/edges_{os.path.basename(image_path)}", edges)
    # Save the segmented image
    cv2.imwrite(f"output/segmented_{os.path.basename(image_path)}", segmented_image)
    # Save the morphologically transformed image
    cv2.imwrite(f"output/morph_{os.path.basename(image_path)}", image_morph)
    # Save the final image with emoji overlay
    cv2.imwrite(f"output/final_{os.path.basename(image_path)}", image_altered)

    # Resize the final image for a larger display
    display_scale = 1.5
    display_width = int(image_altered.shape[1] * display_scale)
    display_height = int(image_altered.shape[0] * display_scale)
    image_display = cv2.resize(image_altered, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    # Show the enlarged image
    cv2.imshow('Emoji Face Swap', image_display)
    cv2.moveWindow('Emoji Face Swap', 200, 200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
