import os
import cv2
from fer import FER

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=True)

# Path to the input image and output folder
image_path = 'Data/crying_stock_photo.png'  
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Load Image
image = cv2.imread(image_path)
if image is None:
    print(f"Error loading image: {image_path}")
    exit()

# Detect emotions and bounding box using FER
emotion_results = emotion_detector.detect_emotions(image)

if not emotion_results:
    print("No face detected.")
else:
    for result in emotion_results:
        # Extract bounding box and emotions
        bbox = result['box']  # (x, y, width, height)
        emotions = result['emotions']
        detected_emotion = max(emotions, key=emotions.get)
        print(f"Detected emotion: {detected_emotion} | Emotion scores: {emotions}")

        # Draw bounding box
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add emotion text
        text = f"{detected_emotion.capitalize()}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save and display the output image
output_path = os.path.join(output_folder, f"FER_Output_{os.path.basename(image_path)}")
cv2.imwrite(output_path, image)
print(f"Saved output to {output_path}")

# Optional: Display the image
cv2.imshow("Emotion Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
