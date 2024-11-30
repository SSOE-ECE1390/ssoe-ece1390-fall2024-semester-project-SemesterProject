import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def img_overlay(image, icon_path, icon_mask_path=os.path.abspath("Output/SeparateIcon/test2.jpeg"), output_path="test"):
    # Load Haar cascade classifiers for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # Optional: Use for eye detection

    # Load the input image and convert to grayscale
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the input image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Load the overlay image (black mask with white background)
    face_overlay = cv2.imread(icon_path)

    # Convert the overlay to grayscale and create a binary mask
    overlay_gray = cv2.cvtColor(face_overlay, cv2.COLOR_BGR2GRAY)
    # _, binary_mask = cv2.threshold(overlay_gray, 128, 255, cv2.THRESH_BINARY_INV)
    icon_mask = cv2.bitwise_not(cv2.imread(icon_mask_path))
    binary_mask = np.all(icon_mask == [255, 255, 255], axis=-1).astype(np.uint8) * 255

    if len(faces) > 0:
        # Use the first detected face only
        (x, y, w, h) = faces[0]

        # Dynamically adjust the overlay mask size based on the detected face
        overlay_width = int(w * 1.5) 
        overlay_height = int(h * 2.0) 
        overlay_resized = cv2.resize(face_overlay, (overlay_width, overlay_height))
        binary_mask_resized = cv2.resize(binary_mask, (overlay_width, overlay_height))

        # Estimate the eyes' positions in the face bounding box
        eye_y_position = int(y) # assuming the eyes are at the very top of the bounding box
        mask_eye_y_position = int(overlay_height * 0.35)  # Assume the mask's eyes are similarly positioned

        # Center the mask based on the estimated eye position
        overlay_x = x + w // 2 - overlay_width // 2
        overlay_y = eye_y_position - mask_eye_y_position

        # Ensure overlay fits within image boundaries
        img_h, img_w = img.shape[:2]
        if overlay_x < 0:
            binary_mask_resized = binary_mask_resized[:, -overlay_x:]
            overlay_resized = overlay_resized[:, -overlay_x:]
            overlay_x = 0
        if overlay_y < 0:
            binary_mask_resized = binary_mask_resized[-overlay_y:, :]
            overlay_resized = overlay_resized[-overlay_y:, :]
            overlay_y = 0
        if overlay_x + overlay_resized.shape[1] > img_w:
            binary_mask_resized = binary_mask_resized[:, :img_w - overlay_x]
            overlay_resized = overlay_resized[:, :img_w - overlay_x]
        if overlay_y + overlay_resized.shape[0] > img_h:
            binary_mask_resized = binary_mask_resized[:img_h - overlay_y, :]
            overlay_resized = overlay_resized[:img_h - overlay_y, :]

        # Perform overlay using the binary mask
        for c in range(0, 3):  # Apply to each BGR channel
            img[overlay_y:overlay_y + overlay_resized.shape[0], overlay_x:overlay_x + overlay_resized.shape[1], c] = \
                np.where(binary_mask_resized > 0, overlay_resized[:, :, c], img[overlay_y:overlay_y + overlay_resized.shape[0], overlay_x:overlay_x + overlay_resized.shape[1], c])

    # Save the final image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image Overlay")
    plt.show()
    output_path = os.path.abspath(f"Output/Overlay/{output_path}.jpeg")
    cv2.imwrite(output_path, img)
    print("Output image saved as 'test.jpeg'.")

# # example usage
# face_path = os.path.abspath("Input/Face/1 (1).jpeg")
# face_path = os.path.abspath("Output/BokehEffect/test.png")
# face = cv2.imread(face_path)
# # icon_path = os.path.abspath("Input/Icon/Comiccon_Decals_Square_for_Shopify-42.webp")
# icon_path = os.path.abspath("Input/Icon/spongebob.jpeg")
# icon = cv2.imread(icon_path)
# icon_mask_path = os.path.abspath("Output/SeparateIcon/test2.png")
# icon_mask = cv2.bitwise_not(cv2.imread(icon_mask_path))
# icon = cv2.bitwise_and(icon_mask, icon)
# binary_icon_mask = np.all(icon_mask == [255, 255, 255], axis=-1).astype(np.uint8) * 255
# plt.imshow(icon_mask)
# plt.show()
# img_overlay(face, icon, binary_icon_mask)