import cv2
import face_recognition
import numpy as np
from img_segmentation import img_segmentation

def img_overlay(path1, path2):
    face_image = cv2.imread(path1)
    icon = cv2.imread(path2)
    face_locations = face_recognition.face_locations(face_image)
    icon_path, segmented_image = img_segmentation(path2)
    seg_mask = segmented_image # Assuming the alpha channel is the mask

    # Check if a face is detected
    if face_locations:
        # Get the coordinates of the first detected face
        top, right, bottom, left = face_locations[0]

        # Resize the segmented image to match the face dimensions
        face_width = right - left
        face_height = bottom - top
        resized_segmented_image = cv2.resize(icon, (face_width, face_height))
        resized_mask = cv2.resize(seg_mask, (face_width, face_height))

        # Define the region of interest (ROI) on the face image
        roi = face_image[top:bottom, left:right]

        # Create an inverse mask for blending
        inv_mask = cv2.bitwise_not(resized_mask)

        # Mask out the face area in the ROI and overlay the segmented image
        background = cv2.bitwise_and(roi, roi, mask=inv_mask)
        foreground = cv2.bitwise_and(resized_segmented_image, resized_segmented_image, mask=resized_mask)

        # Combine the background and foreground
        blended = cv2.add(background, foreground)
        face_image[top:bottom, left:right] = blended

        # Save or display the result
        cv2.imwrite("Output/face_with_overlay.jpg", face_image)
        print("Overlay complete and saved as face_with_overlay.jpg")

    else:
        print("No face detected in the image.")