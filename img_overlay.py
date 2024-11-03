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

def img_overlay2(path1, path2):
    face_image = cv2.imread(path1)
    icon = cv2.imread(path2)
    face_locations = face_recognition.face_locations(face_image)
    icon_path, segmented_image = img_segmentation(path2)
    seg_mask = segmented_image # Assuming the alpha channel is the mask

    # Check if a face is detected
    if face_locations:
        # Get the coordinates of the first detected face
        top, right, bottom, left = face_locations[0]

        # Find the contours of the mask to determine points
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the extreme points
        highest_point = face_image.shape[0]  # Height of the image
        lowest_point = 0  # Start at the top of the image
        leftmost_point = face_image.shape[1]  # Width of the image
        rightmost_point = 0  # Start at the left of the image

        # Iterate through contours to find extreme points
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y < highest_point:
                    highest_point = y
                if y > lowest_point:
                    lowest_point = y
                if x < leftmost_point:
                    leftmost_point = x
                if x > rightmost_point:
                    rightmost_point = x

        # Calculate dimensions for the icon based on the mask points
        icon_width = rightmost_point - leftmost_point
        icon_height = lowest_point - highest_point

        # Calculate new positions based on the face detection
        new_top = top
        new_bottom = new_top + icon_height
        new_left = left
        new_right = new_left + icon_width

        # Resize the icon to match the calculated dimensions
        resized_icon = cv2.resize(icon, (icon_width, icon_height), interpolation=cv2.INTER_AREA)

        # Resize the segmentation mask to match the dimensions of the detected face area
        resized_seg_mask = cv2.resize(seg_mask, (icon_width, icon_height), interpolation=cv2.INTER_AREA)

        # Create a region of interest (ROI) on the face image
        roi = face_image[new_top:new_bottom, new_left:new_right]

        # Create an inverse mask for blending
        inv_mask = cv2.bitwise_not(resized_seg_mask)  # Invert the resized segmentation mask

        # Check mask types and shapes
        print(f"ROI Shape: {roi.shape}")
        print(f"Inverse Mask Shape: {inv_mask.shape}")
        print(f"Resized Seg Mask Type: {resized_seg_mask.dtype}, Shape: {resized_seg_mask.shape}")
        print(f"Inverse Mask Type: {inv_mask.dtype}, Shape: {inv_mask.shape}")

        # Display the resized icon and the mask
        cv2.imshow("Resized Icon", resized_icon)
        cv2.imshow("Resized Segmentation Mask", resized_seg_mask)

        # Wait for a key press and close the image windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert masks to uint8 if they are not already
        resized_seg_mask = resized_seg_mask.astype(np.uint8)
        inv_mask = inv_mask.astype(np.uint8)
        
        # Mask out the face area in the ROI and overlay the resized icon
        background = cv2.bitwise_and(roi, roi, mask=inv_mask)
        foreground = cv2.bitwise_and(resized_icon, resized_icon, mask=seg_mask)

        # Combine the background and foreground
        blended = cv2.add(background, foreground)

        # Place the blended image back into the original face image
        face_image[new_top:new_bottom, new_left:new_right] = blended

        # Save or display the result
        cv2.imwrite("Output/head_with_overlay_feature_matching.jpg", face_image)
        print("Overlay complete and saved as head_with_overlay_feature_matching.jpg")
    else:
        print("No face detected in the image.")