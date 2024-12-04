import mediapipe as mp
import numpy as np
import cv2

# TODO: Polish hair and facial hair detection

def detect_hair(image, face_outer_points):
    
    # Create face mask using outer face landmark points
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_outer_points = np.array(face_outer_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(face_mask, [face_outer_points], 255)

    # Compute centroid of face points
    centroid = np.mean(face_outer_points, axis=0)

    # Scale points outward relative to the centroid
    hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    scale_factor = 1.85
    hair_outer_points = (face_outer_points - centroid) * scale_factor + centroid
    hair_outer_points = hair_outer_points.astype(np.int32)
    cv2.fillPoly(hair_mask, [hair_outer_points], 255)

    image_face = cv2.bitwise_and(image, image, mask = face_mask)
    image_hair = cv2.bitwise_and(image, image, mask = hair_mask)
    side_by_side = cv2.hconcat([image_face, image_hair])
    cv2.imshow('Hair Mask', side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply the mask to the input image
    not_face_mask = cv2.bitwise_not(face_mask)
    roi = cv2.bitwise_and(image, image, mask = not_face_mask)

    # Convert to grayscale and apply diluation
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3))
    dilated = cv2.dilate(blurred, kernel, iterations=1) 

    cv2.imshow('Dilated', dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract valid marker points
    print("Face outer points: ", face_outer_points[1])
    hair_starting_marker = face_outer_points[1]
    
    # Convert markers to uint8 type as required by cv2.watershed
    markers = np.zeros(image.shape[:2], dtype=np.int32)
    markers[face_mask == 255] = 1  # Face region
    markers[hair_starting_marker] = 2  # Hair region
    # markers[background==1] = 3
    
    # Draw circles on the annotated image
    for point in markers:
        cv2.circle(annotated, (point[1], point[0]), 10, (255, 0, 0), 2)  # (x, y) format

    cv2.imshow('Markers after watershed', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply the watershed algorithm
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(roi_rgb, markers)

    annotated = image.copy()

    # # Color-based segmentation use after adjusting the roi to be a larger region over the face mask
    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # lower_hair = np.array([0, 10, 10])  # Adjust to fit hair color ranges
    # upper_hair = np.array([180, 255, 90])
    # hair_mask = cv2.inRange(hsv, lower_hair, upper_hair)
    
    cv2.imshow('Hair Mask', hair_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Face Mask', face_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combine hair_mask with watershed results
    combined_mask = cv2.bitwise_and(hair_mask, gray)

    cv2.imshow('Combined Mask', combined_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Contour detection and analysis
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hair_detected = False
    for contour in contours:
        # Analyze contour area and shape (e.g., aspect ratio, circularity)
        area = cv2.contourArea(contour)
        if area > 100 and area < 200:  # Filter small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            hair_detected = True
            hair_color = contour.mean()
        

    return image

    