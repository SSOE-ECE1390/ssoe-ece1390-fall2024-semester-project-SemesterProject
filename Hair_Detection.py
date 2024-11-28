import mediapipe as mp
import numpy as np
import cv2

# TODO: Polish hair and facial hair detection

def detect_hair(image, face_outer_points):
    
    # Create face mask using outer face landmark points
    face_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_outer_points = np.array(face_outer_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(face_mask, [face_outer_points], 255)

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

    # Binary thresholding for Watershed markers REMOVE AND JUST USE MANUAL MARKERS?
    _, binary = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Extract valid marker points
    marker_points = np.column_stack(np.where(markers > 0))  # Extract (y, x) indices
    annotated = image.copy()

    # Draw circles on the annotated image
    for point in marker_points:
        cv2.circle(annotated, (point[1], point[0]), 10, (255, 0, 0), 2)  # (x, y) format

    cv2.imshow('Markers', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert markers to int32 type as required by cv2.watershed
    markers = np.uint8(markers)
    markers = cv2.connectedComponents(markers)[1]  # Ensure markers are unique regions
    markers = markers.astype(np.int32)

    # Watershed segmentation
    roi[markers == -1] = [0, 0, 255]  # Highlight boundaries (optional)
    markers = cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), markers)
    
    # Color-based segmentation use after adjusting the roi to be a larger region over the face mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_hair = np.array([0, 10, 10])  # Adjust to fit hair color ranges
    upper_hair = np.array([180, 255, 90])
    hair_mask = cv2.inRange(hsv, lower_hair, upper_hair)
    
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
    for contour in contours:
        # Analyze contour area and shape (e.g., aspect ratio, circularity)
        area = cv2.contourArea(contour)
        if area > 100 and area < 400:  # Filter small contours
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return image

    