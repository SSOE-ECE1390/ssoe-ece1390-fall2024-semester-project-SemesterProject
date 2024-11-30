import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import face_recognition

# src: https://stackoverflow.com/questions/66309089/replacing-cv2-face-detection-photo-with-overlayed-image
def detect_faces(path):
    # Load the image
    image = face_recognition.load_image_file(path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Print the locations of each face in this image
    print("Found {} face(s) in this image.".format(len(face_locations)))
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        print(f"Face {i + 1} found at Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")

        # Optionally, draw a rectangle around the detected face
        image_with_rectangle = cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Show the image with face rectangles (optional)
    cv2.imwrite("Output/img_face_detection.png", image_with_rectangle)

    return image_with_rectangle 