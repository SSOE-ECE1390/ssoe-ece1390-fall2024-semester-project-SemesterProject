import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import face_recognition

# # src: https://stackoverflow.com/questions/66309089/replacing-cv2-face-detection-photo-with-overlayed-image
# def upload_files():
   
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#     img = cv2.imread('face.jpg')
#     img_to_place = cv2.imread('img.png')

#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gray_to_place = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
#     img_h, img_w = gray.shape
#     img_to_place_h, img_to_place_w = gray_to_place.shape

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             resized_img = cv2.resize(img_to_place, (eh, ew), interpolation = cv2.INTER_AREA)
#             resized_img_h, resized_img_w, _ = resized_img.shape

#             roi_color[ey:ey+resized_img_h, ex:ex+resized_img_w, :] = resized_img

#     cv2.imwrite('out.png', img)

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