from img_segmentation import icon_segmentation
from img_segmentation import person_segmentation
from img_overlay import img_overlayv2
from bokeh_effect import bokeh
from extra_features import resize

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def bokeh_bg(face_path, icon_path, background_path, bokeh_selector=1, bokeh_effect="star", icon_mask_path=os.path.abspath("Output/SeparateIcon/test2.jpeg"), output_path="test"):
    face = cv2.imread(face_path)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.title("Face")
    plt.show()

    icon = cv2.imread(icon_path)
    plt.imshow(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB))
    plt.title("Icon")
    plt.show()

    background = cv2.imread(background_path)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title("Background")
    plt.show()

    # resize background to fit face image
    background = resize.resize(background, face)
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.title("Resized Background")
    plt.show()

    # get mask
    face_mask_gray = person_segmentation.segment_person(face_path)
    face_mask = cv2.cvtColor(face_mask_gray, cv2.COLOR_GRAY2BGR)
    plt.imshow(face_mask)
    plt.title("Face Mask")
    plt.show()

    background_mask = cv2.bitwise_not(face_mask)
    plt.imshow(background_mask)
    plt.title("Background Mask")
    plt.show()

    icon_mask = cv2.bitwise_not(cv2.imread(icon_mask_path))
    plt.imshow(icon_mask)
    plt.title("Icon Mask")
    plt.show()

    icon = cv2.bitwise_and(icon_mask, icon)
    plt.imshow(cv2.cvtColor(icon, cv2.COLOR_BGR2RGB))
    plt.title("Masked Icon")
    plt.show()

    if bokeh_selector == 1:
        kernel_name = bokeh_effect
        kernel_file = f"Input/Effect/{kernel_name}.png"
        kernel = np.float32(cv2.imread(kernel_file))
    else:
        target_height, target_width = face.shape[:2]
        resized_icon = cv2.resize(icon_mask, (int(target_height*.10), int(target_width*0.10)))
        kernel = np.float32(resized_icon)
        plt.imshow(resized_icon)
        plt.title("Resized Icon (for Bokeh Effect)")
        plt.show()
        
    background_with_bokeh = bokeh.bokeh_blur(background, kernel, cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    plt.imshow(cv2.cvtColor(background_with_bokeh, cv2.COLOR_BGR2RGB))
    plt.title("Bokeh Background")
    plt.show()

    just_the_face = cv2.bitwise_and(face, face_mask)
    just_the_bokeh_bg = cv2.bitwise_and(background_with_bokeh, background_mask)
    face_plus_bokeh = cv2.add(just_the_face, just_the_bokeh_bg)
    plt.imshow(cv2.cvtColor(face_plus_bokeh, cv2.COLOR_BGR2RGB))
    plt.title("Bokeh Background with Face")
    plt.show()
    return face_plus_bokeh

# NOTE: for some reason, these functions cannot be put into a function and ran together...
# they have to be ran separately, one by one :|
# def compilation(face_path, icon_path, background_path):
#     icon_mask = icon_segmentation.segment_iconv2(icon_path)
#     bokeh_background = bokeh_bg(face_path, icon_path, background_path)
#     result = img_overlayv2.img_overlay(bokeh_background, icon_path)
#     return result

# example usage:
face_path = os.path.abspath("Input/Face/HL-005.jpeg")
icon_path = os.path.abspath("Input/Icon/spongebob.jpeg")
background_path = os.path.abspath("Input/Background/fireworks.jpg")
icon_mask = icon_segmentation.segment_iconv2(icon_path)
bokeh_background = bokeh_bg(face_path, icon_path, background_path, bokeh_selector=0)
result = img_overlayv2.img_overlay(bokeh_background, icon_path)
# result = compilation(face_path, icon_path, background_path)


