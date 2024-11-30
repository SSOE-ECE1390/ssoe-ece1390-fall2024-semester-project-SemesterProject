# from img_segmentation import icon_segmentation
from img_segmentation import icon_segmentation
from img_segmentation import person_segmentation
from img_overlay import img_overlayv2
from bokeh_effect import bokeh
import cv2
import os
import matplotlib.pyplot as plt
from mediapipe.tasks import python
import numpy as np

face_path = os.path.abspath("Input/Face/HL-005.jpeg")
icon_path = os.path.abspath("Input/Icon/spongebob.jpeg")
face = cv2.imread(face_path)
icon = cv2.imread(icon_path)
face_mask = cv2.cvtColor(person_segmentation.segment_person(face_path), cv2.COLOR_GRAY2BGR)
background_mask = cv2.bitwise_not(face_mask)
plt.imshow(background_mask)
plt.title("Background mask")
plt.show()

background = cv2.bitwise_and(face, background_mask)
plt.imshow(background)
plt.title("Background")
plt.show()

# icon_mask = icon_segmentation.segment_iconv2(icon_path)
icon_mask_path = os.path.abspath("Output/SeparateIcon/test2.png")
icon_mask = cv2.bitwise_not(cv2.imread(icon_mask_path))
icon = cv2.bitwise_and(icon_mask, icon)
plt.imshow(icon)
plt.title("Icon")
plt.show()

image_with_bokeh = bokeh.bokeh_blur(background, icon, (person_segmentation.segment_person(face_path)))
plt.imshow(image_with_bokeh)
plt.title("Bokeh effect on background")
plt.show()

just_the_face = cv2.bitwise_and(face, face_mask)
just_the_bokeh_bg = cv2.bitwise_and(image_with_bokeh, background_mask)
face_plus_bokeh = cv2.add(just_the_face, just_the_bokeh_bg)
face_plus_bokeh_rgb = cv2.cvtColor(face_plus_bokeh,cv2.COLOR_BGR2RGB)
plt.imshow(face_plus_bokeh_rgb)
plt.title("Face with bokeh background")
plt.show()
output_path = os.path.abspath("Output/BokehEffect/test.png")
cv2.imwrite(output_path, face_plus_bokeh)

binary_icon_mask = np.all(icon_mask == [255, 255, 255], axis=-1).astype(np.uint8) * 255
bokeh_image_with_overlay = img_overlayv2.img_overlay(face_plus_bokeh, icon, binary_icon_mask)

output_path = os.path.abspath("Output/Overlay/test.jpeg")
bokeh_image_with_overlay = cv2.imread(output_path)
bokeh_image_with_overlay_rgb = cv2.cvtColor(bokeh_image_with_overlay, cv2.COLOR_BGR2RGB)
plt.imshow(bokeh_image_with_overlay)
plt.title("Face with bokeh background and overlay")
plt.show()

output_path = os.path.abspath("Output/Overlay/test2.png")
cv2.imwrite(output_path, bokeh_image_with_overlay)

