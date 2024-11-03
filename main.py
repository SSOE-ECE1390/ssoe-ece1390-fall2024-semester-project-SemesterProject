import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import face_recognition
import img_enhance as f1
import img_filter as f2
import face_detection as f3
import img_segmentation as f4
import img_overlay as f5

path = 'Input/1 (1).jpeg'
path2 = 'Input/Comiccon_Decals_Square_for_Shopify-42.webp'
# img1 = plt.imread(path)[:,:,:3]
# new_img = f1.random_adjust_brightness(img1)
# plt.imshow(new_img)
# plt.axis("off")
# plt.show()

# new_img = f2.random_smoothing_filter(img1)
# plt.imshow(new_img)
# plt.axis("off")
# plt.show()

f5.img_overlay2(path, path2)
