import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import img_enhance as f1
import img_filter as f2

img1 = plt.imread('Input/1 (1).jpeg')[:,:,:3]
new_img = f1.random_adjust_brightness(img1)
plt.imshow(new_img)
plt.axis("off")
plt.show()

new_img = f2.random_smoothing_filter(img1)
plt.imshow(new_img)
plt.axis("off")
plt.show()

