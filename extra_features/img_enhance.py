import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def random_adjust_brightness(img):
    img = img.astype("int16")
    print("Runnning random_adjust_brightness...")
    
    val = 0 
    while val == 0: # exclude 0 from possible val range
        val = random.randrange(-255, 255, 51)
    print("using random value: ", val)
    print("\n")

    matrix = np.ones(img.shape, dtype="int16") * val

    new_img = np.clip(cv2.add(img, matrix), 0, 255).astype("uint8")
    plt.imsave("Output/randomly_enhanced_img.png", new_img)
    return new_img