import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# mean filter operation for smoothing (kernel_size = 1 is 3x3)
def random_smoothing_filter(img):
    print("Runnning random_smoothing_filter...")
    x, y, z = img.shape  
    # random kernel size
    kernel_size = np.clip(random.randrange(1, 25, 5), 0, min(x, y))
    print("using random kernel size: ", kernel_size)
    print("\n")

    new_img = np.zeros(img.shape, dtype=img.dtype)
    
    for i in range(x):
        for j in range(y):
            x_range = np.clip(range(i-kernel_size, i+kernel_size+1), 0, x-1)
            y_range = np.clip(range(j-kernel_size, j+kernel_size+1), 0, y-1)
            r = np.mean(img[x_range, y_range, 0])
            g = np.mean(img[x_range, y_range, 1])
            b = np.mean(img[x_range, y_range, 2])
            new_img[i, j] = [r, g, b]

    plt.imsave("Output/randomly_smoothed_img.png", new_img)
    return new_img