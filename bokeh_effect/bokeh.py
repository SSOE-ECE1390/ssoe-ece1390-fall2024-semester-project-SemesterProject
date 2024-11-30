import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def bokeh_blur(image, kernel, mask, gamma=3.3):

    # Mask original image for background and foreground
    mask_inv = cv2.bitwise_not(mask)
    image_background = image
    image_foreground = cv2.bitwise_and(image, image, mask=mask_inv)

    # Convert to linear (gamma correction, as images are stored in sRGB)
    # 2.2 is the "default" gamma, higher values produce brighter results for
    # larger radius blur
    image_linear = np.power(np.float32(image_background)/255, gamma)


    # Generate circular kernel
    # kernel_size = 51
    # kernel_radius = int(kernel_size/2)
    # x,y = np.indices([kernel_size, kernel_size])
    # x -= kernel_radius
    # y -= kernel_radius
    # r = np.sqrt(x**2+y**2)

    # kernel = (np.logical_and(r <= kernel_radius, x >= 0)).astype(float) / (kernel_size*kernel_size)

    # Manipulate kernel to right format
    kernel = np.float32(kernel) / np.sum(kernel)
    kernel = np.flip(kernel)

    # Apply Bokeh Filter and mask to result to background
    image_bokeh_linear = cv2.filter2D(image_linear, ddepth=-1, kernel=kernel)
    image_bokeh_linear = cv2.bitwise_and(image_bokeh_linear, image_bokeh_linear, mask=mask)

    # Apply inverse gamma correction
    image_bokeh = np.ubyte(np.power(image_bokeh_linear, 1/gamma) * 255)

    image_final = cv2.add(image_bokeh, image_foreground)
    # image_final = image_foreground
    # image_final = image_bg_painted

    return image_final



# # def __main__():
# # Read Image
# filename = "fireworks"
# file_extension = "jpg"
# image_bgr = cv2.imread(os.path.relpath(f"Input/Background/{filename}.{file_extension}"))
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# _,image_mask = cv2.threshold(image_rgb[:,:,0], 100, 255, cv2.THRESH_BINARY_INV)
# image_mask = cv2.erode(image_mask, np.ones((5,5)))

# kernel_name = "star"
# kernel_file = f"Input/Effect/{kernel_name}.png"
# kernel = np.float32(cv2.imread(os.path.relpath(kernel_file), cv2.IMREAD_GRAYSCALE))

# image_bokeh = bokeh_blur(image_rgb, kernel, image_mask)

# # Store results to file
# cv2.imwrite(f"Output/BokehEffect/{filename}_bokeh_{kernel_name}.{file_extension}", cv2.cvtColor(image_bokeh, cv2.COLOR_RGB2BGR))

# # Plot results
# plt.subplot(121);plt.imshow(image_rgb)
# plt.subplot(122);plt.imshow(image_bokeh)
# plt.show()