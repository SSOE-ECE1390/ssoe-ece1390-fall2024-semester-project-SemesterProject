import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Read Image
filename = "fireworks"
file_extension = "jpg"
image_bgr = cv2.imread(os.path.relpath(f"Input/{filename}.{file_extension}"))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to linear (gamma correction, as images are stored in sRGB)
# 2.2 is the "default" gamma, higher values produce brighter results for
# larger radius blur
gamma = 3.3
image_linear = np.pow(np.float32(image_rgb)/255, gamma)


# Generate circular kernel
# kernel_size = 51
# kernel_radius = int(kernel_size/2)
# x,y = np.indices([kernel_size, kernel_size])
# x -= kernel_radius
# y -= kernel_radius
# r = np.sqrt(x**2+y**2)

# kernel = (np.logical_and(r <= kernel_radius, x >= 0)).astype(float) / (kernel_size*kernel_size)


# Generate kernel from file
kernel_file = "Input/star.png"
kernel = np.float32(cv2.imread(os.path.relpath(kernel_file), cv2.IMREAD_GRAYSCALE)) / 255
kernel /= kernel.shape[0] * kernel.shape[1]
kernel = np.flip(kernel)

# Apply Bokeh Filter
image_bokeh_linear = cv2.filter2D(image_linear, ddepth=-1, kernel=kernel)

# Apply inverse gamma correction
image_bokeh = np.ubyte(np.pow(image_bokeh_linear, 1/gamma) * 255)


# Store results to file
cv2.imwrite(f"Output/{filename}_bokeh.{file_extension}", cv2.cvtColor(image_bokeh, cv2.COLOR_RGB2BGR))

# Plot results
plt.subplot(121);plt.imshow(image_rgb)
plt.subplot(122);plt.imshow(image_bokeh)
plt.show()