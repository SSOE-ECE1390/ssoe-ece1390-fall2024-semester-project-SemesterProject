import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Read Image
image_bgr = cv2.imread(os.path.relpath("Input/fireworks.jpg"))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to linear (gamma correction, as images are stored in sRGB)
# 2.2 is the "default" gamma, higher values produce brighter results for
# larger radius blur
gamma = 3.3
image_linear = np.pow(np.float32(image_rgb)/255, gamma)


# Generate circular kernel
kernel_size = 21
kernel_radius = int(kernel_size/2)
x,y = np.indices([kernel_size, kernel_size])
x -= kernel_radius
y -= kernel_radius
r = np.sqrt(x**2+y**2)

kernel = (r <= kernel_radius).astype(float) / (kernel_size*kernel_size)

# Apply Bokeh Filter
image_bokeh_linear = cv2.filter2D(image_linear, ddepth=-1, kernel=kernel)

# Apply inverse gamma correction
image_bokeh = np.ubyte(np.pow(image_bokeh_linear, 1/gamma) * 255)


# Store results to file
cv2.imwrite("Output/fireworks_bokeh.jpg", cv2.cvtColor(image_bokeh, cv2.COLOR_RGB2BGR))

# Plot results
plt.subplot(121);plt.imshow(image_rgb)
plt.subplot(122);plt.imshow(image_bokeh)
plt.show()