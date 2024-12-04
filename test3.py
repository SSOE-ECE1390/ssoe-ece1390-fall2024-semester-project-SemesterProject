import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

# Image paths
image_paths = ['Data/jim.jpg', 'Data/crying_stock_photo.png']

# Create the options for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:

    # Loop through demo image(s)
    for image_file_name in image_paths:
        
        # Read image 
        image = cv2.imread(image_file_name)
     
        # Convert the image to MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Retrieve the segmentation result
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        # Convert the BGR image to RGB for processing
        image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply effects (blurring the background)
        blurred_image = cv2.GaussianBlur(image_data, (55, 55), 0)
        
        # Create the mask condition
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        
        # Combine the images (foreground is original image, background is blurred)
        output_image = np.where(condition, image_data, blurred_image)

        # Convert back to BGR for OpenCV display
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Show the final image
        cv2.imshow(f'Segmented Image: {image_file_name}', output_image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
