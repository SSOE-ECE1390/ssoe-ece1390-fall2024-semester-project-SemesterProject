import pixellib
from pixellib.instance import instance_segmentation
import cv2

def img_segment_people(path):
    segment_image = instance_segmentation()
    segment_image.load_model("mask_rcnn_coco.h5") 
    segmask, output = segment_image.segmentImage(path, show_bboxes=True)

    # Access the segmented masks
    person_masks = []
    for i, class_id in enumerate(segmask['class_ids']):
        if class_id == 0:  # 0 represents 'person' in COCO dataset
            person_masks.append(segmask['masks'][:, :, i])
    output_path = "Output/people_segmented_img.png"
    cv2.imwrite(output_path, person_masks[0])