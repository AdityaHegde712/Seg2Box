'''
Utility functions for bounding box creation and modification.
'''
import numpy as np
import torch
from skimage.measure import label, regionprops
import cv2


# Convert bounding boxes to YOLO format (x, y, width, height)
def convert(size, box):
    x_center = ((box[0] + box[2]) / 2.0) / size[0]
    y_center = ((box[1] + box[3]) / 2.0) / size[1]
    width = abs((box[2] - box[0]) / size[0])
    height = abs((box[3] - box[1]) / size[1])
    return (x_center, y_center, width, height)


# Save bounding boxes to a text file
def save_boxes_to_txt(boxes, filepath):
    with open(filepath, 'w') as f:
        for box in boxes:
            f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")


# Remove microboxes (i.e., boxes with an area less than 0.1% of the image size)
def remove_microboxes(boxes, size):
    image_area = size[0] * size[1]
    min_area_threshold = int(0.001 * image_area)
    # return [box for box in boxes if abs((box[2] - box[0]) * (box[3] - box[1])) >= min_area_threshold]
    saved_boxes = []
    for box in boxes:
        area = abs((box[2] - box[0]) * (box[3] - box[1]))
        if area >= min_area_threshold:
            saved_boxes.append(box)
    return saved_boxes


# Remove nested boxes (i.e., one box fully inside another)
def remove_nested_boxes(boxes):
    new_boxes = []
    for box in boxes:
        is_nested = False
        for other_box in boxes:
            if box != other_box:
                # Check if box is inside other_box
                if (box[0] >= other_box[0] and box[1] >= other_box[1] and
                        box[2] <= other_box[2] and box[3] <= other_box[3]):
                    is_nested = True
                    break
        if not is_nested:
            new_boxes.append(box)
    return new_boxes


# Compute IoU (Intersection over Union)
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def min_iou(box1, box2):
    """A custom iou where iou is computed with respect to the smaller box.

    Args:
        box1 (Tuple[int, int, int, int]): First bounding box.
        box2 (Tuple[int, int, int, int]): Second bounding box.

    Raises:
        ValueError: If the box coordinates are not integers.

    Returns:
        float: iou value.
    """
    # if not all(isinstance(coord, int) for coord in box1 + box2):
    #     raise ValueError("Box coordinates must be integers.")
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(min(box1_area, box2_area))
    
    return iou


# Perform NMS with box merging for overlapping boxes
def merge_overlapping_boxes(boxes, iou_threshold=0.4):
    if len(boxes) == 1:
        return boxes

    while True:
        merged = False
        for idx, box in enumerate(boxes):
            for other_idx, other_box in enumerate(boxes[idx+1:]):
                iou = min_iou(box, other_box)
                # found a nested box, remove the smaller one
                if iou == 1:
                    boxes = remove_nested_boxes(boxes)
                    merged = True
                    break
                if iou > iou_threshold:
                    # Merge the boxes
                    new_box = [min(box[0], other_box[0]), min(box[1], other_box[1]),
                               max(box[2], other_box[2]), max(box[3], other_box[3])]
                    # Replace the two old boxes with the new merged box
                    boxes[idx] = new_box
                    del boxes[idx + 1 + other_idx]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return boxes


def get_bboxes(mask: np.ndarray) -> np.ndarray:    
    # Ensure mask is a numpy array if it's a torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # If the mask has more than 2 dimensions (e.g., a color image), we need to reduce it to 2D
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask)
    elif len(mask.shape) == 3 and mask.shape[0] > 1:
        raise ValueError("Mask has more than one channel. Please provide a single-channel mask. We haven't built this for multichannel masks yet.")

    # Label connected components in the mask
    labeled_mask = label(mask == 255)

    # Get properties of labeled regions (connected components)
    props = regionprops(labeled_mask)

    # Extract bounding boxes
    bounding_boxes = []
    for prop in props:
        ymin, xmin, ymax, xmax = prop.bbox
        bounding_boxes.append((xmin, ymin, xmax, ymax))

    return bounding_boxes


# Draw bounding boxes on the image
def draw_bounding_boxes(image, boxes):
    # Rearrange the image from (3, 1132, 674) -> (1132, 674, 3)
    if image.shape[0] == 3:
        image_rgb = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
    else:
        image_rgb = image.astype(np.uint8)
    
    # Draw the bounding boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)  # Ensure coordinates are integers
        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Red bounding box
    
    return image_rgb
