[![DOI](https://zenodo.org/badge/888979325.svg)](https://doi.org/10.5281/zenodo.14169702)

# Segmentation-to-Bounding Box Converter

## Description

This repository features a novel algorithm for converting binary segmentation masks into YOLO-compatible bounding boxes. It removes microboxes, eliminates nested boxes, and merges overlapping boxes using a custom IoU metric. Designed for efficiency and precision, this tool addresses the lack of automated solutions for dataset preparation.

Note: Additional dataset for testing, containing 12-band TIF images and 1-band TIF masks can be downloaded here: [TIF Data split](https://drive.google.com/drive/folders/17RkzkwMRBFIY10Md_9etoS8AepjQdeza?usp=sharing).

## Features

-   Converts binary segmentation masks to YOLO-compatible bounding boxes.
-   Removes microboxes smaller than 0.1% of the image area.
-   Eliminates nested bounding boxes.
-   Merges overlapping bounding boxes using a custom IoU metric, "MinIoU".

## Installation

This tool was tested on Python 3.10.15. To set up the environment, first ensure you are using the correct Python version, then run the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the script on `seg_ds_pngs`, containing PNG images and PNG masks, with the code saving masks and verbose output visualizations, use:

```bash
python seg2box.py --image-dir seg_ds_pngs/images --mask-dir seg_ds_pngs/masks --img-ext png --mask-ext png --output-dir output_pngs --save-mask -v 1
```

To view usage instructions, use:

```bash
python seg2box.py --help
```

## MinIoU Explanation

Unlike the traditional Intersection over Union (IoU), **MinIoU** is calculated as:

```
min_iou = intersection_area / area_of_smaller_box
```

This ensures that if a certain threshold area of either of the bounding boxes overlaps with another, they are merged into one box. This approach guarantees more accurate bounding box representations by reducing unnecessary splits and merging boxes with significant overlap.
