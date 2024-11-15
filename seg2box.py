import os
from glob import glob
import rasterio
from tqdm import tqdm
import numpy as np
from utils.image import (
    normalize_uint8,
    process_mask,
    save_image,
    clip_stretch
)
from utils.bbox import (
    get_bboxes,
    remove_microboxes,
    remove_nested_boxes,
    merge_overlapping_boxes,
    draw_bounding_boxes,
    convert,
    save_boxes_to_txt
)
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DRIVER_DICT = {
    "tif": "GTiff",
    "tiff": "GTiff",
    "png": "PNG",
    "jpg": "JPEG"
}


def main(opt):
    # Variable allocations
    output_dir: str = opt.output_dir
    iou_threshold: str = opt.iou
    selected_bands: list = [i-1 for i in opt.selected_bands]  # Convert to 0-based indexing from band numbers
    verbose: str = opt.verbose
    source_images: str = opt.image_dir
    source_masks: str = opt.mask_dir
    mask_suffix: str = opt.mask_suffix
    image_suffix: str = opt.img_suffix
    image_ext: str = opt.img_ext
    mask_ext: str = opt.mask_ext
    output_dtype: str = "uint8"
    try:
        image_driver: str = DRIVER_DICT[opt.img_ext.lower()]
        mask_driver: str = DRIVER_DICT[opt.mask_ext.lower()]
    except KeyError:
        raise ValueError(f"Invalid image or mask extension. Supported extensions are: {list(DRIVER_DICT.keys())}")
    except Exception as e:
        raise e
    
    # Fix band discrepancy for JPGs and PNGs
    if image_ext.lower() in ["jpg", "jpeg", "png"]:
        selected_bands = [-1]  # Use all bands for JPGs and PNGs

    # Set up directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Make mask dir if needed
    mask_dir = None
    if opt.save_mask:
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
    
    # Find all masks
    mask_pattern = os.path.join(source_masks, f"**/*{mask_suffix}.{mask_ext}")
    all_masks = sorted(glob(mask_pattern, recursive=True))

    # Decide between drawing on images or masks
    all_images = []
    drawing_on_masks = False
    if not source_images:
        print("No images directory provided. Proceeding to draw on masks.", end="\n\n")
        all_images = all_masks
        image_driver = mask_driver
        image_ext = mask_ext
        drawing_on_masks = True
        selected_bands = [0]  # Use gray band for masks

    # Find images if needed
    if not drawing_on_masks:
        image_pattern = os.path.join(source_images, f"**/*{image_suffix}.{image_ext}")
        all_images = sorted(glob(image_pattern, recursive=True))
        # If no images are found, draw on masks
        if not all_images:
            print("No images found. Proceeding to draw on masks.", end="\n\n")
            all_images = all_masks
            image_driver = mask_driver
            image_ext = mask_ext
            drawing_on_masks = True
            selected_bands = [0]  # Use gray band for masks
    
    # Check if georeferenced
    sample = None
    if all_images:
        sample = all_images[0]
        chosen_driver = image_driver
    else:
        sample = all_masks[0]
        chosen_driver = mask_driver
    with rasterio.open(sample, "r", driver=chosen_driver) as src:  # Change driver as per extension argument via a dictionary.
        if src.crs is None:
            sample_name = sample.replace("\\", "/")
            print(f"Proceeding without georeferencing, as \n{sample_name}, \nhas no crs property in RASTERIO metadata.", end="\n\n")

    # Main section
    if all_masks:
        print(f"Processing {len(all_masks)} images...")
        count = 0

        for file, mask_file_path in tqdm(zip(all_images, all_masks), total=len(all_masks)):
            file_root = os.path.splitext(os.path.basename(file))[0]

            # Read the processed image
            with rasterio.open(file, "r", driver=image_driver) as src:
                # Keep an argument for selected bands in images if any. 
                if not src.count > max(selected_bands):
                    raise ValueError(f"Indexes for selected bands > number of bands. Please check number of input bands in the image. \nCurrent status: \nNumber of bands: {src.count}\nSelected bands: {selected_bands} \nImages found: {not drawing_on_masks}\n\n")

                if selected_bands == [-1]:  # Drawing on png or jpeg
                    image_data = np.transpose(normalize_uint8(src.read()), (1, 2, 0))
                    image_metadata = src.meta
                elif not drawing_on_masks:  # Drawing on tif
                    image_data = clip_stretch(src.read()[selected_bands])
                    image_metadata = src.meta
                    image_metadata["count"] = len(selected_bands) if isinstance(selected_bands, list) else 1
                else:  # Drawing on mask
                    image_data = process_mask(src.read(), bands=3)  # Get 3 band version for drawing
                    if len(image_data.shape) == 4 and image_data.shape[0] == 1:  # Correct additional axis dimension, (1, h, w, c) -> (h, w, c)
                        image_data = image_data[0]
                    image_metadata = src.meta
                    image_metadata["count"] = 3

                image_metadata["dtype"] = output_dtype
            
            size = (image_metadata["width"], image_metadata["height"])

            # Read the mask
            with rasterio.open(mask_file_path, "r", driver=mask_driver) as mask_src:
                mask_data = process_mask(mask_src.read())  # Assuming single-band mask
                mask_metadata = mask_src.meta

                if opt.save_mask:
                    mask_metadata_copy = mask_metadata.copy()
                    mask_metadata_copy["count"] = 1
                    mask_metadata_copy["dtype"] = output_dtype
                    mask_metadata_copy["crs"] = image_metadata["crs"]
                    mask_metadata_copy["transform"] = image_metadata["transform"]
                
                    mask_dest = mask_file_path.replace(source_masks, mask_dir)
                    if len(mask_data.shape) == 4:
                        mask_data = mask_data[0]
                    if min(mask_data.shape) == mask_data.shape[2]:  # If no. channels are the last dimension, move it to first for rasterio
                        mask_data = np.transpose(mask_data, (2, 0, 1))
                    save_image(mask_dest, mask_data, mask_metadata_copy)

            # Create bounding boxes
            raw_boxes = get_bboxes(mask_data)
            
            # Remove microboxes
            removed_microboxes = remove_microboxes(raw_boxes, (mask_metadata["width"], mask_metadata["height"]))
            
            # Remove nested boxes
            removed_nested_boxes = remove_nested_boxes(removed_microboxes)
            
            # Merge overlapping boxes
            final_boxes = merge_overlapping_boxes(removed_nested_boxes, iou_threshold=iou_threshold)
            
            # Save intermediary stages if verbose
            if verbose:
                image_with_raw_boxes = np.transpose(draw_bounding_boxes(image_data, raw_boxes), (2, 0, 1))
                raw_boxes_name = os.path.join(images_dir, file_root + "_raw_boxes." + image_ext)
                save_image(raw_boxes_name, image_with_raw_boxes, image_metadata)
                
                image_no_microboxes = np.transpose(draw_bounding_boxes(image_data, removed_microboxes), (2, 0, 1))
                removed_microboxes_name = os.path.join(images_dir, file_root + "_removed_microboxes." + image_ext)
                save_image(removed_microboxes_name, image_no_microboxes, image_metadata)
                
                image_removed_nested_boxes = np.transpose(draw_bounding_boxes(image_data, removed_nested_boxes), (2, 0, 1))
                removed_nested_boxes_name = os.path.join(images_dir, file_root + "_removed_nested_boxes." + image_ext)
                save_image(removed_nested_boxes_name, image_removed_nested_boxes, image_metadata)

            if not final_boxes:
                continue
            
            # Clip boxes
            boxes = [[max(0, box[0]), max(0, box[1]), min(image_metadata["width"], box[2]), min(image_metadata["height"], box[3])] for box in final_boxes]
            
            # Draw final bboxes and save the image
            drawn_image = np.transpose(draw_bounding_boxes(image_data, boxes), (2, 0, 1))
            if verbose:
                drawn_image_path = os.path.join(images_dir, file_root + "_final." + image_ext)
                save_image(drawn_image_path, drawn_image, image_metadata)
            else:
                drawn_image_path = os.path.join(images_dir, file_root + "." + image_ext)
                save_image(drawn_image_path, drawn_image, image_metadata)
            
            # Convert bounding boxes to YOLO format
            yolo_boxes = [convert(size, box) for box in boxes]

            # Write the YOLO annotations to a .txt file
            annotation_file = drawn_image_path.replace(images_dir, labels_dir).replace("." + image_ext, ".txt")
            save_boxes_to_txt(yolo_boxes, annotation_file)
            
            count += 1
    else:
        print("No masks found. Exiting...", end="\n\n")


if __name__ == "__main__":
    parser = ArgumentParser(exit_on_error=True, description="Arguments to convert segmentation masks to bounding boxes.\n")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing images.")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing masks.")
    parser.add_argument("--img-ext", type=str, default="tif", help="Extension of the images (eg: tif, tiff, png).")
    parser.add_argument("--mask-ext", type=str, default="tif", help="Extension of the masks.")
    parser.add_argument("--img-suffix", type=str, default="", help="Suffix of the images.")
    parser.add_argument("--mask-suffix", type=str, default="", help="Suffix of the masks.")
    parser.add_argument("--output-dir", type=str, default="bbox_dataset", help="Output directory for the drawn images and generated labels.")
    parser.add_argument("--save-mask", action="store_true", help="Save the mask files in the output directory.")
    parser.add_argument("--iou", type=float, default=0.2, help="IOU threshold for merging overlapping boxes.")
    parser.add_argument("--selected-bands", type=int, nargs="+", default=[3, 2, 1], help="Selected bands for the visualized images.")  # Select RGB bands by default as per Sentinel-2 imagery
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbosity level for the script, choose between 0 and 1. Choosing 1 will print intermediary visualisations for every bbox removal level. Intended for debugging purposes. Choosing 0 will only visualise the final bboxes and save the images.")
    args = parser.parse_args()
    dict_args = vars(args)
    
    for key, value in dict_args.items():
        print(f"{key}: {value}")
    print("\n")
    
    main(args)
