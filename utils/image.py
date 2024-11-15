'''
Utility functions related to image processing.
'''
import numpy as np
import rasterio

target = 'uint8'

def normalize_uint8(image: np.ndarray) -> np.ndarray:
    return ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)


def normalize_min_max(image: np.ndarray) -> np.ndarray:
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def process_mask(mask: np.ndarray, bands: int = 1) -> np.ndarray:
    intermediary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    # If single band, convert to 3 band
    if bands == 3 and (len(intermediary_mask.shape) == 2 or 1 in intermediary_mask.shape):
        return np.stack((intermediary_mask,) * 3, axis=-1)
    return intermediary_mask


def _clip_percentiles(image: np.ndarray, lower_percentile: float = 2, upper_percentile: float = 98) -> np.ndarray:
    """
    Clip the image based on the percentiles (Cumulative Count Cut).
    
    Args:
        image (np.ndarray): Input image.
        lower_percentile (float): Lower percentile cut-off.
        upper_percentile (float): Upper percentile cut-off.
    
    Returns:
        np.ndarray: Image with values clipped to the specified percentiles.
    """
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, lower_bound, upper_bound)
    return clipped_image


def _stretch_to_minmax(image: np.ndarray) -> np.ndarray:
    """
    Stretch image values to the [0, 1] range based on the image's min and max values.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Stretched image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        return np.zeros_like(image, dtype=target)
    
    stretched_image = (image - min_val) / (max_val - min_val) * 255
    return stretched_image


def clip_stretch(image: np.ndarray) -> np.ndarray:
    clipped_image = _clip_percentiles(image)
    stretched_image = _stretch_to_minmax(clipped_image)
    return stretched_image.astype(target)


def prepare_image(file: str, output_name: str):
    with rasterio.open(file, "r", driver="GTiff") as src:
        data = src.read()
        
        # Print the shape of the data array
        # print(f"Shape of data array for {file}: {data.shape}")
        
        # Get the number of bands
        num_bands = data.shape[0]
        # print(f"Number of bands: {num_bands}")

        # Select a single band (e.g., Band 8 - typically NIR for Sentinel-2, adjust index as needed)
        band_index = 8  # Adjust this index if needed based on your specific requirements
        if band_index < num_bands:
            single_band_image = clip_stretch(data[band_index])  # Process the specified band
        else:
            print(f"Warning: {file} does not have the specified band index {band_index}. Skipping...")
            return  # Skip this file if it doesn't have enough bands

        # Define output path in the new directory
        # output_file = os.path.join(output_dir, os.path.basename(file).replace(".tiff", "_clip_stretched_1band.png"))

        # Ensure metadata includes width and height
        metadata = src.meta.copy()  # Make a copy of the original metadata
        metadata.update({
            'driver': 'PNG',
            'count': 1,
            'dtype': target,
            'width': single_band_image.shape[1],  # Set width
            'height': single_band_image.shape[0]  # Set height
        })

        with rasterio.open(output_name, "w", **metadata) as dst:
            dst.write(single_band_image, 1)


def save_image(filepath, image, metadata):
    with rasterio.open(filepath, "w", **metadata) as dst:
        dst.write(image)
