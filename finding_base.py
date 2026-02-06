import pandas as pd 
import numpy as np
import cv2 
import re
import os
from typing import Tuple, Union
from io import BytesIO

import matplotlib.pyplot as plt
from skimage import io

def load_and_crop_image(
        image_path: Union[str, BytesIO],
        crop_coords: Tuple[int, int, int, int]= (400, 900, 0, 1200)
    ) -> Tuple[np.ndarray, int]:
    """
    Load an image, crop it to specified coordinates,
      and extract the 't' value from the filename.
 
     Args:
     image_path (str or io.BytesIO): Path to the input image or a file buffer.
     # if we do a streamlit app , we can pass io.BytesIO object, but we will need to find solutions to thr t value  
     crop_coords (tuple): Tuple of (y_start, y_end, x_start, x_end) for cropping.
     # we need to decide coords 

     Returns:
     Tuple[np.ndarray, int]: Cropped grayscale image and the extracted 't' value.
     """
    # Load the image in grayscale
    if isinstance(image_path,str): 
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    elif isinstance(image_path, BytesIO):
        image = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("image_path must be a string or io.BytesIO object.")
    if image is None:
        raise ValueError("Could not load image. Check the file path or buffer.")
    
    # extract the time value from the filename
    # the images are saved as frame[numner]
    # so we use the letter e to look for the number
    if isinstance(image_path,str):
        t_match = re.search(r"e(\d+)", image_path)
    else: 
        t_match = re.search(r"e(\d+)", str(image_path))
    if t_match:
        t_value = int(t_match.group(1)) # extract the number after 'e' , convert to int
    else:
        t_value = -1  # default value if 'e' not found
    
    # Crop the image using the provided coordinates
    y_start, y_end, x_start, x_end = crop_coords
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image, t_value


def blur_and_stretch(image) -> np.ndarray:
    """"
    apply gaussian blur and stretch the contrast of the image
    gets image from load_and_crop_image function (np.ndarray)

    returns processed image (np.ndarray)
    """
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Stretch the contrast of the image
    stretched_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return stretched_image

def blur_and_clahe(image) -> np.ndarray:
    """"
    apply gaussian blur and CLAHE to the image
    gets image from load_and_crop_image function (np.ndarray)

    returns processed image (np.ndarray)
    """
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(blurred_image)
    
    return clahe_image



def otsu_standard(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to the input image.

    Args:
    image (np.ndarray): Grayscale input image.

    Returns:
    np.ndarray: Binary image after applying Otsu's thresholding.
    """
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

def otsu_partial(image: np.ndarray, 
                lower_percentile: float=2.5,
                upper_percentile: float=97.5) -> np.ndarray:
    """
    Apply Otsu's thresholding to a specified percentile range of the input image.

    Args:
    image (np.ndarray): Grayscale input image.
    lower_percentile (float): Lower percentile for thresholding.
    upper_percentile (float): Upper percentile for thresholding.

    Returns:
    np.ndarray: Binary image after applying Otsu's thresholding on the specified range.
    """
    # Calculate the lower and upper bounds based on percentiles
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    
    # Create a mask for the specified range
    mask_range = (image >= lower_bound) & (image <= upper_bound)
    pixels_in_range = image[mask_range]
    
    
    if pixels_in_range.size == 0:
        raise ValueError("No pixels found in the specified percentile range.")
    
    thersh_val,_= cv2.threshold(pixels_in_range, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply Otsu's thresholding on the pixels within the specified range
    _, binary_masked = cv2.threshold(image, thersh_val, 255, cv2.THRESH_BINARY)
    
    return binary_masked

def canny_edge_detection(image: np.ndarray, 
                         lower_threshold: int=100, 
                         upper_threshold: int=200) -> np.ndarray:
    """
    Apply Canny edge detection to the input image.

    Args:
    image (np.ndarray): Grayscale input image.
    lower_threshold (int): Lower threshold for the hysteresis procedure.
    upper_threshold (int): Upper threshold for the hysteresis procedure.

    Returns:
    np.ndarray: Binary image after applying Canny edge detection.
    """
    # Apply Canny edge detection
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    
    return edges


# cleaning
def refine_image(image: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to refine the binary image.

    Args:
    image (np.ndarray): Binary input image.

    Returns:
    np.ndarray: Refined binary image after morphological operations.
    """
    # Define a kernel for morphological operations
    #kernel = np.ones((5, 5), np.uint8)
    kernal_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernal_size, kernal_size))
    
    # Apply morphological closing to fill small holes
    closed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=5)
    
    # Apply morphological opening to remove small noise
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return opened_image

def refine_image_by_filling(binary_mask: np.ndarray) -> np.ndarray:
    """
    Identifies the external contour of the droplet and fills all internal holes.
    Particularly effective for droplets on glass with strong reflections.
    """
    # 1. Invert the image for contour detection
    # OpenCV detects white objects on a black background
    inverted_mask = cv2.bitwise_not(binary_mask)
    
    # 2. Find contours
    # RETR_EXTERNAL retrieves only the outermost contours
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary_mask
    
    # 3. Create a new mask and represent the droplet as a solid area
    # We select the largest contour (assuming it represents the droplet)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Create a fully white image (matching your original background)
    filled_mask = np.ones_like(binary_mask) * 255
    
    # 4. Draw the contour and fill its interior with black (0)
    # thickness=-1 fills the entire internal area of the contour
    cv2.drawContours(filled_mask, [main_contour], -1, 0, thickness=-1)
    
    return filled_mask


def find_droplet_base(binary_mask, bottom_percent=0.3):
    """
    Find droplet baseline as the widest row,
    but only in the lower part of the droplet.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary image (droplet = 255 or True)
    bottom_percent : float
        Fraction of lower droplet height to search in (0–1)

    Returns
    -------
    dict with keys: y, x_left, x_right, width
    """

    mask = binary_mask == 0
    rows_with_drop = np.where(mask.any(axis=1))[0]

    if len(rows_with_drop) == 0:
        return None

    y_top = rows_with_drop[0]
    y_bottom = rows_with_drop[-1]
    droplet_height = y_bottom - y_top

    # search only in lower part of droplet
    y_start = int(y_bottom - bottom_percent * droplet_height)

    widths = np.sum(mask[y_start:y_bottom+1], axis=1)
    y_rel = np.argmax(widths)
    y_base = y_start + y_rel

    row_pixels = np.where(mask[y_base])[0]
    x_left = row_pixels[0]
    x_right = row_pixels[-1]

    return {
        'y': y_base,
        'x_left': x_left,
        'x_right': x_right,
        'width': x_right - x_left
    }
def crop_surface(image, y_surface):
    """"
    Crops the image to the region above the surface.
    """
    crop_limit = y_surface - 1
    cropped_image = image[0:crop_limit, :]
    return cropped_image

def extract_width_profile(image: np.ndarray) -> np.ndarray:
    """
    Extracts the width profile of the droplet from a binary mask.
    
    Iterates through each horizontal row (scanline) of the image to find 
    the distance between the leftmost and rightmost droplet pixels.
    
    Returns:
        y_vals (np.ndarray): The y-coordinates (row indices).
        widths (np.ndarray): The calculated width of the droplet at each row.
    """
    height, width = image.shape
    y_vals = []
    widths = []

    for y in range(height):
        # Identify indices of black pixels (0) in the current row (representing the droplet)
        black_pixels = np.where(image[y, :] == 0)[0]
        
        # If at least two pixels are found, calculate the width between boundaries
        if len(black_pixels) >= 2:
            x_left = black_pixels[0]
            x_right = black_pixels[-1]
            row_width = x_right - x_left
            
            y_vals.append(y)
            widths.append(row_width)
        else:
            # If no droplet is detected in this row, set width to 0
            widths.append(0)
            y_vals.append(y)
            
    return np.array(y_vals), np.array(widths)


from scipy.signal import savgol_filter

def smooth_width_profile(widths, window_size=11, poly_order=2):
    """
    Step 3.2: Smoothing the droplet width profile.
    
    Args:
        widths: The raw width data.
        window_size: The size of the filter window (must be an odd integer). 
        poly_order: The degree of the polynomial used for fitting (typically 2 or 3).
    """
    # Ensure the window size does not exceed the total number of data points
    if len(widths) < window_size:
        return widths
        
    # Apply the Savitzky-Golay filter
    smoothed_widths = savgol_filter(widths, window_size, poly_order)
    
    # Clip negative values to 0 (handling potential artifacts created by the smoothing process)
    smoothed_widths = np.maximum(smoothed_widths, 0)
    
    return smoothed_widths

def calculate_derivative(widths: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of the width profile.
    """
    derivative = np.gradient(widths) 
    return derivative

def find_and_visualize_kink(y_coords, smoothed_widths, smooth_derivative, original_img, binary_mask):
    """
    Step 3.4: Locate the derivative peak.
    Step 3.5: Map to X-coordinates.
    Step 3.6: Visual representation.
    """
    # 3.4: Find the index of the maximum value in the derivative
    # We restrict the search to the bottom half of the data 
    # to avoid interference from the top/apex of the droplet.
    search_start_idx = len(smooth_derivative) // 2
    relative_idx = np.argmax(smooth_derivative[search_start_idx:])
    peak_idx = search_start_idx + relative_idx
    
    y_baseline = y_coords[peak_idx]
    
    # 3.5: Locate X-coordinates for the selected row (baseline)
    # Use the original binary mask to find the exact edges at this y-coordinate
    black_pixels = np.where(binary_mask[y_baseline, :] == 0)[0]
    
    if len(black_pixels) < 2:
        print("Could not find baseline pixels.")
        return None
    
    x_left = black_pixels[0]
    x_right = black_pixels[-1]
    
    # 3.6: Visualization
    # Convert grayscale original image to BGR for color drawing
    display_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # Draw the baseline in red (BGR: 0, 0, 255)
    cv2.line(display_img, (x_left, y_baseline), (x_right, y_baseline), (0, 0, 255), 3)
    
    # Mark the contact points in green (BGR: 0, 255, 0)
    cv2.circle(display_img, (x_left, y_baseline), 6, (0, 255, 0), -1)
    cv2.circle(display_img, (x_right, y_baseline), 6, (0, 255, 0), -1)
    
    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(display_img)
    plt.title(f"Final Baseline Detection (Kink Point at y={y_baseline})")
    plt.axis('off')
    plt.show()
    
    return y_baseline, x_left, x_right

def extract_height_profile(binary_mask: np.ndarray):
    """
    שלב א: חילוץ פרופיל הגובה.
    עבור כל עמודה x, מחשב את המרחק בין הקצה העליון לתחתון.
    """
    height, width = binary_mask.shape
    x_vals = np.arange(width)
    heights = []
    
    for x in x_vals:
        # מוצאים את כל הפיקסלים ששייכים לטיפה בעמודה x (ערך 0 ב-Otsu שלנו)
        column_pixels = np.where(binary_mask[:, x] == 0)[0]
        if len(column_pixels) > 0:
            heights.append(np.max(column_pixels) - np.min(column_pixels))
        else:
            heights.append(0)
            
    return x_vals, np.array(heights)



from scipy.signal import savgol_filter

def smooth_profile(data, window_size=15, poly_order=2):
    """
    שלב ב: החלקת הנתונים למניעת רעשים בנגזרת.
    """
    if len(data) < window_size:
        return data
    return savgol_filter(data, window_size, poly_order)


def calculate_derivative(profile: np.ndarray):
    """
    שלב ג: חישוב הנגזרת של הפרופיל.
    """
    return np.gradient(profile)



def find_x_contact_points(derivative):
    """
    שלב ד: איתור נקודות המגע משמאל ומימין.
    """
    x_left = np.argmax(derivative)
    x_right = np.argmin(derivative)
    return x_left, x_right


def find_droplet_window(binary_mask):
    # 2. מציאת Baseline (y) - לפי שיטת רוחב מקסימלי או נגזרת y
    y_vals, w_profile = extract_width_profile(binary_mask)
    smoothed_w = smooth_profile(w_profile)
    deriv_w = calculate_derivative(smoothed_w)
    y_baseline = y_vals[len(deriv_w)//2 + np.argmax(deriv_w[len(deriv_w)//2:])] # מציאת הקינק ב-y
    
    # 3. מציאת נקודות מגע (x) - השיטה החדשה "על Y"
    x_vals, h_profile = extract_height_profile(binary_mask)
    smoothed_h = smooth_profile(h_profile)
    deriv_h = calculate_derivative(smoothed_h)
    x_left, x_right = find_x_contact_points(deriv_h)
    return y_baseline, x_left, x_right


def compute_mm_per_pixel(
    image_path,
    crop_coords=(0, 200, 400, 1000),
    dw_frac=0.005,
    min_run=3,
    plot=False,
    tip_diameter_mm=0.8
):
    """
    Compute mm-per-pixel conversion using tip width.

    Robust version:
    - Uses cylinder detection if possible
    - Falls back to percentile-based width if not
    - Handles cases with few detected rows
    """

    # -------------------------------------------------
    # 1. Load + crop image
    # -------------------------------------------------
    cropped_img, _ = load_and_crop_image(image_path, crop_coords)

    # -------------------------------------------------
    # 2. Otsu thresholding
    # -------------------------------------------------
    _, binary_img = cv2.threshold(
        cropped_img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -------------------------------------------------
    # 3. Measure width per row
    # -------------------------------------------------
    H, W = binary_img.shape
    widths = []
    ys = []

    for y in range(H):
        xs = np.where(binary_img[y] == 0)[0]
        if xs.size > 0:
            widths.append(xs[-1] - xs[0])
            ys.append(y)

    widths = np.array(widths)
    ys = np.array(ys)

    # -------------------------------------------------
    # FALLBACK 1: not enough rows
    # -------------------------------------------------
    if len(widths) < 10:
        if len(widths) == 0:
            raise ValueError("No tip detected in cropped image.")

        width_px = np.percentile(widths, 90)

        if width_px <= 0:
            raise ValueError("Fallback failed: invalid width.")

        mm_per_pixel = tip_diameter_mm / width_px

        if plot:
            plt.figure(figsize=(5, 5))
            plt.imshow(binary_img, cmap='gray')
            plt.title(
                f"Fallback (few rows)\n"
                f"{tip_diameter_mm} mm = {width_px:.1f} px  |  "
                f"{mm_per_pixel:.5f} mm/px"
            )
            plt.axis('off')
            plt.show()

        print("⚠️ Fallback calibration used (insufficient rows)")
        return mm_per_pixel

    # -------------------------------------------------
    # 4. Detect transition to cylinder
    # -------------------------------------------------
    dw = np.abs(np.diff(widths))
    dw_thresh = dw_frac * np.median(widths)

    rep_idx = None
    for i in range(len(dw) - min_run):
        if np.all(dw[i:i + min_run] < dw_thresh):
            rep_idx = i
            break

    # -------------------------------------------------
    # FALLBACK 2: no cylinder detected
    # -------------------------------------------------
    if rep_idx is None:
        width_px = np.percentile(widths, 90)

        if width_px <= 0:
            raise ValueError("Fallback failed: invalid width.")

        mm_per_pixel = tip_diameter_mm / width_px

        if plot:
            plt.figure(figsize=(5, 5))
            plt.imshow(binary_img, cmap='gray')
            plt.title(
                f"Fallback (no cylinder)\n"
                f"{tip_diameter_mm} mm = {width_px:.1f} px  |  "
                f"{mm_per_pixel:.5f} mm/px"
            )
            plt.axis('off')
            plt.show()

        print("⚠️ Fallback calibration used (no cylinder detected)")
        return mm_per_pixel

    # -------------------------------------------------
    # 5. Cylinder-based calibration (original logic)
    # -------------------------------------------------
    y_rep = ys[rep_idx]

    cyl_mask = dw[rep_idx:] < dw_thresh
    cyl_indices = np.where(cyl_mask)[0] + rep_idx

    runs = np.split(
        cyl_indices,
        np.where(np.diff(cyl_indices) != 1)[0] + 1
    )

    cyl_run = max(runs, key=len)

    if len(cyl_run) < min_run:
        # fallback instead of failing
        width_px = np.percentile(widths, 90)

        if width_px <= 0:
            raise ValueError("Fallback failed after short cylinder.")

        mm_per_pixel = tip_diameter_mm / width_px
        print("⚠️ Fallback calibration used (short cylinder)")
        return mm_per_pixel

    y_cyl_rows = ys[cyl_run]

    left_edges = []
    right_edges = []

    for y in y_cyl_rows:
        xs = np.where(binary_img[y] == 0)[0]
        if xs.size > 0:
            left_edges.append(xs[0])
            right_edges.append(xs[-1])

    x_left = int(np.median(left_edges))
    x_right = int(np.median(right_edges))
    width_px = x_right - x_left

    if width_px <= 0:
        raise ValueError("Invalid cylinder width detected.")

    mm_per_pixel = tip_diameter_mm / width_px

    # -------------------------------------------------
    # 6. Optional visualization
    # -------------------------------------------------
    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(binary_img, cmap='gray')
        plt.axhline(y_rep, color='red', linewidth=2, label='Arc → Cylinder')
        plt.axvline(x_left, color='blue', linewidth=2)
        plt.axvline(x_right, color='blue', linewidth=2)
        plt.title(
            f"{tip_diameter_mm} mm = {width_px} px  |  "
            f"{mm_per_pixel:.5f} mm/px"
        )
        plt.axis('off')
        plt.legend()
        plt.show()

    return mm_per_pixel



def find_surface_line(image, debug=False):
    raw, t = load_and_crop_image(image, [300,900,0,250])

    # Blur עדין
    raw_blur = cv2.GaussianBlur(raw, (5,5), 0)

    # גרדיאנט אנכי (מעבר רקע → משטח)
    grad_y = cv2.Sobel(raw_blur, cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad_y = np.abs(grad_y)

    y_candidates = []

    # לכל עמודה: מציאת y עם גרדיאנט מקסימלי
    for x in range(grad_y.shape[1]):
        col = grad_y[:, x]
        y_candidates.append(np.argmax(col))

    # baseline יציב = median
    y_baseline = int(np.median(y_candidates))

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(raw, cmap='gray')
        plt.axhline(y_baseline, color='red')
        plt.title("Detected surface baseline")
        plt.show()

    return y_baseline

    


