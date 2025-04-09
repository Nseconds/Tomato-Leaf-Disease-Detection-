import cv2
import numpy as np

def preprocess_for_leaf_detection(image: np.ndarray) -> np.ndarray:
    """Preprocess image for edge-based leaf detection."""
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Step 3: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (7, 7), 0)  # Increased kernel size for better noise reduction

    # Step 4: Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 20, 100)  # Lowered thresholds to capture more edges

    # Step 5: Dilate edges to connect broken segments
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Increased iterations to connect more edges

    # Step 6: Erode to remove small noise
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded