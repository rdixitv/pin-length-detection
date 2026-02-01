import cv2
import numpy as np

def find_pin_extremes(image, window_name="Debug"):
    """Finds the coordinates of the top and bottom of the pin"""

    # Preprocessing to increase the contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    

    _, mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"DEBUG: No white objects found in {window_name}")
        return None

    # Get the largest white object
    cnt = max(contours, key=cv2.contourArea)

    # Get Extreme Points along the X-axis
    # leftmost point
    tip1 = tuple(cnt[cnt[:, :, 0].argmin()][0])
    # rightmost point
    tip2 = tuple(cnt[cnt[:, :, 0].argmax()][0])

    # Draw the detected points
    debug_img = image.copy()
    cv2.circle(debug_img, tip1, 25, (0, 255, 0), -1)
    cv2.circle(debug_img, tip2, 25, (0, 0, 255), -1)
    cv2.imshow(f"Detection_{window_name}", cv2.resize(debug_img, (800, 600)))
    
    return tip1, tip2

def run_measurement(path_l, path_r, baseline_mm):
    # Load Images
    imgL = cv2.imread(path_l)
    imgR = cv2.imread(path_r)

    if imgL is None or imgR is None:
        print("Error: Could not load images. Check your filenames.")
        return

    # Image dimensions
    width, height = 4032, 3024

    # Focal length
    f_px = (26 * width) / 36 
    cx, cy = width / 2, height / 2

    # Detect tips
    pts_L = find_pin_extremes(imgL, "Left")
    pts_R = find_pin_extremes(imgR, "Right")

    if pts_L is None or pts_R is None:
        print("Error: Detection failed")
        cv2.waitKey(0)
        return

    # Projection Matrices
    P1 = np.array([[f_px, 0, cx, 0], 
                   [0, f_px, cy, 0], 
                   [0, 0, 1, 0]], dtype=np.float32)
    
    P2 = np.array([[f_px, 0, cx, -f_px * baseline_mm], 
                   [0, f_px, cy, 0], 
                   [0, 0, 1, 0]], dtype=np.float32)

    ptsL_arr = np.array([pts_L[0], pts_L[1]], dtype=np.float32).T
    ptsR_arr = np.array([pts_R[0], pts_R[1]], dtype=np.float32).T

    # Triangulate to 4D Homogeneous
    points4D = cv2.triangulatePoints(P1, P2, ptsL_arr, ptsR_arr)
    
    # Normalize to 3D (Divide X, Y, Z by W)
    points3D = (points4D[:3, :] / points4D[3, :]).T

    # Tip 1 to Tip 2 distance
    length = np.linalg.norm(points3D[0] - points3D[1])

    print(f"\n--- STEREO MEASUREMENT REPORT ---")
    print(f"Calculated Depth (Z): {points3D[0][2]:.2f} mm") 
    print(f"Tip 1 Coord: {points3D[0]}")
    print(f"Tip 2 Coord: {points3D[1]}")
    print("-" * 33)
    print(f"FINAL CALCULATED LENGTH: {length:.2f} mm")
    print("-" * 33)
    
    print("\nPress any key in the image windows to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Baseline is 335 cm
run_measurement("left.jpg", "right.jpg", 335.0)
