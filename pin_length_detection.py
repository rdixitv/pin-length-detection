import cv2
import numpy as np

def find_pin_robust(image, window_name="Debug"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 5)
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow(f"1_Mask_{window_name}", cv2.resize(mask, (800, 600)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"DEBUG: No objects found in {window_name}")
        return None

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    pin_cnt = None
    for cnt in sorted_contours:
        area = cv2.contourArea(cnt)
        if 500 < area < (image.shape[0] * image.shape[1] * 0.1): 
            pin_cnt = cnt
            break

    if pin_cnt is None:
        print(f"DEBUG: Could not isolate pin from background in {window_name}")
        return None

    tip1 = tuple(pin_cnt[pin_cnt[:, :, 0].argmin()][0])
    tip2 = tuple(pin_cnt[pin_cnt[:, :, 0].argmax()][0])

    debug_img = image.copy()
    cv2.circle(debug_img, tip1, 10, (0, 255, 0), -1) # Green
    cv2.circle(debug_img, tip2, 10, (0, 0, 255), -1) # Red
    cv2.imshow(f"2_Detect_{window_name}", cv2.resize(debug_img, (800, 600)))
    
    return tip1, tip2

def run_automated_measurement(path_l, path_r, baseline_mm):
    imgL = cv2.imread(path_l)
    imgR = cv2.imread(path_r)

    if imgL is None or imgR is None:
        print("Error: Could not load images.")
        return

    width, height = 3024, 4032
    # f_px = (26 * width) / 36 * 1.3
    f_px = (5.7 * width) / 7.15
    cx, cy = width / 2, height / 2

    pts_L = find_pin_robust(imgL, "Left")
    pts_R = find_pin_robust(imgR, "Right")
    pts_R = (pts_R[1], pts_R[0])

    if pts_L is None or pts_R is None:
        cv2.waitKey(0)
        return

    # Triangulation Matrices
    P1 = np.array([[f_px, 0, cx, 0], [0, f_px, cy, 0], [0, 0, 1, 0]], dtype=np.float32)
    P2 = np.array([[f_px, 0, cx, -f_px * baseline_mm], [0, f_px, cy, 0], [0, 0, 1, 0]], dtype=np.float32)

    # Convert to 3D
    ptsL_arr = np.array([pts_L[0], pts_L[1]], dtype=np.float32).T
    ptsR_arr = np.array([pts_R[0], pts_R[1]], dtype=np.float32).T

    points4D = cv2.triangulatePoints(P1, P2, ptsL_arr, ptsR_arr)
    points3D = (points4D[:3, :] / points4D[3, :]).T
    
    print(points3D)

    print(pts_L)
    print(pts_R)

    # Distance calculation
    length = np.linalg.norm(points3D[0] - points3D[1])
    length2 = points3D[0][2] - points3D[1][2]
    # depth_z = (points3D[0][2] + points3D[1][2]) / 2

    print(f"\n--- STEREO REPORT ---")
    print(f"Z top: {points3D[0][2]:.2f} mm")
    print(f"Z bottom: {points3D[1][2]} mm")
    print(f"FINAL CALCULATED LENGTH: {length:.2f} mm")
    print(f"FINAL CALCULATED LENGTH using only z values: {length2:.2f} mm")
    print("-" * 21)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_automated_measurement("39l.jpg", "39r.jpg", 278.0)
