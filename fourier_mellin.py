import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from math import exp, log, pow

# Derived from reference code generated as follows:
#   Prompt: "give me an example implementation of using a fourier-mellin transform to correct for rotation and scale between two images"
#   Model: ChatGPT 4o

def log_polar_transform(image):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_radius = min(center[0], center[1])
    log_base = max_radius / np.log(max_radius)
    return cv2.logPolar(image, center, log_base, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

def phase_correlation(image1, image2):
    f1 = fft2(image1)
    f2 = fft2(image2)
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    inverse_fft = ifft2(cross_power_spectrum)
    return np.abs(inverse_fft)

def find_rotation_and_scale(image1, image2):
    # Apply log-polar transformation to convert rotation and scaling to translation
    image1_logpolar = log_polar_transform(image1)
    image2_logpolar = log_polar_transform(image2)

    # Compute the phase correlation to find translation (rotation and scale)
    result = phase_correlation(image1_logpolar, image2_logpolar)

    if False:
        cv2.imshow("img1 refpolar", image1_logpolar)
        cv2.imshow("img2 refpolar", image2_logpolar)
        cv2.imshow("phase_correlation", cv2.normalize(result, 0, 255, cv2.NORM_MINMAX))
        cv2.waitKey(0)

    # Find the peak of the correlation to get rotation and scaling difference
    max_loc = np.unravel_index(np.argmax(result), result.shape)

    # Compute scale and rotation
    rotation_angle = (max_loc[0] - image1.shape[0] / 2) * (360.0 / image1.shape[0])

    # from https://github.com/sthoduka/imreg_fmt/tree/master
    rows = image1.shape[0] # height
    cols = image1.shape[1] # width
    logbase = exp(log(rows * 1.1 / 2.0) / max(rows, cols))

    scale = pow(logbase, max_loc[1])
    scale = 1.0 / scale

    return rotation_angle, scale

def correct_rotation_and_scale(image, rotation_angle, scale_factor):
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Correct for rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Correct for scale
    corrected_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return corrected_image

def snap_to_max(img, max_dim):
    snap = np.zeros((max_dim, max_dim), np.uint8)
    offset_x = (snap.shape[1] - img.shape[1]) // 2
    offset_y = (snap.shape[0] - img.shape[0]) // 2
    snap[offset_y:offset_y + img.shape[0], offset_x:offset_x+img.shape[1]] = img
    return snap

# Example usage
if __name__ == "__main__":
    # Load two images (they should be grayscale for simplicity)
    img1 = cv2.imread("imaging/housekeeping-poly.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("imaging/housekeeping_-0.7deg_crop.png", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("imaging/housekeeping_0deg_crop.png", cv2.IMREAD_GRAYSCALE)

    max_dim = max(max(img1.shape), max(img2.shape))
    img1_snap = snap_to_max(img1, max_dim)
    img2_snap = snap_to_max(img2, max_dim)

    # Find rotation and scale difference
    rotation_angle, scale_factor = find_rotation_and_scale(img1_snap, img2_snap)
    print(f"Rotation angle: {rotation_angle}, Scale factor: {scale_factor}")

    # Correct img2 for rotation and scale
    corrected_img2 = correct_rotation_and_scale(img2_snap, -rotation_angle + 180, 1/scale_factor)

    # now template match the reference onto the image so we can determine the offset
    corr = cv2.matchTemplate(corrected_img2, img1, cv2.TM_CCOEFF)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(corr)
    # create the composite
    composite_overlay = np.zeros(corrected_img2.shape, np.uint8)
    composite_overlay[max_loc[1]:max_loc[1] + img1.shape[0], max_loc[0]: max_loc[0] + img1.shape[1]] = img1
    blended = cv2.addWeighted(corrected_img2, 1.0, composite_overlay, 0.5, 0)

    # Display the corrected image
    # cv2.imshow("Corrected Image", corrected_img2)
    # cv2.imshow("Reference image", img1)
    cv2.imshow("Correlation", cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    cv2.imshow("Composite", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
