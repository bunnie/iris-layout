import cv2
import numpy as np
from numpy.fft import fft2, ifft2

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

    # Find the peak of the correlation to get rotation and scaling difference
    max_loc = np.unravel_index(np.argmax(result), result.shape)

    # Compute scale and rotation
    rotation_angle = (max_loc[0] - image1.shape[0] / 2) * (360.0 / image1.shape[0])
    scale_factor = np.exp((max_loc[1] - image1.shape[1] / 2) / image1.shape[1])

    return rotation_angle, scale_factor

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
    # img2 = cv2.imread("imaging/housekeeping_-0.7deg_crop.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("imaging/housekeeping_0deg_crop.png", cv2.IMREAD_GRAYSCALE)

    max_dim = max(max(img1.shape), max(img2.shape))
    img1_snap = snap_to_max(img1, max_dim)
    img2_snap = snap_to_max(img2, max_dim)

    # Find rotation and scale difference
    rotation_angle, scale_factor = find_rotation_and_scale(img1_snap, img2_snap)
    print(f"Rotation angle: {rotation_angle}, Scale factor: {scale_factor}")

    # Correct img2 for rotation and scale
    corrected_img2 = correct_rotation_and_scale(img2_snap, -rotation_angle, 1/scale_factor)

    # Display the corrected image
    cv2.imshow("Corrected Image", corrected_img2)
    cv2.imshow("Reference image", img1_snap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
