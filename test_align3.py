import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage.draw import line

PIX_PER_UM_10X = 3330 / 700
PIX_PER_UM = PIX_PER_UM_10X
SCALE=0.25

def pyramid_decomposition(image, levels):
    pyramid = [image]

    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)

    return pyramid

def pyramid_reconstruction(pyramid):
    # Start with the smallest image in the pyramid
    reconstructed_image = pyramid[-1]
    reconstructed = [reconstructed_image]

    # Iteratively upsample and add the previous levels
    for i in range(len(pyramid) - 2, -1, -1):
        # Upsample the current image to the size of the next larger image
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        reconstructed.append(reconstructed_image)

        # Add the upsampled image to the current level of the pyramid
        reconstructed_image = cv2.add(reconstructed_image, pyramid[i])

    return reconstructed

def tile_pyramid(pyramid):
    # Determine the size of the canvas
    h, w = pyramid[0].shape[:2]
    canvas_width = w + w // 2 + 1
    canvas_height = h

    # Initialize the canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Place the largest image
    current_x, current_y = 0, 0
    canvas[current_y:current_y + h, current_x:current_x + w] = pyramid[0]

    for i in range(1, len(pyramid)):
        img = pyramid[i]
        h, w = img.shape[:2]

        if i % 2 == 1: # place the image to the right
            next_x = current_x + pyramid[i-1].shape[1]
            next_y = current_y
        else: # place it below
            next_x = current_x
            next_y = current_y + pyramid[i-1].shape[0]

        # Place the image
        canvas[next_y:next_y + h, next_x:next_x + w] = img

        # Update the position for the next tile
        current_x, current_y = next_x, next_y

    return canvas

# Function to perform thinning using morphological operations
def thinning(image):
    size = np.size(image)
    skeleton = np.zeros(image.shape, np.uint8)

    ret, image = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True

    return skeleton

def filter_contours(contours, min_area=1000, sides=4, epsilon=0.02):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if area > min_area and len(approx) == sides:
            filtered_contours.append(approx)

    return filtered_contours

# this routine is sloooooooow
def compute_adaptive_thresholds(image, blockSize, C, method='mean'):
    # Ensure blockSize is odd
    assert blockSize % 2 == 1, "blockSize must be odd."

    # Compute the integral image for fast mean calculation
    integral_image = cv2.integral(image)

    # Padding to maintain the same size output
    padded_image = cv2.copyMakeBorder(image, blockSize // 2, blockSize // 2, blockSize // 2, blockSize // 2, cv2.BORDER_REPLICATE)

    # Output thresholds
    thresholds = np.zeros_like(image, dtype=np.float32)

    # Compute the thresholds
    if method == 'mean':
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                y1 = y
                y2 = y + blockSize
                x1 = x
                x2 = x + blockSize

                # Compute the sum of the block using the integral image
                block_sum = integral_image[y2, x2] - integral_image[y1, x2] - integral_image[y2, x1] + integral_image[y1, x1]
                # Calculate mean
                block_area = blockSize * blockSize
                mean = block_sum / block_area
                thresholds[y, x] = mean - C

    elif method == 'gaussian':
        # Compute Gaussian weighted sum
        # Create a Gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(blockSize, -1)
        gaussian_kernel = gaussian_kernel * gaussian_kernel.T
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                y1 = y
                y2 = y + blockSize
                x1 = x
                x2 = x + blockSize

                # Apply Gaussian kernel to the block
                block = padded_image[y:y2, x:x2].astype(np.float32)
                gaussian_weighted_sum = np.sum(block * gaussian_kernel)
                thresholds[y, x] = gaussian_weighted_sum - C

    thresholds = thresholds - thresholds.min()
    thresh_int = thresholds.astype(np.uint8)
    return thresh_int

def generate_sinusoid_image(image_shape, period, phase, angle_deg):
    """
    Generates a 2D image of a sinusoid with a given period, phase, and angle relative to the Y-axis.

    Parameters:
    - image_shape: Tuple (height, width) defining the shape of the image.
    - period: The period of the sinusoid in pixels.
    - phase: The phase offset of the sinusoid in radians.
    - angle_deg: The angle of the sinusoid relative to the Y-axis in degrees.

    Returns:
    - A 2D numpy array representing the image with the sinusoid.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Create coordinate grids for the image
    y, x = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')

    # Rotate the coordinates based on the given angle
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)

    # Generate the sinusoidal pattern
    sinusoid = np.sin((2 * np.pi * x_rot / period) + phase)

    return sinusoid

# Notes:
#  - Log-polar transform can help find both translation and rotation
#  - Fourier-Mellin transform may be a viable method, but we have to generate a reference image
#
# Next steps - generate the reference image to align against.

if __name__ == '__main__':
    # Specify the directory containing PNG images
    image_directory = Path('imaging/')

    # Iterate through all PNG files in the directory
    for image_path in image_directory.glob('*.png'):
        # Read the image using OpenCV
        img = cv2.imread(str(image_path))

        # show original
        cv2.imshow(str(image_path), cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # filtered_image = cv2.bilateralFilter(norm, d=5, sigmaColor=75, sigmaSpace=75)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0) # blur at wavelength-res to reduce pixel noise
        norm = cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)

        ret, thresh = cv2.threshold(norm, 80, 255, cv2.THRESH_BINARY)
        adaptive_thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 31, 2)

        src = cv2.convertScaleAbs(norm, alpha=1/255.0)
        # show what goes into the pipe
        cv2.imshow("adaptive", cv2.resize(src, (0, 0), fx=SCALE, fy=SCALE))

        angles = np.arange(89.5, 90.5, 0.1)
        for angle in angles:
            print(f"angle: {angle-90.0}")
            basis = generate_sinusoid_image(src.shape, 22, 0, angle)
            convolved = cv2.filter2D(src, -1, basis)
            cv2.imshow("convolved", cv2.resize(convolved, (0, 0), fx=SCALE, fy=SCALE))
            cv2.imshow("basis", cv2.resize(basis, (0, 0), fx=SCALE, fy=SCALE))
            cv2.waitKey(0)

        if False:
            thresholds = compute_adaptive_thresholds(gray, 127, 2, method='gaussian')
            cv2.imshow("thresholds", cv2.resize(gray - thresholds, (0, 0), fx=SCALE, fy=SCALE))

        if False: # this was the previous best candidate
            # next steps - maybe apply this to a pyramidally decomposed image??
            # the problem is currently that the image is too complicated, and too high-resolution -
            # the curve tracer yields way more than the specified number of points.

            # another option is to recurse...

            # Find contours in the image
            contours, hierarchy = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #for i in range(len(contours)):
            #    cv2.drawContours(gray, contours, i, (0, 255, 0), 1)
            #cv2.imshow('all contours', cv2.resize(gray, (0, 0), fx=SCALE, fy=SCALE))

            # Filter the contours by area and aspect ratio
            for param in range(3,10):
                p = param
                print(f"parameter: {p}")
                filtered_contours = filter_contours(contours, 1000, p, 0.02)
                # Draw the rectangles on the image
                img_c = img.copy()
                for i in range(len(filtered_contours)):
                    cv2.drawContours(img_c, filtered_contours, i, (0, 255, 0), 1)
                cv2.imshow('filtered contours', cv2.resize(img_c, (0, 0), fx=SCALE, fy=SCALE))
                cv2.waitKey(0)



        if False: # compute pyramids for multi-scale texture analysis
            pyramid = pyramid_decomposition(adaptive_thresh, 4)
            tiled_pyramid = tile_pyramid(pyramid)
            cv2.imshow("pyramid", cv2.resize(tiled_pyramid, (0, 0), fx=SCALE, fy=SCALE))

            r_pyramid = pyramid_reconstruction(pyramid)
            delta = [a - b for a, b in zip(pyramid, r_pyramid[::-1])]
            tiled_delta = tile_pyramid(delta)
            cv2.imshow("r_pyramid", cv2.resize(tiled_delta, (0, 0), fx=SCALE, fy=SCALE))

        cv2.destroyAllWindows()
