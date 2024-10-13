import argparse
import logging

import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from math import exp, log, pow
import json
import pickle
import operator

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS GDS to pixels helper")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=True, help="Base name of file series",
    )
    parser.add_argument(
        "--layer", required=False, help="Layer to process", choices=['poly', 'm1'], default='poly'
    )
    parser.add_argument(
        "--tech", required=False, help="Tech library", choices=['sky130'], default='sky130'
    )

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    name = args.name
    layer = args.layer
    tech = args.tech

    # Load two images (they should be grayscale for simplicity)
    gds_png = cv2.imread(f"imaging/{name}-{layer}.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(f"imaging/{name}.png", cv2.IMREAD_GRAYSCALE)

    max_dim = max(max(gds_png.shape), max(image.shape))
    gds_png_snap = snap_to_max(gds_png, max_dim)
    image_snap = snap_to_max(image, max_dim)

    # Find rotation and scale difference
    rotation_angle, scale_factor = find_rotation_and_scale(gds_png_snap, image_snap)
    print(f"Rotation angle: {rotation_angle}, Scale factor: {scale_factor}")

    # Correct img2 for rotation and scale
    corrected_image = correct_rotation_and_scale(image_snap, -rotation_angle + 180, 1/scale_factor)

    # now template match the reference onto the image so we can determine the offset
    corr = cv2.matchTemplate(corrected_image, gds_png, cv2.TM_CCOEFF)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(corr)
    # create the composite
    composite_overlay = np.zeros(corrected_image.shape, np.uint8)
    composite_overlay[max_loc[1]:max_loc[1] + gds_png.shape[0], max_loc[0]: max_loc[0] + gds_png.shape[1]] = gds_png
    blended = cv2.addWeighted(corrected_image, 1.0, composite_overlay, 0.5, 0)

    with open(f'imaging/{args.tech}_cells.json', 'r') as f:
        cell_names = json.load(f)

    with open(f'imaging/{name}-{layer}_lib.json', 'r') as f:
        cells = json.load(f)

    # check alignment by drawing the rectangles
    cell_overlay = np.zeros(corrected_image.shape, np.uint8)
    max_x = 0
    max_y = 0
    entry = {}
    entry['labels'] = []
    entry['data'] = []
    labels = {} # this is no longer used, we pull the label index from the master label list
    label_count = 0
    for cell in cells.values():
        if cell[2] not in labels:
            labels[cell[2]] = label_count
            label_count += 1
        cv2.rectangle(
            cell_overlay,
            [cell[0][0][0] + max_loc[0], cell[0][0][1] + max_loc[1]],
            [cell[0][1][0] + max_loc[0], cell[0][1][1] + max_loc[1]],
            cell[1]
        )
        # this is used to sanity check the statically coded dimensions of the x/y image crops
        if abs(cell[0][0][0] - cell[0][1][0]) > max_x:
            max_x = abs(cell[0][0][0] - cell[0][1][0])
        if abs(cell[0][0][1] - cell[0][1][1]) > max_y:
            max_y = abs(cell[0][0][1] - cell[0][1][1])

        # extract a rectangle around the center of each standard cell and save it in a labelled training set
        data = np.zeros((32, 64, 3), dtype=np.uint8)
        center_x = (cell[0][0][0] + cell[0][1][0]) // 2
        center_y = (cell[0][0][1] + cell[0][1][1]) // 2
        if center_y - 16 >= 0 and center_y + 16 < corrected_image.shape[0] \
           and center_x - 32 >= 0 and center_x + 32 < corrected_image.shape[1]:
            data = cv2.cvtColor(corrected_image[center_y - 16:center_y + 16, center_x - 32:center_x + 32], cv2.COLOR_GRAY2RGB)
            # cv2.imshow('data', data)
            # cv2.waitKey(0)
            try:
                label_index = cell_names.index(cell[2])
            except ValueError:
                print(f'Cell not in master cell list: {cell[2]}; skipping')
                continue
            entry['labels'].append(label_index) # substitute with a numeric value so it can be converted to a tensor
            entry['data'].append(data)

    # dump the data into pickle files for consumption by downstream CNN pipeline
    print(f'max_x: {max_x}, max_y: {max_y}')
    with open(f'imaging/{name}-{layer}.pkl', 'wb') as f:
        pickle.dump(entry, f)
    meta = {
        'num_cases_per_batch' : len(entry['data']),
        'label_names' : cell_names,
        'num_vis' : 64 * 32 * 3,
    }
    with open(f'imaging/{name}-{layer}.meta', 'wb') as f:
        pickle.dump(meta, f)

    # Quality check the alignment
    blended_rect = cv2.addWeighted(corrected_image, 1.0, cell_overlay, 0.5, 0)

    # Display the corrected image
    # cv2.imshow("Corrected Image", corrected_img2)
    # cv2.imshow("Reference image", img1)
    cv2.imshow("Correlation", cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    cv2.imshow("Composite", blended)
    cv2.imshow("Rectangles", blended_rect)
    cv2.waitKey(0)

    # Now read in the JSON file with cell locations, and use this to create a training set of data
    # This consists of:
    #  - A monochrome rectangle that defines the region of interest; the color is correlated to the gate type
    #  - The underlying source image, cropped to a fixed size that represents the maximum search window for a
    #    gate of any size (equal to the biggest standard cell plus some alignment margin)
    #  - The representation of the "true gate" as a black and white image, correlated to the source image
    #
    # The input to the classifier would be a source image area, that is the same as the fixed size used in training
    # The output of the classifier is a tensor of potential gate matches, which we will threshold into "most likely match"

