import argparse
import logging
import importlib.util
from progressbar.bar import ProgressBar
import json

import cv2
import numpy as np
from pathlib import Path
import gdspy
from math import ceil

# Measured effective zoom of the optical system
PIX_PER_UM_10X = 3330 / 700
PIX_PER_UM = PIX_PER_UM_10X

# Effective measured trace/space from the GDS file. Used to
# up-scale the GDS prior to imaging to remove artifacts due to
# the polygon fill algorithm over-filling drawn edges.
LAMBDA = 0.17

# Scaling factors based on the above parameters
SCALE_OVER_MICRONS = 16.0 / LAMBDA # keep track of this as a factor over micron because we eventually have to scale to pixels/micron
OPTICAL_FACTOR = SCALE_OVER_MICRONS / PIX_PER_UM # this is what we scale by to achieve parity with imaged data

# Maximum intermediate tile resolution. Limited by available memory to process tiles.
MAX_X = 10_000
MAX_Y = 10_000
# Max tile sizes translated into a final optical resolution, accounting for intermediate scaling over microns
OPTICAL_RES = (int(MAX_X / OPTICAL_FACTOR), int(MAX_Y / OPTICAL_FACTOR))

# Compute Airy Disk approximation constants
NA = 0.28 # From the lens datasheet
F_NUMBER = 1 / (2 * 0.28) # approximation for small angles
AIRY_GAUSSIAN = 0.84 * F_NUMBER * 1.050 * PIX_PER_UM

# Black on white
#BG_COLOR = (255, 255, 255)
#FG_COLOR = (16, 16, 16)

# White on black
BG_COLOR = (0, 0, 0)
FG_COLOR = (128, 128, 128)

def is_intersecting(r1, r2):
    r1_x, r1_y, r1_w, r1_h = r1
    r2_x, r2_y, r2_w, r2_h = r2

    if (r1_x < r2_x + r2_w and
        r1_x + r1_w > r2_x and
        r1_y < r2_y + r2_h and
        r1_y + r1_h > r2_y):
        return True
    return False

def map_orientation(rotation, reflection):
    # rotation: in radians, could be None
    # reflection: boolean, true/false

    if rotation == None:
        rotation = 0.0
    # normalize to 0-to-2pi range
    rotation = rotation % (2 * np.pi)

    if ((rotation > 7.0 * np.pi / 4.0) and (rotation <= 8.0 * np.pi / 4.0)) or ((rotation >= 0.0) and (rotation < np.pi / 4.0)):
        if reflection:
            return 'FN'
        else:
            return 'N'
    elif (rotation > np.pi / 4.0) and (rotation < 3.0 * np.pi / 4.0):
        if reflection:
            return 'FE'
        else:
            return 'E'
    elif (rotation > 3.0 * np.pi / 4.0) and (rotation < 5.0 * np.pi / 4.0):
        if reflection:
            return 'FS'
        else:
            return 'S'
    elif (rotation > 5.0 * np.pi / 4.0) and (rotation < 7.0 * np.pi / 4.0):
        if reflection:
            return 'FW'
        else:
            return 'W'
    else:
        logging.error(f"Unhandled rotation: {rotation}")

def export_png(cells):
    all_polygons = []
    for polygon in cells.get_polygons():
        all_polygons.append(polygon)

    # Determine layout bounds (for setting image size)
    min_x = min(polygon[:, 0].min() for polygon in all_polygons)
    min_y = min(polygon[:, 1].min() for polygon in all_polygons)
    max_x = max(polygon[:, 0].max() for polygon in all_polygons)
    max_y = max(polygon[:, 1].max() for polygon in all_polygons)

    block_width = ceil((max_x - min_x))
    block_height = ceil((max_y - min_y))

    steps_x = ceil(block_width * SCALE_OVER_MICRONS / MAX_X)
    steps_y = ceil(block_height * SCALE_OVER_MICRONS / MAX_Y)
    image = np.full((ceil((steps_y + 1) * MAX_Y / OPTICAL_FACTOR), ceil((steps_x + 1) * MAX_X / OPTICAL_FACTOR), 3), BG_COLOR, dtype=np.uint8)

    total_steps = steps_x * steps_y
    progress = ProgressBar(min_value = 0, max_value=total_steps, prefix = f'Extracting {gds_file.stem} ')
    step = 0
    for x_base in range(steps_x):
        for y_base in range(steps_y):
            tile = np.full((MAX_Y, MAX_X, 3), BG_COLOR, dtype = np.uint8)
            x_offset = x_base * MAX_X
            y_offset = y_base * MAX_Y
            tile_bounds = (x_offset, y_offset, x_offset + MAX_X, y_offset + MAX_Y)
            # cells.write_svg(image_directory / (gds_file.stem + '.svg'))
            # svg_to_png(image_directory / (gds_file.stem + '.svg'), image_directory / (gds_file.stem + '.png'))
            for polygon in cells.get_polygons():
                # problem: fillPoly will fill in the "lines" as well as the center, causing
                # the amount of 'metal' to be overdrawn by the width of a pixel. There seems to be
                # no native method call to fill in just the polygon centers. However, we also want
                # to "over-render" the polygons anyways and then convolve them down into average
                # reflectance areas, so in the end the most performant method may be to divide the
                # polygon list into polygons that overlap a large region, oversample it, then blend
                # it down into an equivalent image for template matching.
                points = np.rint(polygon * SCALE_OVER_MICRONS).astype(int)
                bounding_rect = cv2.boundingRect(points)
                if is_intersecting(tile_bounds, bounding_rect):
                    cv2.fillPoly(tile, [points + [-x_offset, -y_offset]], FG_COLOR, lineType=cv2.LINE_AA)
            # tile now contains a highly upsampled version of the target image
            ksize = int(OPTICAL_FACTOR / 2.0) * 2 + 1
            tile = cv2.blur(tile, (ksize, ksize))
            tile = cv2.resize(tile, OPTICAL_RES, interpolation=cv2.INTER_CUBIC)

            image[y_base * OPTICAL_RES[1]:(y_base+1) * OPTICAL_RES[1], x_base * OPTICAL_RES[0]:(x_base + 1) * OPTICAL_RES[0]] = tile
            # cv2.imshow('tile', tile)
            # cv2.waitKey(0)
            step += 1
            progress.update(step)
    progress.finish()

    cv2.imshow(f'{gds_file.stem} rendered', image)
    cv2.imwrite(str(image_directory / (gds_file.stem + '.png')), image)
    # apply a gaussian blur that approximates the effect of an airy disk, which simulates the effect of
    # diffraction-limited optics. The Airy parameters are computed in the constants at the top of the file.
    airy = cv2.GaussianBlur(image, (0, 0), AIRY_GAUSSIAN)
    cv2.imshow(f'{gds_file.stem} airy', airy)
    cv2.imwrite(str(image_directory / (gds_file.stem + '_airy_' + '.png')), airy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def export_lib(cell_list):
    boxes = []
    export = {}
    for ref in cell_list.references:
        color = tm.pallette.str_to_rgb(ref.ref_cell.name, map_orientation(ref.rotation, ref.x_reflection))
        color_int = list(map(int, color))
        if ref.get_bounding_box() is not None:
            boxes += [(ref.get_bounding_box(), color_int)]

    min_x = min(polygon[0][:, 0].min() for polygon in boxes)
    min_y = min(polygon[0][:, 1].min() for polygon in boxes)
    max_x = max(polygon[0][:, 0].max() for polygon in boxes)
    max_y = max(polygon[0][:, 1].max() for polygon in boxes)

    block_width = ceil((max_x - min_x))
    block_height = ceil((max_y - min_y))
    image = np.full((ceil(block_height * PIX_PER_UM), ceil(block_width * PIX_PER_UM), 3), BG_COLOR, dtype=np.uint8)
    offset = (int(round(min_x * PIX_PER_UM)), int(round(min_y * PIX_PER_UM)))
    progress = ProgressBar(min_value = 0, max_value=len(boxes), prefix = f'Library mapping {gds_file.stem} ')
    for (i, (rect, color)) in enumerate(boxes):
        r = np.rint(rect * PIX_PER_UM).astype(int)
        cv2.rectangle(image, r[0] - offset, r[1] - offset, color, thickness=-1, lineType=cv2.LINE_8)
        progress.update(i)
        # pixel offsets and colors
        export[i] = ([(r[0] - offset).tolist(), (r[1] - offset).tolist()], color)
    progress.finish()

    cv2.imshow(f'{gds_file.stem} library', image)
    cv2.imwrite(str(image_directory / (gds_file.stem + '_lib.png')), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open(str(image_directory / (gds_file.stem + '_lib.json')), 'w') as f:
        json.dump(export, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRIS GDS to pixels helper")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--tech", required=True, help="Technology name", choices=['gf180', 'sky130', 'tsmc22ull']
    )
    parser.add_argument(
        "--regenerate-lef", default=False, action="store_true", help="Force regeneration of LEF database"
    )
    parser.add_argument(
        "--redact", default=False, action="store_true", help="Redact details"
    )

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    # load the technology description
    tech_module_spec = importlib.util.spec_from_file_location('Tech', f'./{args.tech}.py')
    tech_module = importlib.util.module_from_spec(tech_module_spec)
    tech_module_spec.loader.exec_module(tech_module)
    tm = tech_module.Tech(args)

    image_directory = Path('imaging/')
    for gds_file in image_directory.glob('*.gds'):
        # Load the GDS file
        gds_lib = gdspy.GdsLibrary(infile=str(gds_file))
        # Get all cells in the GDS
        cells = gds_lib.top_level()
        # This seems to always be the case for a GDS file that's read in? unclear; maybe just a quirk of the test cases I have.
        # Catch it if it's not the case, so I can find the test case and understand what it even means to have two top cells.
        assert len(cells) == 1

        if False:
            # Export the GDS as PNG files
            export_png(cells[0])

        if True:
            # Export the GDS as abstract library tiles
            export_lib(cells[0])
