import cv2
import numpy as np
from pathlib import Path
import gdspy
import matplotlib.pyplot as plt
from math import ceil

PIX_PER_UM_10X = 3330 / 700
PIX_PER_UM = PIX_PER_UM_10X
LAMBDA = 0.17
NA = 0.28
F_NUMBER = 1 / (2 * 0.28) # approximation for small angles
AIRY_GAUSSIAN = 0.84 * F_NUMBER * 1.050 * PIX_PER_UM

#BG_COLOR = (255, 255, 255)
#FG_COLOR = (16, 16, 16)

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

if __name__ == '__main__':
    image_directory = Path('imaging/')
    for gds_file in image_directory.glob('*.gds'):
        # Load the GDS file
        gds_lib = gdspy.GdsLibrary(infile=str(gds_file))
        # Get all cells in the GDS
        cells = gds_lib.top_level()

        # Extract the bounding box of the layout for scaling
        all_polygons = []
        for cell in cells:
            for polygon in cell.get_polygons():
                all_polygons.append(polygon)

        # Determine layout bounds (for setting image size)
        min_x = min(polygon[:, 0].min() for polygon in all_polygons)
        min_y = min(polygon[:, 1].min() for polygon in all_polygons)
        max_x = max(polygon[:, 0].max() for polygon in all_polygons)
        max_y = max(polygon[:, 1].max() for polygon in all_polygons)

        block_width = ceil((max_x - min_x))
        block_height = ceil((max_y - min_y))

        MAX_X = 10_000
        MAX_Y = 10_000

        scale_up_over_um = 16.0 / LAMBDA # keep track of this as a factor over micron because we eventually have to scale to pixels/micron
        optical_factor = scale_up_over_um / PIX_PER_UM # this is what we scale by to achieve parity with imaged data

        steps_x = ceil(block_width * scale_up_over_um / MAX_X)
        steps_y = ceil(block_height * scale_up_over_um / MAX_Y)

        image = np.full((ceil((steps_y + 1) * MAX_Y / optical_factor), ceil((steps_x + 1) * MAX_X / optical_factor), 3), BG_COLOR, dtype=np.uint8)
        target_res = (int(MAX_X / optical_factor), int(MAX_Y / optical_factor))

        for x_base in range(steps_x):
            for y_base in range(steps_y):
                tile = np.full((MAX_Y, MAX_X, 3), BG_COLOR, dtype = np.uint8)
                x_offset = x_base * MAX_X
                y_offset = y_base * MAX_Y
                tile_bounds = (x_offset, y_offset, x_offset + MAX_X, y_offset + MAX_Y)
                for cell in cells:
                    # cell.write_svg(image_directory / (gds_file.stem + '.svg'))
                    # svg_to_png(image_directory / (gds_file.stem + '.svg'), image_directory / (gds_file.stem + '.png'))
                    for polygon in cell.get_polygons():
                        # problem: fillPoly will fill in the "lines" as well as the center, causing
                        # the amount of 'metal' to be overdrawn by the width of a pixel. There seems to be
                        # no native method call to fill in just the polygon centers. However, we also want
                        # to "over-render" the polygons anyways and then convolve them down into average
                        # reflectance areas, so in the end the most performant method may be to divide the
                        # polygon list into polygons that overlap a large region, oversample it, then blend
                        # it down into an equivalent image for template matching.
                        points = np.rint(polygon * scale_up_over_um).astype(int)
                        bounding_rect = cv2.boundingRect(points)
                        if is_intersecting(tile_bounds, bounding_rect):
                            cv2.fillPoly(tile, [points + [-x_offset, -y_offset]], FG_COLOR, lineType=cv2.LINE_AA)
                # tile now contains a highly upsampled version of the target image
                ksize = int(optical_factor / 2.0) * 2 + 1
                tile = cv2.blur(tile, (ksize, ksize))
                tile = cv2.resize(tile, target_res, interpolation=cv2.INTER_CUBIC)

                image[y_base * target_res[1]:(y_base+1) * target_res[1], x_base * target_res[0]:(x_base + 1) * target_res[0]] = tile
                # cv2.imshow('tile', tile)
                # cv2.waitKey(0)


        cv2.imshow('rendered', image)
        cv2.imwrite(str(image_directory / (gds_file.stem + '.png')), image)
        airy = cv2.GaussianBlur(image, (0, 0), AIRY_GAUSSIAN)
        cv2.imshow('airy', airy)
        cv2.imwrite(str(image_directory / (gds_file.stem + '_airy_' + '.png')), airy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()