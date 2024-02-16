#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2
import sys
import json
import importlib.util
from progressbar.bar import ProgressBar

from schema import Schema
from prims import Rect, Point
from pallette import HashPallette
from design import Design

# derived from reference image "full-H"
# NOTE: this may change with improvements in the microscope hardware.
# be sure to re-calibrate after adjustments to the hardware.
PIX_PER_UM_20X = 3535 / 370 # 20x objective
PIX_PER_UM_5X = 2350 / 1000 # 5x objective, 3.94 +/- 0.005 ratio to 20x
PIX_PER_UM_10X = 3330 / 700 # 10x objective, ~4.757 pix/um
PIX_PER_UM = None

def main():
    parser = argparse.ArgumentParser(description="IRIS LEF/DEF to pixels helper")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--def-file", required=True, help="DEF file containing the top-level layout"
    )
    parser.add_argument(
        "--tech", required=True, help="Technology name", choices=['gf180', 'sky130', 'tsmc22ull']
    )
    parser.add_argument(
        "--mag", help="Magnification of target image", choices=['5x', '10x', '20x'], default='10x'
    )
    parser.add_argument(
        "--regenerate-lef", default=False, action="store_true", help="Force regeneration of LEF database"
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    tech_module_spec = importlib.util.spec_from_file_location('Tech', f'./{args.tech}.py')
    tech_module = importlib.util.module_from_spec(tech_module_spec)
    tech_module_spec.loader.exec_module(tech_module)
    tm = tech_module.Tech(args)

    if args.mag == '5x':
        PIX_PER_UM = PIX_PER_UM_5X
    elif args.mag == '10x':
        PIX_PER_UM = PIX_PER_UM_10X
    elif args.mag == '20x':
        PIX_PER_UM = PIX_PER_UM_20X

    design_path = Path(args.def_file).parent
    top_def = Design(args.def_file, PIX_PER_UM)

    logging.info("gathering top level stats...")
    tm.gather_stats(top_def)

    logging.info("generating image...")
    # now generate a PNG of the cell map that we can use to manually overlay
    # on the stitched image to validate that our parsing makes sense.
    die_ll = top_def.schema['die_area_ll']
    die_ur = top_def.schema['die_area_ur']
    die = Rect(Point(die_ll[0], die_ll[1]), Point(die_ur[0], die_ur[1]))
    canvas = np.zeros((int(die.height() * PIX_PER_UM), int(die.width() * PIX_PER_UM), 3), dtype=np.uint8)

    missing_cells = top_def.render_layer(tm)
    for missing_cell in missing_cells:
        d = Design(design_path / (missing_cell['cell'] + '.def'), PIX_PER_UM)
        tm.gather_stats(d)
        d.render_layer(tm)
        top_def.merge_subdesign(d, missing_cell)

    logging.info("generating legend...")
    top_def.generate_legend(tm)

    # final output artifacts
    top_def.save_layout()
    tm.print_stats()

if __name__ == "__main__":
    main()
