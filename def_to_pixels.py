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

DEF_TO_PIXELS_VERSION = '1.0.0'
# derived from reference image "full-H"
# NOTE: this may change with improvements in the microscope hardware.
# be sure to re-calibrate after adjustments to the hardware.
PIX_PER_UM_20X = 3535 / 370 # 20x objective
PIX_PER_UM_5X = 2350 / 1000 # 5x objective, 3.94 +/- 0.005 ratio to 20x
PIX_PER_UM_10X = 3330 / 700 # 10x objective, ~4.757 pix/um
PIX_PER_UM = None

def build_json(schema, df):
    with open(df, 'r') as def_file:
        total_lines = 0
        for line in def_file:
            total_lines += 1
    with open(df, 'r') as def_file:
        state = 'HEADER'
        progress = ProgressBar(min_value = 0, max_value = total_lines, prefix='Processing...')
        processed_lines = 0
        for line in def_file:
            processed_lines += 1
            progress.update(processed_lines)
            line = line.strip().lstrip()
            tokens = line.split(' ')
            if state == 'HEADER':
                if tokens[0] == 'DESIGN':
                    schema['name'] = tokens[1]
                elif tokens[0] == 'UNITS':
                    schema['units'] = float(tokens[3])
                elif tokens[0] == 'DIEAREA':
                    # reduce any die area polygon into a rectangle that encompasses the maximum extents
                    # regex has...problems doing an arbitrary list length, so we do this with a stupid
                    # iterative construct
                    da_state = 'SEARCH_L'
                    da_coords = []
                    coord = []
                    for token in tokens:
                        if da_state == 'SEARCH_L':
                            if token == '(':
                                da_state = 'X'
                        elif da_state == 'X':
                            coord += [float(token) / schema['units']]
                            da_state = 'Y'
                        elif da_state == 'Y':
                            coord += [float(token) / schema['units']]
                            da_coords += [coord]
                            coord = []
                            da_state = 'SEARCH_L'
                    min_x = 10**20
                    min_y = 10**20
                    max_x = 0
                    max_y = 0
                    for coord in da_coords:
                        if coord[0] > max_x:
                            max_x = coord[0]
                        if coord[0] < min_x:
                            min_x = coord[0]
                        if coord[1] > max_y:
                            max_y = coord[1]
                        if coord[1] < min_y:
                            min_y = coord[1]
                    schema['die_area_ll'] = [min_x, min_y]
                    schema['die_area_ur'] = [max_x, max_y]
                elif tokens[0] == 'COMPONENTS':
                    state = 'COMPONENTS'
            if state == 'COMPONENTS':
                if tokens[0] == 'END' and len(tokens) > 1 and tokens[1] == 'COMPONENTS':
                    state = 'DONE'
                elif tokens[0] == '-':
                    name = tokens[1]
                    cell = tokens[2]
                    # now do an iterative search through tokens for subsections that we care about
                    comp_state = 'SEARCH'
                    skip = False
                    for token in tokens:
                        if comp_state == 'SEARCH':
                            if token == 'PLACED':
                                comp_state = 'PLACED'
                            elif token == 'SOURCE':
                                comp_state = 'SOURCE'
                            elif token == ';':
                                if not skip:
                                    schema['cells'][name] = {
                                        'cell': cell,
                                        'loc' : [x, y],
                                        'orientation' : orientation
                                    }
                                comp_state = 'END'
                        elif comp_state == 'PLACED':
                            assert token == '('
                            comp_state = 'PLACED_X'
                        elif comp_state == 'PLACED_X':
                            x = float(token) / schema['units']
                            comp_state = 'PLACED_Y'
                        elif comp_state == 'PLACED_Y':
                            y = float(token) / schema['units']
                            comp_state = 'PLACED_)'
                        elif comp_state == 'PLACED_)':
                            assert token == ')'
                            comp_state = 'PLACED_ORIENTATION'
                        elif comp_state == 'PLACED_ORIENTATION':
                            orientation = token
                            comp_state = 'SEARCH'

                        elif comp_state == 'SOURCE':
                            if token != 'DIST' and token != 'NETLIST':
                                skip = True
                            comp_state = 'SEARCH'

        progress.finish()
        with open(df.stem + '.json', 'w+') as def_out:
            def_out.write(json.dumps(schema, indent=2))


def main():
    parser = argparse.ArgumentParser(description="IRIS LEF/DEF to pixels helper")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--def-file", required=True, help="DEF file containing the layout"
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
    tm = tech_module.Tech()

    if args.mag == '5x':
        PIX_PER_UM = PIX_PER_UM_5X
    elif args.mag == '10x':
        PIX_PER_UM = PIX_PER_UM_10X
    elif args.mag == '20x':
        PIX_PER_UM = PIX_PER_UM_20X

    tech = Schema('tech' / Path(args.tech))
    if not tech.read() or args.regenerate_lef:
        logging.info("Can't read db.json in tech directory; generating it automatically.")
        tech.scan()
        tech.overwrite()

    df = Path(args.def_file)

    def_json = Path(df.stem + '.json')
    if not def_json.is_file():
        schema = {
            'version': DEF_TO_PIXELS_VERSION,
            'cells' : {},
        }
        logging.info("building json from def...")
        build_json(schema, df)
    else:
        with open(def_json, 'r') as db_file:
            schema = json.loads(db_file.read())

    logging.info("gathering stats...")
    tm.gather_stats(schema, tech)

    logging.info("generating image...")
    # now generate a PNG of the cell map that we can use to manually overlay
    # on the stitched image to validate that our parsing makes sense.
    die_ll = schema['die_area_ll']
    die_ur = schema['die_area_ur']
    die = Rect(Point(die_ll[0], die_ll[1]), Point(die_ur[0], die_ur[1]))

    canvas = np.zeros((int(die.height() * PIX_PER_UM), int(die.width() * PIX_PER_UM), 3), dtype=np.uint8)
    pallette = HashPallette(tm)
    for cell, data in schema['cells'].items():
        color = pallette.str_to_rgb(data['cell'], data['orientation'])
        loc = data['loc']
        try:
            cell_size = tech.schema['cells'][data['cell']]['size']
        except:
            logging.warning(f"Cell not found: {data['cell']}")
            continue
        tl = (
            int(loc[0] * PIX_PER_UM),
            int((loc[1] + cell_size[1]) * PIX_PER_UM),
        )
        br = (
            int((loc[0] + cell_size[0]) * PIX_PER_UM),
            int(loc[1] * PIX_PER_UM),
        )
        cv2.rectangle(
            canvas,
            tl,
            br,
            color,
            thickness = -1,
        )
    cv2.imwrite(df.stem + '.png', canvas)
    logging.info("generating legend...")
    pallette.generate_legend(df.stem + '_legend.png')

if __name__ == "__main__":
    main()
