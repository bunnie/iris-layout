import logging
import json
import cv2
import numpy as np
from progressbar.bar import ProgressBar
from pathlib import Path

from prims import Rect, Point

DEF_TO_PIXELS_VERSION = '1.0.0'

class Design():
    def build_json(self):
        with open(self.df, 'r') as def_file:
            total_lines = 0
            for line in def_file:
                total_lines += 1
        with open(self.df, 'r') as def_file:
            state = 'HEADER'
            progress = None
            processed_lines = 0
            for line in def_file:
                processed_lines += 1
                if progress:
                    progress.update(processed_lines)
                line = line.strip().lstrip()
                tokens = line.split(' ')
                if state == 'HEADER':
                    if tokens[0] == 'DESIGN':
                        self.schema['name'] = tokens[1]
                        progress = ProgressBar(min_value = 0, max_value = total_lines, prefix=f"Extracting {self.schema['name']} DEF...")
                    elif tokens[0] == 'UNITS':
                        self.schema['units'] = float(tokens[3])
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
                                coord += [float(token) / self.schema['units']]
                                da_state = 'Y'
                            elif da_state == 'Y':
                                coord += [float(token) / self.schema['units']]
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
                        self.schema['die_area_ll'] = [min_x, min_y]
                        self.schema['die_area_ur'] = [max_x, max_y]
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
                                if token == 'PLACED' or token == 'FIXED':
                                    comp_state = 'PLACED'
                                elif token == 'SOURCE':
                                    comp_state = 'SOURCE'
                                elif token == ';':
                                    if not skip:
                                        self.schema['cells'][name] = {
                                            'cell': cell,
                                            'loc' : [x, y],
                                            'orientation' : orientation
                                        }
                                    comp_state = 'END'
                            elif comp_state == 'PLACED':
                                assert token == '('
                                comp_state = 'PLACED_X'
                            elif comp_state == 'PLACED_X':
                                x = float(token) / self.schema['units']
                                comp_state = 'PLACED_Y'
                            elif comp_state == 'PLACED_Y':
                                y = float(token) / self.schema['units']
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
            if progress:
                progress.finish()

            logging.info("Saving to json (may take a while)...")
            with open(self.def_json, 'w+') as def_out:
                def_out.write(json.dumps(self.schema, indent=2))

    def __init__(self, file_path, pix_per_um):
        self.pix_per_um = pix_per_um
        self.df = Path(file_path)
        self.extension = self.df.suffix
        self.design_path = Path(file_path).parent
        self.name = self.df.stem

        if self.extension == 'def':
            self.def_json = self.df.with_name(self.df.stem + '.json')
            if not self.def_json.is_file():
                self.schema = {
                    'version': DEF_TO_PIXELS_VERSION,
                    'cells' : {},
                }
                logging.info("building json from def...")
                self.build_json()
            else:
                with open(self.def_json, 'r') as db_file:
                    self.schema = json.loads(db_file.read())
        elif self.extension == 'gds':
            pass
        else:
            logging.error(f"Unhandled design extension {self.extension}")
            assert False

        die_ll = self.schema['die_area_ll']
        die_ur = self.schema['die_area_ur']
        die = Rect(Point(die_ll[0], die_ll[1]), Point(die_ur[0], die_ur[1]))
        self.canvas = np.zeros((int(die.height() * self.pix_per_um), int(die.width() * self.pix_per_um), 3), dtype=np.uint8)

    def render_layer(self, tech):
        do_progress = len(self.schema['cells'].keys()) > 1000
        if do_progress:
            progress = ProgressBar(min_value = 0, max_value=len(self.schema['cells'].keys()), prefix='Rendering layout...')
        count = 0
        missing_cells = []
        for cell, data in self.schema['cells'].items():
            count += 1
            if do_progress:
                progress.update(count)
            color = tech.pallette.str_to_rgb(data['cell'], data['orientation'])
            loc = data['loc']
            try:
                cell_size = tech.tech.schema['cells'][data['cell']]['size']
            except:
                missing_cells += [data]
                continue
            tl = (
                int(loc[0] * self.pix_per_um),
                int((loc[1] + cell_size[1]) * self.pix_per_um),
            )
            br = (
                int((loc[0] + cell_size[0]) * self.pix_per_um),
                int(loc[1] * self.pix_per_um),
            )
            cv2.rectangle(
                self.canvas,
                tl,
                br,
                color,
                thickness = -1,
            )
        if do_progress:
            progress.finish()
        return missing_cells

    def generate_legend(self, tech):
        tech.pallette.generate_legend(str(self.df.with_name(self.df.stem + '_legend.png')))

    def save_layout(self):
        cv2.imwrite(str(self.df.with_name(self.df.stem + '.png')), self.canvas)

    def get_layout(self, orientation='N'):
        if orientation == 'N':
            return self.canvas
        elif orientation == 'S':
            return cv2.rotate(self.canvas, cv2.ROTATE_180)
        elif orientation == 'W':
            return cv2.rotate(self.canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == 'E':
            return cv2.rotate(self.canvas, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 'FN':
            return cv2.flip(self.canvas, 1)
        elif orientation == 'FS':
            return cv2.flip(cv2.rotate(self.canvas, cv2.ROTATE_180), 1)
        elif orientation == 'FW':
            return cv2.flip(cv2.rotate(self.canvas, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
        elif orientation == 'FE':
            return cv2.flip(cv2.rotate(self.canvas, cv2.ROTATE_90_CLOCKWISE), 1)
        else:
            logging.error(f"unknown orientation: {orientation}")
            assert False # cause a crash

    def merge_subdesign(self, d, info):
        img = d.get_layout(orientation=info['orientation'])
        if False:
            self.canvas[
                int(info['loc'][1] * self.pix_per_um) : int(info['loc'][1] * self.pix_per_um) + img.shape[0],
                int(info['loc'][0] * self.pix_per_um) : int(info['loc'][0] * self.pix_per_um) + img.shape[1],
                :
            ] = img
        else:
            try:
                merged = cv2.addWeighted(self.canvas[
                    int(info['loc'][1] * self.pix_per_um) : int(info['loc'][1] * self.pix_per_um) + img.shape[0],
                    int(info['loc'][0] * self.pix_per_um) : int(info['loc'][0] * self.pix_per_um) + img.shape[1],
                    :
                ], 1.0, img, 1.0, 0.0)
                self.canvas[
                    int(info['loc'][1] * self.pix_per_um) : int(info['loc'][1] * self.pix_per_um) + img.shape[0],
                    int(info['loc'][0] * self.pix_per_um) : int(info['loc'][0] * self.pix_per_um) + img.shape[1],
                    :
                ] = merged
            except:
                logging.warning(f"Couldn't merge sub-block {d.name} into {self.name}")

    def generate_missing(self, missing_cells, tm):
        for missing_cell in missing_cells:
            def_file = self.design_path / (missing_cell['cell'] + '.def')
            if def_file.exists(): # prefer DEF over GDS
                d = Design(def_file, self.pix_per_um)
            else:
                gds_file = self.design_path / (missing_cell['cell'] + '.gds')
                if gds_file.exists():
                    d = Design(def_file, self.pix_per_um)
                else:
                    logging.warning(f"Couldn't find design file for {missing_cell['cell']}")
                    continue
            tm.gather_stats(d)
            next_missing = d.render_layer(tm)
            if len(next_missing) > 0:
                d.generate_missing(next_missing, tm)
            self.merge_subdesign(d, missing_cell)
