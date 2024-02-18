import logging
import json
import cv2
import numpy as np
import struct
import io
import string

from progressbar.bar import ProgressBar
from pathlib import Path

from prims import Rect, Point

# GDS parser based off of https://github.com/mikaeloduh/gds2ascii-tool-project
# Note: this repo has no explicit license, so I just used the general structure
# and facts and expressed it in my own style.

DEF_TO_PIXELS_VERSION = '1.0.0'

class Design():
    def def_to_json(self):
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
            with open(self.design_json, 'w+') as def_out:
                def_out.write(json.dumps(self.schema, indent=2))

    # note that we cant just pack into a 8 byte python double
    # because gdsii 8-byte real number has the form of
    # SEEEEEEE MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
    # that's different than the IEEE 754 binary64 double format that python "struct" uses
    def unpack_8byte_real(self, data):
        e = (data[0] & 0x7F) - 64
        s = (data[0] & 0x80) >> 7

        m = 0
        for i in range(7):
            m |= (data[i + 1] & 0xFF) << ((6 - i) * 8)

        d = m
        d = d * (16.0 ** (e - 14))
        d = -d if s == 1 else d
        return d

    def extract_data(self, record):
        datlen = self.gds_dat_size[record['dat_type']]
        if record['dat_type'] == 0x00:
            return []
        elif record['dat_type'] == 0x01:
            return record['data']
        elif record['dat_type'] == 0x02 or record['dat_type'] == 0x03:
            return [int.from_bytes(record['data'][pos:pos + datlen], 'big', signed=True)
                for pos in range(0, len(record['data']), datlen)
            ]
        elif record['dat_type'] == 0x04:
            return [struct.unpack('>f', record['data'][pos:pos + datlen])[0]
                for pos in range(0, len(record['data']), datlen)
            ]
        elif record['dat_type'] == 0x05:
            return [self.unpack_8byte_real(record['data'][pos:pos + datlen])
                for pos in range(0, len(record['data']), datlen)
            ]
        elif record['dat_type'] == 0x06:
            non_null = bytes(filter(lambda val: val !=0, record['data']))
            return [non_null.decode("utf-8", errors='ignore')]
        else:
            logging.error(f"Illegal record type in GDS stream: {record['dat_type']}")
            assert False

    def gds_to_json(self):
        with open(self.df, 'rb') as gds_file:
            gds_len = gds_file.seek(0, io.SEEK_END)
            gds_file.seek(0)
            do_progress = gds_len > 1_000_000
            if do_progress:
                progress = ProgressBar(min_value= 0, max_value= gds_len, prefix = f"Importing {self.df.name} GDS...")
            structure = {
                'orientation': 'N' # default to 'N' in case an angle isn't specified
            }
            while True:
                # break the stream down into records
                if do_progress:
                    pos = gds_file.tell()
                    progress.update(pos)
                rec_size = int.from_bytes(gds_file.read(2), byteorder='big')
                rec_type = int.from_bytes(gds_file.read(1), byteorder='big')
                dat_type = int.from_bytes(gds_file.read(1), byteorder='big')
                rec_data = gds_file.read(rec_size - 4)
                record = {
                    'size': rec_size,
                    'rec_type': rec_type,
                    'dat_type': dat_type,
                    'data': rec_data
                }

                # extract the data from a record
                data = self.extract_data(record)
                rec_name = self.gds_name_lut[record['rec_type']]
                if rec_name == 'LIBNAME':
                    self.schema['name'] = data[0]
                elif rec_name == 'UNITS':
                    self.schema['units'] = data[0] / data[1]
                elif rec_name == 'LAYER':
                    structure['layer'] = data[0]
                elif rec_name == 'SREF':
                    structure = {
                        'orientation': 'N'
                    }
                elif rec_name == 'ENDEL':
                    if 'name' in structure:
                        if 'reflect' in structure and structure['reflect']:
                            # apply reflection: GDS reflects on the *X* axis *before* rotation.
                            # LEF/DEF reflects on the *Y* axis *after* rotation. Go figure. :-/
                            if structure['orientation'] == 'N':
                                structure['orientation'] = 'FS'
                            elif structure['orientation'] == 'W':
                                structure['orientation'] = 'FW'
                            elif structure['orientation'] == 'E':
                                structure['orientation'] = 'FE'
                            elif structure['orientation'] == 'S':
                                structure['orientation'] = 'FN'
                            else:
                                logging.error("Bad internal orientation")
                                assert False
                        self.schema['cells'][structure['name']] = {
                            'cell': structure['cell'],
                            'loc': [structure['x'][0], structure['y'][0]],
                            'orientation': structure['orientation'],
                        }
                    elif 'layer' in structure:
                        if structure['layer'] == 235: # object boundary
                            self.schema['die_area_ll'] = [min(structure['x']), min(structure['y'])]
                            self.schema['die_area_ur'] = [max(structure['x']), max(structure['y'])]
                    else:
                        pass # not a record of concern
                    structure = {
                        'orientation': 'N'
                    }
                elif rec_name == 'SNAME':
                    structure['cell'] = data[0]
                elif rec_name == 'PROPVALUE':
                    structure['name'] = data[0]
                elif rec_name == 'STRANS':
                    assert data[1] == 0, "STRANS magnification or other parameter not handled!"
                    if data[0] == 0:
                        structure['reflect'] = False
                    elif data[0] == 128:
                        structure['reflect'] = True
                    else:
                        logging.error("Malformed STRANS record")
                        assert False
                elif rec_name == 'ANGLE':
                    if int(data[0]) == 0:
                        structure['orientation'] = 'N'
                    elif int(data[0]) == 90:
                        structure['orientation'] = 'W'
                    elif int(data[0]) == 180:
                        structure['orientation'] = 'S'
                    elif int(data[0]) == 270:
                        structure['orientation'] = 'E'
                    else:
                        logging.warning(f"unhandled orientation angle: {data[0]}")
                elif rec_name == 'XY':
                    structure['x'] = [(data[i] / self.schema['units']) * 1000 for i in range(len(data)) if i % 2 == 0]
                    structure['y'] = [(data[i] / self.schema['units']) * 1000 for i in range(len(data)) if i % 2 != 0]
                if record['rec_type'] == 0x04: # ENDLIB
                    break
            progress.finish()

            logging.info("Saving to json (may take a while)...")
            with open(self.design_json, 'w+') as def_out:
                def_out.write(json.dumps(self.schema, indent=2))

    def __init__(self, file_path, pix_per_um):
        self.pix_per_um = pix_per_um
        self.df = Path(file_path)
        self.extension = self.df.suffix
        self.design_path = Path(file_path).parent
        self.name = self.df.stem
        self.gds_dat_size = {0x00: 1, 0x01: 1, 0x02: 2, 0x03: 4, 0x04: 4, 0x05: 8, 0x06: 1}
        self.gds_name_lut = {
            0x00 : 'HEADER',
            0x01 : 'BGNLIB',
            0x02 : 'LIBNAME',
            0x03 : 'UNITS',
            0x04 : 'ENDLIB',
            0x05 : 'BGNSTR',
            0x06 : 'STRNAME',
            0x07 : 'ENDSTR',
            0x08 : 'BOUNDARY',
            0x09 : 'PATH',
            0x0A : 'SREF',
            0x0B : 'AREF',
            0x0C : 'TEXT',
            0x0D : 'LAYER',
            0x0E : 'DATATYPE',
            0x0F : 'WIDTH',
            0x10 : 'XY',
            0x11 : 'ENDEL',
            0x12 : 'SNAME',
            0x13 : 'COLROW',
            0x15 : 'NODE',
            0x16 : 'TEXTTYPE',
            0x17 : 'PRESENTATION',
            0x19 : 'STRING',
            0x1A : 'STRANS',
            0x1B : 'MAG',
            0x1C : 'ANGLE',
            0x1F : 'REFLIBS',
            0x20 : 'FONTS',
            0x21 : 'PATHTYPE',
            0x22 : 'GENERATIONS',
            0x23 : 'ATTRATABLE',
            0x26 : 'ELFLAGS',
            0x2A : 'NODETYPE',
            0x2B : 'PROPATTR',
            0x2C : 'PROPVALUE',
            0x2D : 'BOX',
            0x2E : 'BOXTYPE',
            0x2F : 'PLEX',
            0x32 : 'TAPENUM',
            0x33 : 'TAPECODE',
            0x36 : 'FORMAT',
            0x37 : 'MASK',
            0x38 : 'ENDMASKS'
        }
        self.design_json = self.df.with_name(self.df.stem + '.json')
        if not self.design_json.is_file():
            self.schema = {
                'version': DEF_TO_PIXELS_VERSION,
                'cells' : {},
            }
            if self.extension == '.def':
                logging.info("building json from def...")
                self.def_to_json()
            elif self.extension == '.gds':
                logging.info("building json from gds...")
                self.gds_to_json()
            else:
                logging.error(f"Unhandled design extension {self.extension}")
                assert False
        else:
            with open(self.design_json, 'r') as db_file:
                self.schema = json.loads(db_file.read())

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
        tech.pallette.generate_legend(str(self.df.with_name(self.df.stem)))

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
                    d = Design(gds_file, self.pix_per_um)
                else:
                    logging.warning(f"Couldn't find design file for {missing_cell['cell']}")
                    continue
            tm.gather_stats(d)
            next_missing = d.render_layer(tm)
            if len(next_missing) > 0:
                d.generate_missing(next_missing, tm)
            self.merge_subdesign(d, missing_cell)
