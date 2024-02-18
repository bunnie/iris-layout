import json
from prims import Rect, Point, ROUNDING
import logging
from scipy.spatial import distance
import numpy as np
import math
from pathlib import Path
import cv2
import re

class Schema():
    SCHEMA_VERSION = "1.0.0"

    def __init__(self, path):
        self.schema = {
            'version' : Schema.SCHEMA_VERSION,
            'cells' : {},
        }
        self.path = path

    def read(self):
        fullpath = self.path / Path('db.json')
        if not fullpath.is_file():
            # For some reason the FileNotFoundError is not being propagated
            # back to the caller, I'm having to do this weird thing. Probably
            # some import has...changed the behaviors of exceptions in a way
            # I don't expect and I don't know which one it was. Fucking Python.
            return False
        with open(self.path / Path('db.json'), 'r') as config:
            self.schema = json.loads(config.read())
            return True

    def overwrite(self):
        with open(self.path / Path('db.json'), 'w+') as config:
            config.write(json.dumps(self.schema, indent=2))

    def scan(self):
        files = [file for file in self.path.glob('*.lef') if file.is_file()]
        cell = {}
        name = None
        for file in files:
            if '.magic.lef' in str(file):
                continue
            state = 'FIND_MACRO'
            with open(file, 'r') as lef_file:
                for line in lef_file:
                    line = line.strip().lstrip()
                    if state == 'FIND_MACRO':
                        if 'MACRO' in line:
                            name = re.split('\s+', line)[1]
                            cell = {} # reset the cell properties, they should have been saved by now
                            state = 'MACRO'
                    elif state == 'MACRO':
                        tokens = re.split('\s+', line)
                        if 'FOREIGN' in line:
                            cell['foreign_name'] = tokens[1]
                            if len(tokens) > 3:
                                cell['foreign_origin'] = [float(tokens[2]), float(tokens[3])]
                            else:
                                cell['foreign_origin'] = [0.0, 0.0]
                        elif 'ORIGIN' in line:
                            cell['origin'] = [float(tokens[1]), float(tokens[2])]
                        elif line.startswith('SIZE'):
                            cell['size'] = [float(tokens[1]), float(tokens[3])]
                        if 'END' in line and len(tokens) > 1 and tokens[1] == name:
                            self.schema['cells'][name] = cell
                            state = 'FIND_MACRO'

# This is just for documentation purposes
sample_schema = {
    'version': Schema.SCHEMA_VERSION,
    'cells': [
        {
            'gf180mcu_fd_sc_mcu7t5v0_or2_1' : # standard cell macro name
            {
                'size' : [0.0, 0.0],  # size in microns of the standard cell
                'origin' : [0.0, 0.0], # origin offset in microns
                'foreign_origin' : [0.0, 0.0], # foreign origin
                'foreign_name' : 'gf180mcu_fd_sc_mcu7t5v0_or2_1',
            }
        },
        # more 'tile' objects
    ],
}

