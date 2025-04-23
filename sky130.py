import logging
import re
import pprint

from techbase import TechBase

class Tech(TechBase):
    def __init__(self, args):
        # cell type goes by hue
        self.hue_lut = {
            'ff' : [0, 10],
            'lat' : [11, 20],
            'mux' : [21, 30],
            'aoi' : [31, 40],
            'oai' : [41, 50],
            'add' : [51, 60],
            'buf' : [61, 70],
            'and' : [71, 80],
            'or'  : [81, 90],
            'inv' : [101, 110],
            'dly' : [111, 120],
            'clk' : [121, 130],
            'xor' : [131, 140],
            'fill' : [150, 169],
            'other' : [170, 180]
        }
        # cell family goes by saturation
        self.sat_lut = {
            'hs' : [228, 255],
            'ms' : [200, 227],
            'hd' : [172, 199],
            'ls' : [144, 171],
            'lp' : [116, 143],
            'hvl' : [88, 115],
            'hdll' : [60, 87],
            'other' : [32, 59],
        }
        super().__init__(args)

    def is_ff(self, cell_name):
        if '_df' in cell_name or ('_dl' in cell_name and not '_dly' in cell_name):
            return 1
        else:
            return 0

    def map_name_to_celltype(self, cell_name):
        cell_name = cell_name.lower()
        cell_match = re.search('__(.*)', cell_name)
        if cell_match is None:
            return 'other'
        nm = cell_match.group(1)

        if nm.startswith('xor') or nm.startswith('xnor'):
            return 'xor'
        elif nm.startswith('sed') or nm.startswith('sd') or nm.startswith('df') or nm.startswith('edf'):
            return 'ff'
        elif nm.startswith('dly'):
            return 'dly'
        elif nm.startswith('dl'): # this must be after 'dly'
            return 'lat'
        elif nm.startswith('or') or nm.startswith('nor'):
            return 'or'
        elif nm.startswith('and') or nm.startswith('nand'):
            return 'and'
        elif nm.startswith('mux'):
            return 'mux'
        elif nm.startswith('inv') or nm.startswith('einv'):
            return 'inv'
        elif nm.startswith('buf') or nm.startswith('ebuf'):
            return 'buf'
        elif nm.startswith('fa'):
            return 'add'
        elif nm.startswith('mux'):
            return 'mux'
        elif nm.startswith('clk'):
            return 'clk'
        elif nm.startswith('a'): # this is last in the if/else chain so more specific patterns evaluate first
            return 'aoi'
        elif nm.startswith('o'): # this is last in the if/else chain so more specific patterns evaluate first
            return 'oai'
        elif nm.startswith('decap'):
            return 'fill'
        else:
            return 'other'

    def map_name_to_family(self, cell_name):
        cell_name = cell_name.lower()
        match = re.search('sky130_fd_sc_([a-z]*)_(.*)', cell_name)
        if match:
            if match.group(1) in self.sat_lut.keys():
                return match.group(1)
            else:
                return 'other'
        else:
            return 'other'

    def shorten_cellname(self, name):
        return name.replace('sky130_fd_', '').replace('sc_', '')