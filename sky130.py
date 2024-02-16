import logging
import re
from techbase import TechBase

class Tech(TechBase):
    def __init__(self):
        super().__init__()
        # cell type goes by hue
        self.hue_lut = {
            'ff' : [0, 10],
            'lat' : [11, 20],
            'clk' : [21, 30],
            'aoi' : [31, 40],
            'oai' : [41, 50],
            'add' : [51, 60],
            'buf' : [61, 70],
            'and' : [71, 80],
            'or'  : [81, 90],
            'inv' : [101, 110],
            'dly' : [111, 120],
            'mux' : [121, 130],
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

    # print some statistics -- just because it's interesting?
    def gather_stats(self, schema, tech):
        stats = {
            'fill' : 0,
            'antenna' : 0,
            'tap' : 0,
            'ff' : 0,
            'logic' : 0,
            'other' : 0,
        }
        stats_count = {
            'fill' : 0,
            'antenna' : 0,
            'tap' : 0,
            'ff' : 0,
            'logic' : 0,
            'other' : 0,
        }
        for cell, data in schema['cells'].items():
            if 'FILLER' in cell:
                try:
                    s = tech.schema['cells'][data['cell']]['size']
                    stats['fill'] += s[0] * s[1]
                    stats_count['fill'] += 1
                except:
                    pass
            elif 'ANTENNA' in cell:
                try:
                    s = tech.schema['cells'][data['cell']]['size']
                    stats['antenna'] += s[0] * s[1]
                    stats_count['antenna'] += 1
                except:
                    pass
            elif 'TAP' in cell:
                try:
                    s = tech.schema['cells'][data['cell']]['size']
                    stats['tap'] += s[0] * s[1]
                    stats_count['tap'] += 1
                except:
                    pass
            elif 'PHY' in cell:
                try:
                    s = tech.schema['cells'][data['cell']]['size']
                    stats['other'] += s[0] * s[1]
                    stats_count['other'] += 1
                except:
                    pass
            else:
                if '_df' in data['cell'] or ('_dl' in data['cell'] and not '_dly' in data['cell']):
                    try:
                        s = tech.schema['cells'][data['cell']]['size']
                        stats['ff'] += s[0] * s[1]
                        stats_count['ff'] += 1
                    except:
                        logging.info(f"cell not found {data['cell']}")
                else:
                    try:
                        s = tech.schema['cells'][data['cell']]['size']
                        stats['logic'] += s[0] * s[1]
                        stats_count['logic'] += 1
                    except:
                        logging.info(f"cell not found: {data['cell']}")

        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(stats)
        pp.pprint(stats_count)

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