import logging
from techbase import TechBase

class Tech(TechBase):
    def __init__(self):
        super().__init__()
        # cell types go by hue
        self.hue_lut = {
            'ff' : [0, 10],
            'lat' : [11, 20],
            'clk' : [21, 30],
            'add' : [51, 60],
            'aoi' : [31, 40],
            'oai' : [41, 50],
            'buf' : [61, 70],
            'and' : [71, 80],
            'or'  : [81, 90],
            'inv' : [101, 110],
            'dly' : [111, 120],
            'mux' : [121, 130],
            'fill' : [150, 169],
            'other' : [170, 180]
        }
        # cell family goes by saturation
        self.sat_lut = {
            'default' : [32, 255],
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
                if 'ff' in data['cell']:
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
        mn = 'other' # clamp all unknown names to "other"
        mapped_names = self.hue_lut.keys()
        for subtype in mapped_names:
            if subtype in cell_name:
                mn = subtype
                break
        return mn

    def map_name_to_family(self, cell_name):
        return 'default'