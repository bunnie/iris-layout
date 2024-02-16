import logging
from techbase import TechBase

class Tech(TechBase):
    def __init__(self, args):
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
        super().__init__(args)

    def is_ff(self, cell_name):
        return 'ff' in cell_name

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