import pprint
import logging
from pathlib import Path
from schema import Schema
from pallette import HashPallette
import hashlib

class TechBase():
    def __init__(self, args):
        self.stats = {
            'fill' : 0,
            'antenna' : 0,
            'tap' : 0,
            'ff' : 0,
            'logic' : 0,
            'other' : 0,
        }
        self.stats_count = {
            'fill' : 0,
            'antenna' : 0,
            'tap' : 0,
            'ff' : 0,
            'logic' : 0,
            'other' : 0,
        }
        self.tech_name = args.tech
        self.redact = args.redact
        self.tech = Schema('tech' / Path(self.tech_name))
        if not self.tech.read() or args.regenerate_lef:
            logging.info("Can't read db.json in tech directory; generating it automatically.")
            self.tech.scan()
            self.tech.overwrite()
        self.pallette = HashPallette(self)

    def hue_ranges(self):
        return self.hue_lut

    def sat_ranges(self):
        return self.sat_lut

    def gather_stats(self, design):
        for cell, data in design.schema['cells'].items():
            if 'FILLER' in cell:
                try:
                    s = self.tech.schema['cells'][data['cell']]['size']
                    self.stats['fill'] += s[0] * s[1]
                    self.stats_count['fill'] += 1
                except:
                    pass
            elif 'ANTENNA' in cell:
                try:
                    s = self.tech.schema['cells'][data['cell']]['size']
                    self.stats['antenna'] += s[0] * s[1]
                    self.stats_count['antenna'] += 1
                except:
                    pass
            elif 'TAP' in cell:
                try:
                    s = self.tech.schema['cells'][data['cell']]['size']
                    self.stats['tap'] += s[0] * s[1]
                    self.stats_count['tap'] += 1
                except:
                    pass
            elif 'PHY' in cell:
                try:
                    s = self.tech.schema['cells'][data['cell']]['size']
                    self.stats['other'] += s[0] * s[1]
                    self.stats_count['other'] += 1
                except:
                    pass
            else:
                ff_cnt = self.is_ff(data['cell'])
                if ff_cnt != 0:
                    try:
                        s = self.tech.schema['cells'][data['cell']]['size']
                        self.stats['ff'] += s[0] * s[1]
                        self.stats_count['ff'] += ff_cnt
                    except:
                        logging.debug(f"non-primitive cell: {data['cell']}")
                else:
                    # this covers the tsmc22ull style names. I think it doesn't conflict with the other base types?
                    ctype = self.map_name_to_celltype(data['cell'])
                    if ctype == 'fill' or ctype == 'other':
                        try:
                            s = self.tech.schema['cells'][data['cell']]['size']
                            self.stats[ctype] += s[0] * s[1]
                            self.stats_count[ctype] += 1
                        except:
                            logging.debug(f"non-primitive cell: {data['cell']}")
                    else:
                        try:
                            s = self.tech.schema['cells'][data['cell']]['size']
                            self.stats['logic'] += s[0] * s[1]
                            self.stats_count['logic'] += 1
                        except:
                            logging.debug(f"non-primitive cell: {data['cell']}")


    def print_stats(self):
            pp = pprint.PrettyPrinter(indent=2)
            logging.info("By area (drawn micron^2):")
            pp.pprint(self.stats)
            logging.info("By count:")
            pp.pprint(self.stats_count)

    def shorten_cellname(self, name):
        return name

    def redact_cellname(self, name):
        m = hashlib.sha256()
        m.update(name.encode('utf-8'))
        return m.hexdigest()[:20]

    def set_func_count(self, count):
        self.pallette.set_func_count(count)