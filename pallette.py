import numpy as np
import cv2
import logging

class HashPallette():
    def __init__(self, tech):
        self.tech = tech
        self.lut = {}
        self.hue_ranges = tech.hue_ranges()
        self.sat_ranges = tech.sat_ranges()
        self.next_color = {}
        for family, srange in self.sat_ranges.items():
            self.next_color[family] = {}
            for name, range in self.hue_ranges.items():
                if name != 'fill' and name != 'other':
                    self.next_color[family][name] = [range[0], srange[1], 255]
                else: # desaturate fill and other
                    self.next_color[family][name] = [range[0], 32, 32]

    # Map standard cells to colors.
    # Names are mapped into H/S space of HSV
    # Orientations are mapped into V-space
    def str_to_rgb(self, name, orientation):
        if name in self.lut:
            return self.lut[name]
        else:
            mn = self.tech.map_name_to_celltype(name)
            mf = self.tech.map_name_to_family(name)

            # now use the subtype to pick a color category
            hsv_color = self.next_color[mf][mn]
            # modulate the v based on the orientation
            if orientation == 'N':
                v = 255
            elif orientation == 'S':
                v = 255 - 16
            elif orientation == 'W':
                v = 255 - 32
            elif orientation == 'E':
                v = 255 - 48
            elif orientation == 'FN':
                v = 255 - 64
            elif orientation == 'FS':
                v = 255 - 80
            elif orientation == 'FW':
                v = 255 - 96
            elif orientation == 'FE':
                v = 255 - 128
            else:
                logging.error(f"unknown orientation: {orientation}")
                assert False # cause a crash
            hsv_color[2] = v

            # update to the next color in the series
            srange = self.sat_ranges[mf]
            next_h = self.next_color[mf][mn][0] + 1
            next_s = self.next_color[mf][mn][1]
            next_v = self.next_color[mf][mn][2]
            if next_h >= self.hue_ranges[mn][1]:
                next_h = self.hue_ranges[mn][0] # reset to beginning of range
                next_s -= 1 # decrement saturation by 1
                if mn != 'fill' and mn != 'other':
                    assert next_s >= srange[0], "ran out of unique H/S combos"
                else:
                    assert next_s >= 0, "ran out of unique H/S combos"
            self.next_color[mf][mn] = [next_h, next_s, next_v]
            # convert to RGB
            rgb_color = cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0]
            self.lut[name] = (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))
            return self.lut[name]

    # this path is much slower, but we also don't expect to use it very often at this point.
    # this is mostly just for checking
    def rgb_to_str(self, r, g, b):
        for name, (r_x, g_x, b_x) in self.lut.items():
            if r_x == r and g_x == g and b_x == b:
                return name
        return None