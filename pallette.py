import numpy as np
import cv2
import logging
from math import ceil, floor, sqrt

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

    def generate_legend(self, fname):
        font_scale = 1.0
        thickness = 1
        font_face = cv2.FONT_HERSHEY_PLAIN
        k = sorted(self.lut.keys())
        longest_name = ''
        for n in k:
            if len(n) > len(longest_name):
                longest_name = n
        ((w, th), baseline) = cv2.getTextSize(longest_name, font_face, font_scale, thickness)
        v_spacing = int(th + baseline)
        h_spacing = int(w * 1.15)
        single_col_height = (v_spacing * (len(k) + 2))
        desired_ratio = 16/9
        cols = ceil(sqrt(single_col_height / (desired_ratio * h_spacing)))
        wrap_height = floor(single_col_height / cols)
        canvas = np.zeros((wrap_height + v_spacing, cols * h_spacing, 3))
        y = v_spacing
        x = 0
        for n in k:
            color = self.lut[n]
            cv2.rectangle(
                canvas,
                (x + 5,y),
                (x + 50, y+th),
                color,
                thickness = -1,
                lineType = cv2.LINE_4
            )
            cv2.putText(
                canvas,
                n,
                (x + 65, y+th),
                font_face,
                font_scale,
                (255, 255, 255),
                thickness,
                bottomLeftOrigin=False
            )
            y += v_spacing
            if y > wrap_height:
                x += h_spacing
                y = v_spacing
        cv2.imwrite(fname, canvas)