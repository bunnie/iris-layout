import numpy as np
import cv2
import logging
import json
from math import ceil, floor, sqrt

class HashPallette():
    def __init__(self, tech):
        self.tech = tech
        self.lut = {} # this stores RGB values
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

        self.function_lut = {} # this stores HSV values for $reasons
        self.function_cur_hsv = (0, 255, 255)
        self.func_region_count = 0
        self.h_inc = 1

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

    def save(self, fname):
        save_struct = {
            'lut': self.lut,
            'hue_ranges': self.hue_ranges,
            "sat_ranges": self.sat_ranges,
            "next_color": self.next_color,
            "function_lut" : self.function_lut,
            "f_cur_hsv": self.function_cur_hsv,
        }
        with open(fname, 'w') as f:
            f.write(json.dumps(save_struct, indent=2))

    # this path is much slower, but we also don't expect to use it very often at this point.
    # this is mostly just for checking
    def rgb_to_str(self, r, g, b):
        for name, (r_x, g_x, b_x) in self.lut.items():
            if r_x == r and g_x == g and b_x == b:
                return name
        return None

    def set_func_count(self, f):
        self.func_region_count = f
        h_inc = floor(180 / self.func_region_count)
        if floor(180 / self.func_region_count) > 0:
            self.h_inc = h_inc
        else:
            self.h_inc = 1

    def function_to_rgb(self, func):
        if func not in self.function_lut:
            self.function_lut[func] = self.function_cur_hsv
            (h, s, v) = self.function_cur_hsv
            # compute the next color
            h = h + self.h_inc
            if h >= 180:
                logging.warning("Function mapping ran out of color space, wrapping colors!")
                h = 0
                v = v - 32
                if v < 0:
                    logging.warning("Wrapping v around, you might want to rethink the granularity of the function mapping!")
                    v = 255
            # alternate s-values to make more perceptual distance between adjacent hues
            if s == 255:
                s = 85
            elif s == 85:
                s = 170
            else:
                s = 255
            self.function_cur_hsv = (h, s, v)

        rgb_color = cv2.cvtColor(np.array([[self.function_lut[func]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0]
        return (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))

    # key_names == None implies rendering of the function lut
    def _generate_core(self, fname, display_names, key_names=None):
        font_scale = 1.0
        thickness = 1
        font_face = cv2.FONT_HERSHEY_PLAIN

        longest_name = ''
        for n in display_names:
            if len(n) > len(longest_name):
                longest_name = n
        ((w, th), baseline) = cv2.getTextSize(longest_name, font_face, font_scale, thickness)
        v_spacing = int(th + baseline)
        h_spacing = int(w * 1.15)
        single_col_height = (v_spacing * (len(display_names) + 2))
        desired_ratio = 16/9
        cols = ceil(sqrt(single_col_height / (desired_ratio * h_spacing)))
        wrap_height = floor(single_col_height / cols)
        canvas = np.zeros((wrap_height + v_spacing, cols * h_spacing, 3))
        y = v_spacing
        x = 0
        for (i, n) in enumerate(display_names):
            if key_names is not None:
                color = self.lut[key_names[i]]
            else:
                rgb_color = cv2.cvtColor(np.array([[self.function_lut[display_names[i]]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0]
                color = (float(rgb_color[0]), float(rgb_color[1]), float(rgb_color[2]))
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

    def generate_legend(self, fname_stem):
        long_k = sorted(self.lut.keys())
        sk = []
        rk = []
        for lk in long_k:
            sk += [self.tech.shorten_cellname(lk)]
            rk += [self.tech.redact_cellname(lk)]
        fk = []
        for func_k in sorted(self.function_lut.keys()):
            if func_k != 'top':
                fk += [func_k]

        self._generate_core(fname_stem + '_legend.png', sk, long_k)
        self._generate_core(fname_stem + '_redacted_legend.png', rk, long_k)
        self._generate_core(fname_stem + '_function_legend.png', fk, None)