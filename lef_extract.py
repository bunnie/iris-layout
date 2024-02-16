#! /usr/bin/env python3

# NOTE: this is deprecated, use `def_to_pixels --regenerate-lef` to perform this function

import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2
import sys

from schema import Schema

def main():
    parser = argparse.ArgumentParser(description="IRIS LEF primitive extraction - process LEF DB into a JSON file to reduce load times of other scripts")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--tech", required=True, help="Path to directory containing the LEF files"
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    schema = Schema(Path(args.tech))

    schema.scan()
    schema.overwrite()

if __name__ == "__main__":
    main()
