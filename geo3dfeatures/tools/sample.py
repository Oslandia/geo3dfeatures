"""Extract a subsample of a .las point cloud and save it into another .las file
"""

import argparse
from pathlib import Path
import sys

import daiquiri
import laspy
import numpy as np


logger = daiquiri.getLogger(__name__)


def main(opts):
    logger.info(
        "Sample %s points from file %s...", opts.sample_points, opts.input_file
    )
    input_path = Path(opts.datapath, "input", opts.input_file)
    basename, ext = opts.input_file.split(".")
    output_file = basename + "-" + str(opts.sample_points) + "." + ext
    output_path = Path(opts.datapath, "input", output_file)
    input_lasfile = laspy.file.File(input_path, mode="r")
    output_lasfile = laspy.file.File(
        output_path, mode="w", header=input_lasfile.header
    )
    input_points = input_lasfile.points
    sample_mask = np.random.choice(
        np.arange(len(input_points)), size=opts.sample_points, replace=False
    )
    output_lasfile.points = input_points[sample_mask]
    input_lasfile.close()
    output_lasfile.close()
