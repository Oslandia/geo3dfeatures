"""Extract a subsample of a .las point cloud and save it into another .las file
"""

import argparse
from pathlib import Path
import sys

import laspy
import numpy as np


def _parse_args(args):
    parser = argparse.ArgumentParser(description=("3D point cloud sampling"))
    parser.add_argument("-d", "--datapath", default="./data",
                        help="Data folder on the file system")
    parser.add_argument("-i", "--input-file",
                        help="Input point cloud file")
    parser.add_argument('-p', '--sample-points',
                        type=int,
                        help="Number of sample points to evaluate")
    return parser.parse_args(args)


def main(argv=sys.argv[1:]):
    opts = _parse_args(argv)
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


if __name__ == "__main__":
    main()
