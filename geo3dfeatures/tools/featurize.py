import os
import sys
import argparse
from pathlib import Path

import numpy as np

from geo3dfeatures.io import (
    xyz as read_xyz,
    las as read_las,
    ply as read_ply,
    )
from geo3dfeatures.extract import extract
from geo3dfeatures import FEATURE_SETS


def main(opts):
    input_path = Path(opts.datapath, "input", opts.input_file)
    if input_path.suffix == ".xyz":
        data = read_xyz(str(input_path))
    elif input_path.suffix == ".las":
        data = read_las(str(input_path))
    elif input_path.suffix == ".ply":
        data = read_ply(str(input_path))
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")

    if len(opts.input_columns) != data.shape[1]:
        raise ValueError("The given input columns does not match data shape.")

    if opts.sample_points is not None:
        sample_mask = np.random.choice(np.arange(data.shape[0]),
                                       size=opts.sample_points,
                                       replace=False)
        data = data[sample_mask]

    features = extract(
        data, opts.neighbors, opts.input_columns,
        opts.kdtree_leafs, opts.feature_set, opts.nb_process
    )

    experiment = (
        opts.experiment
        if opts.experiment is not None
        else opts.input_file.split(".")[0]
        )
    instance = (
        "features-" + str(len(data)) + "-" + str(opts.neighbors) + "-"
        + str(opts.feature_set) + "-" + str(opts.nb_process)
        )
    output_path = Path(opts.datapath, "output", experiment, "features")
    os.makedirs(output_path, exist_ok=True)
    output_file = Path(output_path, instance + ".csv")
    features.to_csv(output_file, index=False)
