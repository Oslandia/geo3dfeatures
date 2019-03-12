"""Compute k-means clustering on 3D point cloud with geometric features
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

from sklearn.cluster import KMeans

from geo3dfeatures.io import xyz as read_xyz
from geo3dfeatures.features import normalize
from geo3dfeatures import FEATURE_SETS


SEED = 1337


def _parse_args(args):
    parser = argparse.ArgumentParser(description=("3D point cloud geometric"
                                                  " feature extraction"))
    parser.add_argument("-d", "--datapath", default="./data",
                        help="Data folder on the file system")
    parser.add_argument("-e", "--experiment", required=True,
                        help="Name of the feature extraction experiment")
    parser.add_argument('-f', '--feature-set', choices=FEATURE_SETS,
                        help="Set of computed features")
    parser.add_argument('-n', '--neighbors',
                        type=int, default=50,
                        help="Number of neighbors to consider")
    parser.add_argument('-p', '--sample-points',
                        type=int,
                        help="Number of sample points to evaluate")
    parser.add_argument("-k", "--nb-clusters", type=int, default=2,
                        help="Desired amount of clusters")
    return parser.parse_args(args)


def main(argv=sys.argv[1:]):
    opts = _parse_args(argv)
    instance = (
        str(opts.sample_points) + "-"
        + str(opts.neighbors) + "-"
        + opts.feature_set
    )
    filepath = Path(
        opts.datapath, "output", opts.experiment, "features",
        "features-" + instance + ".csv"
        )
    data = pd.read_csv(filepath)

    color_columns = ["r", "g", "b"]
    for c in color_columns:
        assert c in data.columns
        data[c] /= 255

    column_to_normalize = [
        "density", "density_2D", "bin_density",
        "eigenvalue_sum", "eigenvalue_sum_2D", "eigenentropy"
    ]
    for c in column_to_normalize:
        assert c in data.columns
        data[c] = normalize(data[c])

    assert "bin_z_range" in data.columns
    data["bin_z_range"].fillna(0, inplace=True)

    result = data[["x", "y", "z"]].copy()
    data.drop(["x", "y", "z"], axis=1, inplace=True)
    model = KMeans(opts.nb_clusters, random_state=SEED)
    model.fit(data)

    result["r"] = 20
    result["g"] = 200
    result["b"] = 20
    mask_floor = model.labels_ == 1
    result.loc[mask_floor, "r"] = 100
    result.loc[mask_floor, "g"] = 100
    result.loc[mask_floor, "b"] = 100

    output_path = Path(
        opts.datapath, "output", opts.experiment, "clustering",
    )
    os.makedirs(output_path, exist_ok=True)
    output_file = Path(output_path, "kmeans-" + instance + ".xyz")
    result.to_csv(str(output_file), sep=" ", index=False, header=False)

if __name__ == '__main__':
    main()
