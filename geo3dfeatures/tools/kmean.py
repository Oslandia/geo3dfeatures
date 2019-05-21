"""Compute k-means clustering on 3D point cloud with geometric features
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from geo3dfeatures.features import normalize


SEED = 1337
MAX_COLOR = 10
PALETTE = sns.color_palette("colorblind", MAX_COLOR)  # deep, muted, dark, ...


def main(opts):
    if opts.neighbors is not None:
        neighborhood = "n" + str(opts.neighbors)
    elif opts.radius is not None:
        neighborhood = "r" + str(opts.radius)
    else:
        raise ValueError(
            "Error in input neighborhood definition: "
            "neighbors and radius arguments can't be both undefined"
            )
    instance = (
        neighborhood + "-" + opts.feature_set
        + "-binsize-" + str(opts.bin_size)
    )
    filepath = Path(
        opts.datapath, "output", opts.experiment, "features",
        "features-" + instance + ".csv"
        )
    data = pd.read_csv(filepath)

    color_columns = ["r", "g", "b"]
    for c in color_columns:
        if c in data.columns:
            data[c] /= 255

    column_to_normalize = [
        "density", "density_2D", "bin_density",
        "eigenvalue_sum", "eigenvalue_sum_2D", "eigenentropy"
    ]
    for c in column_to_normalize:
        if c in data.columns:
            data[c] = normalize(data[c])

    if "bin_z_range" in data.columns:
        data["bin_z_range"].fillna(0, inplace=True)

    result = data[["x", "y", "z"]].copy()
    data.drop(["x", "y", "z"], axis=1, inplace=True)
    model = KMeans(opts.nb_clusters, random_state=SEED)
    model.fit(data)

    colors = np.array([PALETTE[l] for l in model.labels_]) * 255
    colors = pd.DataFrame(colors, columns=["r", "g", "b"], dtype=int)
    result = result.join(colors)

    output_path = Path(
        opts.datapath, "output", opts.experiment, "clustering",
    )
    os.makedirs(output_path, exist_ok=True)
    output_file = Path(
        output_path,
        "kmeans-" + instance + "-" + str(opts.nb_clusters) + ".xyz"
    )
    result.to_csv(str(output_file), sep=" ", index=False, header=False)
